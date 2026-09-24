"""Prepare the prespecified 10-per-subfield SuperGPQA study sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cpis.datasets import DatasetItem, PARTITIONS, Partition

DIFFICULTIES = ("easy", "middle", "hard")


def _hash_key(*values: str) -> bytes:
    return hashlib.sha256("\0".join(values).encode("utf-8")).digest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    required = {
        "uuid",
        "question",
        "options",
        "answer",
        "answer_letter",
        "discipline",
        "field",
        "subfield",
        "difficulty",
        "is_calculation",
    }
    if any(set(row) != required for row in rows):
        raise ValueError("SuperGPQA source schema differs from the pinned contract")
    if len(rows) != 26_529 or len({row["uuid"] for row in rows}) != len(rows):
        raise ValueError("expected 26,529 unique SuperGPQA source rows")
    for row in rows:
        answer_index = ord(row["answer_letter"]) - ord("A")
        if answer_index < 0 or answer_index >= len(row["options"]):
            raise ValueError(f"invalid answer letter for {row['uuid']}")
        if row["options"][answer_index] != row["answer"]:
            raise ValueError(f"answer text mismatch for {row['uuid']}")
        if row["difficulty"] not in DIFFICULTIES:
            raise ValueError(f"unknown difficulty for {row['uuid']}")
    return rows


def balanced_subfield_sample(
    rows: list[dict[str, Any]], per_subfield: int, salt: str
) -> list[dict[str, Any]]:
    by_subfield: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subfield[row["subfield"]].append(row)
    if len(by_subfield) != 285:
        raise ValueError(f"expected 285 subfields, got {len(by_subfield)}")

    selected: list[dict[str, Any]] = []
    for subfield, candidates in sorted(by_subfield.items()):
        buckets: dict[str, list[dict[str, Any]]] = {}
        for difficulty in DIFFICULTIES:
            bucket = [row for row in candidates if row["difficulty"] == difficulty]
            buckets[difficulty] = sorted(
                bucket,
                key=lambda row: (_hash_key(salt, "select", row["uuid"]), row["uuid"]),
            )
        counts = Counter({difficulty: 0 for difficulty in DIFFICULTIES})
        for draw in range(per_subfield):
            available = [difficulty for difficulty in DIFFICULTIES if buckets[difficulty]]
            if not available:
                raise ValueError(f"subfield {subfield!r} has fewer than {per_subfield} rows")
            minimum = min(counts[difficulty] for difficulty in available)
            tied = [difficulty for difficulty in available if counts[difficulty] == minimum]
            difficulty = min(
                tied,
                key=lambda value: _hash_key(
                    salt, "difficulty", subfield, str(draw), value
                ),
            )
            selected.append(buckets[difficulty].pop(0))
            counts[difficulty] += 1
    return selected


def assign_stratified_partitions(
    rows: list[dict[str, Any]], salt: str
) -> dict[str, Partition]:
    by_subfield: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subfield[row["subfield"]].append(row)
    assignments: dict[str, Partition] = {}
    capacities: dict[Partition, int] = {
        "development": 4,
        "certification": 3,
        "test": 3,
    }
    for subfield, candidates in sorted(by_subfield.items()):
        if len(candidates) != 10:
            raise ValueError(f"prepared subfield {subfield!r} must contain exactly 10 rows")
        total_counts: Counter[str] = Counter()
        difficulty_counts: dict[str, Counter[str]] = {
            difficulty: Counter() for difficulty in DIFFICULTIES
        }
        ordered = sorted(
            candidates,
            key=lambda row: (
                _hash_key(salt, "partition-order", row["uuid"]),
                row["uuid"],
            ),
        )
        for row in ordered:
            available = [
                partition
                for partition in PARTITIONS
                if total_counts[partition] < capacities[partition]
            ]
            partition = min(
                available,
                key=lambda value: (
                    difficulty_counts[row["difficulty"]][value],
                    total_counts[value] / capacities[value],
                    _hash_key(salt, "partition-tie", row["uuid"], value),
                ),
            )
            assignments[row["uuid"]] = partition
            total_counts[partition] += 1
            difficulty_counts[row["difficulty"]][partition] += 1
    return assignments


def _format_question(row: dict[str, Any]) -> str:
    options = "\n".join(
        f"{chr(ord('A') + index)}. {option}"
        for index, option in enumerate(row["options"])
    )
    return f"{row['question']}\n\nOptions:\n{options}"


def _canonical_lines(items: list[DatasetItem]) -> bytes:
    return (
        "\n".join(
            json.dumps(
                item.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            for item in sorted(items, key=lambda value: value.item_id)
        )
        + "\n"
    ).encode("utf-8")


def _write_deterministic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"refusing to replace non-identical artifact: {path}")
        return
    path.write_bytes(data)


def prepare(
    source: Path,
    output: Path,
    evidence: Path,
    source_revision: str,
    expected_source_sha256: str,
    salt: str,
    partition_output_directory: Path | None = None,
) -> dict[str, Any]:
    source_bytes = source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != expected_source_sha256:
        raise ValueError(
            f"source SHA-256 mismatch: expected {expected_source_sha256}, got {source_sha256}"
        )
    rows = _read_rows(source)
    sampled = balanced_subfield_sample(rows, per_subfield=10, salt=salt)
    assignments = assign_stratified_partitions(sampled, salt=salt)
    items = [
        DatasetItem(
            item_id=row["uuid"],
            question=_format_question(row),
            answer=row["answer_letter"],
            success_criterion=(
                "The proposed answer is exactly the letter of the correct option."
            ),
            partition=assignments[row["uuid"]],
            metadata={
                "discipline": row["discipline"],
                "field": row["field"],
                "subfield": row["subfield"],
                "difficulty": row["difficulty"],
                "is_calculation": row["is_calculation"],
                "option_count": len(row["options"]),
            },
        )
        for row in sampled
    ]
    output_bytes = _canonical_lines(items)
    output_sha256 = hashlib.sha256(output_bytes).hexdigest()
    _write_deterministic(output, output_bytes)

    partition_outputs: dict[str, dict[str, Any]] = {}
    if partition_output_directory is not None:
        for partition in PARTITIONS:
            partition_bytes = _canonical_lines(
                [item for item in items if item.partition == partition]
            )
            partition_path = partition_output_directory / f"items-{partition}.jsonl"
            _write_deterministic(partition_path, partition_bytes)
            partition_outputs[partition] = {
                "path": partition_path.name,
                "rows": sum(item.partition == partition for item in items),
                "sha256": hashlib.sha256(partition_bytes).hexdigest(),
            }

    partition_counts = Counter(item.partition for item in items)
    difficulty_counts = Counter(item.metadata["difficulty"] for item in items)
    discipline_counts = Counter(item.metadata["discipline"] for item in items)
    manifest = {
        "schema_version": "2.0" if partition_output_directory else "1.0",
        "dataset_id": "m-a-p/SuperGPQA",
        "source_revision": source_revision,
        "source_sha256": source_sha256,
        "source_rows": len(rows),
        "algorithm": "balanced-10-per-subfield-stratified-4-3-3-v1",
        "salt": salt,
        "selected_rows": len(items),
        "subfields": len({item.metadata["subfield"] for item in items}),
        "partition_counts": dict(sorted(partition_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "discipline_counts": dict(sorted(discipline_counts.items())),
        "output_sha256": output_sha256,
        "output_format": "cpis DatasetItem JSONL v1",
    }
    if partition_output_directory is not None:
        manifest["partition_outputs"] = partition_outputs
    evidence_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _write_deterministic(evidence, evidence_bytes)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--salt", required=True)
    parser.add_argument("--partition-output-directory", type=Path)
    args = parser.parse_args()
    manifest = prepare(
        source=args.source,
        output=args.output,
        evidence=args.evidence,
        source_revision=args.source_revision,
        expected_source_sha256=args.source_sha256,
        salt=args.salt,
        partition_output_directory=args.partition_output_directory,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
