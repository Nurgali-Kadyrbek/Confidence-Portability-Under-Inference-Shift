"""Prepare frozen internal splits of the pinned MedXpertQA Text test set."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cpis.datasets import DatasetItem, PARTITIONS, Partition

SOURCE_FIELDS = {
    "id",
    "question",
    "options",
    "label",
    "medical_task",
    "body_system",
    "question_type",
}
FRACTIONS: dict[Partition, float] = {
    "development": 0.4,
    "certification": 0.3,
    "test": 0.3,
}


def _hash_key(*values: str) -> bytes:
    return hashlib.sha256("\0".join(values).encode("utf-8")).digest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 2_450:
        raise ValueError(f"expected 2,450 MedXpertQA Text test rows, got {len(rows)}")
    if any(set(row) != SOURCE_FIELDS for row in rows):
        raise ValueError("MedXpertQA source schema differs from the pinned contract")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("MedXpertQA item IDs are not unique")
    expected_options = [chr(ord("A") + index) for index in range(10)]
    for row in rows:
        if list(row["options"]) != expected_options:
            raise ValueError(f"expected ordered A--J options for {row['id']}")
        if row["label"] not in row["options"]:
            raise ValueError(f"invalid reference option for {row['id']}")
        if "Answer Choices:" not in row["question"]:
            raise ValueError(f"official task text lacks answer choices for {row['id']}")
        for field in ("medical_task", "body_system", "question_type"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"invalid {field} for {row['id']}")
    return rows


def assign_stratified_partitions(
    rows: list[dict[str, Any]], salt: str
) -> dict[str, Partition]:
    """Allocate exact 40/30/30 totals while balancing official strata."""

    by_stratum: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        stratum = (row["medical_task"], row["body_system"], row["question_type"])
        by_stratum[stratum].append(row)

    ordered: list[tuple[int, tuple[str, str, str], dict[str, Any]]] = []
    for stratum, candidates in by_stratum.items():
        candidates = sorted(
            candidates,
            key=lambda row: (_hash_key(salt, "row", row["id"]), row["id"]),
        )
        ordered.extend((index, stratum, row) for index, row in enumerate(candidates))
    ordered.sort(
        key=lambda value: (
            value[0],
            _hash_key(salt, "stratum", *value[1]),
            value[2]["id"],
        )
    )

    capacities: dict[Partition, int] = {
        "development": int(len(rows) * FRACTIONS["development"]),
        "certification": int(len(rows) * FRACTIONS["certification"]),
        "test": 0,
    }
    capacities["test"] = len(rows) - sum(capacities.values())
    global_counts: Counter[str] = Counter()
    stratum_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    stratum_totals: Counter[tuple[str, str, str]] = Counter()
    assignments: dict[str, Partition] = {}
    for _, stratum, row in ordered:
        available = [
            partition
            for partition in PARTITIONS
            if global_counts[partition] < capacities[partition]
        ]
        next_stratum_total = stratum_totals[stratum] + 1
        next_global_total = len(assignments) + 1
        partition = min(
            available,
            key=lambda value: (
                abs(
                    (stratum_counts[stratum][value] + 1)
                    - next_stratum_total * FRACTIONS[value]
                ),
                abs(
                    (global_counts[value] + 1)
                    - next_global_total * FRACTIONS[value]
                ),
                _hash_key(salt, "partition-tie", row["id"], value),
            ),
        )
        assignments[row["id"]] = partition
        stratum_counts[stratum][partition] += 1
        stratum_totals[stratum] += 1
        global_counts[partition] += 1

    if dict(global_counts) != capacities:
        raise AssertionError(f"partition allocation missed exact capacities: {global_counts}")
    return assignments


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
    output_directory: Path,
    evidence: Path,
    source_revision: str,
    expected_source_sha256: str,
    salt: str,
) -> dict[str, Any]:
    source_bytes = source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != expected_source_sha256:
        raise ValueError(
            f"source SHA-256 mismatch: expected {expected_source_sha256}, got {source_sha256}"
        )
    rows = _read_rows(source)
    assignments = assign_stratified_partitions(rows, salt)
    items = [
        DatasetItem(
            item_id=row["id"],
            question=row["question"],
            answer=row["label"],
            success_criterion=(
                "The proposed answer is exactly the letter of the medically correct option."
            ),
            partition=assignments[row["id"]],
            metadata={
                "medical_task": row["medical_task"],
                "body_system": row["body_system"],
                "question_type": row["question_type"],
                "option_count": len(row["options"]),
                "official_subset": "Text",
                "official_split": "test",
            },
        )
        for row in rows
    ]

    outputs: dict[str, dict[str, Any]] = {}
    for partition in PARTITIONS:
        partition_items = [item for item in items if item.partition == partition]
        data = _canonical_lines(partition_items)
        path = output_directory / f"items-{partition}.jsonl"
        _write_deterministic(path, data)
        outputs[partition] = {
            "path": path.name,
            "rows": len(partition_items),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    stratum_counts = Counter(
        (
            item.metadata["medical_task"],
            item.metadata["body_system"],
            item.metadata["question_type"],
        )
        for item in items
    )
    manifest = {
        "schema_version": "1.0",
        "dataset_id": "TsinghuaC3I/MedXpertQA",
        "source_subset": "Text",
        "source_split": "test",
        "source_revision": source_revision,
        "source_sha256": source_sha256,
        "source_rows": len(rows),
        "algorithm": "official-strata-exact-40-30-30-v1",
        "salt": salt,
        "strata": len(stratum_counts),
        "partition_outputs": outputs,
        "medical_task_counts": dict(
            sorted(Counter(item.metadata["medical_task"] for item in items).items())
        ),
        "body_system_counts": dict(
            sorted(Counter(item.metadata["body_system"] for item in items).items())
        ),
        "question_type_counts": dict(
            sorted(Counter(item.metadata["question_type"] for item in items).items())
        ),
        "output_format": "cpis DatasetItem JSONL v1",
    }
    evidence_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _write_deterministic(evidence, evidence_bytes)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--salt", required=True)
    args = parser.parse_args()
    manifest = prepare(
        source=args.source,
        output_directory=args.output_directory,
        evidence=args.evidence,
        source_revision=args.source_revision,
        expected_source_sha256=args.source_sha256,
        salt=args.salt,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
