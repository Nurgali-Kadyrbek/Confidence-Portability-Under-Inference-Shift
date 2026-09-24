"""Prepare the four primary BFCL V3 multi-turn categories without data leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cpis.config import SplitSpec
from cpis.datasets import DatasetItem, PARTITIONS, Partition, deterministic_split


CATEGORIES = ("base", "miss_param", "miss_func", "long_context")
QUESTION_FIELDS = {"id", "question", "initial_config", "path", "involved_classes"}
ANSWER_FIELDS = {"id", "ground_truth"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle_sha256(paths: list[Path], root: Path) -> str:
    entries = [
        f"{path.relative_to(root).as_posix()}\0{_sha256(path)}"
        for path in sorted(paths)
    ]
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def _scenario_index(identifier: str, category: str) -> int:
    prefix = f"multi_turn_{category}_"
    if not identifier.startswith(prefix):
        raise ValueError(f"unexpected BFCL identifier {identifier!r}")
    suffix = identifier.removeprefix(prefix)
    if not suffix.isdigit():
        raise ValueError(f"non-numeric BFCL scenario suffix in {identifier!r}")
    return int(suffix)


def _load_and_validate(source_root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    rows: list[dict[str, Any]] = []
    source_paths: list[Path] = []
    for category in CATEGORIES:
        question_path = source_root / f"BFCL_v3_multi_turn_{category}.json"
        answer_path = (
            source_root
            / "possible_answer"
            / f"BFCL_v3_multi_turn_{category}.json"
        )
        source_paths.extend((question_path, answer_path))
        questions = _read_jsonl(question_path)
        answers = _read_jsonl(answer_path)
        if len(questions) != 200 or len(answers) != 200:
            raise ValueError(f"BFCL {category} must contain exactly 200 cases")
        answer_by_id = {answer["id"]: answer for answer in answers}
        if len(answer_by_id) != 200:
            raise ValueError(f"BFCL {category} possible answers contain duplicate IDs")
        seen_indices: set[int] = set()
        for question in questions:
            allowed_fields = QUESTION_FIELDS | ({"missed_function"} if category == "miss_func" else set())
            if set(question) != allowed_fields:
                raise ValueError(f"BFCL {category} question schema changed")
            identifier = question["id"]
            index = _scenario_index(identifier, category)
            seen_indices.add(index)
            try:
                answer = answer_by_id.pop(identifier)
            except KeyError as exc:
                raise ValueError(f"BFCL possible answer missing for {identifier}") from exc
            if set(answer) != ANSWER_FIELDS:
                raise ValueError(f"BFCL {category} answer schema changed")
            if not isinstance(question["question"], list) or not question["question"]:
                raise ValueError(f"BFCL {identifier} has no conversation turns")
            rows.append(
                {
                    "category": category,
                    "scenario_index": index,
                    "question": question,
                    "possible_answer": answer,
                }
            )
        if answer_by_id or seen_indices != set(range(200)):
            raise ValueError(f"BFCL {category} does not contain scenarios 0--199")
    return rows, source_paths


def _items(rows: list[dict[str, Any]], salt: str) -> list[DatasetItem]:
    group_ids = {f"bfcl-v3-scenario-{row['scenario_index']:03d}" for row in rows}
    if len(group_ids) != 200:
        raise ValueError("BFCL primary categories must map to 200 base scenarios")
    assignments = deterministic_split(
        group_ids,
        SplitSpec(salt=salt, development=0.4, certification=0.3, test=0.3),
    )
    items: list[DatasetItem] = []
    for row in rows:
        question = row["question"]
        group_id = f"bfcl-v3-scenario-{row['scenario_index']:03d}"
        items.append(
            DatasetItem(
                item_id=question["id"],
                group_id=group_id,
                question=json.dumps(
                    question["question"],
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
                answer=None,
                success_criterion=(
                    "The complete multi-turn tool-use episode succeeds under the "
                    "pinned official BFCL V3 evaluator, including required "
                    "clarification or abstention behavior."
                ),
                partition=assignments[group_id],
                scoring_payload={
                    "source_case": question,
                    "possible_answer": row["possible_answer"],
                },
                metadata={
                    "category": row["category"],
                    "base_scenario_index": row["scenario_index"],
                    "conversation_turns": len(question["question"]),
                    "involved_classes": question["involved_classes"],
                },
            )
        )
    return items


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
    source_root: Path,
    output_directory: Path,
    evidence: Path,
    source_revision: str,
    expected_bundle_sha256: str,
    salt: str,
) -> dict[str, Any]:
    rows, source_paths = _load_and_validate(source_root)
    bundle_sha256 = _bundle_sha256(source_paths, source_root)
    if bundle_sha256 != expected_bundle_sha256:
        raise ValueError(
            "BFCL source bundle SHA-256 mismatch: "
            f"expected {expected_bundle_sha256}, got {bundle_sha256}"
        )
    items = _items(rows, salt)
    if len(items) != 800 or len({item.item_id for item in items}) != 800:
        raise ValueError("BFCL preparation must produce 800 unique category cases")

    outputs: dict[str, dict[str, Any]] = {}
    for partition in PARTITIONS:
        partition_items = [item for item in items if item.partition == partition]
        data = _canonical_lines(partition_items)
        path = output_directory / f"items-{partition}.jsonl"
        _write_deterministic(path, data)
        outputs[partition] = {
            "path": path.name,
            "rows": len(partition_items),
            "groups": len({item.group_id for item in partition_items}),
            "category_counts": dict(
                sorted(Counter(item.metadata["category"] for item in partition_items).items())
            ),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    group_partitions: dict[str, set[Partition]] = defaultdict(set)
    group_categories: dict[str, set[str]] = defaultdict(set)
    for item in items:
        assert item.group_id is not None and item.partition is not None
        group_partitions[item.group_id].add(item.partition)
        group_categories[item.group_id].add(str(item.metadata["category"]))
    if any(len(partitions) != 1 for partitions in group_partitions.values()):
        raise AssertionError("a BFCL base scenario crossed the data firewall")
    if any(categories != set(CATEGORIES) for categories in group_categories.values()):
        raise AssertionError("a BFCL base scenario is missing a category variant")

    manifest = {
        "schema_version": "1.0",
        "dataset_id": "gorilla-llm/Berkeley-Function-Calling-Leaderboard:BFCL-v3-multi-turn",
        "source_revision": source_revision,
        "source_bundle_sha256": bundle_sha256,
        "source_files": {
            path.relative_to(source_root).as_posix(): _sha256(path)
            for path in sorted(source_paths)
        },
        "source_rows": len(rows),
        "independent_scenario_groups": len(group_partitions),
        "derived_category_cases": len(items),
        "categories": list(CATEGORIES),
        "algorithm": "scenario-group-preserving-hash-40-30-30-v1",
        "salt": salt,
        "scorer": "bfcl_v3_official_multi_turn_v1",
        "partition_outputs": outputs,
        "output_format": "cpis DatasetItem JSONL v2",
        "dependence_note": (
            "The four category cases sharing a numeric suffix preserve the same "
            "base scenario and are clustered repeated observations, not independent units."
        ),
    }
    evidence_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _write_deterministic(evidence, evidence_bytes)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-bundle-sha256", required=True)
    parser.add_argument("--salt", required=True)
    args = parser.parse_args()
    manifest = prepare(
        source_root=args.source_root,
        output_directory=args.output_directory,
        evidence=args.evidence,
        source_revision=args.source_revision,
        expected_bundle_sha256=args.source_bundle_sha256,
        salt=args.salt,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
