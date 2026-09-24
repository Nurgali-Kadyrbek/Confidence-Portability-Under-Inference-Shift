"""Prepare grouped, sealed ManyIFEval and StyleMBPP partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from cpis.config import SplitSpec
from cpis.datasets import DatasetItem, PARTITIONS, Partition, deterministic_split

Benchmark = Literal["manyifeval", "stylembpp"]

MANYIFEVAL_FIELDS = {
    "key",
    "prompt",
    "instruction_id_list",
    "kwargs",
    "task_prompt",
    "not_conflict_rules_list",
    "rules",
}
STYLEMBPP_FIELDS = {
    "text",
    "code",
    "task_id",
    "test_setup_code",
    "test_list",
    "challenge_test_list",
    "promt_without_inst",
    "instruction_id_list",
    "prompt",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _validate_manyifeval(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 2_160 or any(set(row) != MANYIFEVAL_FIELDS for row in rows):
        raise ValueError("ManyIFEval source differs from the pinned 2,160-row schema")
    by_key: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row["key"], int) or not isinstance(row["prompt"], str):
            raise ValueError("ManyIFEval key/prompt types differ from the contract")
        if len(row["instruction_id_list"]) != len(row["kwargs"]):
            raise ValueError(f"ManyIFEval instruction/kwargs mismatch for {row['key']}")
        by_key[row["key"]].append(row)
    if len(by_key) != 216:
        raise ValueError(f"expected 216 ManyIFEval base prompts, got {len(by_key)}")
    for key, variants in by_key.items():
        loads = sorted(len(row["instruction_id_list"]) for row in variants)
        if loads != list(range(1, 11)):
            raise ValueError(f"base prompt {key} does not have loads 1--10")


def _validate_stylembpp(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 3_000 or any(set(row) != STYLEMBPP_FIELDS for row in rows):
        raise ValueError("StyleMBPP source differs from the pinned 3,000-row schema")
    by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row["task_id"], int) or not isinstance(row["prompt"], str):
            raise ValueError("StyleMBPP task/prompt types differ from the contract")
        if not row["test_list"]:
            raise ValueError(f"StyleMBPP task {row['task_id']} has no functional tests")
        by_task[row["task_id"]].append(row)
    if len(by_task) != 500:
        raise ValueError(f"expected 500 StyleMBPP base tasks, got {len(by_task)}")
    for task_id, variants in by_task.items():
        loads = sorted(len(row["instruction_id_list"]) for row in variants)
        if loads != list(range(1, 7)):
            raise ValueError(f"base task {task_id} does not have loads 1--6")


def _group_assignments(
    group_ids: set[str], salt: str
) -> dict[str, Partition]:
    return deterministic_split(
        group_ids,
        SplitSpec(
            salt=salt,
            development=0.4,
            certification=0.3,
            test=0.3,
        ),
    )


def _items_manyifeval(rows: list[dict[str, Any]], salt: str) -> list[DatasetItem]:
    assignments = _group_assignments(
        {f"manyifeval-{row['key']}" for row in rows}, salt
    )
    return [
        DatasetItem(
            item_id=(
                f"manyifeval-{row['key']}-n{len(row['instruction_id_list']):02d}"
            ),
            group_id=f"manyifeval-{row['key']}",
            question=row["prompt"],
            answer=None,
            success_criterion=(
                "The proposed response satisfies every instruction under the "
                "official strict ManyIFEval evaluator."
            ),
            partition=assignments[f"manyifeval-{row['key']}"],
            scoring_payload=row,
            metadata={
                "base_key": row["key"],
                "instruction_count": len(row["instruction_id_list"]),
            },
        )
        for row in rows
    ]


def _items_stylembpp(rows: list[dict[str, Any]], salt: str) -> list[DatasetItem]:
    assignments = _group_assignments(
        {f"stylembpp-{row['task_id']}" for row in rows}, salt
    )
    return [
        DatasetItem(
            item_id=(
                f"stylembpp-{row['task_id']}-n{len(row['instruction_id_list']):02d}"
            ),
            group_id=f"stylembpp-{row['task_id']}",
            question=row["prompt"],
            answer=None,
            success_criterion=(
                "The proposed Python program passes the official functional tests "
                "and every required StyleMBPP constraint."
            ),
            partition=assignments[f"stylembpp-{row['task_id']}"],
            scoring_payload=row,
            metadata={
                "base_task_id": row["task_id"],
                "instruction_count": len(row["instruction_id_list"]),
            },
        )
        for row in rows
    ]


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
    benchmark: Benchmark,
    source: Path,
    output_directory: Path,
    evidence: Path,
    source_revision: str,
    expected_source_sha256: str,
    source_archive_sha256: str,
    salt: str,
) -> dict[str, Any]:
    source_bytes = source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != expected_source_sha256:
        raise ValueError(
            f"source SHA-256 mismatch: expected {expected_source_sha256}, got {source_sha256}"
        )
    rows = _read_jsonl(source)
    if benchmark == "manyifeval":
        _validate_manyifeval(rows)
        items = _items_manyifeval(rows, salt)
        dataset_id = "kenoharada/Multiple-Instructions-Following:ManyIFEval"
        scorer = "manyifeval_official_v1"
    else:
        _validate_stylembpp(rows)
        items = _items_stylembpp(rows, salt)
        dataset_id = "kenoharada/Multiple-Instructions-Following:StyleMBPP"
        scorer = "stylembpp_official_v1"

    item_ids = [item.item_id for item in items]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError(f"{benchmark} prepared item IDs are not unique")
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
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    group_partitions: dict[str, set[Partition]] = defaultdict(set)
    for item in items:
        assert item.group_id is not None and item.partition is not None
        group_partitions[item.group_id].add(item.partition)
    if any(len(partitions) != 1 for partitions in group_partitions.values()):
        raise AssertionError("a derived-item group crossed the data firewall")

    manifest = {
        "schema_version": "1.0",
        "dataset_id": dataset_id,
        "source_revision": source_revision,
        "source_archive_sha256": source_archive_sha256,
        "source_sha256": source_sha256,
        "source_rows": len(rows),
        "groups": len(group_partitions),
        "algorithm": "group-preserving-hash-40-30-30-v1",
        "salt": salt,
        "scorer": scorer,
        "instruction_load_counts": dict(
            sorted(Counter(item.metadata["instruction_count"] for item in items).items())
        ),
        "partition_outputs": outputs,
        "output_format": "cpis DatasetItem JSONL v2",
    }
    evidence_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _write_deterministic(evidence, evidence_bytes)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("manyifeval", "stylembpp"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--salt", required=True)
    args = parser.parse_args()
    manifest = prepare(
        benchmark=args.benchmark,
        source=args.source,
        output_directory=args.output_directory,
        evidence=args.evidence,
        source_revision=args.source_revision,
        expected_source_sha256=args.source_sha256,
        source_archive_sha256=args.source_archive_sha256,
        salt=args.salt,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
