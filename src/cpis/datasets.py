"""Dataset contracts and deterministic, item-level partitioning."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from cpis.config import DatasetSpec, SplitSpec

Partition = Literal["development", "certification", "test"]
PARTITIONS: tuple[Partition, ...] = ("development", "certification", "test")


class DatasetItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str | None = Field(default=None, min_length=1)
    success_criterion: str = Field(
        default="The proposed answer exactly matches the reference answer.",
        min_length=1,
    )
    partition: Partition | None = None
    group_id: str | None = Field(default=None, min_length=1)
    scoring_payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetLoader(Protocol):
    def load(self) -> list[DatasetItem]: ...


class JsonlLoader:
    """Content-addressed JSONL loader for fixtures or prepared public data."""

    def __init__(self, path: Path, expected_sha256: str) -> None:
        self.path = path
        self.expected_sha256 = expected_sha256

    def load(self) -> list[DatasetItem]:
        raw = self.path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != self.expected_sha256:
            raise ValueError(
                f"dataset content hash mismatch for {self.path}: "
                f"expected {self.expected_sha256}, got {actual}"
            )
        items: list[DatasetItem] = []
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                items.append(DatasetItem.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"invalid JSONL item at line {line_number}: {exc}") from exc
        if not items:
            raise ValueError("dataset is empty")
        ids = [item.item_id for item in items]
        if len(set(ids)) != len(ids):
            raise ValueError("dataset item_id values must be unique")
        return items


def load_dataset(
    spec: DatasetSpec, config_directory: Path, dataset_cache: Path
) -> list[DatasetItem]:
    source = Path(spec.source_path)
    if spec.loader == "jsonl_fixture" and not source.is_absolute():
        source = (config_directory / source).resolve()
    elif spec.loader == "cached_jsonl":
        source = (dataset_cache / source).resolve()
    if spec.loader in {"jsonl_fixture", "cached_jsonl"}:
        return JsonlLoader(source, spec.content_sha256).load()
    raise ValueError(f"unsupported dataset loader: {spec.loader}")


def deterministic_split(
    item_ids: Iterable[str], spec: SplitSpec
) -> dict[str, Partition]:
    """Assign a fixed dataset revision exactly once, independent of input order.

    Seed/configuration never enters this function: all repeated observations for an
    item consequently retain the same partition.
    """
    ids = list(item_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("cannot split duplicate item IDs")
    ranked = sorted(
        ids,
        key=lambda item_id: (
            hashlib.sha256(f"{spec.salt}\0{item_id}".encode("utf-8")).digest(),
            item_id,
        ),
    )
    fractions = [spec.development, spec.certification, spec.test]
    exact_counts = [len(ids) * fraction for fraction in fractions]
    counts = [math.floor(value) for value in exact_counts]
    remainder = len(ids) - sum(counts)
    priority = sorted(
        range(len(PARTITIONS)),
        key=lambda i: (-(exact_counts[i] - counts[i]), i),
    )
    for index in priority[:remainder]:
        counts[index] += 1

    result: dict[str, Partition] = {}
    cursor = 0
    for partition, count in zip(PARTITIONS, counts, strict=True):
        for item_id in ranked[cursor : cursor + count]:
            result[item_id] = partition
        cursor += count
    if len(result) != len(ids):
        raise AssertionError("split assignment lost an item")
    return result


def partition_items(items: list[DatasetItem], spec: DatasetSpec) -> list[DatasetItem]:
    declared = [item.partition for item in items]
    if any(value is not None for value in declared):
        if any(value is None for value in declared):
            raise ValueError("dataset cannot mix declared and computed partitions")
        partition = [item for item in items if item.partition == spec.partition]
    else:
        assignments = deterministic_split((item.item_id for item in items), spec.split)
        partition = [
            item for item in items if assignments[item.item_id] == spec.partition
        ]
    if spec.item_limit is None:
        return partition
    offset = spec.item_offset or 0
    if offset + spec.item_limit > len(partition):
        raise ValueError(
            f"item_offset={offset}+item_limit={spec.item_limit} exceeds "
            f"{spec.partition} partition size {len(partition)}"
        )
    assert spec.item_selection_salt is not None
    ranked = sorted(
        partition,
        key=lambda item: (
            hashlib.sha256(
                f"{spec.item_selection_salt}\0{item.item_id}".encode("utf-8")
            ).digest(),
            item.item_id,
        ),
    )
    window = ranked[offset : offset + spec.item_limit]
    if spec.item_subselection_salt is None:
        return window
    # Rank again inside the window, never over the partition, so the result is
    # a strict subset of the items the broad study already evaluated.
    resampled = sorted(
        window,
        key=lambda item: (
            hashlib.sha256(
                f"{spec.item_subselection_salt}\0{item.item_id}".encode("utf-8")
            ).digest(),
            item.item_id,
        ),
    )
    return resampled[: spec.item_subselection_limit]


def assign_replica(item_id: str, replicas: int) -> int:
    """Keep all repeated observations of an item on one stable model replica."""
    if replicas <= 0:
        raise ValueError("replicas must be positive")
    digest = hashlib.sha256(f"cpis-replica-v1\0{item_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % replicas
