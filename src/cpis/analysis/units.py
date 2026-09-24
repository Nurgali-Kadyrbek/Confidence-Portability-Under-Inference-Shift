"""Prespecified mappings from benchmark rows to independent source units."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass

from cpis.datasets import DatasetItem


@dataclass(frozen=True)
class IndependentUnitSelection:
    selected_item_ids: frozenset[str]
    source_groups: int
    variant_counts: dict[int | str, int]
    method: str
    selector_salt: str


def deterministic_balanced_group_anchors(
    items: list[DatasetItem], selector_salt: str
) -> IndependentUnitSelection:
    """Choose one variant per source group without consulting any outcome.

    Ungrouped datasets already have one row per independent unit and are
    returned unchanged. For derived-variant datasets, source groups are ordered
    by a salted hash and assigned their available instruction-count variants in
    round-robin order. This makes the certification sample independent at the
    source-group level while balancing task load as closely as integer counts
    permit.
    """
    if len(selector_salt) < 16:
        raise ValueError("group-anchor selector salt must contain at least 16 characters")
    if not items:
        raise ValueError("group-anchor selection requires dataset items")
    if len({item.item_id for item in items}) != len(items):
        raise ValueError("dataset item IDs must be unique")
    grouped = any(item.group_id is not None for item in items)
    if not grouped:
        return IndependentUnitSelection(
            selected_item_ids=frozenset(item.item_id for item in items),
            source_groups=len(items),
            variant_counts={},
            method="one_row_per_original_item",
            selector_salt=selector_salt,
        )
    if any(item.group_id is None for item in items):
        raise ValueError("a dataset cannot mix grouped and ungrouped rows")

    members: dict[str, dict[int, DatasetItem]] = {}
    for item in items:
        raw_variant = item.metadata.get("instruction_count")
        if not isinstance(raw_variant, int) or raw_variant < 1:
            raise ValueError("grouped certification requires integer instruction_count")
        variants = members.setdefault(str(item.group_id), {})
        if raw_variant in variants:
            raise ValueError("source group has duplicate instruction_count variants")
        variants[raw_variant] = item

    signatures = {tuple(sorted(variants)) for variants in members.values()}
    if len(signatures) != 1:
        raise ValueError("source groups must expose the same variant-load levels")
    levels = next(iter(signatures))
    ordered_groups = sorted(
        members,
        key=lambda group: (
            hashlib.sha256(f"{selector_salt}\0{group}".encode()).digest(),
            group,
        ),
    )
    selected: list[DatasetItem] = []
    for index, group in enumerate(ordered_groups):
        selected.append(members[group][levels[index % len(levels)]])
    counts = Counter(int(item.metadata["instruction_count"]) for item in selected)
    if max(counts.values()) - min(counts.values()) > 1:
        raise RuntimeError("balanced group-anchor assignment invariant failed")
    return IndependentUnitSelection(
        selected_item_ids=frozenset(item.item_id for item in selected),
        source_groups=len(members),
        variant_counts=dict(sorted(counts.items())),
        method="deterministic_balanced_one_variant_per_source_group",
        selector_salt=selector_salt,
    )


def deterministic_balanced_category_group_anchors(
    items: list[DatasetItem], selector_salt: str, category_key: str
) -> IndependentUnitSelection:
    """Choose one outcome-blind, category-balanced row per source scenario."""

    if len(selector_salt) < 16:
        raise ValueError("group-anchor selector salt must contain at least 16 characters")
    if not items or any(item.group_id is None for item in items):
        raise ValueError("categorical group anchors require grouped items")
    members: dict[str, dict[str, DatasetItem]] = {}
    for item in items:
        category = item.metadata.get(category_key)
        if not isinstance(category, str) or not category:
            raise ValueError("categorical group anchor is missing its category")
        variants = members.setdefault(str(item.group_id), {})
        if category in variants:
            raise ValueError("source group has a duplicate category")
        variants[category] = item
    signatures = {tuple(sorted(variants)) for variants in members.values()}
    if len(signatures) != 1:
        raise ValueError("source groups must expose the same category levels")
    levels = next(iter(signatures))
    ordered_groups = sorted(
        members,
        key=lambda group: (
            hashlib.sha256(f"{selector_salt}\0{group}".encode()).digest(),
            group,
        ),
    )
    selected = [
        members[group][levels[index % len(levels)]]
        for index, group in enumerate(ordered_groups)
    ]
    counts = Counter(str(item.metadata[category_key]) for item in selected)
    if max(counts.values()) - min(counts.values()) > 1:
        raise RuntimeError("balanced categorical group-anchor invariant failed")
    return IndependentUnitSelection(
        selected_item_ids=frozenset(item.item_id for item in selected),
        source_groups=len(members),
        variant_counts=dict(sorted(counts.items())),
        method="deterministic_balanced_one_category_per_source_group",
        selector_salt=selector_salt,
    )
