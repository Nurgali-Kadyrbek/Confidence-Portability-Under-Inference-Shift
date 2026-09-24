#!/usr/bin/env python3
"""Materialise the two-phase core study from the frozen-core configs.

A development partition is a data pool, not a workload. Running every item in it
through every decoder condition is the Cartesian-product failure that produced
first a twenty-seven-day campaign and then a twelve-hour "development phase".

Decoder shifts are the held-out intervention. Development needs only enough to
validate the machinery and to choose the reference threshold, and both of those
happen at the reference coordinate:

  smoke        32 items/dataset, all three decoder conditions, engineering only
  calibration  300 items/dataset, reference decoder only, selects tau_0
  test         600 items/dataset, all three decoder conditions, the experiment

Smoke and calibration draw disjoint windows from one salt, so no item used to
shake out parsers is later used to choose a threshold. There is no certification
phase: the thesis needs a held-out paired contrast with tau_0 frozen beforehand,
not a certified risk guarantee.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from cpis.matrix_config import load_matrix_config

SALT = "cpis-core-phase-subsets-v1"
REFERENCE = "reference"

PHASES = {
    # name: (partition, item_limit, item_offset, reference_only)
    "smoke": ("development", 32, 0, False),
    "calibration": ("development", 300, 32, True),
    # Materialised only at the test freeze, once the calibration thresholds are
    # immutable. The test partition stays locked behind the phase gate until a
    # committed gate names the frozen files.
    "test": ("test", 600, 0, False),
}
MATRICES = (
    "qwen35-9b",
    "qwen35-9b-thinking",
    "ministral3-8b-instruct",
    "ministral3-8b-reasoning",
)
DATASETS = ("supergpqa", "manyifeval")


def sealed_partition(reference, partition: str, repository_root: Path) -> dict:
    """The sealed file and hash the integration record pins for a partition."""
    record = yaml.safe_load(
        (repository_root / reference.record_path).read_text(encoding="utf-8")
    )
    for entry in record.get("prepared_partitions", []):
        if entry.get("partition") == partition or partition in entry.get(
            "relative_cache_path", ""
        ):
            return entry
    raise RuntimeError(
        f"{reference.record_path} declares no sealed {partition} partition"
    )


def build(source: Path, phase: str, model: str, dataset: str) -> dict:
    partition, limit, offset, reference_only = PHASES[phase]
    config = load_matrix_config(source)
    payload = config.model_dump(mode="json")

    # A partition is a different sealed file with its own hash, not just a
    # label. Changing the label alone points a test config at development bytes,
    # which the integration firewall correctly refuses.
    sealed = sealed_partition(config.dataset_integration, partition, source.parents[2])
    payload["dataset"]["partition"] = partition
    payload["dataset"]["source_path"] = sealed["relative_cache_path"]
    payload["dataset"]["content_sha256"] = sealed["sha256"]
    payload["dataset"]["item_limit"] = limit
    payload["dataset"]["item_offset"] = offset
    payload["dataset"]["item_selection_salt"] = SALT

    inference = payload["inference"]
    design = inference["design"]
    if reference_only:
        inference["answer_conditions"] = [
            c for c in inference["answer_conditions"] if c["condition_id"] == REFERENCE
        ]
        keep = {REFERENCE, design["standardized_readout_id"]}
        inference["confidence_readouts"] = [
            r for r in inference["confidence_readouts"] if r["readout_id"] in keep
        ]
        design["coupled_readout_by_answer"] = {REFERENCE: REFERENCE}

    if partition == "test":
        payload["study"] = "core_test"
        payload["confirmatory"] = True
    payload["experiment_id"] = f"cpis-{phase}-{model}-{dataset}-v1"
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--phases", nargs="*", default=["smoke", "calibration"])
    args = parser.parse_args()

    root = args.repository_root.resolve()
    for phase in args.phases:
        destination = root / "configs/core" / phase
        destination.mkdir(parents=True, exist_ok=True)
        records = 0
        for model in MATRICES:
            for dataset in DATASETS:
                source = root / "configs/core" / f"{model}-{dataset}-core-v1.yaml"
                payload = build(source, phase, model, dataset)
                out = destination / f"{model}-{dataset}-{phase}-v1.yaml"
                out.write_text(
                    yaml.safe_dump(payload, sort_keys=False, width=100), encoding="utf-8"
                )
                written = load_matrix_config(out)
                cells = (
                    written.dataset.item_limit
                    * len(written.inference.answer_conditions)
                    * len(written.inference.seeds)
                )
                records += cells * 2
        print(f"{phase:<12} 8 configs  {records:>7,} records (answers + confidence)")


if __name__ == "__main__":
    main()
