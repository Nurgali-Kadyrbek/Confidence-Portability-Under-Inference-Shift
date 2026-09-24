#!/usr/bin/env python3
"""Materialise the seed-robustness subset from the frozen held-out configs.

The broad study runs one sampling seed, so its intervals describe uncertainty
over items at a fixed draw from the decoder. This subset answers the separate
question of how much a contrast moves when the decoder is resampled.

Three constraints shape it and none of them is negotiable here:

The items are a strict subset of the held-out window. A fresh ranking over the
test partition would reach items the broad study never evaluated - both test
partitions are larger than 600 - so the draw is taken inside the window the
held-out configs already selected.

All three seeds are generated in this run, including the one the broad study
already used. Generation is not invariant to batch composition on this engine
(evidence/environment/batch_size_generation_invariance_v1.json: 17.9% of
answers byte-identical across two batch sizes with all thirteen control fields
equal), so reusing the 600-item run's records for seed 1729 would confound the
seed contrast with a change of batching. The cost of regenerating that seed is
a third of a small run and it buys a clean comparison.

Nothing else moves: thresholds, decoder values, prompts, readout, scoring and
pairing are the held-out ones. No recalibration happens here and no threshold
is reselected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from cpis.matrix_config import load_matrix_config

# Declared before any generation and never varied. The first is the broad
# study's seed, regenerated here for batch comparability; the other two are
# fixed constants chosen without reference to any outcome.
SEEDS = (1729, 2718, 3141)
SUBSET_SALT = "cpis-seed-robustness-subset-v1"
SUBSET_SIZE = 128


def build(source: Path) -> dict:
    config = load_matrix_config(source)
    payload = config.model_dump(mode="json")

    payload["study"] = "core_seed_robustness"
    payload["confirmatory"] = False
    payload["experiment_id"] = config.experiment_id.replace("cpis-test-", "cpis-seeds-")

    # The window stays exactly as the held-out config selected it; the subset is
    # drawn from inside that window.
    payload["dataset"]["item_subselection_salt"] = SUBSET_SALT
    payload["dataset"]["item_subselection_limit"] = SUBSET_SIZE
    payload["inference"]["seeds"] = list(SEEDS)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--source", type=Path, default=Path("configs/core/test"))
    parser.add_argument("--destination", type=Path, default=Path("configs/core/seeds"))
    args = parser.parse_args()

    root = args.repository_root.resolve()
    destination = root / args.destination
    destination.mkdir(parents=True, exist_ok=True)
    written = 0
    for source in sorted((root / args.source).glob("*.yaml")):
        payload = build(source)
        out = destination / source.name.replace("-test-v1.yaml", "-seeds-v1.yaml")
        out.write_text(
            yaml.safe_dump(payload, sort_keys=False, width=100), encoding="utf-8"
        )
        # Reload through the validator so a materialised config that the schema
        # would reject fails here rather than at launch.
        reloaded = load_matrix_config(out)
        assert reloaded.study == "core_seed_robustness"
        written += 1
    print(
        f"wrote {written} seed-robustness configs: {SUBSET_SIZE} items/dataset, "
        f"seeds {SEEDS}, salt {SUBSET_SALT}"
    )


if __name__ == "__main__":
    main()
