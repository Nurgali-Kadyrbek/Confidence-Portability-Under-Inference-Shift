#!/usr/bin/env python3
"""Materialise the frozen-core broad study from the existing matrix configs.

The core is four model conditions by two datasets by three decoder conditions
by one stochastic replicate, with exactly one native coupled confidence per
answer. The older configs carry five decoder conditions, three seeds and 3.6
confidence cells per answer, which is a mechanistic and mitigation study
multiplied across the whole panel.

The retained decoder conditions are a subset of what was already prespecified,
not new values, and each is defined relative to that model's own native
reference: `high-temperature` moves only temperature and `low-top-p` moves only
top-p. The retained seed is the first already-declared replicate. Both
selections are mechanical and recorded before any outcome is consulted, so this
is a design decision rather than a choice among observed results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from cpis.matrix_config import load_matrix_config

KEEP_CONDITIONS = ("reference", "high-temperature", "low-top-p")
KEEP_SEED_INDEX = 0

SOURCES = {
    "qwen35-9b": "v1",
    "qwen35-9b-thinking": "v1",
    "ministral3-8b-instruct": "v1",
    # The reasoning conditions were re-versioned to batch 64 on measured
    # throughput; the core inherits that rather than the batch-8 original.
    "ministral3-8b-reasoning": "v2",
}
DATASETS = ("supergpqa", "manyifeval")


def build(source_path: Path, model: str, dataset: str) -> dict:
    config = load_matrix_config(source_path)
    payload = config.model_dump(mode="json")
    inference = payload["inference"]

    declared = {c["condition_id"] for c in inference["answer_conditions"]}
    missing = set(KEEP_CONDITIONS) - declared
    if missing:
        raise RuntimeError(f"{source_path.name} does not declare {sorted(missing)}")

    inference["answer_conditions"] = [
        c for c in inference["answer_conditions"] if c["condition_id"] in KEEP_CONDITIONS
    ]
    design = inference["design"]
    standardized = design["standardized_readout_id"]
    # Keep the coupled readouts for the retained coordinates. The standardized
    # readout stays declared because the design references it, but under
    # coupled_only no standardized cell is generated.
    keep_readouts = set(KEEP_CONDITIONS) | {standardized}
    inference["confidence_readouts"] = [
        r for r in inference["confidence_readouts"] if r["readout_id"] in keep_readouts
    ]
    design["coupled_readout_by_answer"] = {
        k: v for k, v in design["coupled_readout_by_answer"].items()
        if k in KEEP_CONDITIONS
    }
    design["report_only_readout_ids"] = [design["reference_confidence_readout_id"]]
    design["confidence_cells"] = "coupled_only"

    inference["seeds"] = [config.inference.seeds[KEEP_SEED_INDEX]]
    payload["experiment_id"] = f"dev-core-{model}-{dataset}-v1"
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--destination", type=Path, default=Path("configs/core"))
    args = parser.parse_args()

    root = args.repository_root.resolve()
    destination = (root / args.destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for model, version in SOURCES.items():
        for dataset in DATASETS:
            source = (
                root
                / "configs/development"
                / f"{model}-{dataset}-development-matrix-{version}.yaml"
            )
            payload = build(source, model, dataset)
            out = destination / f"{model}-{dataset}-core-v1.yaml"
            out.write_text(
                yaml.safe_dump(payload, sort_keys=False, width=100), encoding="utf-8"
            )
            written = load_matrix_config(out)
            print(
                f"{out.relative_to(root)}  conditions="
                f"{len(written.inference.answer_conditions)} "
                f"seeds={list(written.inference.seeds)} "
                f"cells={written.inference.design.confidence_cells} "
                f"run_id={written.run_id}"
            )


if __name__ == "__main__":
    main()
