#!/usr/bin/env python3
"""Seed robustness: how much does a contrast move when the decoder is resampled?

The broad study's intervals resample items at a fixed draw from the decoder, so
they answer "would this hold on other items?" and not "would this hold on
another run?". This estimates the second.

For item i and seed s, the paired contribution to a contrast is

    D_is = Z_{i,theta1,s} - Z_{i,theta0,s}

and the law of total variance splits its variability into a within-item part
and a between-item part:

    Var_{i,s}(D) = E_i[Var_s(D | i)] + Var_i(E_s[D | i])

The first term is what resampling the decoder contributes; the second is item
heterogeneity, which is what the broad study's bootstrap already captures.

Two honesty constraints. The three seeds are generated in one run, so batch
composition is shared across them but a given item still lands beside different
neighbours at different seeds; what is measured is therefore the variability a
practitioner would see on a rerun, not a pure sampling-seed effect isolated
from batching. And this runs on 128 items per dataset, so its per-seed point
estimates are noisier than the 600-item held-out estimates and are not a
replacement for them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pvariance

from cpis.analysis.matrix import _as_parsed, _resolve_official_scores
from cpis.analysis.metrics import operational_endpoints, outcome_partition
from cpis.matrix_config import load_matrix_config
from cpis.matrix_records import MatrixParsedRecord
from cpis.storage import StorageLayout, read_json

ENDPOINTS = ("q", "K", "R", "S", "E", "aC", "aTau")
CONTRASTS = {
    "reference_to_high_temperature": "high-temperature",
    "reference_to_low_top_p": "low-top-p",
}


def endpoint_vector(records, threshold: float) -> dict[str, float | None]:
    e = operational_endpoints(records, threshold)
    p = outcome_partition(records, threshold)
    return {
        "q": e.answer_failure_rate,
        "K": e.coverage,
        "R": e.selective_risk,
        "S": e.operational_success,
        "E": p.accepted_error,
        "aC": p.confidence_failure,
        "aTau": p.low_confidence_abstention,
    }


def load(config_path: Path, root: Path):
    config = load_matrix_config(config_path)
    storage = StorageLayout.from_spec(config.storage, root)
    run = storage.output_root / config.run_id
    paths = sorted((run / "parsed").glob("*.json"))
    if not paths:
        raise RuntimeError(f"no parsed records for {config.run_id}")
    records = [MatrixParsedRecord.model_validate(read_json(p)) for p in paths]
    records = _resolve_official_scores(
        type("R", (), {"derived": run / "derived"})(), records, paths
    )
    unscored = [r for r in records if r.correct is None]
    if unscored:
        raise RuntimeError(f"{config.run_id}: {len(unscored)} unscored records")
    coupled = [r for r in records if "coupled" in r.design_roles]
    by_seed_condition = defaultdict(list)
    for r in coupled:
        by_seed_condition[(r.seed, r.answer_condition_id)].append(_as_parsed(r))
    return config, by_seed_condition


def item_level_deltas(reference, shifted, threshold: float) -> dict[str, dict]:
    """Per-item contribution to each endpoint, as an indicator difference.

    An item's contribution to a rate endpoint is its own indicator, so the
    per-item delta is well defined for the partition components and for
    coverage. Selective risk is a ratio of two of them and has no per-item
    decomposition; it is handled at the aggregate level instead.
    """
    ref_by_item = {r.dataset_item_id: r for r in reference}
    shf_by_item = {r.dataset_item_id: r for r in shifted}
    shared = sorted(set(ref_by_item) & set(shf_by_item))
    out = {}
    for item in shared:
        a = endpoint_vector([ref_by_item[item]], threshold)
        b = endpoint_vector([shf_by_item[item]], threshold)
        out[item] = {
            k: (None if a[k] is None or b[k] is None else b[k] - a[k])
            for k in ENDPOINTS
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--configs", type=Path, default=Path("configs/core/seeds"))
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()

    root = args.repository_root.resolve()
    thresholds = json.loads((root / args.thresholds).read_text(encoding="utf-8"))
    tau = {
        (p["model_id"], p["model_mode"], p["dataset_id"]): p for p in thresholds["pairs"]
    }

    results = []
    for path in sorted((root / args.configs).glob("*.yaml")):
        config, by_seed_condition = load(path, root)
        key = (config.model.model_id, config.model.model_mode, config.dataset.dataset_id)
        entry_tau = tau[key]
        if entry_tau["status"] != "selected":
            results.append(
                {
                    "experiment_id": config.experiment_id,
                    "status": "no threshold; calibration infeasible",
                }
            )
            continue
        threshold = float(entry_tau["threshold"])
        seeds = sorted(config.inference.seeds)
        entry = {
            "experiment_id": config.experiment_id,
            "run_id": config.run_id,
            "model_id": config.model.model_id,
            "model_mode": config.model.model_mode,
            "dataset_id": config.dataset.dataset_id,
            "threshold": threshold,
            "seeds": seeds,
            "items": config.dataset.item_subselection_limit,
            "status": "estimated",
            "contrasts": {},
        }
        for contrast, condition in CONTRASTS.items():
            per_seed = {}
            per_item_by_seed = {}
            for seed in seeds:
                reference = by_seed_condition.get((seed, "reference"), [])
                shifted = by_seed_condition.get((seed, condition), [])
                if not reference or not shifted:
                    continue
                a = endpoint_vector(reference, threshold)
                b = endpoint_vector(shifted, threshold)
                per_seed[seed] = {
                    k: (None if a[k] is None or b[k] is None else round(b[k] - a[k], 6))
                    for k in ENDPOINTS
                }
                per_item_by_seed[seed] = item_level_deltas(reference, shifted, threshold)
            if len(per_seed) < 2:
                continue

            across = {}
            for k in ENDPOINTS:
                values = [v[k] for v in per_seed.values() if v[k] is not None]
                if len(values) < 2:
                    continue
                signs = {1 if v > 0 else (-1 if v < 0 else 0) for v in values}
                across[k] = {
                    "per_seed": {str(s): per_seed[s][k] for s in per_seed},
                    "mean": round(fmean(values), 6),
                    "sd": round(pvariance(values) ** 0.5, 6),
                    "range": round(max(values) - min(values), 6),
                    "sign_consistent": len(signs - {0}) <= 1,
                }

            # Law of total variance over the per-item, per-seed contributions.
            decomposition = {}
            items = sorted(
                set.intersection(*(set(d) for d in per_item_by_seed.values()))
            )
            # R is a ratio of two partition components, not a per-item rate.
            # Decomposing it here would silently restrict to items accepted at
            # two or more seeds, which is a selected subset, so it is left out.
            for k in (e for e in ENDPOINTS if e != "R"):
                cells = {
                    item: [
                        per_item_by_seed[s][item][k]
                        for s in per_item_by_seed
                        if per_item_by_seed[s][item][k] is not None
                    ]
                    for item in items
                }
                usable = {i: v for i, v in cells.items() if len(v) >= 2}
                if len(usable) < 2:
                    continue
                within = fmean(pvariance(v) for v in usable.values())
                between = pvariance([fmean(v) for v in usable.values()])
                total = within + between
                decomposition[k] = {
                    "within_item_across_seeds": round(within, 8),
                    "between_item": round(between, 8),
                    "total": round(total, 8),
                    "seed_share_of_variance": (
                        None if total == 0 else round(within / total, 6)
                    ),
                    "items": len(usable),
                }
            entry["contrasts"][contrast] = {
                "across_seed_summary": across,
                "variance_decomposition": decomposition,
            }
        results.append(entry)

    payload = {
        "schema_version": "1.0",
        "artifact": "cpis-seed-robustness-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "secondary robustness; not confirmatory and not multiplicity corrected",
        "threshold_artifact_sha256": thresholds.get("artifact_sha256"),
        "interpretation": (
            "seed_share_of_variance is the fraction of the per-item contrast "
            "variability attributable to resampling the decoder, under the "
            "batching regime of this run. Selective risk R is a ratio and has "
            "no per-item decomposition; its across-seed spread is reported but "
            "it carries no variance split"
        ),
        "pairs": results,
    }
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    payload["artifact_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    destination = (root / args.destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"pairs": len(results), "destination": str(destination)}))


if __name__ == "__main__":
    main()
