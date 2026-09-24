#!/usr/bin/env python3
"""Why selective-risk equivalence was hard to establish, and why coverage overshot.

Two structural facts explain most of the held-out result and neither is a
finding about portability.

Selective risk is estimated on accepted items only, so its precision is set by
NK rather than by N. A pair that accepts a quarter of its items has a quarter
of the sample behind its risk estimate, and the prespecified equivalence margin
was fixed without reference to that. This reports, per pair, the accepted count,
the asymptotic standard error it implies, and the realised bootstrap interval
width against the equivalence band it would have to fit inside.

Elicited confidence is heavily tied, so coverage is a step function of the
threshold. A nominal fifty-percent coverage target cannot be met when a single
confidence value carries more than half the mass; the achieved coverage jumps
past it. This reports the atoms, their mass, the entropy of the reported
distribution and the coverage steps actually reachable.

Descriptive and secondary. Nothing here is corrected for multiplicity and
nothing here changes an estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from cpis.analysis.matrix import _as_parsed, _resolve_official_scores
from cpis.analysis.metrics import has_valid_confidence
from cpis.analysis_plan import load_analysis_plan
from cpis.matrix_config import load_matrix_config
from cpis.matrix_records import MatrixParsedRecord
from cpis.storage import StorageLayout, read_json


def load_coupled(config_path: Path, root: Path) -> list[MatrixParsedRecord]:
    config = load_matrix_config(config_path)
    storage = StorageLayout.from_spec(config.storage, root)
    run = storage.output_root / config.run_id
    paths = sorted((run / "parsed").glob("*.json"))
    records = [MatrixParsedRecord.model_validate(read_json(p)) for p in paths]
    records = _resolve_official_scores(
        type("R", (), {"derived": run / "derived"})(), records, paths
    )
    return config, [r for r in records if "coupled" in r.design_roles]


def quantisation(records: list[MatrixParsedRecord]) -> dict:
    """The reported-confidence distribution as a discrete measure."""
    valid = [float(r.confidence) for r in records if has_valid_confidence(r)]
    if not valid:
        return {"reported_values": 0}
    counts = Counter(valid)
    n = len(valid)
    mass = {f"{value:.3f}": count / n for value, count in sorted(counts.items())}
    entropy = -sum(
        (count / n) * math.log2(count / n) for count in counts.values()
    )
    largest_value, largest_count = max(counts.items(), key=lambda kv: kv[1])
    # Coverage is a step function of tau: only these values are reachable.
    reachable = []
    for value in sorted(counts, reverse=True):
        reachable.append(
            {
                "threshold": value,
                "coverage_at_or_above": sum(c for v, c in counts.items() if v >= value)
                / n,
            }
        )
    return {
        "observations": n,
        "distinct_reported_values": len(counts),
        "entropy_bits": entropy,
        "max_entropy_bits": math.log2(len(counts)) if len(counts) > 1 else 0.0,
        "largest_atom_value": largest_value,
        "largest_atom_mass": largest_count / n,
        "mass_by_value": mass,
        "reachable_coverage_steps": reachable,
    }


def risk_precision(entry: dict, contrasts: dict, margin: float, items: int) -> dict:
    """What precision the accepted sample can buy for a risk estimate."""
    coverage = entry["reference_coverage"]
    risk = entry["reference_risk"]
    accepted = entry["reference_accepted"]
    # The asymptotic standard error a single-arm binomial of this size implies.
    # The realised estimator is paired and bootstrapped, so this is a scale
    # reference rather than the quantity actually used for inference.
    se = math.sqrt(risk * (1.0 - risk) / accepted) if accepted else None
    out = {
        "reference_coverage": coverage,
        "reference_risk": risk,
        "accepted_observations_at_calibration": accepted,
        "held_out_items": items,
        "implied_accepted_at_held_out": round(items * coverage),
        "asymptotic_se_at_held_out": (
            math.sqrt(risk * (1.0 - risk) / (items * coverage))
            if coverage and risk is not None
            else None
        ),
        "calibration_asymptotic_se": se,
        "equivalence_band_width": 2 * margin,
        "contrasts": {},
    }
    for name, payload in (contrasts or {}).items():
        d = payload["deltas"]["R"]
        if d["ci_low"] is None:
            continue
        width = d["ci_high"] - d["ci_low"]
        out["contrasts"][name] = {
            "bootstrap_ci_width": round(width, 6),
            "width_over_equivalence_band": round(width / (2 * margin), 3),
            "band_could_contain_interval": bool(width < 2 * margin),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--calibration-configs", type=Path, default=Path("configs/core/calibration")
    )
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument(
        "--analysis-plan", type=Path, default=Path("configs/analysis/core-v5.yaml")
    )
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan = load_analysis_plan((root / args.analysis_plan).resolve())
    margin = plan.equivalence_margins.selective_risk
    thresholds = json.loads((root / args.thresholds).read_text(encoding="utf-8"))
    analysis = json.loads((root / args.analysis).read_text(encoding="utf-8"))
    by_key = {
        (p["model_id"], p["model_mode"], p["dataset_id"]): p for p in analysis["pairs"]
    }

    entries = []
    for path in sorted((root / args.calibration_configs).glob("*.yaml")):
        config, records = load_coupled(path, root)
        key = (config.model.model_id, config.model.model_mode, config.dataset.dataset_id)
        threshold_entry = next(
            p
            for p in thresholds["pairs"]
            if (p["model_id"], p["model_mode"], p["dataset_id"]) == key
        )
        held_out = by_key.get(key) or {}
        entries.append(
            {
                "model_id": config.model.model_id,
                "model_mode": config.model.model_mode,
                "dataset_id": config.dataset.dataset_id,
                "threshold": threshold_entry["threshold"],
                "endpoint": threshold_entry.get("endpoint"),
                "confidence_quantisation": quantisation(records),
                "selective_risk_precision": risk_precision(
                    threshold_entry,
                    held_out.get("contrasts"),
                    margin,
                    held_out.get("items") or 0,
                ),
            }
        )

    payload = {
        "schema_version": "1.0",
        "artifact": "cpis-measurement-properties-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "secondary descriptive; explains precision and threshold granularity",
        "equivalence_margin_selective_risk": margin,
        "pairs": entries,
    }
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    payload["artifact_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    destination = (root / args.destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for e in entries:
        q = e["confidence_quantisation"]
        p = e["selective_risk_precision"]
        widths = [c["width_over_equivalence_band"] for c in p["contrasts"].values()]
        print(
            f"  {e['model_id'].split('/')[-1][:22]:<24}{e['model_mode']:<13}"
            f"{('ManyIFEval' if 'Instruct' in e['dataset_id'] else 'SuperGPQA'):<12}"
            f"values={q['distinct_reported_values']:<4}"
            f"top_atom={q['largest_atom_value']:.2f}@{q['largest_atom_mass']:.2f}  "
            f"n_acc={p['implied_accepted_at_held_out']:<5}"
            f"SE~{p['asymptotic_se_at_held_out']:.3f}  "
            f"CI/band={'/'.join(f'{w:.1f}' for w in widths)}"
        )


if __name__ == "__main__":
    main()
