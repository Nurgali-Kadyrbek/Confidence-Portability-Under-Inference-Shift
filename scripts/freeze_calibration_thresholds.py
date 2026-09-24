#!/usr/bin/env python3
"""Apply the frozen threshold rule to calibration and publish an immutable record.

For each model condition and dataset the outcome is exactly one of two things:
a threshold selected by the frozen rule, or no usable threshold under that rule.
Nothing here may be widened to rescue an infeasible pair - not the target risk,
not the sample, not the threshold grid, not the confidence definition.

The record it writes is the artifact the held-out phase reads, so it carries the
inputs that produced it: run id, config hash, analysis-plan hash and the
reference coverage and risk at the selected threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from cpis.analysis.matrix import _as_parsed, _resolve_official_scores
from cpis.analysis.metrics import (
    operational_endpoints,
    select_empirical_threshold,
    select_fixed_coverage_threshold,
)
from cpis.analysis_plan import load_analysis_plan
from cpis.matrix_config import load_matrix_config
from cpis.matrix_records import MatrixParsedRecord
from cpis.storage import StorageLayout, read_json


def evaluate(config_path: Path, plan, repository_root: Path) -> dict:
    config = load_matrix_config(config_path)
    storage = StorageLayout.from_spec(config.storage, repository_root)
    run = storage.output_root / config.run_id
    paths = sorted((run / "parsed").glob("*.json"))
    if not paths:
        raise RuntimeError(f"calibration run has no parsed records: {config.run_id}")
    records = [MatrixParsedRecord.model_validate(read_json(p)) for p in paths]
    records = _resolve_official_scores(
        type("R", (), {"derived": run / "derived"})(), records, paths
    )
    pending = [r for r in records if r.correct is None]
    if pending:
        raise RuntimeError(
            f"{config.run_id} has {len(pending)} unscored records; run score-matrix first"
        )
    coupled = [r for r in records if "coupled" in r.design_roles]
    parsed = [_as_parsed(r) for r in coupled]

    entry = {
        "experiment_id": config.experiment_id,
        "run_id": config.run_id,
        "config_sha256": config.config_sha256,
        "config_path": str(config_path.relative_to(repository_root)),
        "model_id": config.model.model_id,
        "model_mode": config.model.model_mode,
        "dataset_id": config.dataset.dataset_id,
        "scorer": config.dataset.scorer,
        "items": len(parsed),
        "target_risk": plan.risk_contract.target_risk,
        "minimum_coverage": plan.risk_contract.minimum_coverage,
        "threshold_selection": plan.risk_contract.threshold_selection,
    }
    # The frozen endpoint rule is certified_risk_threshold_else_fixed_coverage_
    # target: when the risk contract cannot be met, the policy falls back to the
    # prespecified universal fixed-coverage target rather than vanishing. This is
    # the frozen rule applied as written, not a rescue; the risk contract is
    # still recorded as infeasible for that pair.
    target = plan.fixed_coverage.universal_confirmatory_target
    contract = "attained"
    reason = None
    try:
        selected = select_empirical_threshold(parsed, plan.risk_contract.target_risk)
    except ValueError as exc:
        selected, contract, reason = None, "infeasible", str(exc)
    if selected is not None and selected.coverage < plan.risk_contract.minimum_coverage:
        reason = (
            f"selected coverage {selected.coverage:.4f} is below the frozen "
            f"minimum {plan.risk_contract.minimum_coverage}"
        )
        selected, contract = None, "infeasible"

    endpoint = "certified_risk_threshold"
    if selected is None:
        fallback = select_fixed_coverage_threshold(parsed, target)
        if fallback.threshold is None:
            return entry | {
                "status": "threshold infeasible",
                "risk_contract": contract,
                "reason": (
                    f"{reason}; the fixed-coverage fallback at {target} is also "
                    f"unattainable (best coverage {fallback.achieved_coverage:.4f})"
                ),
                "threshold": None,
            }
        endpoint = f"fixed_coverage_{target}"
        from cpis.analysis.metrics import selective_risk as _sr
        selected = _sr(parsed, fallback.threshold)
    endpoints = operational_endpoints(parsed, selected.threshold)
    return entry | {
        "status": "selected",
        "endpoint": endpoint,
        "risk_contract": contract,
        "risk_contract_reason": reason,
        "fixed_coverage_target": target if endpoint != "certified_risk_threshold" else None,
        "threshold": selected.threshold,
        "reference_coverage": round(selected.coverage, 6),
        "reference_risk": round(selected.risk, 6),
        "reference_accepted": selected.accepted_observations,
        "reference_answer_failure_rate": round(endpoints.answer_failure_rate, 6),
        "reference_confidence_invalid_rate": round(endpoints.confidence_invalid_rate, 6),
        "reference_operational_success": round(endpoints.operational_success, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--configs", type=Path, default=Path("configs/core/calibration"))
    parser.add_argument("--analysis-plan", type=Path, default=Path("configs/analysis/core-v5.yaml"))
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan = load_analysis_plan((root / args.analysis_plan).resolve())
    entries = [
        evaluate(path, plan, root)
        for path in sorted((root / args.configs).glob("*.yaml"))
    ]
    payload = {
        "schema_version": "1.0",
        "artifact": "cpis-calibration-thresholds-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_plan_id": plan.plan_id,
        "analysis_plan_sha256": plan.sha256,
        "rule": (
            "the frozen endpoint rule certified_risk_threshold_else_fixed_coverage"
            "_target: maximum empirical coverage subject to empirical selective risk "
            "at or below the frozen target and the frozen minimum coverage; when that "
            "contract cannot be met the policy falls back to the prespecified "
            "universal fixed-coverage target. The risk contract is still recorded as "
            "infeasible for such a pair. Nothing is widened to rescue one"
        ),
        "risk_contract_attained": sum(
            1 for e in entries if e.get("risk_contract") == "attained"
        ),
        "risk_contract_infeasible": sum(
            1 for e in entries if e.get("risk_contract") == "infeasible"
        ),
        "selected": sum(1 for e in entries if e["status"] == "selected"),
        "infeasible": sum(1 for e in entries if e["status"] != "selected"),
        "pairs": entries,
    }
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    payload["artifact_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    destination = (root / args.destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: payload[k] for k in ("selected", "infeasible", "artifact_sha256")}))
    for e in entries:
        tau = "infeasible" if e["threshold"] is None else f"{e['threshold']:.3f}"
        print(
            f"  {e['model_id'].split('/')[-1][:28]:<30}{e['model_mode']:<13}"
            f"{e['dataset_id'].split('/')[-1][:14]:<16}tau={tau:<12}"
            + (
                f"cov={e['reference_coverage']:.3f} risk={e['reference_risk']:.3f}"
                if e["threshold"] is not None
                else e["reason"][:60]
            )
        )


if __name__ == "__main__":
    main()
