"""Exact risk-contract certification on the sealed certification partition."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from cpis.analysis.matrix import (
    _as_parsed,
    _load_replica_manifests,
    _resolve_official_scores,
)
from cpis.analysis.metrics import certify_risk_contract
from cpis.analysis.units import deterministic_balanced_group_anchors
from cpis.analysis_plan import load_analysis_plan
from cpis.datasets import load_dataset, partition_items
from cpis.manifest import git_metadata
from cpis.matrix_config import load_matrix_config
from cpis.matrix_records import MatrixParsedRecord
from cpis.policy import FrozenReadoutPolicy, validate_policy_reference
from cpis.phase_gate import (
    authorize_phase_analysis_plan,
    authorize_phase_config,
    record_dataset_access,
    require_phase_gate,
)
from cpis.storage import StorageLayout, read_json, write_immutable_json


def _certify_readout(
    policy: FrozenReadoutPolicy,
    records: list[MatrixParsedRecord],
    plan,
) -> dict:
    if policy.risk_contract.status == "risk-contract infeasible":
        return {
            "status": "risk-contract infeasible",
            "stage": "development_selection",
            "reason": policy.risk_contract.reason,
            "certification": None,
            "incidental_records_not_used": len(records),
        }
    threshold = policy.risk_contract.threshold
    assert threshold is not None
    result = certify_risk_contract(
        [_as_parsed(record) for record in records],
        threshold=threshold,
        target_risk=plan.risk_contract.target_risk,
        minimum_coverage=plan.risk_contract.minimum_coverage,
        tail_probability=plan.risk_contract.certification_tail_probability,
        certification_replicate_seed=plan.risk_contract.certification_replicate_seed,
    )
    return {
        "status": result.status,
        "stage": "certification",
        "reason": result.reason,
        "certification": asdict(result),
    }


def analyze_certification(
    config_path: Path, analysis_plan_path: Path, repository_root: Path
) -> tuple[Path, dict]:
    """Apply frozen thresholds; never select or modify a policy here."""
    repository_root = repository_root.resolve()
    config_path = config_path.resolve()
    config = load_matrix_config(config_path)
    if config.study != "certification" or config.dataset.partition != "certification":
        raise ValueError("certification analysis requires a certification matrix")
    if config.frozen_policy is None:
        raise ValueError("certification analysis requires a frozen policy")
    gate = require_phase_gate("certification", repository_root)
    authorize_phase_config(gate, config_path, repository_root)
    plan = load_analysis_plan(analysis_plan_path.resolve())
    authorize_phase_analysis_plan(gate, plan.sha256)
    if plan.schema_version not in {"3.0", "4.0"} or plan.grouped_certification is None:
        raise ValueError("certification analysis requires analysis plan v3+")
    policy = validate_policy_reference(config.frozen_policy, repository_root)
    if policy.analysis_plan_sha256 != plan.sha256:
        raise RuntimeError("frozen policy and certification analysis plan differ")

    storage = StorageLayout.from_spec(config.storage, repository_root)
    run = storage.run(config.run_id)
    record_dataset_access(
        run,
        run_id=config.run_id,
        experiment_id=config.experiment_id,
        config_sha256=config.config_sha256,
        dataset_id=config.dataset.dataset_id,
        dataset_revision=config.dataset.revision,
        partition=config.dataset.partition,
        replica_index=0,
        gate=gate,
    )
    manifests = _load_replica_manifests(run, config, repository_root)
    parsed_paths = sorted(run.parsed.glob("*.json"))
    records = [MatrixParsedRecord.model_validate(read_json(path)) for path in parsed_paths]
    records = _resolve_official_scores(run, records, parsed_paths)
    if any(record.correct is None for record in records):
        raise RuntimeError("certification requires completed deterministic scoring")

    all_items = load_dataset(config.dataset, config_path.parent, storage.dataset_cache)
    selected_items = partition_items(all_items, config.dataset)
    units = deterministic_balanced_group_anchors(
        selected_items, plan.grouped_certification.selector_salt
    )
    anchors = units.selected_item_ids
    reference_answer = config.inference.design.reference_answer_condition_id
    coupled = [
        record
        for record in records
        if record.dataset_item_id in anchors
        and record.answer_condition_id == reference_answer
        and "coupled" in record.design_roles
    ]
    standardized = [
        record
        for record in records
        if record.dataset_item_id in anchors
        and record.answer_condition_id == reference_answer
        and "standardized" in record.design_roles
    ]
    if policy.coupled.risk_contract.status == "candidate" and len(coupled) != len(anchors):
        raise RuntimeError("certification coupled readout is incomplete")
    if (
        policy.standardized_deterministic.risk_contract.status == "candidate"
        and len(standardized) != len(anchors)
    ):
        raise RuntimeError("certification standardized readout is incomplete")

    result = {
        "schema_version": "1.0",
        "analysis_id": f"{config.experiment_id}-risk-certification-v1",
        "confirmatory": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": config.run_id,
        "generation_git_commit": manifests[0]["git"]["commit"],
        "analysis_git": git_metadata(repository_root),
        "config_sha256": config.config_sha256,
        "analysis_plan_id": plan.plan_id,
        "analysis_plan_sha256": plan.sha256,
        "frozen_policy_id": policy.policy_id,
        "frozen_policy_sha256": policy.canonical_sha256,
        "unit_selection": {
            "method": units.method,
            "selector_salt": units.selector_salt,
            "source_groups": units.source_groups,
            "selected_rows": len(units.selected_item_ids),
            "variant_counts": units.variant_counts,
        },
        "coupled": _certify_readout(policy.coupled, coupled, plan),
        "standardized_deterministic": _certify_readout(
            policy.standardized_deterministic, standardized, plan
        ),
    }
    output_path = run.statistics / "risk-contract-certification-v1.json"
    if output_path.exists():
        existing = read_json(output_path)
        comparable = dict(result)
        comparable["created_at_utc"] = existing.get("created_at_utc")
        comparable["analysis_git"] = existing.get("analysis_git")
        if existing != comparable:
            raise RuntimeError(f"certification result collision at {output_path}")
        return output_path, existing
    write_immutable_json(output_path, result)
    return output_path, result
