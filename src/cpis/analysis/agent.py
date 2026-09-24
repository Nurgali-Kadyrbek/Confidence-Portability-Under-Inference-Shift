"""Saved-output development analysis for the BFCL agent validation study."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from cpis.agent_config import load_agent_config
from cpis.agent_pipeline import _build_observations
from cpis.agent_records import AgentEpisodeRecord, AgentParsedRecord
from cpis.analysis.certification import _certify_readout
from cpis.analysis.confirmatory import (
    _certified_risk_analysis,
    _confidence_diagnostics,
    _confidence_portability,
    _frozen_coverage_analysis,
    _randomization_p_value,
    _same_answer_analysis,
)
from cpis.analysis.matrix import (
    _accuracy,
    _as_parsed,
    _bootstrap_paired,
    _decomposition,
    _fixed_coverage_analysis,
    _interval,
    _load_replica_manifests,
    _mean_confidence,
    _resampling_design,
    _summary,
)
from cpis.analysis.metrics import holm_adjust, select_risk_contract_threshold
from cpis.analysis.units import deterministic_balanced_category_group_anchors
from cpis.analysis_plan import load_analysis_plan
from cpis.datasets import load_dataset, partition_items
from cpis.manifest import git_metadata
from cpis.matrix_records import MatrixParsedRecord
from cpis.phase_gate import (
    authorize_phase_analysis_plan,
    authorize_phase_config,
    record_dataset_access,
    require_phase_gate,
)
from cpis.policy import (
    validate_certification_decision_reference,
    validate_policy_reference,
)
from cpis.records import GenerationRecord
from cpis.storage import StorageLayout, read_json, write_immutable_json


def _transcript_sha256(messages: tuple[dict, ...]) -> str:
    return hashlib.sha256(
        json.dumps(
            messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _as_matrix(
    parsed: AgentParsedRecord,
    episode: AgentEpisodeRecord,
    confidence_raw: GenerationRecord,
) -> MatrixParsedRecord:
    answer_valid = not episode.force_terminated and episode.invalid_tool_outputs == 0
    confidence_valid = parsed.confidence_parse_status == "valid"
    first_seed = episode.steps[0].sampling_seed if episode.steps else 0
    return MatrixParsedRecord(
        record_id=parsed.record_id,
        answer_observation_id=parsed.episode_observation_id,
        answer_raw_record_id=parsed.episode_raw_record_id,
        answer_raw_record_sha256=parsed.episode_raw_record_sha256,
        confidence_raw_record_id=parsed.confidence_raw_record_id,
        confidence_raw_record_sha256=parsed.confidence_raw_record_sha256,
        run_id=parsed.run_id,
        experiment_id=parsed.experiment_id,
        dataset_id=parsed.dataset_id,
        dataset_revision=parsed.dataset_revision,
        dataset_item_id=parsed.dataset_item_id,
        dataset_partition=parsed.dataset_partition,
        answer_condition_id=parsed.answer_condition_id,
        confidence_readout_id=parsed.confidence_readout_id,
        design_roles=parsed.design_roles,
        seed=parsed.seed,
        answer_sampling_seed=first_seed,
        confidence_sampling_seed=confidence_raw.sampling_seed,
        answer_parser_version=episode.parser_version,
        confidence_parser_version="final_json_confidence_v2",
        scorer=parsed.scorer,
        answer_parse_status="valid" if answer_valid else "invalid",
        confidence_parse_status=parsed.confidence_parse_status,
        parse_status="valid" if answer_valid and confidence_valid else "invalid",
        parsed_answer=_transcript_sha256(episode.final_messages),
        confidence=parsed.confidence,
        scoring_status="scored",
        correct=parsed.correct,
        answer_parse_error=(
            None
            if answer_valid
            else "forced termination or malformed native tool-call attempt"
        ),
        confidence_parse_error=parsed.confidence_parse_error,
    )


def analyze_agent_development(
    config_path: Path, analysis_plan_path: Path, repository_root: Path
) -> tuple[Path, dict]:
    """Analyze a complete BFCL development matrix without model execution."""

    config_path = config_path.resolve()
    repository_root = repository_root.resolve()
    config = load_agent_config(config_path)
    if (
        config.phase != "development"
        or config.dataset.partition != "development"
        or config.dataset.item_limit is not None
    ):
        raise ValueError("agent development analysis requires the full development data")
    gate = require_phase_gate(config.dataset.partition, repository_root)
    authorize_phase_config(gate, config_path, repository_root)
    plan = load_analysis_plan(analysis_plan_path.resolve())
    authorize_phase_analysis_plan(gate, plan.sha256)
    if plan.schema_version != "4.0" or plan.agent_grouped_certification is None:
        raise ValueError("agent analysis requires analysis plan v4")
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
    manifests = _load_replica_manifests(  # type: ignore[arg-type]
        run, config, repository_root
    )
    parsed_paths = sorted(run.parsed.glob("*.json"))
    parsed = [AgentParsedRecord.model_validate(read_json(path)) for path in parsed_paths]
    episode_records = {
        record.observation_id: record
        for path in sorted(run.raw_answer.glob("*.json"))
        if (record := AgentEpisodeRecord.model_validate(read_json(path)))
    }
    confidence_records = {
        record.record_id: record
        for path in sorted(run.raw_confidence.glob("*.json"))
        if (record := GenerationRecord.model_validate(read_json(path)))
    }
    records: list[MatrixParsedRecord] = []
    for record in parsed:
        try:
            episode = episode_records[record.episode_observation_id]
        except KeyError as exc:
            raise RuntimeError("agent parsed record lacks its immutable episode") from exc
        if episode.sha256 != record.episode_raw_record_sha256:
            raise RuntimeError("agent episode hash differs from parsed provenance")
        try:
            confidence_raw = confidence_records[record.confidence_raw_record_id]
        except KeyError as exc:
            raise RuntimeError("agent parsed record lacks its confidence raw record") from exc
        if confidence_raw.sha256 != record.confidence_raw_record_sha256:
            raise RuntimeError("agent confidence hash differs from parsed provenance")
        records.append(_as_matrix(record, episode, confidence_raw))

    items = partition_items(
        load_dataset(config.dataset, config_path.parent, storage.dataset_cache),
        config.dataset,
    )
    episodes, cells = _build_observations(config, items)
    if len(episode_records) != len(episodes) or len(records) != len(cells):
        raise RuntimeError(
            f"agent matrix is incomplete: expected {len(episodes)} episodes and "
            f"{len(cells)} cells, got {len(episode_records)} and {len(records)}"
        )
    if any(record.run_id != config.run_id for record in records):
        raise RuntimeError("agent parsed matrix contains another run ID")
    cluster_by_item, strata_by_cluster, _ = _resampling_design(
        items, config.dataset.dataset_id
    )
    anchors = deterministic_balanced_category_group_anchors(
        items,
        plan.agent_grouped_certification.selector_salt,
        plan.agent_grouped_certification.variant_key,
    )
    condition_ids = [entry.condition_id for entry in config.inference.answer_conditions]
    coupled = {
        condition: [
            record
            for record in records
            if record.answer_condition_id == condition
            and "coupled" in record.design_roles
        ]
        for condition in condition_ids
    }
    standardized = {
        condition: [
            record
            for record in records
            if record.answer_condition_id == condition
            and "standardized" in record.design_roles
        ]
        for condition in condition_ids
    }
    design = config.inference.design
    reference_id = design.reference_answer_condition_id
    reference_coupled = coupled[reference_id]
    reference_standardized = standardized[reference_id]
    shifted_coupled = {key: value for key, value in coupled.items() if key != reference_id}
    shifted_standardized = {
        key: value for key, value in standardized.items() if key != reference_id
    }
    anchor_ids = anchors.selected_item_ids
    risk_coupled = [
        record for record in reference_coupled if record.dataset_item_id in anchor_ids
    ]
    risk_standardized = [
        record
        for record in reference_standardized
        if record.dataset_item_id in anchor_ids
    ]
    coupled_selection = select_risk_contract_threshold(
        [_as_parsed(record) for record in risk_coupled],
        plan.risk_contract.target_risk,
        plan.risk_contract.minimum_coverage,
    )
    standardized_selection = select_risk_contract_threshold(
        [_as_parsed(record) for record in risk_standardized],
        plan.risk_contract.target_risk,
        plan.risk_contract.minimum_coverage,
    )
    condition_effects = {}
    for index, condition in enumerate(condition_ids):
        if condition == reference_id:
            continue
        ref_mean = _mean_confidence(reference_coupled)
        shift_mean = _mean_confidence(coupled[condition])
        condition_effects[condition] = {
            "delta_success": _accuracy(coupled[condition])
            - _accuracy(reference_coupled),
            "delta_success_cluster_bootstrap_95": _interval(
                _bootstrap_paired(
                    reference_coupled,
                    coupled[condition],
                    _accuracy,
                    plan.bootstrap.replicates,
                    plan.bootstrap.seed + 200_000 + index,
                    cluster_by_item,
                    strata_by_cluster,
                )
            ),
            "delta_mean_confidence_valid_only": (
                shift_mean - ref_mean
                if shift_mean is not None and ref_mean is not None
                else None
            ),
        }
    result = {
        "schema_version": "2.0",
        "analysis_id": f"{config.experiment_id}-analysis-v2",
        "exploratory": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": config.run_id,
        "generation_git_commit": manifests[0]["git"]["commit"],
        "analysis_git": git_metadata(repository_root),
        "config_sha256": config.config_sha256,
        "analysis_plan_id": plan.plan_id,
        "analysis_plan_sha256": plan.sha256,
        "experimental_units": len(set(cluster_by_item.values())),
        "dataset_rows": len(items),
        "replicate_seeds": list(config.inference.seeds),
        "parsed_cells": len(records),
        "invalid_tool_episode_observations": sum(
            record.invalid_tool_outputs > 0 or record.force_terminated
            for record in episode_records.values()
        ),
        "invalid_confidence_cells": sum(record.confidence is None for record in records),
        "bootstrap": {
            "replicates": plan.bootstrap.replicates,
            "unit": "bfcl_base_scenario",
            "stratified": False,
        },
        "risk_contract_unit_selection": {
            "method": anchors.method,
            "selector_salt": anchors.selector_salt,
            "source_groups": anchors.source_groups,
            "selected_rows": len(anchors.selected_item_ids),
            "variant_counts": anchors.variant_counts,
        },
        "coupled": {
            "summaries": {
                key: _summary(value, cluster_by_item) for key, value in coupled.items()
            },
            "reference_risk_contract_selection": asdict(coupled_selection),
            "condition_effects": condition_effects,
            "fixed_coverage": _fixed_coverage_analysis(
                reference_coupled,
                shifted_coupled,
                plan.fixed_coverage.targets,
                plan.bootstrap.replicates,
                plan.bootstrap.seed + 201_000,
                cluster_by_item,
                strata_by_cluster,
            ),
        },
        "standardized_deterministic": {
            "summaries": {
                key: _summary(value, cluster_by_item)
                for key, value in standardized.items()
            },
            "reference_risk_contract_selection": asdict(standardized_selection),
            "fixed_coverage": _fixed_coverage_analysis(
                reference_standardized,
                shifted_standardized,
                plan.fixed_coverage.targets,
                plan.bootstrap.replicates,
                plan.bootstrap.seed + 202_000,
                cluster_by_item,
                strata_by_cluster,
            ),
        },
        "decomposition": _decomposition(
            records,
            config,  # type: ignore[arg-type]
            plan.bootstrap.replicates,
            plan.bootstrap.seed + 203_000,
            cluster_by_item,
            strata_by_cluster,
        ),
    }
    output = run.statistics / "development-agent-analysis-v2.json"
    if output.exists():
        existing = read_json(output)
        comparable = dict(result)
        comparable["created_at_utc"] = existing.get("created_at_utc")
        comparable["analysis_git"] = existing.get("analysis_git")
        if existing != comparable:
            raise RuntimeError(f"agent analysis collision at {output}")
        return output, existing
    write_immutable_json(output, result)
    return output, result


def analyze_agent_certification(
    config_path: Path, analysis_plan_path: Path, repository_root: Path
) -> tuple[Path, dict]:
    """Certify BFCL policies on one balanced row per independent scenario."""

    config_path = config_path.resolve()
    repository_root = repository_root.resolve()
    config = load_agent_config(config_path)
    if config.phase != "certification" or config.dataset.partition != "certification":
        raise ValueError("agent certification analysis requires certification data")
    if config.frozen_policy is None:
        raise ValueError("agent certification requires a frozen policy")
    gate = require_phase_gate("certification", repository_root)
    authorize_phase_config(gate, config_path, repository_root)
    plan = load_analysis_plan(analysis_plan_path.resolve())
    authorize_phase_analysis_plan(gate, plan.sha256)
    if plan.schema_version != "4.0" or plan.agent_grouped_certification is None:
        raise ValueError("agent certification requires analysis plan v4")
    policy = validate_policy_reference(config.frozen_policy, repository_root)
    if policy.analysis_plan_sha256 != plan.sha256:
        raise RuntimeError("agent policy and certification plan differ")
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
    manifests = _load_replica_manifests(  # type: ignore[arg-type]
        run, config, repository_root
    )
    parsed = [
        AgentParsedRecord.model_validate(read_json(path))
        for path in sorted(run.parsed.glob("*.json"))
    ]
    episodes = {
        record.observation_id: record
        for path in sorted(run.raw_answer.glob("*.json"))
        if (record := AgentEpisodeRecord.model_validate(read_json(path)))
    }
    confidence_records = {
        record.record_id: record
        for path in sorted(run.raw_confidence.glob("*.json"))
        if (record := GenerationRecord.model_validate(read_json(path)))
    }
    records = []
    for parsed_record in parsed:
        try:
            episode = episodes[parsed_record.episode_observation_id]
            confidence = confidence_records[parsed_record.confidence_raw_record_id]
        except KeyError as exc:
            raise RuntimeError("agent certification is missing immutable raw data") from exc
        if (
            episode.sha256 != parsed_record.episode_raw_record_sha256
            or confidence.sha256 != parsed_record.confidence_raw_record_sha256
        ):
            raise RuntimeError("agent certification raw/parsed provenance mismatch")
        records.append(_as_matrix(parsed_record, episode, confidence))
    items = partition_items(
        load_dataset(config.dataset, config_path.parent, storage.dataset_cache),
        config.dataset,
    )
    expected_episodes, expected_cells = _build_observations(config, items)
    if len(episodes) != len(expected_episodes) or len(records) != len(expected_cells):
        raise RuntimeError("agent certification output is incomplete")
    units = deterministic_balanced_category_group_anchors(
        items,
        plan.agent_grouped_certification.selector_salt,
        plan.agent_grouped_certification.variant_key,
    )
    anchors = units.selected_item_ids
    reference = config.inference.design.reference_answer_condition_id
    coupled = [
        record
        for record in records
        if record.dataset_item_id in anchors
        and record.answer_condition_id == reference
        and "coupled" in record.design_roles
    ]
    standardized = [
        record
        for record in records
        if record.dataset_item_id in anchors
        and record.answer_condition_id == reference
        and "standardized" in record.design_roles
    ]
    if policy.coupled.risk_contract.status == "candidate" and len(coupled) != len(
        anchors
    ):
        raise RuntimeError("agent coupled certification readout is incomplete")
    if (
        policy.standardized_deterministic.risk_contract.status == "candidate"
        and len(standardized) != len(anchors)
    ):
        raise RuntimeError("agent standardized certification readout is incomplete")
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
    output = run.statistics / "risk-contract-certification-v1.json"
    if output.exists():
        existing = read_json(output)
        comparable = dict(result)
        comparable["created_at_utc"] = existing.get("created_at_utc")
        comparable["analysis_git"] = existing.get("analysis_git")
        if existing != comparable:
            raise RuntimeError(f"agent certification collision at {output}")
        return output, existing
    write_immutable_json(output, result)
    return output, result


def analyze_agent_test(
    config_path: Path, analysis_plan_path: Path, repository_root: Path
) -> tuple[Path, dict]:
    """Analyze untouched BFCL test episodes with scenario-clustered inference."""

    config_path = config_path.resolve()
    repository_root = repository_root.resolve()
    config = load_agent_config(config_path)
    if config.phase != "test" or config.dataset.partition != "test" or not config.confirmatory:
        raise ValueError("agent test analysis requires untouched confirmatory test data")
    if config.frozen_policy is None or config.certification_decision is None:
        raise ValueError("agent test analysis requires both frozen gate artifacts")
    gate = require_phase_gate("test", repository_root)
    authorize_phase_config(gate, config_path, repository_root)
    plan = load_analysis_plan(analysis_plan_path.resolve())
    authorize_phase_analysis_plan(gate, plan.sha256)
    if plan.schema_version != "4.0" or plan.robustness is None:
        raise ValueError("agent test analysis requires analysis plan v4")
    policy = validate_policy_reference(config.frozen_policy, repository_root)
    decision = validate_certification_decision_reference(
        config.certification_decision, repository_root
    )
    if (
        policy.analysis_plan_sha256 != plan.sha256
        or decision.analysis_plan_sha256 != plan.sha256
        or decision.frozen_policy_sha256 != policy.canonical_sha256
    ):
        raise RuntimeError("agent test artifacts do not share one frozen plan")

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
    manifests = _load_replica_manifests(  # type: ignore[arg-type]
        run, config, repository_root
    )
    parsed = [
        AgentParsedRecord.model_validate(read_json(path))
        for path in sorted(run.parsed.glob("*.json"))
    ]
    episodes = {
        record.observation_id: record
        for path in sorted(run.raw_answer.glob("*.json"))
        if (record := AgentEpisodeRecord.model_validate(read_json(path)))
    }
    confidence_records = {
        record.record_id: record
        for path in sorted(run.raw_confidence.glob("*.json"))
        if (record := GenerationRecord.model_validate(read_json(path)))
    }
    records: list[MatrixParsedRecord] = []
    for parsed_record in parsed:
        try:
            episode = episodes[parsed_record.episode_observation_id]
            confidence = confidence_records[parsed_record.confidence_raw_record_id]
        except KeyError as exc:
            raise RuntimeError("agent test is missing immutable raw data") from exc
        if (
            episode.sha256 != parsed_record.episode_raw_record_sha256
            or confidence.sha256 != parsed_record.confidence_raw_record_sha256
        ):
            raise RuntimeError("agent test raw/parsed provenance mismatch")
        records.append(_as_matrix(parsed_record, episode, confidence))

    items = partition_items(
        load_dataset(config.dataset, config_path.parent, storage.dataset_cache),
        config.dataset,
    )
    expected_episodes, expected_cells = _build_observations(config, items)
    if len(episodes) != len(expected_episodes) or len(records) != len(expected_cells):
        raise RuntimeError(
            f"agent test is incomplete: expected {len(expected_episodes)} episodes and "
            f"{len(expected_cells)} cells, got {len(episodes)} and {len(records)}"
        )
    cluster_by_item, strata_by_cluster, stratum_counts = _resampling_design(
        items, config.dataset.dataset_id
    )
    design = config.inference.design
    condition_ids = [condition.condition_id for condition in config.inference.answer_conditions]
    expected_conditions = {
        "reference",
        "low-temperature",
        "high-temperature",
        "low-top-p",
        "high-diversity",
    }
    if set(condition_ids) != expected_conditions:
        raise RuntimeError("agent test does not contain the four frozen contrasts")
    coupled = {
        condition: [
            record
            for record in records
            if record.answer_condition_id == condition and "coupled" in record.design_roles
        ]
        for condition in condition_ids
    }
    standardized = {
        condition: [
            record
            for record in records
            if record.answer_condition_id == condition
            and "standardized" in record.design_roles
        ]
        for condition in condition_ids
    }
    reference_id = design.reference_answer_condition_id
    shifted_coupled = {key: value for key, value in coupled.items() if key != reference_id}
    shifted_standardized = {
        key: value for key, value in standardized.items() if key != reference_id
    }
    coupled_risk = _certified_risk_analysis(
        coupled[reference_id],
        shifted_coupled,
        decision.coupled,
        plan.risk_contract.target_risk,
        plan,
        cluster_by_item,
        strata_by_cluster,
    )
    standardized_risk = _certified_risk_analysis(
        standardized[reference_id],
        shifted_standardized,
        decision.standardized_deterministic,
        plan.risk_contract.target_risk,
        plan,
        cluster_by_item,
        strata_by_cluster,
    )
    coupled_fixed = _frozen_coverage_analysis(
        coupled[reference_id],
        shifted_coupled,
        policy.coupled.fixed_coverage,
        config,  # type: ignore[arg-type]
        plan,
        cluster_by_item,
        strata_by_cluster,
    )
    standardized_fixed = _frozen_coverage_analysis(
        standardized[reference_id],
        shifted_standardized,
        policy.standardized_deterministic.fixed_coverage,
        config,  # type: ignore[arg-type]
        plan,
        cluster_by_item,
        strata_by_cluster,
    )

    certified = decision.coupled.status == "certified"
    if certified:
        if decision.coupled.threshold is None:
            raise RuntimeError("certified agent policy has no threshold")
        test_threshold = decision.coupled.threshold
    else:
        frozen = next(
            entry
            for entry in policy.coupled.fixed_coverage
            if math.isclose(
                entry.target_coverage,
                plan.fixed_coverage.universal_confirmatory_target,
            )
        )
        if frozen.threshold is None:
            raise RuntimeError("universal agent fixed-coverage target is unattainable")
        test_threshold = frozen.threshold
    randomization = {
        condition: _randomization_p_value(
            coupled[reference_id],
            coupled[condition],
            test_threshold,
            cluster_by_item,
            plan.multiplicity.randomization_replicates,
            plan.multiplicity.randomization_seed + index,
            two_sided=not certified,
        )
        for index, condition in enumerate(
            ("low-temperature", "high-temperature", "low-top-p", "high-diversity")
        )
    }
    holm = holm_adjust(
        {condition: value["p_value"] for condition, value in randomization.items()},
        plan.multiplicity.familywise_alpha,
    )
    summaries = {
        condition: {
            **_summary(rows, cluster_by_item),
            **_confidence_diagnostics(
                rows,
                plan.robustness.ece_bins,
                plan.robustness.calibration_sensitivity_epsilon,
            ),
        }
        for condition, rows in coupled.items()
    }
    result = {
        "schema_version": "1.0",
        "analysis_id": f"{config.experiment_id}-confirmatory-test-analysis-v1",
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
        "certification_decision_id": decision.decision_id,
        "certification_decision_sha256": decision.canonical_sha256,
        "experimental_units": len(set(cluster_by_item.values())),
        "dataset_rows": len(items),
        "replicate_seeds": list(config.inference.seeds),
        "parsed_cells": len(records),
        "bootstrap": {
            "replicates": plan.bootstrap.replicates,
            "unit": "bfcl_base_scenario",
            "stratified": strata_by_cluster is not None,
            "strata": len(stratum_counts),
        },
        "coupled": {
            "summaries": summaries,
            "risk_contract_transport": coupled_risk,
            "fixed_coverage_transport": coupled_fixed,
            "confidence_portability": _confidence_portability(
                coupled[reference_id],
                shifted_coupled,
                plan,
                cluster_by_item,
                strata_by_cluster,
                260_000,
            ),
        },
        "standardized_deterministic": {
            "summaries": {
                condition: {
                    **_summary(rows, cluster_by_item),
                    **_confidence_diagnostics(
                        rows,
                        plan.robustness.ece_bins,
                        plan.robustness.calibration_sensitivity_epsilon,
                    ),
                }
                for condition, rows in standardized.items()
            },
            "risk_contract_transport": standardized_risk,
            "fixed_coverage_transport": standardized_fixed,
            "confidence_portability": _confidence_portability(
                standardized[reference_id],
                shifted_standardized,
                plan,
                cluster_by_item,
                strata_by_cluster,
                360_000,
            ),
        },
        "multiplicity": {
            "endpoint": (
                "certified_risk_threshold"
                if certified
                else f"fixed_coverage_{plan.fixed_coverage.universal_confirmatory_target:.2f}"
            ),
            "threshold": test_threshold,
            "randomization": randomization,
            "holm": holm,
        },
        "robustness": {
            "same_trajectory": _same_answer_analysis(
                coupled[reference_id], shifted_coupled
            ),
            "success_matched": {
                condition: {
                    "delta_success": _accuracy(rows) - _accuracy(coupled[reference_id]),
                    "within_prespecified_tolerance": abs(
                        _accuracy(rows) - _accuracy(coupled[reference_id])
                    )
                    <= plan.robustness.accuracy_match_absolute_tolerance,
                    "absolute_tolerance": plan.robustness.accuracy_match_absolute_tolerance,
                }
                for condition, rows in shifted_coupled.items()
            },
        },
        "decomposition": _decomposition(
            records,
            config,  # type: ignore[arg-type]
            plan.bootstrap.replicates,
            plan.bootstrap.seed + 50_000,
            cluster_by_item,
            strata_by_cluster,
        ),
    }
    output = run.statistics / "confirmatory-agent-test-analysis-v1.json"
    if output.exists():
        existing = read_json(output)
        comparable = dict(result)
        comparable["created_at_utc"] = existing.get("created_at_utc")
        comparable["analysis_git"] = existing.get("analysis_git")
        if existing != comparable:
            raise RuntimeError(f"agent test analysis collision at {output}")
        return output, existing
    write_immutable_json(output, result)
    return output, result
