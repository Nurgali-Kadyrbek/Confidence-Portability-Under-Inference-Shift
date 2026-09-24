"""Deterministically derive certification and test configs from frozen artifacts."""

from __future__ import annotations

from pathlib import Path

import yaml

from cpis.agent_config import AgentExperimentConfig, load_agent_config
from cpis.integration import (
    DatasetIntegrationRecord,
    IntegrationReference,
    validate_integration_reference,
)
from cpis.matrix_config import MatrixExperimentConfig, load_matrix_config
from cpis.policy import (
    load_certification_decision,
    load_frozen_policy,
)


def _repository_reference(path: Path, repository_root: Path, record) -> IntegrationReference:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repository_root.resolve())
    except ValueError as exc:
        raise ValueError("frozen phase artifact must live inside the repository") from exc
    return IntegrationReference(
        record_id=(record.policy_id if hasattr(record, "policy_id") else record.decision_id),
        record_path=str(relative),
        canonical_sha256=record.canonical_sha256,
    )


def _partition_payload(
    payload: dict,
    config: MatrixExperimentConfig,
    partition: str,
    repository_root: Path,
) -> None:
    assert config.dataset_integration is not None
    record = validate_integration_reference(
        config.dataset_integration, repository_root, "dataset_integration"
    )
    assert isinstance(record, DatasetIntegrationRecord)
    selected = next(entry for entry in record.prepared_partitions if entry.partition == partition)
    payload["dataset"]["partition"] = partition
    payload["dataset"]["source_path"] = selected.relative_cache_path
    payload["dataset"]["content_sha256"] = selected.sha256
    payload["dataset"]["item_limit"] = None
    payload["dataset"]["item_selection_salt"] = None


def _experiment_id(development_id: str, prefix: str) -> str:
    if not development_id.startswith("dev-matrix-"):
        raise ValueError("phase derivation requires a full dev-matrix experiment ID")
    return f"{prefix}-matrix-{development_id.removeprefix('dev-matrix-')}"


def _agent_experiment_id(development_id: str, prefix: str) -> str:
    if not development_id.startswith("dev-agent-"):
        raise ValueError("agent phase derivation requires a full dev-agent experiment ID")
    return f"{prefix}-agent-{development_id.removeprefix('dev-agent-')}"


def _write_config(
    model: MatrixExperimentConfig, destination: Path
) -> MatrixExperimentConfig:
    rendered = yaml.safe_dump(
        model.model_dump(mode="json"), sort_keys=False, width=100
    )
    destination = destination.resolve()
    if destination.exists() and destination.read_text(encoding="utf-8") != rendered:
        raise RuntimeError(f"refusing to replace a different phase config: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")
    return model


def _write_agent_config(
    model: AgentExperimentConfig, destination: Path
) -> AgentExperimentConfig:
    rendered = yaml.safe_dump(
        model.model_dump(mode="json"), sort_keys=False, width=100
    )
    destination = destination.resolve()
    if destination.exists() and destination.read_text(encoding="utf-8") != rendered:
        raise RuntimeError(f"refusing to replace a different agent phase config: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")
    return model


def _agent_partition_payload(
    payload: dict,
    config: AgentExperimentConfig,
    partition: str,
    repository_root: Path,
) -> None:
    record = validate_integration_reference(
        config.dataset_integration, repository_root, "dataset_integration"
    )
    assert isinstance(record, DatasetIntegrationRecord)
    selected = next(entry for entry in record.prepared_partitions if entry.partition == partition)
    payload["dataset"]["partition"] = partition
    payload["dataset"]["source_path"] = selected.relative_cache_path
    payload["dataset"]["content_sha256"] = selected.sha256
    payload["dataset"]["item_limit"] = None
    payload["dataset"]["item_selection_salt"] = None


def derive_certification_config(
    development_config_path: Path,
    policy_path: Path,
    repository_root: Path,
    destination: Path,
) -> MatrixExperimentConfig:
    """Create the reference-only certification config for a candidate policy."""

    repository_root = repository_root.resolve()
    development = load_matrix_config(development_config_path.resolve())
    if (
        development.study != "development_matrix"
        or development.dataset.partition != "development"
        or development.dataset.item_limit is not None
    ):
        raise ValueError("certification derivation requires a full development config")
    policy = load_frozen_policy(policy_path.resolve())
    if (
        policy.development_run_id != development.run_id
        or policy.development_config_sha256 != development.config_sha256
    ):
        raise RuntimeError("frozen policy does not originate from the development config")
    if all(
        readout.risk_contract.status == "risk-contract infeasible"
        for readout in (policy.coupled, policy.standardized_deterministic)
    ):
        raise ValueError("development-infeasible policies must not open certification data")
    policy_reference = _repository_reference(policy_path, repository_root, policy)
    payload = development.model_dump(mode="json")
    payload["experiment_id"] = _experiment_id(development.experiment_id, "cert")
    payload["study"] = "certification"
    payload["confirmatory"] = True
    payload["frozen_policy"] = policy_reference.model_dump(mode="json")
    payload["certification_decision"] = None
    _partition_payload(payload, development, "certification", repository_root)
    design = development.inference.design
    reference_answer = next(
        condition
        for condition in development.inference.answer_conditions
        if condition.condition_id == design.reference_answer_condition_id
    )
    permitted = {
        design.reference_confidence_readout_id,
        design.standardized_readout_id,
    }
    readouts = [
        readout.model_dump(mode="json")
        for readout in development.inference.confidence_readouts
        if readout.readout_id in permitted
    ]
    if {entry["readout_id"] for entry in readouts} != permitted:
        raise RuntimeError("development config lacks a required certification readout")
    payload["inference"]["seeds"] = [1729]
    payload["inference"]["answer_conditions"] = [
        reference_answer.model_dump(mode="json")
    ]
    payload["inference"]["confidence_readouts"] = readouts
    payload["inference"]["design"] = {
        "reference_answer_condition_id": design.reference_answer_condition_id,
        "reference_confidence_readout_id": design.reference_confidence_readout_id,
        "coupled_readout_by_answer": {
            design.reference_answer_condition_id: design.reference_confidence_readout_id
        },
        "standardized_readout_id": design.standardized_readout_id,
        "report_only_readout_ids": sorted(permitted),
    }
    return _write_config(MatrixExperimentConfig.model_validate(payload), destination)


def derive_test_config(
    development_config_path: Path,
    policy_path: Path,
    decision_path: Path,
    repository_root: Path,
    destination: Path,
) -> MatrixExperimentConfig:
    """Create the full four-contrast untouched-test configuration."""

    repository_root = repository_root.resolve()
    development = load_matrix_config(development_config_path.resolve())
    if (
        development.study != "development_matrix"
        or development.dataset.partition != "development"
        or development.dataset.item_limit is not None
    ):
        raise ValueError("test derivation requires a full development config")
    policy = load_frozen_policy(policy_path.resolve())
    decision = load_certification_decision(decision_path.resolve())
    if (
        policy.development_run_id != development.run_id
        or policy.development_config_sha256 != development.config_sha256
        or decision.frozen_policy_id != policy.policy_id
        or decision.frozen_policy_sha256 != policy.canonical_sha256
    ):
        raise RuntimeError("test gate artifacts do not match the development config")
    policy_reference = _repository_reference(policy_path, repository_root, policy)
    decision_reference = _repository_reference(decision_path, repository_root, decision)
    payload = development.model_dump(mode="json")
    payload["experiment_id"] = _experiment_id(development.experiment_id, "test")
    payload["study"] = "broad"
    payload["confirmatory"] = True
    payload["frozen_policy"] = policy_reference.model_dump(mode="json")
    payload["certification_decision"] = decision_reference.model_dump(mode="json")
    _partition_payload(payload, development, "test", repository_root)
    return _write_config(MatrixExperimentConfig.model_validate(payload), destination)


def derive_agent_certification_config(
    development_config_path: Path,
    policy_path: Path,
    repository_root: Path,
    destination: Path,
) -> AgentExperimentConfig:
    """Create a reference-only BFCL certification config from frozen development."""

    repository_root = repository_root.resolve()
    development = load_agent_config(development_config_path.resolve())
    if (
        development.phase != "development"
        or development.dataset.partition != "development"
        or development.dataset.item_limit is not None
    ):
        raise ValueError("agent certification derivation requires full development")
    policy = load_frozen_policy(policy_path.resolve())
    if (
        policy.development_run_id != development.run_id
        or policy.development_config_sha256 != development.config_sha256
    ):
        raise RuntimeError("frozen policy does not originate from the agent development config")
    if all(
        readout.risk_contract.status == "risk-contract infeasible"
        for readout in (policy.coupled, policy.standardized_deterministic)
    ):
        raise ValueError("development-infeasible agent policies must not open certification")
    policy_reference = _repository_reference(policy_path, repository_root, policy)
    payload = development.model_dump(mode="json")
    payload["experiment_id"] = _agent_experiment_id(development.experiment_id, "cert")
    payload["phase"] = "certification"
    payload["confirmatory"] = True
    payload["frozen_policy"] = policy_reference.model_dump(mode="json")
    payload["certification_decision"] = None
    _agent_partition_payload(payload, development, "certification", repository_root)
    design = development.inference.design
    reference_answer = next(
        condition
        for condition in development.inference.answer_conditions
        if condition.condition_id == design.reference_answer_condition_id
    )
    permitted = {
        design.reference_confidence_readout_id,
        design.standardized_readout_id,
    }
    readouts = [
        readout.model_dump(mode="json")
        for readout in development.inference.confidence_readouts
        if readout.readout_id in permitted
    ]
    if {entry["readout_id"] for entry in readouts} != permitted:
        raise RuntimeError("agent development lacks a required certification readout")
    payload["inference"]["seeds"] = [1729]
    payload["inference"]["answer_conditions"] = [
        reference_answer.model_dump(mode="json")
    ]
    payload["inference"]["confidence_readouts"] = readouts
    payload["inference"]["design"] = {
        "reference_answer_condition_id": design.reference_answer_condition_id,
        "reference_confidence_readout_id": design.reference_confidence_readout_id,
        "coupled_readout_by_answer": {
            design.reference_answer_condition_id: design.reference_confidence_readout_id
        },
        "standardized_readout_id": design.standardized_readout_id,
        "report_only_readout_ids": sorted(permitted),
    }
    return _write_agent_config(
        AgentExperimentConfig.model_validate(payload), destination
    )


def derive_agent_test_config(
    development_config_path: Path,
    policy_path: Path,
    decision_path: Path,
    repository_root: Path,
    destination: Path,
) -> AgentExperimentConfig:
    """Create a full untouched BFCL test config from both frozen gate artifacts."""

    repository_root = repository_root.resolve()
    development = load_agent_config(development_config_path.resolve())
    if (
        development.phase != "development"
        or development.dataset.partition != "development"
        or development.dataset.item_limit is not None
    ):
        raise ValueError("agent test derivation requires full development")
    policy = load_frozen_policy(policy_path.resolve())
    decision = load_certification_decision(decision_path.resolve())
    if (
        policy.development_run_id != development.run_id
        or policy.development_config_sha256 != development.config_sha256
        or decision.frozen_policy_id != policy.policy_id
        or decision.frozen_policy_sha256 != policy.canonical_sha256
    ):
        raise RuntimeError("agent test gate artifacts do not match development")
    payload = development.model_dump(mode="json")
    payload["experiment_id"] = _agent_experiment_id(development.experiment_id, "test")
    payload["phase"] = "test"
    payload["confirmatory"] = True
    payload["frozen_policy"] = _repository_reference(
        policy_path, repository_root, policy
    ).model_dump(mode="json")
    payload["certification_decision"] = _repository_reference(
        decision_path, repository_root, decision
    ).model_dump(mode="json")
    _agent_partition_payload(payload, development, "test", repository_root)
    return _write_agent_config(AgentExperimentConfig.model_validate(payload), destination)
