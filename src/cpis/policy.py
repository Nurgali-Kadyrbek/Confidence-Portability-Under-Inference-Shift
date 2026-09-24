"""Immutable policies selected on development data before certification access."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from cpis.analysis_plan import load_analysis_plan
from cpis.config import Sha256, Slug, StrictModel
from cpis.integration import IntegrationReference
from cpis.matrix_config import load_matrix_config
from cpis.storage import StorageLayout, read_json


class FrozenRiskThreshold(StrictModel):
    status: Literal["candidate", "risk-contract infeasible"]
    threshold: float | None = Field(default=None, ge=0, le=1)
    development_risk: float | None = Field(default=None, ge=0, le=1)
    development_coverage: float | None = Field(default=None, ge=0, le=1)
    accepted_observations: int | None = Field(default=None, ge=1)
    total_observations: int | None = Field(default=None, ge=1)
    reason: str | None = None

    @model_validator(mode="after")
    def status_matches_values(self) -> "FrozenRiskThreshold":
        numeric = (
            self.threshold,
            self.development_risk,
            self.development_coverage,
            self.accepted_observations,
            self.total_observations,
        )
        if self.status == "candidate" and any(value is None for value in numeric):
            raise ValueError("candidate risk policy requires its development estimate")
        if self.status == "risk-contract infeasible" and self.threshold is not None:
            raise ValueError("an infeasible risk policy cannot freeze a threshold")
        return self


class FrozenCoverageThreshold(StrictModel):
    target_coverage: float = Field(gt=0, le=1)
    status: Literal["selected", "target unattainable"]
    threshold: float | None = Field(default=None, ge=0, le=1)
    achieved_development_coverage: float = Field(ge=0, le=1)
    development_risk: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def selected_has_threshold(self) -> "FrozenCoverageThreshold":
        if self.status == "selected" and (
            self.threshold is None or self.development_risk is None
        ):
            raise ValueError("selected fixed-coverage policy requires threshold and risk")
        if self.status == "target unattainable" and self.threshold is not None:
            raise ValueError("unattainable coverage target cannot have a threshold")
        return self


class FrozenReadoutPolicy(StrictModel):
    readout_role: Literal["coupled", "standardized_deterministic"]
    risk_contract: FrozenRiskThreshold
    fixed_coverage: tuple[FrozenCoverageThreshold, ...] = Field(min_length=1)


class FrozenPolicy(StrictModel):
    schema_version: Literal["1.0"]
    policy_id: Slug
    analysis_plan_id: Slug
    analysis_plan_sha256: Sha256
    model_id: str
    model_revision: str
    model_mode: str
    dataset_id: str
    dataset_revision: str
    confidence_protocol_id: str
    prompt_template_version: str
    development_run_id: str
    development_config_sha256: Sha256
    development_analysis_sha256: Sha256
    reference_answer_condition_id: str
    reference_confidence_readout_id: str
    standardized_readout_id: str
    risk_contract_unit_selection: dict
    coupled: FrozenReadoutPolicy
    standardized_deterministic: FrozenReadoutPolicy

    @property
    def canonical_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )


class ReadoutCertificationDecision(StrictModel):
    status: Literal["certified", "risk-contract infeasible"]
    stage: Literal["development_selection", "certification"]
    threshold: float | None = Field(default=None, ge=0, le=1)
    reason: str | None = None
    certification: dict | None = None

    @model_validator(mode="after")
    def certification_state_matches(self) -> "ReadoutCertificationDecision":
        if self.status == "certified" and (
            self.stage != "certification"
            or self.threshold is None
            or self.certification is None
        ):
            raise ValueError("certified decision requires its threshold and exact result")
        return self


class CertificationDecision(StrictModel):
    schema_version: Literal["1.0"]
    decision_id: Slug
    frozen_policy_id: Slug
    frozen_policy_sha256: Sha256
    analysis_plan_id: Slug
    analysis_plan_sha256: Sha256
    certification_run_id: str | None = None
    certification_config_sha256: Sha256 | None = None
    certification_analysis_sha256: Sha256 | None = None
    coupled: ReadoutCertificationDecision
    standardized_deterministic: ReadoutCertificationDecision

    @property
    def canonical_sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _risk_threshold(payload: dict) -> FrozenRiskThreshold:
    estimate = payload.get("estimate")
    if payload["status"] == "candidate":
        if not isinstance(estimate, dict):
            raise RuntimeError("candidate development policy is missing its estimate")
        return FrozenRiskThreshold(
            status="candidate",
            threshold=estimate["threshold"],
            development_risk=estimate["risk"],
            development_coverage=estimate["coverage"],
            accepted_observations=estimate["accepted_observations"],
            total_observations=estimate["total_observations"],
            reason=None,
        )
    return FrozenRiskThreshold(
        status="risk-contract infeasible",
        reason=payload.get("reason") or "no development candidate",
    )


def _coverage_thresholds(payload: dict) -> tuple[FrozenCoverageThreshold, ...]:
    result = []
    for target, entry in sorted(payload.items(), key=lambda value: float(value[0])):
        selection = entry["selection"]
        result.append(
            FrozenCoverageThreshold(
                target_coverage=float(target),
                status=selection["status"],
                threshold=selection["threshold"],
                achieved_development_coverage=selection["achieved_coverage"],
                development_risk=selection["reference_risk"],
            )
        )
    return tuple(result)


def freeze_development_policy(
    config_path: Path,
    analysis_plan_path: Path,
    repository_root: Path,
    destination: Path,
) -> FrozenPolicy:
    """Freeze selected thresholds from an immutable completed development run."""
    repository_root = repository_root.resolve()
    config_path = config_path.resolve()
    config = load_matrix_config(config_path)
    if config.study != "development_matrix" or config.dataset.partition != "development":
        raise ValueError("policy freezing requires a development matrix")
    plan = load_analysis_plan(analysis_plan_path.resolve())
    if plan.schema_version not in {"3.0", "4.0"}:
        raise ValueError("controlled-study policies require analysis plan v3+")
    storage = StorageLayout.from_spec(config.storage, repository_root)
    analysis_path = storage.run(config.run_id).statistics / "development-matrix-analysis-v2.json"
    if not analysis_path.is_file():
        raise RuntimeError("development analysis is incomplete")
    analysis = read_json(analysis_path)
    if (
        analysis.get("run_id") != config.run_id
        or analysis.get("config_sha256") != config.config_sha256
        or analysis.get("analysis_plan_sha256") != plan.sha256
        or analysis.get("exploratory") is not True
    ):
        raise RuntimeError("development analysis provenance does not match the policy inputs")
    unit_selection = analysis.get("risk_contract_unit_selection")
    if not isinstance(unit_selection, dict) or unit_selection.get("method") in {
        None,
        "legacy_all_rows",
    }:
        raise RuntimeError("development analysis lacks the v2 independent-unit selection")

    design = config.inference.design
    policy = FrozenPolicy(
        schema_version="1.0",
        policy_id=f"{config.experiment_id}-policy-v2",
        analysis_plan_id=plan.plan_id,
        analysis_plan_sha256=plan.sha256,
        model_id=config.model.model_id,
        model_revision=config.model.revision,
        model_mode=config.model.model_mode,
        dataset_id=config.dataset.dataset_id,
        dataset_revision=config.dataset.revision,
        confidence_protocol_id=config.confidence.protocol_id,
        prompt_template_version=config.confidence.prompt_template_version,
        development_run_id=config.run_id,
        development_config_sha256=config.config_sha256,
        development_analysis_sha256=_file_sha256(analysis_path),
        reference_answer_condition_id=design.reference_answer_condition_id,
        reference_confidence_readout_id=design.reference_confidence_readout_id,
        standardized_readout_id=design.standardized_readout_id,
        risk_contract_unit_selection=unit_selection,
        coupled=FrozenReadoutPolicy(
            readout_role="coupled",
            risk_contract=_risk_threshold(
                analysis["coupled"]["reference_risk_contract_selection"]
            ),
            fixed_coverage=_coverage_thresholds(analysis["coupled"]["fixed_coverage"]),
        ),
        standardized_deterministic=FrozenReadoutPolicy(
            readout_role="standardized_deterministic",
            risk_contract=_risk_threshold(
                analysis["standardized_deterministic"][
                    "reference_risk_contract_selection"
                ]
            ),
            fixed_coverage=_coverage_thresholds(
                analysis["standardized_deterministic"]["fixed_coverage"]
            ),
        ),
    )
    destination = destination.resolve()
    try:
        destination.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError("frozen policy must be stored inside the repository") from exc
    rendered = json.dumps(policy.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"refusing to overwrite a different frozen policy: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    return policy


def freeze_agent_development_policy(
    config_path: Path,
    analysis_plan_path: Path,
    repository_root: Path,
    destination: Path,
) -> FrozenPolicy:
    """Freeze BFCL development thresholds before any agent certification access."""

    from cpis.agent_config import load_agent_config

    repository_root = repository_root.resolve()
    config_path = config_path.resolve()
    config = load_agent_config(config_path)
    if (
        config.phase != "development"
        or config.dataset.partition != "development"
        or config.dataset.item_limit is not None
    ):
        raise ValueError("agent policy freezing requires a full development matrix")
    plan = load_analysis_plan(analysis_plan_path.resolve())
    if plan.schema_version != "4.0" or plan.agent_grouped_certification is None:
        raise ValueError("agent policies require analysis plan v4")
    storage = StorageLayout.from_spec(config.storage, repository_root)
    analysis_path = storage.run(config.run_id).statistics / "development-agent-analysis-v2.json"
    if not analysis_path.is_file():
        raise RuntimeError("agent development analysis is incomplete")
    analysis = read_json(analysis_path)
    if (
        analysis.get("run_id") != config.run_id
        or analysis.get("config_sha256") != config.config_sha256
        or analysis.get("analysis_plan_sha256") != plan.sha256
        or analysis.get("exploratory") is not True
    ):
        raise RuntimeError("agent development analysis provenance does not match")
    unit_selection = analysis.get("risk_contract_unit_selection")
    if not isinstance(unit_selection, dict) or unit_selection.get("method") is None:
        raise RuntimeError("agent analysis lacks independent-scenario selection")
    design = config.inference.design
    policy = FrozenPolicy(
        schema_version="1.0",
        policy_id=f"{config.experiment_id}-policy-v2",
        analysis_plan_id=plan.plan_id,
        analysis_plan_sha256=plan.sha256,
        model_id=config.model.model_id,
        model_revision=config.model.revision,
        model_mode=config.model.model_mode,
        dataset_id=config.dataset.dataset_id,
        dataset_revision=config.dataset.revision,
        confidence_protocol_id=config.confidence.protocol_id,
        prompt_template_version=config.confidence.prompt_template_version,
        development_run_id=config.run_id,
        development_config_sha256=config.config_sha256,
        development_analysis_sha256=_file_sha256(analysis_path),
        reference_answer_condition_id=design.reference_answer_condition_id,
        reference_confidence_readout_id=design.reference_confidence_readout_id,
        standardized_readout_id=design.standardized_readout_id,
        risk_contract_unit_selection=unit_selection,
        coupled=FrozenReadoutPolicy(
            readout_role="coupled",
            risk_contract=_risk_threshold(
                analysis["coupled"]["reference_risk_contract_selection"]
            ),
            fixed_coverage=_coverage_thresholds(analysis["coupled"]["fixed_coverage"]),
        ),
        standardized_deterministic=FrozenReadoutPolicy(
            readout_role="standardized_deterministic",
            risk_contract=_risk_threshold(
                analysis["standardized_deterministic"][
                    "reference_risk_contract_selection"
                ]
            ),
            fixed_coverage=_coverage_thresholds(
                analysis["standardized_deterministic"]["fixed_coverage"]
            ),
        ),
    )
    destination = destination.resolve()
    try:
        destination.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError("frozen agent policy must be stored inside the repository") from exc
    rendered = json.dumps(policy.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(
                f"refusing to overwrite a different frozen agent policy: {destination}"
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    return policy


def load_frozen_policy(path: Path) -> FrozenPolicy:
    return FrozenPolicy.model_validate(json.loads(path.read_text(encoding="utf-8")))


def validate_policy_reference(
    reference: IntegrationReference, repository_root: Path
) -> FrozenPolicy:
    path = (repository_root / reference.record_path).resolve()
    try:
        path.relative_to(repository_root.resolve())
    except ValueError as exc:
        raise RuntimeError("frozen policy reference escapes the repository") from exc
    policy = load_frozen_policy(path)
    if policy.policy_id != reference.record_id:
        raise RuntimeError("frozen policy ID does not match its reference")
    if policy.canonical_sha256 != reference.canonical_sha256:
        raise RuntimeError("frozen policy hash does not match its reference")
    return policy


def _resolved_readout(policy_readout, result: dict | None) -> ReadoutCertificationDecision:
    if policy_readout.risk_contract.status == "risk-contract infeasible":
        return ReadoutCertificationDecision(
            status="risk-contract infeasible",
            stage="development_selection",
            threshold=None,
            reason=policy_readout.risk_contract.reason,
            certification=None,
        )
    if result is None or result.get("stage") != "certification":
        raise RuntimeError("candidate policy is missing its certification result")
    certification = result.get("certification")
    if not isinstance(certification, dict):
        raise RuntimeError("candidate policy has malformed certification evidence")
    return ReadoutCertificationDecision(
        status=result["status"],
        stage="certification",
        threshold=policy_readout.risk_contract.threshold,
        reason=result.get("reason"),
        certification=certification,
    )


def resolve_certification_decision(
    policy_reference: IntegrationReference,
    analysis_plan_path: Path,
    repository_root: Path,
    destination: Path,
    certification_config_path: Path | None = None,
) -> CertificationDecision:
    """Resolve a policy after development and, when needed, certification."""
    repository_root = repository_root.resolve()
    policy = validate_policy_reference(policy_reference, repository_root)
    plan = load_analysis_plan(analysis_plan_path.resolve())
    if policy.analysis_plan_sha256 != plan.sha256:
        raise RuntimeError("policy and certification decision analysis plans differ")
    candidate_exists = any(
        readout.risk_contract.status == "candidate"
        for readout in (policy.coupled, policy.standardized_deterministic)
    )
    certification_result = None
    certification_run_id = None
    certification_config_sha256 = None
    certification_analysis_sha256 = None
    if candidate_exists:
        if certification_config_path is None:
            raise RuntimeError("a development candidate requires certification evidence")
        certification_config = load_matrix_config(certification_config_path.resolve())
        if certification_config.study != "certification":
            raise ValueError("certification decision requires a certification config")
        if certification_config.frozen_policy != policy_reference:
            raise RuntimeError("certification config references a different policy")
        storage = StorageLayout.from_spec(certification_config.storage, repository_root)
        result_path = (
            storage.run(certification_config.run_id).statistics
            / "risk-contract-certification-v1.json"
        )
        if not result_path.is_file():
            raise RuntimeError("risk-contract certification analysis is incomplete")
        certification_result = read_json(result_path)
        if (
            certification_result.get("frozen_policy_sha256")
            != policy.canonical_sha256
            or certification_result.get("analysis_plan_sha256") != plan.sha256
        ):
            raise RuntimeError("certification analysis provenance does not match")
        certification_run_id = certification_config.run_id
        certification_config_sha256 = certification_config.config_sha256
        certification_analysis_sha256 = _file_sha256(result_path)
    elif certification_config_path is not None:
        raise ValueError("development-infeasible policies must not open certification data")

    decision = CertificationDecision(
        schema_version="1.0",
        decision_id=f"{policy.policy_id}-certification-decision-v1",
        frozen_policy_id=policy.policy_id,
        frozen_policy_sha256=policy.canonical_sha256,
        analysis_plan_id=plan.plan_id,
        analysis_plan_sha256=plan.sha256,
        certification_run_id=certification_run_id,
        certification_config_sha256=certification_config_sha256,
        certification_analysis_sha256=certification_analysis_sha256,
        coupled=_resolved_readout(
            policy.coupled,
            certification_result.get("coupled") if certification_result else None,
        ),
        standardized_deterministic=_resolved_readout(
            policy.standardized_deterministic,
            (
                certification_result.get("standardized_deterministic")
                if certification_result
                else None
            ),
        ),
    )
    destination = destination.resolve()
    try:
        destination.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError("certification decision must be stored inside the repository") from exc
    rendered = json.dumps(decision.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(
                f"refusing to overwrite a different certification decision: {destination}"
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    return decision


def resolve_agent_certification_decision(
    policy_reference: IntegrationReference,
    analysis_plan_path: Path,
    repository_root: Path,
    destination: Path,
    certification_config_path: Path | None = None,
) -> CertificationDecision:
    """Resolve a BFCL policy after its scenario-level exact certification."""

    from cpis.agent_config import load_agent_config

    repository_root = repository_root.resolve()
    policy = validate_policy_reference(policy_reference, repository_root)
    plan = load_analysis_plan(analysis_plan_path.resolve())
    if policy.analysis_plan_sha256 != plan.sha256:
        raise RuntimeError("agent policy and certification analysis plans differ")
    candidate_exists = any(
        readout.risk_contract.status == "candidate"
        for readout in (policy.coupled, policy.standardized_deterministic)
    )
    certification_result = None
    certification_run_id = None
    certification_config_sha256 = None
    certification_analysis_sha256 = None
    if candidate_exists:
        if certification_config_path is None:
            raise RuntimeError("an agent development candidate requires certification")
        certification_config = load_agent_config(certification_config_path.resolve())
        if certification_config.phase != "certification":
            raise ValueError("agent decision requires a certification config")
        if certification_config.frozen_policy != policy_reference:
            raise RuntimeError("agent certification config references another policy")
        storage = StorageLayout.from_spec(certification_config.storage, repository_root)
        result_path = (
            storage.run(certification_config.run_id).statistics
            / "risk-contract-certification-v1.json"
        )
        if not result_path.is_file():
            raise RuntimeError("agent risk-contract certification is incomplete")
        certification_result = read_json(result_path)
        if (
            certification_result.get("frozen_policy_sha256")
            != policy.canonical_sha256
            or certification_result.get("analysis_plan_sha256") != plan.sha256
        ):
            raise RuntimeError("agent certification analysis provenance does not match")
        certification_run_id = certification_config.run_id
        certification_config_sha256 = certification_config.config_sha256
        certification_analysis_sha256 = _file_sha256(result_path)
    elif certification_config_path is not None:
        raise ValueError("development-infeasible agent policies must not read certification")
    decision = CertificationDecision(
        schema_version="1.0",
        decision_id=f"{policy.policy_id}-certification-decision-v1",
        frozen_policy_id=policy.policy_id,
        frozen_policy_sha256=policy.canonical_sha256,
        analysis_plan_id=plan.plan_id,
        analysis_plan_sha256=plan.sha256,
        certification_run_id=certification_run_id,
        certification_config_sha256=certification_config_sha256,
        certification_analysis_sha256=certification_analysis_sha256,
        coupled=_resolved_readout(
            policy.coupled,
            certification_result.get("coupled") if certification_result else None,
        ),
        standardized_deterministic=_resolved_readout(
            policy.standardized_deterministic,
            (
                certification_result.get("standardized_deterministic")
                if certification_result
                else None
            ),
        ),
    )
    destination = destination.resolve()
    try:
        destination.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError("agent certification decision must live in the repository") from exc
    rendered = json.dumps(decision.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(
                f"refusing to overwrite another agent decision: {destination}"
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    return decision


def load_certification_decision(path: Path) -> CertificationDecision:
    return CertificationDecision.model_validate(json.loads(path.read_text(encoding="utf-8")))


def validate_certification_decision_reference(
    reference: IntegrationReference, repository_root: Path
) -> CertificationDecision:
    path = (repository_root / reference.record_path).resolve()
    try:
        path.relative_to(repository_root.resolve())
    except ValueError as exc:
        raise RuntimeError("certification decision reference escapes the repository") from exc
    decision = load_certification_decision(path)
    if decision.decision_id != reference.record_id:
        raise RuntimeError("certification decision ID does not match its reference")
    if decision.canonical_sha256 != reference.canonical_sha256:
        raise RuntimeError("certification decision hash does not match its reference")
    return decision
