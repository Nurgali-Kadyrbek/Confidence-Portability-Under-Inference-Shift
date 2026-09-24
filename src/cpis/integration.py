"""Strict model and dataset integration records used as scientific preflight gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from cpis.config import Sha256, Slug, StrictModel


class SourceRecord(StrictModel):
    source_type: Literal[
        "official_model_card",
        "official_dataset_card",
        "official_model_artifact",
        "official_dataset_artifact",
        "official_engine_documentation",
        "official_paper",
        "local_validation",
    ]
    url: str = Field(min_length=8)
    revision: str = Field(min_length=1)
    artifact_sha256: Sha256 | None
    note: str = Field(min_length=10)


class InterventionRecord(StrictModel):
    intervention_id: Slug
    changed_parameters: dict[str, float]
    held_fixed_parameters: dict[str, float]
    status: Literal["supported", "supported_with_limitation", "unsupported"]
    interpretation: str = Field(min_length=20)


class EvaluatorArtifact(StrictModel):
    relative_path: str = Field(min_length=1)
    sha256: Sha256

    @model_validator(mode="after")
    def repository_relative_path(self) -> "EvaluatorArtifact":
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("evaluator artifact path must be relative")
        return self


class EvaluatorSandbox(StrictModel):
    required: bool
    engine: Literal["none", "docker"]
    network: Literal["host", "none"]
    read_only_root: bool
    run_as_non_root: bool
    memory_limit_mb: int | None = Field(default=None, gt=0)
    pids_limit: int | None = Field(default=None, gt=0)
    image_reference: str | None = None
    image_id_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def required_sandbox_is_locked(self) -> "EvaluatorSandbox":
        if self.required and (
            self.engine != "docker"
            or self.network != "none"
            or not self.read_only_root
            or not self.run_as_non_root
            or self.image_reference is None
            or self.image_id_sha256 is None
        ):
            raise ValueError("required evaluator sandbox must be locked down")
        if not self.required and (
            self.image_reference is not None or self.image_id_sha256 is not None
        ):
            raise ValueError("non-container evaluator cannot pin a sandbox image")
        return self


class EvaluatorIntegrationRecord(StrictModel):
    schema_version: Literal["1.0"]
    record_type: Literal["evaluator_integration"]
    record_id: Slug
    scorer: Literal["manyifeval_official_v1", "stylembpp_official_v1"]
    source_project: str = Field(min_length=1)
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_root_relative_to_dataset_cache: str = Field(min_length=1)
    source_archive_sha256: Sha256
    artifacts: tuple[EvaluatorArtifact, ...] = Field(min_length=1)
    runtime_assets: tuple[EvaluatorArtifact, ...] = ()
    python_version: str = Field(min_length=3)
    package_versions: dict[str, str] = Field(min_length=1)
    deterministic_controls: tuple[str, ...] = Field(min_length=1)
    sandbox: EvaluatorSandbox

    @model_validator(mode="after")
    def unique_safe_artifacts(self) -> "EvaluatorIntegrationRecord":
        root = Path(self.source_root_relative_to_dataset_cache)
        if root.is_absolute() or ".." in root.parts:
            raise ValueError("evaluator source root must be dataset-cache relative")
        paths = [artifact.relative_path for artifact in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("evaluator artifact paths must be unique")
        asset_paths = [artifact.relative_path for artifact in self.runtime_assets]
        if len(asset_paths) != len(set(asset_paths)):
            raise ValueError("evaluator runtime asset paths must be unique")
        return self

    @property
    def canonical_sha256(self) -> str:
        return _canonical_sha256(self)


class ModelIntegrationRecord(StrictModel):
    schema_version: Literal["1.0"]
    record_type: Literal["model_integration"]
    record_id: Slug
    model_id: str
    revision: str
    tokenizer_id: str
    tokenizer_revision: str
    parameter_count_exact: int = Field(gt=0)
    parameter_count_method: str
    architecture: str
    framework: str
    framework_version: str
    framework_requirement: str
    precision: tuple[str, ...] = Field(min_length=1)
    official_chat_template_sha256: Sha256
    message_format: str
    reasoning_controls: str
    required_special_tokens: tuple[str, ...] = Field(min_length=1)
    eos_and_stop_behavior: str
    recommended_context_limit: int = Field(gt=0)
    configured_context_limit: int = Field(gt=0)
    recommended_output_limits: str
    native_reference_sampling: dict[str, float]
    sampler_constraints: str
    greedy_decoding_caveats: str
    tool_calling_requirements: str
    structured_output_constraints: str
    engine_requirements: str
    unsupported_configurations: tuple[str, ...]
    interventions: tuple[InterventionRecord, ...] = Field(min_length=1)
    validation_status: Literal[
        "preflight_pending", "development_validated", "unsupported"
    ]
    validation_evidence: str
    sources: tuple[SourceRecord, ...] = Field(min_length=2)

    @property
    def canonical_sha256(self) -> str:
        return _canonical_sha256(self)


class PreparedPartition(StrictModel):
    partition: Literal["development", "certification", "test"]
    rows: int = Field(gt=0)
    sha256: Sha256
    relative_cache_path: str


class DatasetIntegrationRecord(StrictModel):
    schema_version: Literal["1.0"]
    record_type: Literal["dataset_integration"]
    record_id: Slug
    dataset_id: str
    revision: str
    source_sha256: Sha256
    license_and_access: str
    official_splits: str
    unit_of_observation: str
    grouping_and_dependence: str
    official_task_representation: str
    official_scoring: str
    allowed_normalization: str
    invalid_output_handling: str
    evaluation_type: str
    duplicate_and_transformed_relationships: str
    contamination_considerations: str
    benchmark_exclusions: str
    evaluator_revision: str
    preparation_algorithm: str
    prepared_partitions: tuple[PreparedPartition, PreparedPartition, PreparedPartition]
    access_firewall: str
    sources: tuple[SourceRecord, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def all_partitions_exist(self) -> "DatasetIntegrationRecord":
        if {part.partition for part in self.prepared_partitions} != {
            "development",
            "certification",
            "test",
        }:
            raise ValueError("dataset record must declare all three frozen partitions")
        return self

    @property
    def canonical_sha256(self) -> str:
        return _canonical_sha256(self)


IntegrationRecord = (
    ModelIntegrationRecord | DatasetIntegrationRecord | EvaluatorIntegrationRecord
)


class IntegrationReference(StrictModel):
    record_id: Slug
    record_path: str = Field(min_length=1)
    canonical_sha256: Sha256

    @model_validator(mode="after")
    def relative_repository_path(self) -> "IntegrationReference":
        path = Path(self.record_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("integration record path must be repository-relative")
        return self


def _canonical_sha256(record: StrictModel) -> str:
    payload = json.dumps(
        record.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_integration_record(path: Path) -> IntegrationRecord:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load integration record {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("integration record must be a YAML mapping")
    record_type = payload.get("record_type")
    if record_type == "model_integration":
        return ModelIntegrationRecord.model_validate(payload)
    if record_type == "dataset_integration":
        return DatasetIntegrationRecord.model_validate(payload)
    if record_type == "evaluator_integration":
        return EvaluatorIntegrationRecord.model_validate(payload)
    raise ValueError(f"unknown integration record type: {record_type!r}")


def validate_integration_reference(
    reference: IntegrationReference,
    repository_root: Path,
    expected_type: Literal["model_integration", "dataset_integration"],
) -> IntegrationRecord:
    path = (repository_root / reference.record_path).resolve()
    try:
        path.relative_to(repository_root.resolve())
    except ValueError as exc:
        raise ValueError("integration record resolves outside repository") from exc
    record = load_integration_record(path)
    if record.record_type != expected_type:
        raise ValueError(f"expected {expected_type}, got {record.record_type}")
    if record.record_id != reference.record_id:
        raise ValueError("integration record ID does not match configuration")
    if record.canonical_sha256 != reference.canonical_sha256:
        raise ValueError("integration record canonical SHA-256 does not match")
    return record
