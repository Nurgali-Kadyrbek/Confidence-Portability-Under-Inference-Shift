"""Strict configuration for the BFCL agentic external-validation study."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from cpis.config import (
    ConfidenceProtocolSpec,
    DatasetSpec,
    ExecutionSpec,
    ModelSpec,
    Sha256,
    Slug,
    StorageSpec,
    StrictModel,
)
from cpis.integration import IntegrationReference
from cpis.matrix_config import MatrixInferenceSpec


class BfclProtocolSpec(StrictModel):
    protocol_id: Slug
    evaluator_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    evaluator_root_relative_to_dataset_cache: str = Field(min_length=1)
    function_docs_relative_to_evaluator_root: str = Field(min_length=1)
    function_docs_bundle_sha256: Sha256
    parser_family: Literal["qwen3", "mistral"]
    parser_version: Slug
    max_step_limit: Literal[20]
    additional_function_message: str = Field(min_length=1)

    @model_validator(mode="after")
    def paths_are_cache_relative(self) -> "BfclProtocolSpec":
        for label, raw in (
            ("evaluator_root", self.evaluator_root_relative_to_dataset_cache),
            ("function_docs", self.function_docs_relative_to_evaluator_root),
        ):
            path = Path(raw)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{label} must be a safe relative path")
        return self


class AgentExperimentConfig(StrictModel):
    schema_version: Literal["1.0"]
    experiment_id: Slug
    phase: Literal["development", "certification", "test"]
    confirmatory: bool
    model_integration: IntegrationReference
    dataset_integration: IntegrationReference
    frozen_policy: IntegrationReference | None = None
    certification_decision: IntegrationReference | None = None
    model: ModelSpec
    dataset: DatasetSpec
    inference: MatrixInferenceSpec
    confidence: ConfidenceProtocolSpec
    bfcl: BfclProtocolSpec
    execution: ExecutionSpec
    storage: StorageSpec

    @model_validator(mode="after")
    def phase_firewall_and_engine(self) -> "AgentExperimentConfig":
        if self.dataset.scorer != "bfcl_v3_official_multi_turn_v1":
            raise ValueError("agent study requires the pinned BFCL V3 scorer")
        if self.inference.backend.name != "vllm":
            raise ValueError("scientific agent study requires GPU vLLM")
        if self.dataset.loader != "cached_jsonl":
            raise ValueError("scientific agent study requires sealed cached data")
        if self.model.model_mode not in {"thinking", "non_thinking", "instruct"}:
            raise ValueError("agent model must retain its documented native mode")
        for label, revision in (
            ("model", self.model.revision),
            ("tokenizer", self.model.tokenizer_revision),
        ):
            if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
                raise ValueError(f"{label} revision must be a full commit SHA")
        if self.phase == "development":
            if self.confirmatory or self.dataset.partition != "development":
                raise ValueError("development agent runs are non-confirmatory development")
            if self.frozen_policy is not None or self.certification_decision is not None:
                raise ValueError("development cannot bind post-development decisions")
        elif self.phase == "certification":
            if not self.confirmatory or self.dataset.partition != "certification":
                raise ValueError("certification requires certification data")
            if self.frozen_policy is None or self.certification_decision is not None:
                raise ValueError("certification requires only a frozen policy")
            if len(self.inference.answer_conditions) != 1:
                raise ValueError("certification runs only the reference agent condition")
            if self.inference.seeds != (1729,):
                raise ValueError("certification uses only replicate seed 1729")
        else:
            if (
                not self.confirmatory
                or self.dataset.partition != "test"
                or self.frozen_policy is None
                or self.certification_decision is None
            ):
                raise ValueError(
                    "test requires confirmatory mode and both frozen gate records"
                )
        if self.execution.replicas * self.model.tensor_parallel_size > 8:
            raise ValueError("replicas * tensor_parallel_size exceeds eight GPUs")
        return self

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )

    @property
    def config_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def run_id(self) -> str:
        return f"{self.experiment_id}-{self.config_sha256[:12]}"

    @property
    def study(self) -> str:
        """Compatibility label for shared manifest and backend infrastructure."""

        return "agentic"


def load_agent_config(path: Path) -> AgentExperimentConfig:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load agent configuration {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("agent configuration must be a YAML mapping")
    return AgentExperimentConfig.model_validate(payload)
