"""Versioned configuration for the development readout-decomposition matrix."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, JsonValue, model_validator

from cpis.config import (
    BackendSpec,
    ConfidenceProtocolSpec,
    DatasetSpec,
    ExecutionSpec,
    ModelSpec,
    SamplingSpec,
    Sha256,
    Slug,
    StorageSpec,
    StrictModel,
)
from cpis.integration import IntegrationReference


class AnswerCondition(StrictModel):
    condition_id: Slug
    sampling: SamplingSpec


class ConfidenceReadout(StrictModel):
    readout_id: Slug
    sampling: SamplingSpec
    chat_template_kwargs: dict[str, JsonValue] | None = None
    model_mode: Literal[
        "mock",
        "thinking",
        "non_thinking",
        "instruct",
        "reasoning",
        "agentic",
    ] | None = None

    @model_validator(mode="after")
    def complete_mode_override(self) -> "ConfidenceReadout":
        if (self.chat_template_kwargs is None) != (self.model_mode is None):
            raise ValueError(
                "readout chat-template kwargs and model mode must be overridden together"
            )
        return self


class MatrixDesign(StrictModel):
    reference_answer_condition_id: Slug
    reference_confidence_readout_id: Slug
    coupled_readout_by_answer: dict[str, str] = Field(min_length=1)
    standardized_readout_id: Slug
    report_only_readout_ids: tuple[str, ...] = Field(min_length=1)
    # The broad confirmatory study records one native coupled confidence per
    # answer. The standardized readout is H5 and the crossed report-only cells
    # are the mechanistic decomposition; both belong to their own nested
    # sub-studies, not to every model, dataset and decoder cell. The legacy
    # value is excluded from the dump so existing run identities do not move.
    confidence_cells: Literal["coupled_standardized_reference", "coupled_only"] = Field(
        default="coupled_standardized_reference",
        exclude_if=lambda value: value == "coupled_standardized_reference",
    )


def readout_ids_for_answer(design: MatrixDesign, condition_id: str) -> set[str]:
    """Which confidence readouts an answer under this condition receives.

    The generation pipeline, the official scorer and the analyses each need this
    and each used to carry its own copy, so a design change reached one of them
    and left the others computing a stale expectation. It lives here once.
    """
    coupled = design.coupled_readout_by_answer[condition_id]
    if design.confidence_cells == "coupled_only":
        return {coupled}
    ids = {
        coupled,
        design.standardized_readout_id,
        design.reference_confidence_readout_id,
    }
    if condition_id == design.reference_answer_condition_id:
        ids.update(design.report_only_readout_ids)
    return ids


def expected_confidence_cells(config: "MatrixExperimentConfig", item_count: int) -> int:
    per_item_seed = sum(
        len(readout_ids_for_answer(config.inference.design, condition.condition_id))
        for condition in config.inference.answer_conditions
    )
    return per_item_seed * item_count * len(config.inference.seeds)


class MatrixInferenceSpec(StrictModel):
    backend: BackendSpec
    engine_seed: int = Field(ge=0)
    seeds: tuple[int, ...] = Field(min_length=1)
    answer_conditions: tuple[AnswerCondition, ...] = Field(min_length=1)
    confidence_readouts: tuple[ConfidenceReadout, ...] = Field(min_length=1)
    design: MatrixDesign

    @model_validator(mode="after")
    def validate_matrix(self) -> "MatrixInferenceSpec":
        if any(seed < 0 for seed in self.seeds) or len(set(self.seeds)) != len(
            self.seeds
        ):
            raise ValueError("matrix seeds must be unique non-negative integers")
        answer_ids = [condition.condition_id for condition in self.answer_conditions]
        readout_ids = [readout.readout_id for readout in self.confidence_readouts]
        if len(set(answer_ids)) != len(answer_ids):
            raise ValueError("answer condition IDs must be unique")
        if len(set(readout_ids)) != len(readout_ids):
            raise ValueError("confidence readout IDs must be unique")
        answer_set, readout_set = set(answer_ids), set(readout_ids)
        if self.design.reference_answer_condition_id not in answer_set:
            raise ValueError("reference answer condition is not declared")
        if self.design.reference_confidence_readout_id not in readout_set:
            raise ValueError("reference confidence readout is not declared")
        if self.design.standardized_readout_id not in readout_set:
            raise ValueError("standardized confidence readout is not declared")
        if set(self.design.coupled_readout_by_answer) != answer_set:
            raise ValueError("coupled readout mapping must cover every answer condition")
        if not set(self.design.coupled_readout_by_answer.values()) <= readout_set:
            raise ValueError("coupled readout mapping names an undeclared readout")
        if not set(self.design.report_only_readout_ids) <= readout_set:
            raise ValueError("report-only design names an undeclared readout")
        if len(set(self.design.report_only_readout_ids)) != len(
            self.design.report_only_readout_ids
        ):
            raise ValueError("report-only readout IDs must be unique")
        if (
            self.design.coupled_readout_by_answer[
                self.design.reference_answer_condition_id
            ]
            != self.design.reference_confidence_readout_id
        ):
            raise ValueError("the reference answer must couple to the reference readout")
        return self


class AnswerSourceReference(StrictModel):
    """A prior run whose immutable answers this run consumes, read-only.

    A confidence-only run is not a resumption of the run that produced the
    answers: it is a new run citing an earlier artifact as an input, so a
    differing Git commit is ordinary provenance rather than an illegal resume.
    """

    run_id: str = Field(min_length=1)
    config_path: str = Field(min_length=1)
    config_sha256: Sha256

    @model_validator(mode="after")
    def repository_relative_path(self) -> "AnswerSourceReference":
        path = Path(self.config_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("answer-source config path must be repository-relative")
        return self


class MatrixExperimentConfig(StrictModel):
    schema_version: Literal["2.0"]
    experiment_id: Slug
    study: Literal[
        "matrix_smoke",
        "development_matrix",
        "certification",
        "broad",
        "factorial",
        "decomposition",
        # The frozen two-phase core: a held-out confirmatory run whose
        # thresholds come from the immutable calibration artifact rather than
        # from a certification decision, because the core has no certification
        # phase to produce one.
        "core_test",
        # Repeated sampling seeds on a strict subset of the held-out items.
        # It reads test-partition data and so passes the same gate, but it is
        # explicitly not confirmatory: it estimates how much a contrast moves
        # when the decoder is resampled, which is a different question from the
        # frozen-threshold contrast and enters no confirmatory family.
        "core_seed_robustness",
    ]
    confirmatory: bool
    model_integration: IntegrationReference | None
    dataset_integration: IntegrationReference | None
    frozen_policy: IntegrationReference | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    certification_decision: IntegrationReference | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    dense_qualification: IntegrationReference | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    answer_source: AnswerSourceReference | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    model: ModelSpec
    dataset: DatasetSpec
    inference: MatrixInferenceSpec
    confidence: ConfidenceProtocolSpec
    execution: ExecutionSpec
    storage: StorageSpec

    @model_validator(mode="after")
    def enforce_development_boundaries(self) -> "MatrixExperimentConfig":
        if self.dense_qualification is not None and self.study not in {
            "development_matrix",
            "factorial",
        }:
            raise ValueError(
                "only dense development and factorial matrices may cite a dense "
                "qualification record"
            )
        if self.study == "matrix_smoke":
            if self.confirmatory:
                raise ValueError("matrix_smoke cannot be confirmatory")
            if self.inference.backend.name != "mock":
                raise ValueError("matrix_smoke requires the mock backend")
            if self.dataset.loader != "jsonl_fixture":
                raise ValueError("matrix_smoke requires a fixture dataset")
        else:
            if self.model_integration is None or self.dataset_integration is None:
                raise ValueError(
                    "development_matrix requires model and dataset integration records"
                )
            if self.inference.backend.name != "vllm":
                raise ValueError("development_matrix requires GPU vLLM inference")
            if self.dataset.loader != "cached_jsonl":
                raise ValueError("scientific matrices require a prepared public dataset")
            if self.study == "development_matrix":
                if self.confirmatory or self.dataset.partition != "development":
                    raise ValueError(
                        "development_matrix must be non-confirmatory development data"
                    )
                if self.frozen_policy is not None:
                    raise ValueError("development matrices cannot reference a frozen policy")
                if self.certification_decision is not None:
                    raise ValueError(
                        "development matrices cannot reference a certification decision"
                    )
            elif self.study == "certification":
                if not self.confirmatory or self.dataset.partition != "certification":
                    raise ValueError("certification study requires certification data")
                if len(self.inference.answer_conditions) != 1:
                    raise ValueError("certification generates only the reference answer condition")
                permitted_readouts = {
                    self.inference.design.reference_confidence_readout_id,
                    self.inference.design.standardized_readout_id,
                }
                actual_readouts = {
                    readout.readout_id for readout in self.inference.confidence_readouts
                }
                if actual_readouts != permitted_readouts:
                    raise ValueError(
                        "certification generates only the reference and standardized "
                        "confidence readouts"
                    )
                if self.inference.seeds != (1729,):
                    raise ValueError("certification uses only prespecified replicate seed 1729")
                if self.frozen_policy is None:
                    raise ValueError("certification requires a frozen development policy")
                if self.certification_decision is not None:
                    raise ValueError(
                        "certification cannot reference a post-certification decision"
                    )
            elif self.study == "core_test":
                if not self.confirmatory or self.dataset.partition != "test":
                    raise ValueError(
                        "the frozen-core held-out test requires confirmatory test data"
                    )
                if (
                    self.frozen_policy is not None
                    or self.certification_decision is not None
                ):
                    raise ValueError(
                        "the frozen-core held-out test takes its thresholds from the "
                        "calibration artifact, not from a certification decision"
                    )
            elif self.study == "core_seed_robustness":
                if self.confirmatory or self.dataset.partition != "test":
                    raise ValueError(
                        "seed robustness reads test data but is not confirmatory"
                    )
                if self.dataset.item_subselection_salt is None:
                    raise ValueError(
                        "seed robustness must draw a declared subset of the "
                        "held-out items, not the full held-out window"
                    )
                if len(self.inference.seeds) < 2:
                    raise ValueError(
                        "seed robustness requires repeated seeds to be informative"
                    )
            elif (
                not self.confirmatory
                or self.dataset.partition != "test"
                or self.frozen_policy is None
                or self.certification_decision is None
            ):
                raise ValueError(
                    "broad/factorial/decomposition matrices require confirmatory test "
                    "data, a frozen policy, and a certification decision"
                )
            for label, revision in (
                ("model", self.model.revision),
                ("tokenizer", self.model.tokenizer_revision),
            ):
                if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
                    raise ValueError(f"{label} revision must be a full commit SHA")
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


def load_matrix_config(path: Path) -> MatrixExperimentConfig:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load matrix configuration {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("matrix configuration must be a YAML mapping")
    return MatrixExperimentConfig.model_validate(payload)
