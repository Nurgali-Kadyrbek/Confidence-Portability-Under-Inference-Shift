"""Strict, versioned experiment configuration."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


PinnedRevision = Annotated[str, Field(min_length=7)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Slug = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,79}$")]


class ModelSpec(StrictModel):
    model_id: str = Field(min_length=1)
    revision: PinnedRevision
    tokenizer_id: str = Field(min_length=1)
    tokenizer_revision: PinnedRevision
    parameter_count_billion: float = Field(gt=0, le=20)
    dtype: Literal["bfloat16", "float16"] = "bfloat16"
    tensor_parallel_size: int = Field(default=1, ge=1, le=8)
    max_model_len: int = Field(gt=0)
    gpu_memory_utilization: float = Field(default=0.90, gt=0, le=1)
    trust_remote_code: bool = False
    language_model_only: bool = True
    enforce_eager: bool = False
    prompt_format: Literal["plain", "chat"] = "chat"
    chat_template_id: str | None
    chat_template_sha256: Sha256 | None
    chat_template_kwargs: dict[str, JsonValue] = Field(default_factory=dict)
    system_prompt: str | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None
    )
    model_mode: Literal[
        "mock",
        "thinking",
        "non_thinking",
        "instruct",
        "reasoning",
        "agentic",
    ]

    @model_validator(mode="after")
    def validate_revisions_and_template(self) -> "ModelSpec":
        floating = {"main", "master", "latest", "head"}
        if self.revision.casefold() in floating:
            raise ValueError("model revision must be immutable, not a floating ref")
        if self.tokenizer_revision.casefold() in floating:
            raise ValueError("tokenizer revision must be immutable, not a floating ref")
        if self.prompt_format == "chat":
            if not self.chat_template_id or not self.chat_template_sha256:
                raise ValueError("chat prompts require a versioned template ID and SHA-256")
        elif self.chat_template_id is not None or self.chat_template_sha256 is not None:
            raise ValueError("plain prompts must not declare a chat template")
        if self.prompt_format == "plain" and (
            self.chat_template_kwargs or self.system_prompt is not None
        ):
            raise ValueError(
                "plain prompts must not declare chat-template kwargs or a system prompt"
            )
        return self


class SplitSpec(StrictModel):
    salt: str = Field(min_length=8)
    development: float = Field(gt=0, lt=1)
    certification: float = Field(gt=0, lt=1)
    test: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def fractions_sum_to_one(self) -> "SplitSpec":
        if abs(self.development + self.certification + self.test - 1.0) > 1e-12:
            raise ValueError("split fractions must sum to exactly 1.0")
        return self


class DatasetSpec(StrictModel):
    dataset_id: str = Field(min_length=1)
    revision: PinnedRevision
    loader: Literal["jsonl_fixture", "cached_jsonl"]
    source_path: str = Field(min_length=1)
    source_url: str | None
    content_sha256: Sha256
    scorer: Literal[
        "exact_match_casefold_v1",
        "manyifeval_official_v1",
        "stylembpp_official_v1",
        "bfcl_v3_official_multi_turn_v1",
    ]
    split: SplitSpec
    partition: Literal["development", "certification", "test"]
    item_limit: int | None = Field(default=None, ge=1)
    # Skip this many ranked items before taking item_limit. One salt then yields
    # disjoint deterministic subsets of the same pool: a phase takes its own
    # window rather than a prefix that nests inside every larger one.
    item_offset: int | None = Field(
        default=None, ge=0, exclude_if=lambda value: value is None
    )
    item_selection_salt: str | None = None
    # A second, independent draw taken from inside the window the fields above
    # already selected. The robustness subset must be a strict subset of the
    # items the broad study evaluated, so it cannot be produced by re-ranking
    # the whole partition: the test partitions are larger than the broad
    # study's window and a fresh ranking over them would reach items that were
    # never run.
    item_subselection_salt: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    item_subselection_limit: int | None = Field(
        default=None, ge=1, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def revision_is_pinned(self) -> "DatasetSpec":
        if self.revision.casefold() in {"main", "master", "latest", "head"}:
            raise ValueError("dataset revision must be immutable")
        source = Path(self.source_path)
        if self.loader == "cached_jsonl":
            if source.is_absolute() or ".." in source.parts:
                raise ValueError(
                    "cached_jsonl source_path must be relative to the dataset cache"
                )
            if not self.source_url or not self.source_url.startswith(("https://", "http://")):
                raise ValueError("cached_jsonl requires its public source URL")
        elif self.source_url is not None:
            raise ValueError("fixture data must not declare a public source URL")
        if self.item_offset is not None and self.item_limit is None:
            raise ValueError("item_offset requires item_limit")
        if (self.item_limit is None) != (self.item_selection_salt is None):
            raise ValueError(
                "item_limit and item_selection_salt must be declared together"
            )
        if self.item_selection_salt is not None and len(self.item_selection_salt) < 8:
            raise ValueError("item_selection_salt must contain at least 8 characters")
        if (self.item_subselection_salt is None) != (
            self.item_subselection_limit is None
        ):
            raise ValueError(
                "item_subselection_salt and item_subselection_limit must be "
                "declared together"
            )
        if self.item_subselection_salt is not None:
            if len(self.item_subselection_salt) < 8:
                raise ValueError(
                    "item_subselection_salt must contain at least 8 characters"
                )
            if self.item_limit is None:
                raise ValueError("item_subselection requires item_limit")
            if self.item_subselection_limit > self.item_limit:
                raise ValueError(
                    f"item_subselection_limit={self.item_subselection_limit} exceeds "
                    f"the selected window item_limit={self.item_limit}"
                )
        return self


class SamplingSpec(StrictModel):
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    top_k: int | None = Field(default=None, ge=1)
    min_p: float | None = Field(ge=0, le=1)
    max_tokens: int = Field(gt=0)
    min_tokens: int = Field(ge=0)
    n: Literal[1] = 1
    presence_penalty: float = Field(ge=-2, le=2)
    frequency_penalty: float = Field(ge=-2, le=2)
    repetition_penalty: float = Field(gt=0, le=2)
    ignore_eos: bool
    detokenize: Literal[True]
    skip_special_tokens: bool
    spaces_between_special_tokens: bool
    include_stop_str_in_output: bool
    stop: tuple[str, ...]

    thinking_token_budget: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def prevent_backend_parameter_rewrites(self) -> "SamplingSpec":
        if 0 < self.temperature < 0.01:
            raise ValueError("vLLM clamps temperatures below 0.01; use 0 or >= 0.01")
        if self.temperature == 0 and (
            self.top_p != 1.0
            or self.top_k is not None
            or self.min_p not in (None, 0.0)
        ):
            raise ValueError(
                "greedy decoding requires top_p=1, top_k=null, and min_p=null/0"
            )
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens cannot exceed max_tokens")
        if any(not value for value in self.stop):
            raise ValueError("stop strings cannot be empty")
        return self


class InferenceCondition(StrictModel):
    condition_id: Slug
    answer: SamplingSpec
    confidence: SamplingSpec


class BackendSpec(StrictModel):
    name: Literal["mock", "vllm"]
    version: str = Field(min_length=1)
    use_flashinfer_sampler: bool | None = None
    mock_answer_response: str | None = None
    mock_confidence_response: str | None = None
    tokenizer_mode: Literal["auto", "slow", "mistral", "custom"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    config_format: Literal["auto", "hf", "mistral"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    load_format: Literal["auto", "safetensors", "mistral"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    worker_multiproc_method: Literal["spawn"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def backend_fields_match(self) -> "BackendSpec":
        responses = (self.mock_answer_response, self.mock_confidence_response)
        if self.name == "mock" and any(response is None for response in responses):
            raise ValueError("mock backend requires answer and confidence responses")
        if self.name != "mock" and any(response is not None for response in responses):
            raise ValueError("mock responses are test-only")
        if self.name == "vllm" and self.use_flashinfer_sampler is None:
            raise ValueError("vLLM backend requires an explicit sampler implementation")
        if self.name == "mock" and self.use_flashinfer_sampler is not None:
            raise ValueError("mock backend cannot configure the vLLM sampler")
        loader_options = (self.tokenizer_mode, self.config_format, self.load_format)
        if self.name != "vllm" and any(option is not None for option in loader_options):
            raise ValueError("model loader modes are supported only by vLLM")
        if any(option is not None for option in loader_options) and any(
            option is None for option in loader_options
        ):
            raise ValueError(
                "tokenizer_mode, config_format, and load_format must be pinned together"
            )
        if self.name != "vllm" and self.worker_multiproc_method is not None:
            raise ValueError("worker multiprocessing is supported only by vLLM")
        return self


class InferenceSpec(StrictModel):
    backend: BackendSpec
    engine_seed: int = Field(ge=0)
    seeds: tuple[int, ...] = Field(min_length=1)
    conditions: tuple[InferenceCondition, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def identifiers_are_unique(self) -> "InferenceSpec":
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("random seeds must be non-negative")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("random seeds must be unique")
        ids = [condition.condition_id for condition in self.conditions]
        if len(set(ids)) != len(ids):
            raise ValueError("inference condition IDs must be unique")
        return self


class ConfidenceProtocolSpec(StrictModel):
    protocol_id: Slug
    prompt_template_version: PinnedRevision
    answer_prompt_template: str = Field(min_length=1)
    confidence_prompt_template: str = Field(min_length=1)
    answer_parser_version: Literal[
        "final_json_answer_v1",
        "final_json_answer_v2",
        "identity_text_v1",
    ]
    confidence_parser_version: Literal[
        "final_json_confidence_v1", "final_json_confidence_v2"
    ]
    answer_response_format: Literal["json_answer_v1", "free_text_v1"]
    confidence_response_format: Literal["json_confidence_v1"]

    @model_validator(mode="after")
    def templates_have_required_fields(self) -> "ConfidenceProtocolSpec":
        if self.answer_prompt_template.count("{question}") != 1:
            raise ValueError(
                "answer_prompt_template must contain {question} exactly once"
            )
        required = ("{question}", "{proposed_answer}", "{success_criterion}")
        if any(self.confidence_prompt_template.count(field) != 1 for field in required):
            raise ValueError(
                "confidence_prompt_template must contain question, proposed_answer, "
                "and success_criterion exactly once"
            )
        identity = self.answer_parser_version == "identity_text_v1"
        if identity != (self.answer_response_format == "free_text_v1"):
            raise ValueError(
                "identity_text_v1 and free_text_v1 must be configured together"
            )
        return self


class StorageSpec(StrictModel):
    root_env: str = Field(default="CPIS_STORAGE_ROOT", pattern=r"^[A-Z][A-Z0-9_]+$")
    model_cache_env: str = Field(default="CPIS_MODEL_CACHE", pattern=r"^[A-Z][A-Z0-9_]+$")
    dataset_cache_env: str = Field(default="CPIS_DATASET_CACHE", pattern=r"^[A-Z][A-Z0-9_]+$")
    output_root_env: str = Field(default="CPIS_OUTPUT_ROOT", pattern=r"^[A-Z][A-Z0-9_]+$")
    log_root_env: str = Field(default="CPIS_LOG_ROOT", pattern=r"^[A-Z][A-Z0-9_]+$")


class ExecutionSpec(StrictModel):
    """Process-level model replicas; each replica is launched independently."""

    replicas: int = Field(ge=1, le=8)
    batch_size: int = Field(default=64, ge=1, le=4096)


class ExperimentConfig(StrictModel):
    schema_version: Literal["1.0"]
    experiment_id: Slug
    study: Literal["smoke", "broad", "factorial", "decomposition", "agentic"]
    confirmatory: bool
    model: ModelSpec
    dataset: DatasetSpec
    inference: InferenceSpec
    confidence: ConfidenceProtocolSpec
    execution: ExecutionSpec
    storage: StorageSpec

    @model_validator(mode="after")
    def prohibit_test_backends_in_scientific_runs(self) -> "ExperimentConfig":
        if self.inference.backend.name == "mock" and self.study != "smoke":
            raise ValueError("mock inference is allowed only for study=smoke")
        if self.dataset.loader == "jsonl_fixture" and self.study != "smoke":
            raise ValueError("fixture datasets are allowed only for study=smoke")
        if self.inference.backend.name == "vllm":
            for label, revision in (
                ("model", self.model.revision),
                ("tokenizer", self.model.tokenizer_revision),
            ):
                if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
                    raise ValueError(
                        f"{label} revision must be a full 40-character commit SHA"
                    )
        if self.study == "smoke" and self.confirmatory:
            raise ValueError("smoke runs cannot be confirmatory")
        if self.execution.replicas * self.model.tensor_parallel_size > 8:
            raise ValueError(
                "replicas * tensor_parallel_size exceeds the eight-GPU target"
            )
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


def load_config(path: Path) -> ExperimentConfig:
    """Load YAML safely and reject every unknown or incomplete field."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load configuration {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("experiment configuration must be a YAML mapping")
    return ExperimentConfig.model_validate(payload)
