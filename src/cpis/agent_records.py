"""Immutable raw and derived records for BFCL agent episodes."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Literal

from pydantic import Field, model_validator

from cpis.config import ModelSpec, SamplingSpec
from cpis.matrix_records import MatrixRole
from cpis.records import FrozenRecord


class AgentStepRecord(FrozenRecord):
    turn_index: int = Field(ge=0)
    step_index: int = Field(ge=0)
    sampling_seed: int = Field(ge=0)
    messages_sha256: str
    tools_sha256: str
    raw_text: str
    finish_reason: str | None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    parse_status: Literal["valid_calls", "valid_no_call", "invalid"]
    parse_error: str | None
    calls: tuple[dict[str, Any], ...]
    execution_results: tuple[str, ...]


class AgentEpisodeRecord(FrozenRecord):
    schema_version: Literal["agent-episode-1.0"] = "agent-episode-1.0"
    record_id: str
    observation_id: str
    run_id: str
    experiment_id: str
    created_at_utc: datetime
    dataset_id: str
    dataset_revision: str
    dataset_item_id: str
    dataset_group_id: str
    dataset_partition: Literal["development", "certification", "test"]
    source_case_sha256: str
    function_docs_bundle_sha256: str
    confidence_protocol_id: str
    prompt_template_version: str
    bfcl_protocol_id: str
    evaluator_revision: str
    parser_family: Literal["qwen3", "mistral"]
    parser_version: str
    model: ModelSpec
    inference_backend: str
    inference_backend_version: str
    engine_seed: int
    answer_condition_id: str
    answer_sampling: SamplingSpec
    replicate_seed: int
    tool_call_pairing_id: str | None = None
    force_terminated: bool
    invalid_tool_outputs: int = Field(ge=0)
    turns_decoded: tuple[tuple[tuple[str, ...], ...], ...]
    steps: tuple[AgentStepRecord, ...]
    final_messages: tuple[dict[str, Any], ...]

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AgentParsedRecord(FrozenRecord):
    schema_version: Literal["agent-parsed-1.0"] = "agent-parsed-1.0"
    record_id: str
    episode_observation_id: str
    episode_raw_record_id: str
    episode_raw_record_sha256: str
    confidence_raw_record_id: str
    confidence_raw_record_sha256: str
    run_id: str
    experiment_id: str
    dataset_id: str
    dataset_revision: str
    dataset_item_id: str
    dataset_group_id: str
    dataset_partition: Literal["development", "certification", "test"]
    answer_condition_id: str
    confidence_readout_id: str
    design_roles: tuple[MatrixRole, ...] = Field(min_length=1)
    seed: int
    scorer: Literal["bfcl_v3_official_multi_turn_v1"]
    score_valid: bool
    score_error_type: str | None = None
    score_details: dict[str, Any]
    confidence_parse_status: Literal["valid", "invalid"]
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_parse_error: str | None = None
    correct: bool

    @model_validator(mode="after")
    def confidence_state_is_consistent(self) -> "AgentParsedRecord":
        if (self.confidence_parse_status == "valid") != (self.confidence is not None):
            raise ValueError("valid agent confidence must have exactly one numeric value")
        return self
