"""Schemas for immutable raw generations and separate parsed outcomes."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cpis.config import InferenceCondition, ModelSpec, SamplingSpec


class FrozenRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GenerationRecord(FrozenRecord):
    schema_version: Literal["2.0"] = "2.0"
    record_id: str
    observation_id: str
    generation_stage: Literal["answer", "confidence"]
    parent_answer_record_id: str | None = None
    run_id: str
    experiment_id: str
    created_at_utc: datetime
    dataset_id: str
    dataset_revision: str
    dataset_item_id: str
    dataset_partition: Literal["development", "certification", "test"]
    prompt_text: str
    prompt_sha256: str
    confidence_protocol_id: str
    prompt_template_version: str
    parser_version: str
    response_format: str
    model: ModelSpec
    inference_backend: str
    inference_backend_version: str
    engine_seed: int
    condition: InferenceCondition
    sampling: SamplingSpec
    seed: int
    sampling_seed: int
    raw_text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def stage_parent_is_consistent(self) -> "GenerationRecord":
        if self.generation_stage == "answer" and self.parent_answer_record_id is not None:
            raise ValueError("answer generations cannot have a parent answer record")
        if self.generation_stage == "confidence" and self.parent_answer_record_id is None:
            raise ValueError("confidence generations require a parent answer record")
        return self

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ParsedRecord(FrozenRecord):
    schema_version: Literal["2.0"] = "2.0"
    record_id: str
    answer_raw_record_id: str
    answer_raw_record_sha256: str
    confidence_raw_record_id: str
    confidence_raw_record_sha256: str
    run_id: str
    experiment_id: str
    dataset_id: str
    dataset_revision: str
    dataset_item_id: str
    dataset_partition: Literal["development", "certification", "test"]
    condition_id: str
    seed: int
    answer_parser_version: str
    confidence_parser_version: str
    scorer: str
    answer_parse_status: Literal["valid", "invalid"]
    confidence_parse_status: Literal["valid", "invalid"]
    parse_status: Literal["valid", "invalid"]
    parsed_answer: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    scoring_status: Literal["scored", "pending_official_evaluator"] = "scored"
    correct: bool | None
    answer_parse_error: str | None = None
    confidence_parse_error: str | None = None

    @model_validator(mode="after")
    def scoring_state_is_consistent(self) -> "ParsedRecord":
        pending = self.scoring_status == "pending_official_evaluator"
        if pending != (self.correct is None):
            raise ValueError("pending benchmark scoring must have correct=null")
        return self


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stable_observation_id(
    run_id: str, item_id: str, condition_id: str, seed: int, model_mode: str
) -> str:
    identity = "\0".join((run_id, item_id, condition_id, str(seed), model_mode))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def stable_generation_id(observation_id: str, stage: str) -> str:
    if stage not in {"answer", "confidence"}:
        raise ValueError(f"unknown generation stage: {stage}")
    return hashlib.sha256(f"{observation_id}\0{stage}".encode("utf-8")).hexdigest()


def derive_sampling_seed(replicate_seed: int, item_id: str, stage: str) -> int:
    """Derive item/stage randomness, shared across paired decoder conditions."""
    if replicate_seed < 0:
        raise ValueError("replicate seed must be non-negative")
    if stage not in {"answer", "confidence"}:
        raise ValueError(f"unknown generation stage: {stage}")
    identity = f"cpis-sampling-seed-v1\0{replicate_seed}\0{item_id}\0{stage}"
    return int.from_bytes(
        hashlib.sha256(identity.encode("utf-8")).digest()[:8], "big"
    ) & ((1 << 63) - 1)
