"""Parsed matrix-cell records linking immutable answer and readout generations."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from cpis.records import FrozenRecord

MatrixRole = Literal[
    "coupled",
    "standardized",
    "decomposition_reference_readout",
    "report_only",
]


class MatrixParsedRecord(FrozenRecord):
    schema_version: Literal["matrix-parsed-1.0"] = "matrix-parsed-1.0"
    record_id: str
    answer_observation_id: str
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
    answer_condition_id: str
    confidence_readout_id: str
    design_roles: tuple[MatrixRole, ...] = Field(min_length=1)
    seed: int
    answer_sampling_seed: int
    confidence_sampling_seed: int
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
    def scoring_state_is_consistent(self) -> "MatrixParsedRecord":
        pending = self.scoring_status == "pending_official_evaluator"
        if pending != (self.correct is None):
            raise ValueError("pending benchmark scoring must have correct=null")
        return self
