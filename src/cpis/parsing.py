"""Prespecified terminal-JSON parsers and deterministic scoring."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class AnswerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: str = Field(min_length=1)


class ConfidenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    confidence: float = Field(ge=0, le=1)

    @field_validator("confidence", mode="before")
    @classmethod
    def require_numeric_confidence(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("confidence must be a JSON number")
        return value


class AnswerParseOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["valid", "invalid"]
    answer: str | None = None
    error: str | None = None


class ConfidenceParseOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["valid", "invalid"]
    confidence: float | None = None
    error: str | None = None


class DuplicateKeyError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _terminal_json(raw_text: str, allow_closing_fence: bool) -> Any:
    """Decode final JSON while retaining preceding reasoning as immutable raw data."""
    start = raw_text.rfind("{")
    if start < 0:
        raise ValueError("response has no terminal JSON object")
    suffix = raw_text[start:]
    if not allow_closing_fence:
        return json.loads(suffix, object_pairs_hook=_unique_object)
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
    payload, end = decoder.raw_decode(suffix)
    remainder = suffix[end:].strip()
    if remainder not in {"", "```"}:
        raise ValueError("terminal JSON object is followed by non-fence text")
    return payload


def parse_answer(raw_text: str, parser_version: str) -> AnswerParseOutcome:
    if parser_version == "identity_text_v1":
        if not raw_text.strip():
            return AnswerParseOutcome(status="invalid", error="response is empty")
        return AnswerParseOutcome(status="valid", answer=raw_text)
    if parser_version not in {"final_json_answer_v1", "final_json_answer_v2"}:
        raise ValueError(f"unsupported answer parser version: {parser_version}")
    try:
        response = AnswerResponse.model_validate(
            _terminal_json(
                raw_text, allow_closing_fence=parser_version.endswith("_v2")
            )
        )
    except (json.JSONDecodeError, DuplicateKeyError, ValidationError, ValueError) as exc:
        return AnswerParseOutcome(status="invalid", error=str(exc))
    return AnswerParseOutcome(status="valid", answer=response.answer)


def parse_confidence(raw_text: str, parser_version: str) -> ConfidenceParseOutcome:
    if parser_version not in {
        "final_json_confidence_v1",
        "final_json_confidence_v2",
    }:
        raise ValueError(f"unsupported confidence parser version: {parser_version}")
    try:
        response = ConfidenceResponse.model_validate(
            _terminal_json(
                raw_text, allow_closing_fence=parser_version.endswith("_v2")
            )
        )
    except (json.JSONDecodeError, DuplicateKeyError, ValidationError, ValueError) as exc:
        return ConfidenceParseOutcome(status="invalid", error=str(exc))
    return ConfidenceParseOutcome(status="valid", confidence=response.confidence)


def score_answer(answer: str | None, reference: str | None, scorer: str) -> bool | None:
    if scorer in {"manyifeval_official_v1", "stylembpp_official_v1"}:
        return None
    if scorer != "exact_match_casefold_v1":
        raise ValueError(f"unsupported scorer: {scorer}")
    if reference is None:
        raise ValueError("exact-match scoring requires a reference answer")
    if answer is None:
        return False
    return answer.strip().casefold() == reference.strip().casefold()
