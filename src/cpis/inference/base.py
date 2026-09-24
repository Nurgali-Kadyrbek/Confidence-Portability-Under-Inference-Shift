"""Small batch inference interface shared by mock and GPU backends."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, JsonValue

from cpis.config import InferenceCondition, SamplingSpec


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    stage: Literal["answer", "confidence"]
    prompt: str
    seed: int
    condition: InferenceCondition
    chat_template_kwargs: dict[str, JsonValue] | None = None


class InferenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ChatInferenceRequest(BaseModel):
    """A native multi-turn/tool request for an agentic benchmark step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    messages: tuple[dict[str, JsonValue], ...]
    tools: tuple[dict[str, JsonValue], ...]
    seed: int
    sampling: SamplingSpec
    chat_template_kwargs: dict[str, JsonValue] | None = None


class InferenceBackend(Protocol):
    @property
    def version(self) -> str: ...

    def generate(self, requests: list[InferenceRequest]) -> list[InferenceResponse]: ...
