"""Versioned two-pass answer and confidence prompt construction."""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

from cpis.config import ConfidenceProtocolSpec
from cpis.datasets import DatasetItem


class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: Literal["answer", "confidence"]
    text: str
    sha256: str


def _prompt(stage: Literal["answer", "confidence"], text: str) -> Prompt:
    return Prompt(
        stage=stage,
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _substitute(template: str, values: dict[str, str]) -> str:
    pattern = re.compile("|".join(re.escape(field) for field in values))
    return pattern.sub(lambda match: values[match.group(0)], template)


def build_answer_prompt(
    item: DatasetItem, protocol: ConfidenceProtocolSpec
) -> Prompt:
    text = _substitute(protocol.answer_prompt_template, {"{question}": item.question})
    return _prompt("answer", text)


def build_confidence_prompt(
    item: DatasetItem,
    proposed_answer: str,
    protocol: ConfidenceProtocolSpec,
) -> Prompt:
    replacements = {
        "{question}": item.question,
        "{proposed_answer}": proposed_answer,
        "{success_criterion}": item.success_criterion,
    }
    text = _substitute(protocol.confidence_prompt_template, replacements)
    return _prompt("confidence", text)
