"""Stage-addressable identity for generated artifacts.

`run_id` hashes the whole configuration, so changing a confidence budget
changes the run directory and orphans answers that the new budget cannot have
affected. An answer does not causally depend on a later readout: the dependency
is item -> answer -> confidence, and identity should follow that edge.

`answer_artifact_key` derives an answer's identity from answer-affecting
inputs alone. A confidence-only run can therefore locate previously generated
answers, consume them read-only, and record their provenance, without resuming
or mutating the run that produced them. Nothing here weakens commit-pinned
resume: a new run is a new run, and it cites the old artifacts as inputs.

Reuse is permitted only when every answer-affecting property matches. Any field
that can change the generated text belongs in the key, including the execution
batch size, because generation is measurably not batch-invariant - see
`evidence/environment/batch_size_generation_invariance_v1.json`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from cpis.matrix_config import AnswerCondition, MatrixExperimentConfig

ANSWER_KEY_VERSION = "answer-artifact-key-1.0"


def answer_identity(
    config: MatrixExperimentConfig, *, item_id: str, condition: AnswerCondition, seed: int
) -> dict[str, Any]:
    """Every input that can change an answer, and nothing that cannot."""

    model = config.model
    protocol = config.confidence
    backend = config.inference.backend
    return {
        "key_version": ANSWER_KEY_VERSION,
        "dataset": {
            "dataset_id": config.dataset.dataset_id,
            "revision": config.dataset.revision,
            "content_sha256": config.dataset.content_sha256,
            "item_id": item_id,
        },
        "model": {
            "model_id": model.model_id,
            "revision": model.revision,
            "tokenizer_id": model.tokenizer_id,
            "tokenizer_revision": model.tokenizer_revision,
            "dtype": model.dtype,
            "prompt_format": model.prompt_format,
            "chat_template_id": model.chat_template_id,
            "chat_template_sha256": model.chat_template_sha256,
            "chat_template_kwargs": model.chat_template_kwargs,
            "system_prompt": model.system_prompt,
            "model_mode": model.model_mode,
            "max_model_len": model.max_model_len,
        },
        # Only the answer half of the protocol. The confidence template, parser
        # and response format are deliberately absent: they cannot reach the
        # answer, and including them is what orphaned answers on a budget change.
        "answer_protocol": {
            "protocol_id": protocol.protocol_id,
            "prompt_template_version": protocol.prompt_template_version,
            "answer_prompt_template": protocol.answer_prompt_template,
            "answer_response_format": protocol.answer_response_format,
            "answer_parser_version": protocol.answer_parser_version,
        },
        "sampling": condition.sampling.model_dump(mode="json"),
        "seed": seed,
        "engine": {
            "name": backend.name,
            "version": backend.version,
            "use_flashinfer_sampler": backend.use_flashinfer_sampler,
            "tokenizer_mode": backend.tokenizer_mode,
            "config_format": backend.config_format,
            "load_format": backend.load_format,
            # Batch composition perturbs reduction order and flips sampled
            # tokens, so it is an answer-affecting input, not an inert knob.
            "batch_size": config.execution.batch_size,
        },
    }


def answer_artifact_key(
    config: MatrixExperimentConfig, *, item_id: str, condition: AnswerCondition, seed: int
) -> str:
    identity = answer_identity(config, item_id=item_id, condition=condition, seed=seed)
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
