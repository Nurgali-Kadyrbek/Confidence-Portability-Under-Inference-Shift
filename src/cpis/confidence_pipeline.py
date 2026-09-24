"""Generate confidence readouts from answers a previous run already produced.

A confidence readout depends only on the item, the answer it reads back and the
readout protocol. Regenerating answers because a readout budget changed wastes
the expensive half of the experiment, and `run_id` hashes the whole config, so
that is exactly what a naive budget change does.

This consumes a prior run's answers read-only. It is not a resumption of that
run: the answers are an immutable input and the confidence records land in a
new run directory, so a differing Git commit is ordinary provenance rather than
an illegal resume. The source run is never opened for writing.

Reuse is gated on `answer_artifact_key`: every answer-affecting input must
agree between the two configs. Anything that could have changed the answer
makes the answers ineligible and the run refuses rather than silently pairing a
readout with an answer produced under different conditions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from cpis.artifacts import answer_artifact_key, answer_identity
from cpis.datasets import DatasetItem
from cpis.matrix_config import MatrixExperimentConfig, load_matrix_config
from cpis.matrix_pipeline import _answer_observation_id, _condition_map
from cpis.records import GenerationRecord, stable_generation_id
from cpis.storage import StorageLayout, read_json


@dataclass(frozen=True)
class SourceAnswer:
    item: DatasetItem
    condition_id: str
    seed: int
    record: GenerationRecord
    path: Path
    artifact_key: str

    @property
    def sha256(self) -> str:
        return self.record.sha256


def load_answer_source(
    config: MatrixExperimentConfig, repository_root: Path
) -> MatrixExperimentConfig:
    """Resolve the cited source config and prove it is the one that ran."""

    reference = config.answer_source
    if reference is None:
        raise ValueError("a confidence-only run must cite an answer source")
    repository_root = repository_root.resolve()
    path = (repository_root / reference.config_path).resolve()
    try:
        path.relative_to(repository_root)
    except ValueError as exc:
        raise RuntimeError("answer-source config escapes the repository") from exc
    if not path.is_file():
        raise RuntimeError(f"answer-source config is missing: {reference.config_path}")
    source = load_matrix_config(path)
    if source.config_sha256 != reference.config_sha256:
        raise RuntimeError("answer-source config hash does not match its reference")
    if source.run_id != reference.run_id:
        raise RuntimeError("answer-source run ID does not match its configuration")
    return source


def _first_difference(left: dict, right: dict, trail: str = "") -> str:
    for key in sorted(set(left) | set(right)):
        here = f"{trail}.{key}" if trail else key
        a, b = left.get(key), right.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            found = _first_difference(a, b, here)
            if found:
                return found
        elif a != b:
            return f"{here}: source={a!r} requested={b!r}"
    return ""


def verify_answer_identity(
    config: MatrixExperimentConfig,
    source: MatrixExperimentConfig,
    *,
    item_id: str,
    condition_id: str,
    seed: int,
) -> str:
    """Require every answer-affecting input to agree, and say what differs."""

    conditions, source_conditions = _condition_map(config), _condition_map(source)
    if condition_id not in source_conditions:
        raise RuntimeError(
            f"answer source does not declare answer condition {condition_id}"
        )
    if seed not in set(source.inference.seeds):
        raise RuntimeError(f"answer source does not declare replicate seed {seed}")
    requested = answer_artifact_key(
        config, item_id=item_id, condition=conditions[condition_id], seed=seed
    )
    available = answer_artifact_key(
        source, item_id=item_id, condition=source_conditions[condition_id], seed=seed
    )
    if requested != available:
        difference = _first_difference(
            answer_identity(
                source, item_id=item_id, condition=source_conditions[condition_id], seed=seed
            ),
            answer_identity(
                config, item_id=item_id, condition=conditions[condition_id], seed=seed
            ),
        )
        raise RuntimeError(
            "answer source was produced under different answer-affecting inputs; "
            f"first difference at {difference}"
        )
    return requested


def resolve_source_answers(
    config: MatrixExperimentConfig,
    source: MatrixExperimentConfig,
    items: list[DatasetItem],
    repository_root: Path,
) -> dict[tuple[str, str, int], SourceAnswer]:
    """Locate and verify every answer this confidence run intends to read."""

    storage = StorageLayout.from_spec(config.storage, repository_root)
    source_answers = storage.output_root / source.run_id / "raw" / "answer"
    if not source_answers.is_dir():
        raise RuntimeError(f"answer source has no saved answers: {source_answers}")
    resolved: dict[tuple[str, str, int], SourceAnswer] = {}
    for item in items:
        for condition in config.inference.answer_conditions:
            for seed in config.inference.seeds:
                key = verify_answer_identity(
                    config,
                    source,
                    item_id=item.item_id,
                    condition_id=condition.condition_id,
                    seed=seed,
                )
                observation_id = _answer_observation_id(
                    source, item.item_id, condition.condition_id, seed
                )
                record_id = stable_generation_id(observation_id, "answer")
                path = source_answers / f"{record_id}.json"
                if not path.is_file():
                    raise RuntimeError(
                        f"answer source is missing a generation for "
                        f"{item.item_id}/{condition.condition_id}/seed {seed}"
                    )
                record = GenerationRecord.model_validate(read_json(path))
                if record.generation_stage != "answer":
                    raise RuntimeError(f"source record is not an answer: {path}")
                resolved[(item.item_id, condition.condition_id, seed)] = SourceAnswer(
                    item=item,
                    condition_id=condition.condition_id,
                    seed=seed,
                    record=record,
                    path=path,
                    artifact_key=key,
                )
    return resolved


def answer_source_provenance(
    source: MatrixExperimentConfig, resolved: dict[tuple[str, str, int], SourceAnswer]
) -> dict:
    """A citation of the consumed answers, for the confidence run's manifest."""

    digest = hashlib.sha256()
    for key in sorted(resolved):
        digest.update(resolved[key].sha256.encode("utf-8"))
    return {
        "source_answer_run_id": source.run_id,
        "source_answer_config_sha256": source.config_sha256,
        "source_answer_count": len(resolved),
        "source_answer_set_sha256": digest.hexdigest(),
        "answer_artifact_key_sample": (
            resolved[sorted(resolved)[0]].artifact_key if resolved else None
        ),
        "consumed_read_only": True,
    }
