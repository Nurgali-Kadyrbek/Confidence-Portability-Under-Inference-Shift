"""Two-pass generation pipeline; inferential statistics are never computed here."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

from cpis.config import ExperimentConfig, InferenceCondition, SamplingSpec, load_config
from cpis.datasets import DatasetItem, assign_replica, load_dataset, partition_items
from cpis.inference.base import InferenceBackend, InferenceRequest
from cpis.inference.mock import MockBackend
from cpis.manifest import Manifest, build_manifest, git_metadata
from cpis.parsing import parse_answer, parse_confidence, score_answer
from cpis.phase_gate import (
    authorize_phase_config,
    record_dataset_access,
    require_phase_gate,
)
from cpis.prompts import Prompt, build_answer_prompt, build_confidence_prompt
from cpis.records import (
    GenerationRecord,
    ParsedRecord,
    derive_sampling_seed,
    stable_generation_id,
    stable_observation_id,
    utc_now,
)
from cpis.storage import RunLayout, StorageLayout, read_json, write_immutable_json

Stage = Literal["answer", "confidence"]
T = TypeVar("T")


@dataclass(frozen=True)
class RunResult:
    run_id: str
    run_directory: Path
    replica_index: int
    replica_count: int
    selected_items: int
    expected_observations: int
    expected_generations: int
    raw_created: int
    raw_reused: int
    parsed_created: int
    parsed_reused: int
    invalid_answer_outputs: int
    invalid_confidence_outputs: int


@dataclass(frozen=True)
class Observation:
    item: DatasetItem
    condition: InferenceCondition
    seed: int
    observation_id: str
    answer_record_id: str
    confidence_record_id: str


@dataclass(frozen=True)
class PendingGeneration:
    observation: Observation
    stage: Stage
    record_id: str
    parent_answer_record_id: str | None
    prompt: Prompt
    path: Path


def _chunks(values: list[T], size: int) -> list[list[T]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def ensure_manifest(
    config: ExperimentConfig,
    config_path: Path,
    repository_root: Path,
    run: RunLayout,
    replica_index: int,
) -> None:
    path = run.manifests / (
        f"replica-{replica_index:03d}-of-{config.execution.replicas:03d}.json"
    )
    if path.exists():
        existing = Manifest.model_validate(read_json(path))
        if (
            existing.run_id != config.run_id
            or existing.config_sha256 != config.config_sha256
            or existing.replica_index != replica_index
            or existing.replica_count != config.execution.replicas
        ):
            raise RuntimeError(f"manifest collision at {path}")
        if config.study not in {"smoke", "matrix_smoke"}:
            current_git = git_metadata(repository_root)
            if current_git.get("commit") != existing.git.get("commit"):
                raise RuntimeError(
                    "scientific run cannot resume under a different Git commit"
                )
            if current_git.get("dirty") or existing.git.get("dirty"):
                raise RuntimeError("scientific runs require a clean Git worktree")
        return
    manifest = build_manifest(config, config_path, repository_root, replica_index)
    if config.study not in {"smoke", "matrix_smoke"} and not manifest.git.get("available"):
        raise RuntimeError("scientific runs require a repository with a Git commit")
    if config.study not in {"smoke", "matrix_smoke"} and manifest.git.get("dirty"):
        raise RuntimeError("scientific runs require a clean Git worktree")
    write_immutable_json(path, manifest.model_dump(mode="json"))


def make_backend(config: ExperimentConfig, storage: StorageLayout) -> InferenceBackend:
    backend = config.inference.backend
    if backend.name == "mock":
        return MockBackend(
            answer_response=backend.mock_answer_response or "",
            confidence_response=backend.mock_confidence_response or "",
            version=backend.version,
        )
    if backend.name == "vllm":
        from cpis.inference.vllm_backend import VllmBackend

        return VllmBackend(
            model=config.model,
            expected_version=backend.version,
            download_dir=str(storage.model_cache / "huggingface" / "hub"),
            engine_seed=config.inference.engine_seed,
            use_flashinfer_sampler=bool(backend.use_flashinfer_sampler),
            tokenizer_mode=backend.tokenizer_mode,
            config_format=backend.config_format,
            load_format=backend.load_format,
            worker_multiproc_method=backend.worker_multiproc_method,
        )
    raise ValueError(f"unsupported inference backend: {backend.name}")


def _stage_fields(
    config: ExperimentConfig, condition: InferenceCondition, stage: Stage
) -> tuple[SamplingSpec, str, str]:
    if stage == "answer":
        return (
            condition.answer,
            config.confidence.answer_parser_version,
            config.confidence.answer_response_format,
        )
    return (
        condition.confidence,
        config.confidence.confidence_parser_version,
        config.confidence.confidence_response_format,
    )


def _validate_resumed_raw(
    path: Path,
    config: ExperimentConfig,
    pending: PendingGeneration,
) -> GenerationRecord:
    record = GenerationRecord.model_validate(read_json(path))
    sampling, parser_version, response_format = _stage_fields(
        config, pending.observation.condition, pending.stage
    )
    expected = (
        record.record_id == pending.record_id
        and record.observation_id == pending.observation.observation_id
        and record.generation_stage == pending.stage
        and record.parent_answer_record_id == pending.parent_answer_record_id
        and record.run_id == config.run_id
        and record.experiment_id == config.experiment_id
        and record.dataset_id == config.dataset.dataset_id
        and record.dataset_revision == config.dataset.revision
        and record.dataset_item_id == pending.observation.item.item_id
        and record.dataset_partition == config.dataset.partition
        and record.prompt_text == pending.prompt.text
        and record.prompt_sha256 == pending.prompt.sha256
        and record.confidence_protocol_id == config.confidence.protocol_id
        and record.prompt_template_version == config.confidence.prompt_template_version
        and record.parser_version == parser_version
        and record.response_format == response_format
        and record.condition == pending.observation.condition
        and record.sampling == sampling
        and record.seed == pending.observation.seed
        and record.sampling_seed
        == derive_sampling_seed(
            pending.observation.seed,
            pending.observation.item.item_id,
            pending.stage,
        )
        and record.model == config.model
        and record.inference_backend == config.inference.backend.name
        and record.inference_backend_version == config.inference.backend.version
        and record.engine_seed == config.inference.engine_seed
    )
    if not expected:
        raise RuntimeError(f"existing raw record does not match its run identity: {path}")
    return record


def _generate_pending(
    pending: list[PendingGeneration],
    config: ExperimentConfig,
    backend: InferenceBackend,
) -> dict[str, GenerationRecord]:
    created: dict[str, GenerationRecord] = {}
    for batch in _chunks(pending, config.execution.batch_size):
        requests = [
            InferenceRequest(
                request_id=entry.record_id,
                stage=entry.stage,
                prompt=entry.prompt.text,
                seed=derive_sampling_seed(
                    entry.observation.seed,
                    entry.observation.item.item_id,
                    entry.stage,
                ),
                condition=entry.observation.condition,
            )
            for entry in batch
        ]
        responses = backend.generate(requests)
        by_id = {response.request_id: response for response in responses}
        if len(by_id) != len(batch) or set(by_id) != {
            entry.record_id for entry in batch
        }:
            raise RuntimeError("inference backend violated request/response identity")
        for entry in batch:
            response = by_id[entry.record_id]
            sampling, parser_version, response_format = _stage_fields(
                config, entry.observation.condition, entry.stage
            )
            record = GenerationRecord(
                record_id=entry.record_id,
                observation_id=entry.observation.observation_id,
                generation_stage=entry.stage,
                parent_answer_record_id=entry.parent_answer_record_id,
                run_id=config.run_id,
                experiment_id=config.experiment_id,
                created_at_utc=utc_now(),
                dataset_id=config.dataset.dataset_id,
                dataset_revision=config.dataset.revision,
                dataset_item_id=entry.observation.item.item_id,
                dataset_partition=config.dataset.partition,
                prompt_text=entry.prompt.text,
                prompt_sha256=entry.prompt.sha256,
                confidence_protocol_id=config.confidence.protocol_id,
                prompt_template_version=config.confidence.prompt_template_version,
                parser_version=parser_version,
                response_format=response_format,
                model=config.model,
                inference_backend=config.inference.backend.name,
                inference_backend_version=backend.version,
                engine_seed=config.inference.engine_seed,
                condition=entry.observation.condition,
                sampling=sampling,
                seed=entry.observation.seed,
                sampling_seed=derive_sampling_seed(
                    entry.observation.seed,
                    entry.observation.item.item_id,
                    entry.stage,
                ),
                raw_text=response.text,
                finish_reason=response.finish_reason,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            )
            write_immutable_json(entry.path, record.model_dump(mode="json"))
            created[entry.observation.observation_id] = record
    return created


def _load_or_generate_stage(
    pending: list[PendingGeneration],
    config: ExperimentConfig,
    storage: StorageLayout,
    backend_holder: list[InferenceBackend],
) -> tuple[dict[str, GenerationRecord], int, int]:
    records: dict[str, GenerationRecord] = {}
    missing: list[PendingGeneration] = []
    for entry in pending:
        if entry.path.exists():
            records[entry.observation.observation_id] = _validate_resumed_raw(
                entry.path, config, entry
            )
        else:
            missing.append(entry)
    if missing:
        if not backend_holder:
            backend_holder.append(make_backend(config, storage))
        records.update(
            _generate_pending(missing, config, backend_holder[0])
        )
    return records, len(missing), len(pending) - len(missing)


def run_experiment(
    config_path: Path, repository_root: Path, replica_index: int = 0
) -> RunResult:
    config_path = config_path.resolve()
    repository_root = repository_root.resolve()
    config = load_config(config_path)
    if not 0 <= replica_index < config.execution.replicas:
        raise ValueError(
            f"replica index must be in [0, {config.execution.replicas - 1}]"
        )
    storage = StorageLayout.from_spec(config.storage, repository_root)
    storage.configure_process_caches()
    run = storage.run(config.run_id)
    ensure_manifest(config, config_path, repository_root, run, replica_index)

    gate = require_phase_gate(config.dataset.partition, repository_root)
    authorize_phase_config(gate, config_path, repository_root)
    record_dataset_access(
        run,
        run_id=config.run_id,
        experiment_id=config.experiment_id,
        config_sha256=config.config_sha256,
        dataset_id=config.dataset.dataset_id,
        dataset_revision=config.dataset.revision,
        partition=config.dataset.partition,
        replica_index=replica_index,
        gate=gate,
    )

    all_items = load_dataset(config.dataset, config_path.parent, storage.dataset_cache)
    partition = partition_items(all_items, config.dataset)
    if not partition:
        raise ValueError(
            f"dataset partition {config.dataset.partition!r} contains no items"
        )
    items = [
        item
        for item in partition
        if assign_replica(item.item_id, config.execution.replicas) == replica_index
    ]
    observations: list[Observation] = []
    for item in items:
        for condition in config.inference.conditions:
            for seed in config.inference.seeds:
                observation_id = stable_observation_id(
                    config.run_id,
                    item.item_id,
                    condition.condition_id,
                    seed,
                    config.model.model_mode,
                )
                observations.append(
                    Observation(
                        item=item,
                        condition=condition,
                        seed=seed,
                        observation_id=observation_id,
                        answer_record_id=stable_generation_id(observation_id, "answer"),
                        confidence_record_id=stable_generation_id(
                            observation_id, "confidence"
                        ),
                    )
                )

    answer_pending = [
        PendingGeneration(
            observation=observation,
            stage="answer",
            record_id=observation.answer_record_id,
            parent_answer_record_id=None,
            prompt=build_answer_prompt(observation.item, config.confidence),
            path=run.raw_answer / f"{observation.answer_record_id}.json",
        )
        for observation in observations
    ]
    backend_holder: list[InferenceBackend] = []
    answer_records, answer_created, answer_reused = _load_or_generate_stage(
        answer_pending, config, storage, backend_holder
    )
    answer_outcomes = {
        observation_id: parse_answer(
            record.raw_text, config.confidence.answer_parser_version
        )
        for observation_id, record in answer_records.items()
    }

    confidence_pending: list[PendingGeneration] = []
    for observation in observations:
        answer_record = answer_records[observation.observation_id]
        answer_outcome = answer_outcomes[observation.observation_id]
        proposed_answer = (
            answer_outcome.answer
            if answer_outcome.status == "valid" and answer_outcome.answer is not None
            else answer_record.raw_text
        )
        confidence_pending.append(
            PendingGeneration(
                observation=observation,
                stage="confidence",
                record_id=observation.confidence_record_id,
                parent_answer_record_id=observation.answer_record_id,
                prompt=build_confidence_prompt(
                    observation.item, proposed_answer, config.confidence
                ),
                path=run.raw_confidence / f"{observation.confidence_record_id}.json",
            )
        )
    confidence_records, confidence_created, confidence_reused = (
        _load_or_generate_stage(
            confidence_pending, config, storage, backend_holder
        )
    )

    parsed_created = parsed_reused = 0
    invalid_answer_outputs = invalid_confidence_outputs = 0
    for observation in observations:
        answer_raw = answer_records[observation.observation_id]
        confidence_raw = confidence_records[observation.observation_id]
        answer_outcome = answer_outcomes[observation.observation_id]
        confidence_outcome = parse_confidence(
            confidence_raw.raw_text, config.confidence.confidence_parser_version
        )
        invalid_answer_outputs += answer_outcome.status == "invalid"
        invalid_confidence_outputs += confidence_outcome.status == "invalid"
        parsed_path = run.parsed / f"{observation.observation_id}.json"
        if parsed_path.exists():
            parsed = ParsedRecord.model_validate(read_json(parsed_path))
            if (
                parsed.record_id != observation.observation_id
                or parsed.answer_raw_record_sha256 != answer_raw.sha256
                or parsed.confidence_raw_record_sha256 != confidence_raw.sha256
            ):
                raise RuntimeError(
                    f"parsed record does not match immutable raw records: {parsed_path}"
                )
            parsed_reused += 1
            continue
        correct = score_answer(
            answer_outcome.answer,
            observation.item.answer,
            config.dataset.scorer,
        )
        parsed = ParsedRecord(
            record_id=observation.observation_id,
            answer_raw_record_id=answer_raw.record_id,
            answer_raw_record_sha256=answer_raw.sha256,
            confidence_raw_record_id=confidence_raw.record_id,
            confidence_raw_record_sha256=confidence_raw.sha256,
            run_id=config.run_id,
            experiment_id=config.experiment_id,
            dataset_id=config.dataset.dataset_id,
            dataset_revision=config.dataset.revision,
            dataset_item_id=observation.item.item_id,
            dataset_partition=config.dataset.partition,
            condition_id=observation.condition.condition_id,
            seed=observation.seed,
            answer_parser_version=config.confidence.answer_parser_version,
            confidence_parser_version=config.confidence.confidence_parser_version,
            scorer=config.dataset.scorer,
            answer_parse_status=answer_outcome.status,
            confidence_parse_status=confidence_outcome.status,
            parse_status=(
                "valid"
                if answer_outcome.status == confidence_outcome.status == "valid"
                else "invalid"
            ),
            parsed_answer=answer_outcome.answer,
            confidence=confidence_outcome.confidence,
            scoring_status=(
                "scored" if correct is not None else "pending_official_evaluator"
            ),
            correct=correct,
            answer_parse_error=answer_outcome.error,
            confidence_parse_error=confidence_outcome.error,
        )
        write_immutable_json(parsed_path, parsed.model_dump(mode="json"))
        parsed_created += 1

    return RunResult(
        run_id=config.run_id,
        run_directory=run.base,
        replica_index=replica_index,
        replica_count=config.execution.replicas,
        selected_items=len(items),
        expected_observations=len(observations),
        expected_generations=2 * len(observations),
        raw_created=answer_created + confidence_created,
        raw_reused=answer_reused + confidence_reused,
        parsed_created=parsed_created,
        parsed_reused=parsed_reused,
        invalid_answer_outputs=invalid_answer_outputs,
        invalid_confidence_outputs=invalid_confidence_outputs,
    )
