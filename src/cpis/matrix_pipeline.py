"""Development-only answer/readout matrix with immutable, resumable stages."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from cpis.config import InferenceCondition, ModelSpec, SamplingSpec
from cpis.datasets import DatasetItem, assign_replica, load_dataset, partition_items
from cpis.inference.base import InferenceBackend, InferenceRequest
from cpis.integration import (
    DatasetIntegrationRecord,
    ModelIntegrationRecord,
    validate_integration_reference,
)
from cpis.matrix_config import (
    readout_ids_for_answer,
    AnswerCondition,
    ConfidenceReadout,
    MatrixExperimentConfig,
    load_matrix_config,
)
from cpis.matrix_records import MatrixParsedRecord, MatrixRole
from cpis.parsing import parse_answer, parse_confidence, score_answer
from cpis.phase_gate import (
    authorize_phase_config,
    record_dataset_access,
    require_phase_gate,
)
from cpis.policy import (
    validate_certification_decision_reference,
    validate_policy_reference,
)
from cpis.pipeline import ensure_manifest, make_backend
from cpis.prompts import Prompt, build_answer_prompt, build_confidence_prompt
from cpis.records import GenerationRecord, derive_sampling_seed, stable_generation_id, utc_now
from cpis.storage import RunLayout, StorageLayout, read_json, write_immutable_json

T = TypeVar("T")


@dataclass(frozen=True)
class MatrixRunResult:
    run_id: str
    run_directory: Path
    replica_index: int
    replica_count: int
    selected_items: int
    expected_answer_observations: int
    expected_confidence_cells: int
    expected_generations: int
    raw_created: int
    raw_reused: int
    parsed_created: int
    parsed_reused: int
    invalid_answer_outputs: int
    invalid_confidence_outputs: int


@dataclass(frozen=True)
class AnswerObservation:
    item: DatasetItem
    condition: AnswerCondition
    coupled_readout: ConfidenceReadout
    seed: int
    observation_id: str
    record_id: str
    inference_condition: InferenceCondition


@dataclass(frozen=True)
class ConfidenceCell:
    answer: AnswerObservation
    readout: ConfidenceReadout
    roles: tuple[MatrixRole, ...]
    observation_id: str
    record_id: str
    inference_condition: InferenceCondition


def _chunks(values: list[T], size: int) -> list[list[T]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def _stable_id(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode("utf-8")).hexdigest()


def _answer_observation_id(
    config: MatrixExperimentConfig,
    item_id: str,
    condition_id: str,
    seed: int,
) -> str:
    return _stable_id(
        "cpis-matrix-answer-v1",
        config.run_id,
        item_id,
        condition_id,
        str(seed),
        config.model.model_mode,
    )


def _confidence_observation_id(answer_observation_id: str, readout_id: str) -> str:
    return _stable_id("cpis-matrix-confidence-v1", answer_observation_id, readout_id)


def _condition_map(config: MatrixExperimentConfig) -> dict[str, AnswerCondition]:
    return {condition.condition_id: condition for condition in config.inference.answer_conditions}


def _readout_map(config: MatrixExperimentConfig) -> dict[str, ConfidenceReadout]:
    return {readout.readout_id: readout for readout in config.inference.confidence_readouts}


def _effective_readout_model(
    config: MatrixExperimentConfig, readout: ConfidenceReadout
) -> ModelSpec:
    if readout.chat_template_kwargs is None:
        return config.model
    return config.model.model_copy(
        update={
            "chat_template_kwargs": readout.chat_template_kwargs,
            "model_mode": readout.model_mode,
        }
    )


def _cell_roles(
    config: MatrixExperimentConfig, answer_id: str, readout_id: str
) -> tuple[MatrixRole, ...]:
    design = config.inference.design
    roles: list[MatrixRole] = []
    if design.coupled_readout_by_answer[answer_id] == readout_id:
        roles.append("coupled")
    if design.standardized_readout_id == readout_id:
        roles.append("standardized")
    if design.reference_confidence_readout_id == readout_id:
        roles.append("decomposition_reference_readout")
    if (
        answer_id == design.reference_answer_condition_id
        and readout_id in design.report_only_readout_ids
    ):
        roles.append("report_only")
    if not roles:
        raise AssertionError("matrix cell has no scientific design role")
    return tuple(roles)


def _build_observations(
    config: MatrixExperimentConfig, items: list[DatasetItem]
) -> tuple[list[AnswerObservation], list[ConfidenceCell]]:
    readouts = _readout_map(config)
    design = config.inference.design
    answers: list[AnswerObservation] = []
    for item in items:
        for condition in config.inference.answer_conditions:
            coupled = readouts[design.coupled_readout_by_answer[condition.condition_id]]
            inference_condition = InferenceCondition(
                condition_id=condition.condition_id,
                answer=condition.sampling,
                confidence=coupled.sampling,
            )
            for seed in config.inference.seeds:
                observation_id = _answer_observation_id(
                    config, item.item_id, condition.condition_id, seed
                )
                answers.append(
                    AnswerObservation(
                        item=item,
                        condition=condition,
                        coupled_readout=coupled,
                        seed=seed,
                        observation_id=observation_id,
                        record_id=stable_generation_id(observation_id, "answer"),
                        inference_condition=inference_condition,
                    )
                )
    cells: list[ConfidenceCell] = []
    for answer in answers:
        readout_ids = readout_ids_for_answer(design, answer.condition.condition_id)
        for readout_id in sorted(readout_ids):
            readout = readouts[readout_id]
            observation_id = _confidence_observation_id(
                answer.observation_id, readout_id
            )
            cells.append(
                ConfidenceCell(
                    answer=answer,
                    readout=readout,
                    roles=_cell_roles(
                        config, answer.condition.condition_id, readout_id
                    ),
                    observation_id=observation_id,
                    record_id=stable_generation_id(observation_id, "confidence"),
                    inference_condition=InferenceCondition(
                        condition_id=(
                            f"a-{answer.condition.condition_id}--r-{readout_id}"
                        ),
                        answer=answer.condition.sampling,
                        confidence=readout.sampling,
                    ),
                )
            )
    return answers, cells


def _raw_matches(
    record: GenerationRecord,
    *,
    config: MatrixExperimentConfig,
    item: DatasetItem,
    observation_id: str,
    record_id: str,
    stage: str,
    parent_answer_record_id: str | None,
    prompt: Prompt,
    condition: InferenceCondition,
    sampling: SamplingSpec,
    model: ModelSpec,
    seed: int,
    sampling_seed: int,
) -> bool:
    parser_version = (
        config.confidence.answer_parser_version
        if stage == "answer"
        else config.confidence.confidence_parser_version
    )
    response_format = (
        config.confidence.answer_response_format
        if stage == "answer"
        else config.confidence.confidence_response_format
    )
    return (
        record.record_id == record_id
        and record.observation_id == observation_id
        and record.generation_stage == stage
        and record.parent_answer_record_id == parent_answer_record_id
        and record.run_id == config.run_id
        and record.experiment_id == config.experiment_id
        and record.dataset_id == config.dataset.dataset_id
        and record.dataset_revision == config.dataset.revision
        and record.dataset_item_id == item.item_id
        and record.dataset_partition == config.dataset.partition
        and record.prompt_text == prompt.text
        and record.prompt_sha256 == prompt.sha256
        and record.confidence_protocol_id == config.confidence.protocol_id
        and record.prompt_template_version == config.confidence.prompt_template_version
        and record.parser_version == parser_version
        and record.response_format == response_format
        and record.model == model
        and record.inference_backend == config.inference.backend.name
        and record.inference_backend_version == config.inference.backend.version
        and record.engine_seed == config.inference.engine_seed
        and record.condition == condition
        and record.sampling == sampling
        and record.seed == seed
        and record.sampling_seed == sampling_seed
    )


def _new_raw_record(
    *,
    config: MatrixExperimentConfig,
    backend: InferenceBackend,
    item: DatasetItem,
    observation_id: str,
    record_id: str,
    stage: str,
    parent_answer_record_id: str | None,
    prompt: Prompt,
    condition: InferenceCondition,
    sampling: SamplingSpec,
    model: ModelSpec,
    seed: int,
    sampling_seed: int,
    response,
) -> GenerationRecord:
    return GenerationRecord(
        record_id=record_id,
        observation_id=observation_id,
        generation_stage=stage,
        parent_answer_record_id=parent_answer_record_id,
        run_id=config.run_id,
        experiment_id=config.experiment_id,
        created_at_utc=utc_now(),
        dataset_id=config.dataset.dataset_id,
        dataset_revision=config.dataset.revision,
        dataset_item_id=item.item_id,
        dataset_partition=config.dataset.partition,
        prompt_text=prompt.text,
        prompt_sha256=prompt.sha256,
        confidence_protocol_id=config.confidence.protocol_id,
        prompt_template_version=config.confidence.prompt_template_version,
        parser_version=(
            config.confidence.answer_parser_version
            if stage == "answer"
            else config.confidence.confidence_parser_version
        ),
        response_format=(
            config.confidence.answer_response_format
            if stage == "answer"
            else config.confidence.confidence_response_format
        ),
        model=model,
        inference_backend=config.inference.backend.name,
        inference_backend_version=backend.version,
        engine_seed=config.inference.engine_seed,
        condition=condition,
        sampling=sampling,
        seed=seed,
        sampling_seed=sampling_seed,
        raw_text=response.text,
        finish_reason=response.finish_reason,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
    )


def validate_matrix_integrations(
    config: MatrixExperimentConfig, repository_root: Path
) -> None:
    if config.study == "matrix_smoke":
        return
    assert config.model_integration is not None
    assert config.dataset_integration is not None
    model_record = validate_integration_reference(
        config.model_integration, repository_root, "model_integration"
    )
    dataset_record = validate_integration_reference(
        config.dataset_integration, repository_root, "dataset_integration"
    )
    if config.frozen_policy is not None:
        policy = validate_policy_reference(config.frozen_policy, repository_root)
        if (
            policy.model_id != config.model.model_id
            or policy.model_revision != config.model.revision
            or policy.model_mode != config.model.model_mode
            or policy.dataset_id != config.dataset.dataset_id
            or policy.dataset_revision != config.dataset.revision
            or policy.confidence_protocol_id != config.confidence.protocol_id
            or policy.prompt_template_version != config.confidence.prompt_template_version
            or policy.reference_answer_condition_id
            != config.inference.design.reference_answer_condition_id
            or policy.reference_confidence_readout_id
            != config.inference.design.reference_confidence_readout_id
        ):
            raise RuntimeError("frozen policy does not match the confirmatory configuration")
        if config.certification_decision is not None:
            decision = validate_certification_decision_reference(
                config.certification_decision, repository_root
            )
            if (
                decision.frozen_policy_id != policy.policy_id
                or decision.frozen_policy_sha256 != policy.canonical_sha256
                or decision.analysis_plan_sha256 != policy.analysis_plan_sha256
            ):
                raise RuntimeError("certification decision does not resolve the frozen policy")
    assert isinstance(model_record, ModelIntegrationRecord)
    assert isinstance(dataset_record, DatasetIntegrationRecord)
    if (
        model_record.model_id != config.model.model_id
        or model_record.revision != config.model.revision
        or model_record.tokenizer_revision != config.model.tokenizer_revision
        or model_record.parameter_count_exact
        != round(config.model.parameter_count_billion * 1_000_000_000)
        or model_record.validation_status != "development_validated"
    ):
        raise RuntimeError("model integration record does not match the matrix")
    reference_id = config.inference.design.reference_answer_condition_id
    reference = next(
        condition
        for condition in config.inference.answer_conditions
        if condition.condition_id == reference_id
    )
    configured_sampling = reference.sampling.model_dump(mode="python")
    mismatches = {
        key: {"documented": value, "configured": configured_sampling.get(key)}
        for key, value in model_record.native_reference_sampling.items()
        if configured_sampling.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "matrix reference sampling differs from the model-native integration "
            f"record: {mismatches}"
        )
    partitions = {part.partition: part for part in dataset_record.prepared_partitions}
    declared = partitions[config.dataset.partition]
    if (
        dataset_record.dataset_id != config.dataset.dataset_id
        or dataset_record.revision != config.dataset.revision
        or declared.sha256 != config.dataset.content_sha256
        or declared.relative_cache_path != config.dataset.source_path
    ):
        raise RuntimeError("dataset integration record does not match the matrix")


def run_matrix_experiment(
    config_path: Path,
    repository_root: Path,
    replica_index: int = 0,
    *,
    generate_only: bool = False,
    answers_only: bool = False,
) -> MatrixRunResult:
    config_path = config_path.resolve()
    repository_root = repository_root.resolve()
    config = load_matrix_config(config_path)
    if not 0 <= replica_index < config.execution.replicas:
        raise ValueError(
            f"replica index must be in [0, {config.execution.replicas - 1}]"
        )
    validate_matrix_integrations(config, repository_root)
    storage = StorageLayout.from_spec(config.storage, repository_root)
    storage.configure_process_caches()
    run = storage.run(config.run_id)
    ensure_manifest(config, config_path, repository_root, run, replica_index)  # type: ignore[arg-type]
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
    if config.study != "matrix_smoke" and {
        item.partition for item in all_items
    } != {config.dataset.partition}:
        raise RuntimeError("development matrix requires a sealed single-partition file")
    partition = partition_items(all_items, config.dataset)
    items = [
        item
        for item in partition
        if assign_replica(item.item_id, config.execution.replicas) == replica_index
    ]
    if not items:
        raise ValueError("selected matrix partition contains no items")
    answers, cells = _build_observations(config, items)
    backend_holder: list[InferenceBackend] = []

    answer_records: dict[str, GenerationRecord] = {}
    missing_answers: list[tuple[AnswerObservation, Prompt, Path]] = []
    for answer in answers:
        prompt = build_answer_prompt(answer.item, config.confidence)
        path = run.raw_answer / f"{answer.record_id}.json"
        sampling_seed = derive_sampling_seed(answer.seed, answer.item.item_id, "answer")
        if path.exists():
            record = GenerationRecord.model_validate(read_json(path))
            if not _raw_matches(
                record,
                config=config,
                item=answer.item,
                observation_id=answer.observation_id,
                record_id=answer.record_id,
                stage="answer",
                parent_answer_record_id=None,
                prompt=prompt,
                condition=answer.inference_condition,
                sampling=answer.condition.sampling,
                model=config.model,
                seed=answer.seed,
                sampling_seed=sampling_seed,
            ):
                raise RuntimeError(f"resumed answer record mismatch: {path}")
            answer_records[answer.observation_id] = record
        else:
            missing_answers.append((answer, prompt, path))
    if missing_answers:
        backend_holder.append(make_backend(config, storage))  # type: ignore[arg-type]
        for batch in _chunks(missing_answers, config.execution.batch_size):
            requests = [
                InferenceRequest(
                    request_id=answer.record_id,
                    stage="answer",
                    prompt=prompt.text,
                    seed=derive_sampling_seed(answer.seed, answer.item.item_id, "answer"),
                    condition=answer.inference_condition,
                    chat_template_kwargs=None,
                )
                for answer, prompt, _ in batch
            ]
            by_id = {response.request_id: response for response in backend_holder[0].generate(requests)}
            if set(by_id) != {answer.record_id for answer, _, _ in batch}:
                raise RuntimeError("backend violated matrix answer request identity")
            for answer, prompt, path in batch:
                record = _new_raw_record(
                    config=config,
                    backend=backend_holder[0],
                    item=answer.item,
                    observation_id=answer.observation_id,
                    record_id=answer.record_id,
                    stage="answer",
                    parent_answer_record_id=None,
                    prompt=prompt,
                    condition=answer.inference_condition,
                    sampling=answer.condition.sampling,
                    model=config.model,
                    seed=answer.seed,
                    sampling_seed=derive_sampling_seed(
                        answer.seed, answer.item.item_id, "answer"
                    ),
                    response=by_id[answer.record_id],
                )
                write_immutable_json(path, record.model_dump(mode="json"))
                answer_records[answer.observation_id] = record

    answer_outcomes = {
        observation_id: parse_answer(
            record.raw_text, config.confidence.answer_parser_version
        )
        for observation_id, record in answer_records.items()
    }
    if answers_only:
        # A confidence readout depends only on the item, the produced answer and
        # the readout protocol, so it can be generated later from these immutable
        # answers under a revised protocol. Stopping here keeps answer generation
        # productive while a confidence budget is being re-decided, and creates no
        # partial cell that a later full run would have to reconcile.
        return MatrixRunResult(
            run_id=config.run_id,
            run_directory=run.base,
            replica_index=replica_index,
            replica_count=config.execution.replicas,
            selected_items=len(items),
            expected_answer_observations=len(answers),
            expected_confidence_cells=len(cells),
            expected_generations=len(answers) + len(cells),
            raw_created=len(missing_answers),
            raw_reused=len(answers) - len(missing_answers),
            parsed_created=0,
            parsed_reused=0,
            invalid_answer_outputs=sum(
                outcome.status == "invalid" for outcome in answer_outcomes.values()
            ),
            invalid_confidence_outputs=0,
        )
    confidence_records: dict[str, GenerationRecord] = {}
    missing_cells: list[tuple[ConfidenceCell, Prompt, Path]] = []
    for cell in cells:
        answer_raw = answer_records[cell.answer.observation_id]
        answer_outcome = answer_outcomes[cell.answer.observation_id]
        proposed = (
            answer_outcome.answer
            if answer_outcome.status == "valid" and answer_outcome.answer is not None
            else answer_raw.raw_text
        )
        prompt = build_confidence_prompt(cell.answer.item, proposed, config.confidence)
        path = run.raw_confidence / f"{cell.record_id}.json"
        sampling_seed = derive_sampling_seed(
            cell.answer.seed, cell.answer.item.item_id, "confidence"
        )
        if path.exists():
            record = GenerationRecord.model_validate(read_json(path))
            if not _raw_matches(
                record,
                config=config,
                item=cell.answer.item,
                observation_id=cell.observation_id,
                record_id=cell.record_id,
                stage="confidence",
                parent_answer_record_id=cell.answer.record_id,
                prompt=prompt,
                condition=cell.inference_condition,
                sampling=cell.readout.sampling,
                model=_effective_readout_model(config, cell.readout),
                seed=cell.answer.seed,
                sampling_seed=sampling_seed,
            ):
                raise RuntimeError(f"resumed confidence record mismatch: {path}")
            confidence_records[cell.observation_id] = record
        else:
            missing_cells.append((cell, prompt, path))
    if missing_cells:
        if not backend_holder:
            backend_holder.append(make_backend(config, storage))  # type: ignore[arg-type]
        for batch in _chunks(missing_cells, config.execution.batch_size):
            requests = [
                InferenceRequest(
                    request_id=cell.record_id,
                    stage="confidence",
                    prompt=prompt.text,
                    seed=derive_sampling_seed(
                        cell.answer.seed, cell.answer.item.item_id, "confidence"
                    ),
                    condition=cell.inference_condition,
                    chat_template_kwargs=cell.readout.chat_template_kwargs,
                )
                for cell, prompt, _ in batch
            ]
            by_id = {response.request_id: response for response in backend_holder[0].generate(requests)}
            if set(by_id) != {cell.record_id for cell, _, _ in batch}:
                raise RuntimeError("backend violated matrix confidence request identity")
            for cell, prompt, path in batch:
                record = _new_raw_record(
                    config=config,
                    backend=backend_holder[0],
                    item=cell.answer.item,
                    observation_id=cell.observation_id,
                    record_id=cell.record_id,
                    stage="confidence",
                    parent_answer_record_id=cell.answer.record_id,
                    prompt=prompt,
                    condition=cell.inference_condition,
                    sampling=cell.readout.sampling,
                    model=_effective_readout_model(config, cell.readout),
                    seed=cell.answer.seed,
                    sampling_seed=derive_sampling_seed(
                        cell.answer.seed, cell.answer.item.item_id, "confidence"
                    ),
                    response=by_id[cell.record_id],
                )
                write_immutable_json(path, record.model_dump(mode="json"))
                confidence_records[cell.observation_id] = record

    if generate_only:
        return MatrixRunResult(
            run_id=config.run_id,
            run_directory=run.base,
            replica_index=replica_index,
            replica_count=config.execution.replicas,
            selected_items=len(items),
            expected_answer_observations=len(answers),
            expected_confidence_cells=len(cells),
            expected_generations=len(answers) + len(cells),
            raw_created=len(missing_answers) + len(missing_cells),
            raw_reused=(
                len(answers)
                + len(cells)
                - len(missing_answers)
                - len(missing_cells)
            ),
            parsed_created=0,
            parsed_reused=0,
            invalid_answer_outputs=sum(
                outcome.status == "invalid" for outcome in answer_outcomes.values()
            ),
            invalid_confidence_outputs=sum(
                parse_confidence(
                    record.raw_text,
                    config.confidence.confidence_parser_version,
                ).status
                == "invalid"
                for record in confidence_records.values()
            ),
        )

    parsed_created = parsed_reused = 0
    invalid_confidence = 0
    for cell in cells:
        answer_raw = answer_records[cell.answer.observation_id]
        confidence_raw = confidence_records[cell.observation_id]
        answer_outcome = answer_outcomes[cell.answer.observation_id]
        confidence_outcome = parse_confidence(
            confidence_raw.raw_text, config.confidence.confidence_parser_version
        )
        invalid_confidence += confidence_outcome.status == "invalid"
        path = run.parsed / f"{cell.observation_id}.json"
        if path.exists():
            record = MatrixParsedRecord.model_validate(read_json(path))
            if (
                record.record_id != cell.observation_id
                or record.answer_raw_record_sha256 != answer_raw.sha256
                or record.confidence_raw_record_sha256 != confidence_raw.sha256
                or record.design_roles != cell.roles
            ):
                raise RuntimeError(f"resumed matrix parsed record mismatch: {path}")
            parsed_reused += 1
            continue
        correct = score_answer(
            answer_outcome.answer, cell.answer.item.answer, config.dataset.scorer
        )
        record = MatrixParsedRecord(
            record_id=cell.observation_id,
            answer_observation_id=cell.answer.observation_id,
            answer_raw_record_id=answer_raw.record_id,
            answer_raw_record_sha256=answer_raw.sha256,
            confidence_raw_record_id=confidence_raw.record_id,
            confidence_raw_record_sha256=confidence_raw.sha256,
            run_id=config.run_id,
            experiment_id=config.experiment_id,
            dataset_id=config.dataset.dataset_id,
            dataset_revision=config.dataset.revision,
            dataset_item_id=cell.answer.item.item_id,
            dataset_partition=config.dataset.partition,
            answer_condition_id=cell.answer.condition.condition_id,
            confidence_readout_id=cell.readout.readout_id,
            design_roles=cell.roles,
            seed=cell.answer.seed,
            answer_sampling_seed=derive_sampling_seed(
                cell.answer.seed, cell.answer.item.item_id, "answer"
            ),
            confidence_sampling_seed=derive_sampling_seed(
                cell.answer.seed, cell.answer.item.item_id, "confidence"
            ),
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
        write_immutable_json(path, record.model_dump(mode="json"))
        parsed_created += 1

    return MatrixRunResult(
        run_id=config.run_id,
        run_directory=run.base,
        replica_index=replica_index,
        replica_count=config.execution.replicas,
        selected_items=len(items),
        expected_answer_observations=len(answers),
        expected_confidence_cells=len(cells),
        expected_generations=len(answers) + len(cells),
        raw_created=len(missing_answers) + len(missing_cells),
        raw_reused=len(answers) + len(cells) - len(missing_answers) - len(missing_cells),
        parsed_created=parsed_created,
        parsed_reused=parsed_reused,
        invalid_answer_outputs=sum(
            outcome.status == "invalid" for outcome in answer_outcomes.values()
        ),
        invalid_confidence_outputs=invalid_confidence,
    )
