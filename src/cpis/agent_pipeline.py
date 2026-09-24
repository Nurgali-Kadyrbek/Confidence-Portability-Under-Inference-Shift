"""Immutable, resumable BFCL agent generation and scoring pipeline."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cpis.agent.bfcl import (
    ADDITIONAL_FUNCTION_MESSAGE,
    AgentResponse,
    EpisodeResult,
    PreparedCase,
    function_docs_bundle_sha256,
    prepare_case,
    run_episode,
    score_episode,
)
from cpis.agent_config import AgentExperimentConfig, load_agent_config
from cpis.agent_records import AgentEpisodeRecord, AgentParsedRecord
from cpis.config import InferenceCondition
from cpis.datasets import DatasetItem, assign_replica, load_dataset, partition_items
from cpis.inference.base import ChatInferenceRequest, InferenceRequest
from cpis.integration import (
    DatasetIntegrationRecord,
    ModelIntegrationRecord,
    validate_integration_reference,
)
from cpis.matrix_config import AnswerCondition, ConfidenceReadout
from cpis.parsing import parse_confidence
from cpis.phase_gate import (
    authorize_phase_config,
    record_dataset_access,
    require_phase_gate,
)
from cpis.pipeline import ensure_manifest, make_backend
from cpis.policy import (
    validate_certification_decision_reference,
    validate_policy_reference,
)
from cpis.prompts import build_confidence_prompt
from cpis.records import GenerationRecord, stable_generation_id, utc_now
from cpis.storage import StorageLayout, read_json, write_immutable_json


@dataclass(frozen=True)
class AgentRunResult:
    run_id: str
    run_directory: Path
    replica_index: int
    replica_count: int
    selected_items: int
    expected_episodes: int
    expected_confidence_cells: int
    raw_created: int
    raw_reused: int
    parsed_created: int
    parsed_reused: int
    unsuccessful_episodes: int
    invalid_tool_outputs: int
    invalid_confidence_outputs: int


@dataclass(frozen=True)
class AgentDerivationResult:
    """Result of scoring/parsing saved BFCL records without model execution."""

    run_id: str
    run_directory: Path
    replica_index: int
    expected_episodes: int
    expected_confidence_cells: int
    parsed_created: int
    parsed_reused: int
    unsuccessful_episodes: int
    invalid_confidence_outputs: int


@dataclass(frozen=True)
class EpisodeObservation:
    item: DatasetItem
    condition: AnswerCondition
    seed: int
    observation_id: str
    record_id: str


@dataclass(frozen=True)
class ConfidenceObservation:
    episode: EpisodeObservation
    readout: ConfidenceReadout
    roles: tuple[str, ...]
    observation_id: str
    record_id: str


def _stable_id(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def derive_agent_step_seed(
    replicate_seed: int, item_id: str, turn_index: int, step_index: int
) -> int:
    """Common-random-number seed shared across agent decoder conditions."""

    identity = (
        f"cpis-agent-answer-seed-v1\0{replicate_seed}\0{item_id}"
        f"\0{turn_index}\0{step_index}"
    )
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def derive_agent_pairing_id(
    model_id: str,
    model_revision: str,
    model_mode: str,
    replicate_seed: int,
    item_id: str,
) -> str:
    """Condition-invariant identity for paired tool-call message IDs."""

    return _stable_id(
        "cpis-agent-tool-call-pair-v1",
        model_id,
        model_revision,
        model_mode,
        str(replicate_seed),
        item_id,
    )


def _episode_result(record: AgentEpisodeRecord) -> EpisodeResult:
    from cpis.agent.bfcl import EpisodeStep
    from cpis.agent.tool_calls import ToolCall

    return EpisodeResult(
        item_id=record.dataset_item_id,
        force_terminated=record.force_terminated,
        invalid_tool_outputs=record.invalid_tool_outputs,
        turns_decoded=record.turns_decoded,
        steps=tuple(
            EpisodeStep(
                turn_index=step.turn_index,
                step_index=step.step_index,
                sampling_seed=step.sampling_seed,
                messages_sha256=step.messages_sha256,
                tools_sha256=step.tools_sha256,
                raw_text=step.raw_text,
                finish_reason=step.finish_reason,
                prompt_tokens=step.prompt_tokens,
                completion_tokens=step.completion_tokens,
                parse_status=step.parse_status,
                parse_error=step.parse_error,
                calls=tuple(ToolCall(**call) for call in step.calls),
                execution_results=step.execution_results,
            )
            for step in record.steps
        ),
        final_messages=record.final_messages,
    )


def _prepared(item: DatasetItem, docs_root: Path) -> PreparedCase:
    source = dict(item.scoring_payload["source_case"])
    source["ground_truth"] = item.scoring_payload["possible_answer"]["ground_truth"]
    return prepare_case(source, docs_root)


def _build_observations(
    config: AgentExperimentConfig, items: list[DatasetItem]
) -> tuple[list[EpisodeObservation], list[ConfidenceObservation]]:
    episodes: list[EpisodeObservation] = []
    for item in items:
        for condition in config.inference.answer_conditions:
            for seed in config.inference.seeds:
                observation_id = _stable_id(
                    "cpis-agent-episode-v1",
                    config.run_id,
                    item.item_id,
                    condition.condition_id,
                    str(seed),
                    config.model.model_mode,
                )
                episodes.append(
                    EpisodeObservation(
                        item=item,
                        condition=condition,
                        seed=seed,
                        observation_id=observation_id,
                        record_id=_stable_id(observation_id, "raw-episode"),
                    )
                )
    readouts = {
        readout.readout_id: readout for readout in config.inference.confidence_readouts
    }
    design = config.inference.design
    cells: list[ConfidenceObservation] = []
    for episode in episodes:
        readout_ids = {
            design.coupled_readout_by_answer[episode.condition.condition_id],
            design.standardized_readout_id,
            design.reference_confidence_readout_id,
        }
        if episode.condition.condition_id == design.reference_answer_condition_id:
            readout_ids.update(design.report_only_readout_ids)
        for readout_id in sorted(readout_ids):
            roles: list[str] = []
            if design.coupled_readout_by_answer[episode.condition.condition_id] == readout_id:
                roles.append("coupled")
            if design.standardized_readout_id == readout_id:
                roles.append("standardized")
            if design.reference_confidence_readout_id == readout_id:
                roles.append("decomposition_reference_readout")
            if (
                episode.condition.condition_id == design.reference_answer_condition_id
                and readout_id in design.report_only_readout_ids
            ):
                roles.append("report_only")
            observation_id = _stable_id(
                "cpis-agent-confidence-v1", episode.observation_id, readout_id
            )
            cells.append(
                ConfidenceObservation(
                    episode=episode,
                    readout=readouts[readout_id],
                    roles=tuple(roles),
                    observation_id=observation_id,
                    record_id=stable_generation_id(observation_id, "confidence"),
                )
            )
    return episodes, cells


def validate_agent_integrations(
    config: AgentExperimentConfig, repository_root: Path, *, preflight: bool
) -> None:
    model_record = validate_integration_reference(
        config.model_integration, repository_root, "model_integration"
    )
    dataset_record = validate_integration_reference(
        config.dataset_integration, repository_root, "dataset_integration"
    )
    assert isinstance(model_record, ModelIntegrationRecord)
    assert isinstance(dataset_record, DatasetIntegrationRecord)
    accepted_status = {"preflight_pending", "development_validated"} if preflight else {
        "development_validated"
    }
    if (
        model_record.model_id != config.model.model_id
        or model_record.revision != config.model.revision
        or model_record.tokenizer_revision != config.model.tokenizer_revision
        or model_record.parameter_count_exact
        != round(config.model.parameter_count_billion * 1_000_000_000)
        or model_record.configured_context_limit != config.model.max_model_len
        or model_record.validation_status not in accepted_status
    ):
        raise RuntimeError("model integration record does not match the agent run")
    reference = next(
        condition
        for condition in config.inference.answer_conditions
        if condition.condition_id
        == config.inference.design.reference_answer_condition_id
    )
    configured = reference.sampling.model_dump(mode="python")
    mismatches = {
        key: {"documented": value, "configured": configured.get(key)}
        for key, value in model_record.native_reference_sampling.items()
        if configured.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"agent reference differs from model-native sampling: {mismatches}")
    partitions = {part.partition: part for part in dataset_record.prepared_partitions}
    declared = partitions[config.dataset.partition]
    if (
        dataset_record.dataset_id != config.dataset.dataset_id
        or dataset_record.revision != config.dataset.revision
        or declared.sha256 != config.dataset.content_sha256
        or declared.relative_cache_path != config.dataset.source_path
    ):
        raise RuntimeError("dataset integration record does not match the agent run")
    if config.frozen_policy is not None:
        policy = validate_policy_reference(config.frozen_policy, repository_root)
        if (
            policy.model_id != config.model.model_id
            or policy.model_revision != config.model.revision
            or policy.model_mode != config.model.model_mode
            or policy.dataset_id != config.dataset.dataset_id
            or policy.dataset_revision != config.dataset.revision
            or policy.confidence_protocol_id != config.confidence.protocol_id
            or policy.prompt_template_version
            != config.confidence.prompt_template_version
            or policy.reference_answer_condition_id
            != config.inference.design.reference_answer_condition_id
            or policy.reference_confidence_readout_id
            != config.inference.design.reference_confidence_readout_id
        ):
            raise RuntimeError("frozen policy does not match the agent configuration")
        if config.certification_decision is not None:
            decision = validate_certification_decision_reference(
                config.certification_decision, repository_root
            )
            if (
                decision.frozen_policy_id != policy.policy_id
                or decision.frozen_policy_sha256 != policy.canonical_sha256
                or decision.analysis_plan_sha256 != policy.analysis_plan_sha256
            ):
                raise RuntimeError("agent certification decision does not match policy")


def _episode_record_matches(
    record: AgentEpisodeRecord,
    config: AgentExperimentConfig,
    observation: EpisodeObservation,
    prepared: PreparedCase,
) -> bool:
    return (
        record.record_id == observation.record_id
        and record.observation_id == observation.observation_id
        and record.run_id == config.run_id
        and record.dataset_item_id == observation.item.item_id
        and record.dataset_group_id == observation.item.group_id
        and record.source_case_sha256 == _canonical_sha256(prepared.source_case)
        and record.function_docs_bundle_sha256
        == config.bfcl.function_docs_bundle_sha256
        and record.answer_condition_id == observation.condition.condition_id
        and record.answer_sampling == observation.condition.sampling
        and record.replicate_seed == observation.seed
        and (
            record.tool_call_pairing_id is None
            or record.tool_call_pairing_id
            == derive_agent_pairing_id(
                config.model.model_id,
                config.model.revision,
                config.model.model_mode,
                observation.seed,
                observation.item.item_id,
            )
        )
        and record.model == config.model
        and record.inference_backend_version == config.inference.backend.version
    )


def run_agent_experiment(
    config_path: Path,
    repository_root: Path,
    replica_index: int = 0,
    *,
    preflight: bool = False,
) -> AgentRunResult:
    config_path = config_path.resolve()
    repository_root = repository_root.resolve()
    config = load_agent_config(config_path)
    if not 0 <= replica_index < config.execution.replicas:
        raise ValueError("agent replica index is outside the configured range")
    validate_agent_integrations(config, repository_root, preflight=preflight)
    if config.bfcl.additional_function_message != ADDITIONAL_FUNCTION_MESSAGE:
        raise RuntimeError("BFCL additional-function prompt differs from pinned official text")
    storage = StorageLayout.from_spec(config.storage, repository_root)
    storage.configure_process_caches()
    evaluator_root = storage.dataset_cache / config.bfcl.evaluator_root_relative_to_dataset_cache
    docs_root = evaluator_root / config.bfcl.function_docs_relative_to_evaluator_root
    actual_docs_sha256 = function_docs_bundle_sha256(docs_root)
    if actual_docs_sha256 != config.bfcl.function_docs_bundle_sha256:
        raise RuntimeError(
            "BFCL function-document bundle differs from the pinned configuration: "
            f"expected {config.bfcl.function_docs_bundle_sha256}, "
            f"got {actual_docs_sha256}"
        )
    if str(evaluator_root) not in sys.path:
        sys.path.insert(0, str(evaluator_root))
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
    if {item.partition for item in all_items} != {config.dataset.partition}:
        raise RuntimeError("agent run requires a sealed single-partition file")
    items = [
        item
        for item in partition_items(all_items, config.dataset)
        if assign_replica(item.item_id, config.execution.replicas) == replica_index
    ]
    if not items:
        raise ValueError("selected agent partition contains no items")
    episodes, cells = _build_observations(config, items)
    backend_holder: list[Any] = []
    episode_records: dict[str, AgentEpisodeRecord] = {}
    raw_created = raw_reused = 0
    for observation in episodes:
        prepared = _prepared(observation.item, docs_root)
        path = run.raw_answer / f"{observation.record_id}.json"
        if path.exists():
            record = AgentEpisodeRecord.model_validate(read_json(path))
            if not _episode_record_matches(record, config, observation, prepared):
                raise RuntimeError(f"resumed agent episode mismatch: {path}")
            episode_records[observation.observation_id] = record
            raw_reused += 1
            continue
        if not backend_holder:
            backend_holder.append(make_backend(config, storage))  # type: ignore[arg-type]
        backend = backend_holder[0]
        if not hasattr(backend, "generate_chat"):
            raise RuntimeError("configured backend lacks native agent chat generation")

        def generate(
            messages: list[dict[str, Any]], tools: list[dict[str, Any]], seed: int
        ) -> AgentResponse:
            request_id = _stable_id(observation.observation_id, str(seed))
            response = backend.generate_chat(
                [
                    ChatInferenceRequest(
                        request_id=request_id,
                        messages=tuple(messages),
                        tools=tuple(tools),
                        seed=seed,
                        sampling=observation.condition.sampling,
                        chat_template_kwargs=config.model.chat_template_kwargs,
                    )
                ]
            )[0]
            return AgentResponse(
                text=response.text,
                finish_reason=response.finish_reason,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            )

        result = run_episode(
            prepared=prepared,
            observation_id=observation.observation_id,
            pairing_id=derive_agent_pairing_id(
                config.model.model_id,
                config.model.revision,
                config.model.model_mode,
                observation.seed,
                observation.item.item_id,
            ),
            parser_family=config.bfcl.parser_family,
            generate=generate,
            sampling_seed=lambda turn, step: derive_agent_step_seed(
                observation.seed, observation.item.item_id, turn, step
            ),
            max_step_limit=config.bfcl.max_step_limit,
        )
        if observation.item.group_id is None:
            raise RuntimeError("BFCL item lacks its base-scenario group")
        record = AgentEpisodeRecord(
            record_id=observation.record_id,
            observation_id=observation.observation_id,
            run_id=config.run_id,
            experiment_id=config.experiment_id,
            created_at_utc=utc_now(),
            dataset_id=config.dataset.dataset_id,
            dataset_revision=config.dataset.revision,
            dataset_item_id=observation.item.item_id,
            dataset_group_id=observation.item.group_id,
            dataset_partition=config.dataset.partition,
            source_case_sha256=_canonical_sha256(prepared.source_case),
            function_docs_bundle_sha256=config.bfcl.function_docs_bundle_sha256,
            confidence_protocol_id=config.confidence.protocol_id,
            prompt_template_version=config.confidence.prompt_template_version,
            bfcl_protocol_id=config.bfcl.protocol_id,
            evaluator_revision=config.bfcl.evaluator_revision,
            parser_family=config.bfcl.parser_family,
            parser_version=config.bfcl.parser_version,
            model=config.model,
            inference_backend=config.inference.backend.name,
            inference_backend_version=config.inference.backend.version,
            engine_seed=config.inference.engine_seed,
            answer_condition_id=observation.condition.condition_id,
            answer_sampling=observation.condition.sampling,
            replicate_seed=observation.seed,
            tool_call_pairing_id=derive_agent_pairing_id(
                config.model.model_id,
                config.model.revision,
                config.model.model_mode,
                observation.seed,
                observation.item.item_id,
            ),
            force_terminated=result.force_terminated,
            invalid_tool_outputs=result.invalid_tool_outputs,
            turns_decoded=result.turns_decoded,
            steps=tuple(asdict(step) for step in result.steps),
            final_messages=result.final_messages,
        )
        write_immutable_json(path, record.model_dump(mode="json"))
        episode_records[observation.observation_id] = record
        raw_created += 1

    confidence_records: dict[str, GenerationRecord] = {}
    for cell in cells:
        episode_raw = episode_records[cell.episode.observation_id]
        transcript = json.dumps(
            episode_raw.final_messages,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        prompt = build_confidence_prompt(
            cell.episode.item, transcript, config.confidence
        )
        path = run.raw_confidence / f"{cell.record_id}.json"
        confidence_seed = derive_agent_step_seed(
            cell.episode.seed, cell.episode.item.item_id, 2**16 - 1, 0
        )
        condition = InferenceCondition(
            condition_id=(
                f"a-{cell.episode.condition.condition_id}--r-{cell.readout.readout_id}"
            ),
            answer=cell.episode.condition.sampling,
            confidence=cell.readout.sampling,
        )
        if path.exists():
            record = GenerationRecord.model_validate(read_json(path))
            if (
                record.record_id != cell.record_id
                or record.observation_id != cell.observation_id
                or record.parent_answer_record_id != episode_raw.record_id
                or record.prompt_sha256 != prompt.sha256
                or record.condition != condition
                or record.sampling != cell.readout.sampling
                or record.sampling_seed != confidence_seed
            ):
                raise RuntimeError(f"resumed agent confidence mismatch: {path}")
            confidence_records[cell.observation_id] = record
            raw_reused += 1
            continue
        if not backend_holder:
            backend_holder.append(make_backend(config, storage))  # type: ignore[arg-type]
        response = backend_holder[0].generate(
            [
                InferenceRequest(
                    request_id=cell.record_id,
                    stage="confidence",
                    prompt=prompt.text,
                    seed=confidence_seed,
                    condition=condition,
                    chat_template_kwargs=cell.readout.chat_template_kwargs,
                )
            ]
        )[0]
        effective_model = (
            config.model
            if cell.readout.chat_template_kwargs is None
            else config.model.model_copy(
                update={
                    "chat_template_kwargs": cell.readout.chat_template_kwargs,
                    "model_mode": cell.readout.model_mode,
                }
            )
        )
        record = GenerationRecord(
            record_id=cell.record_id,
            observation_id=cell.observation_id,
            generation_stage="confidence",
            parent_answer_record_id=episode_raw.record_id,
            run_id=config.run_id,
            experiment_id=config.experiment_id,
            created_at_utc=utc_now(),
            dataset_id=config.dataset.dataset_id,
            dataset_revision=config.dataset.revision,
            dataset_item_id=cell.episode.item.item_id,
            dataset_partition=config.dataset.partition,
            prompt_text=prompt.text,
            prompt_sha256=prompt.sha256,
            confidence_protocol_id=config.confidence.protocol_id,
            prompt_template_version=config.confidence.prompt_template_version,
            parser_version=config.confidence.confidence_parser_version,
            response_format=config.confidence.confidence_response_format,
            model=effective_model,
            inference_backend=config.inference.backend.name,
            inference_backend_version=config.inference.backend.version,
            engine_seed=config.inference.engine_seed,
            condition=condition,
            sampling=cell.readout.sampling,
            seed=cell.episode.seed,
            sampling_seed=confidence_seed,
            raw_text=response.text,
            finish_reason=response.finish_reason,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
        write_immutable_json(path, record.model_dump(mode="json"))
        confidence_records[cell.observation_id] = record
        raw_created += 1

    scores: dict[str, dict[str, Any]] = {}
    for observation in episodes:
        episode_raw = episode_records[observation.observation_id]
        scores[observation.observation_id] = score_episode(
            episode=_episode_result(episode_raw),
            prepared=_prepared(observation.item, docs_root),
            evaluation_namespace=_stable_id("cpis-bfcl-score-v1", episode_raw.sha256),
        )
    parsed_created = parsed_reused = invalid_confidence = 0
    for cell in cells:
        episode_raw = episode_records[cell.episode.observation_id]
        confidence_raw = confidence_records[cell.observation_id]
        confidence = parse_confidence(
            confidence_raw.raw_text, config.confidence.confidence_parser_version
        )
        invalid_confidence += confidence.status == "invalid"
        score = scores[cell.episode.observation_id]
        path = run.parsed / f"{cell.observation_id}.json"
        if path.exists():
            record = AgentParsedRecord.model_validate(read_json(path))
            if (
                record.episode_raw_record_sha256 != episode_raw.sha256
                or record.confidence_raw_record_sha256 != confidence_raw.sha256
                or record.design_roles != cell.roles
            ):
                raise RuntimeError(f"resumed agent parsed record mismatch: {path}")
            parsed_reused += 1
            continue
        if cell.episode.item.group_id is None:
            raise RuntimeError("BFCL item lacks group ID")
        record = AgentParsedRecord(
            record_id=cell.observation_id,
            episode_observation_id=cell.episode.observation_id,
            episode_raw_record_id=episode_raw.record_id,
            episode_raw_record_sha256=episode_raw.sha256,
            confidence_raw_record_id=confidence_raw.record_id,
            confidence_raw_record_sha256=confidence_raw.sha256,
            run_id=config.run_id,
            experiment_id=config.experiment_id,
            dataset_id=config.dataset.dataset_id,
            dataset_revision=config.dataset.revision,
            dataset_item_id=cell.episode.item.item_id,
            dataset_group_id=cell.episode.item.group_id,
            dataset_partition=config.dataset.partition,
            answer_condition_id=cell.episode.condition.condition_id,
            confidence_readout_id=cell.readout.readout_id,
            design_roles=cell.roles,
            seed=cell.episode.seed,
            scorer="bfcl_v3_official_multi_turn_v1",
            score_valid=bool(score["valid"]),
            score_error_type=score.get("error_type"),
            score_details=score,
            confidence_parse_status=confidence.status,
            confidence=confidence.confidence,
            confidence_parse_error=confidence.error,
            correct=bool(score["valid"]),
        )
        write_immutable_json(path, record.model_dump(mode="json"))
        parsed_created += 1

    unique_scores = list(scores.values())
    return AgentRunResult(
        run_id=config.run_id,
        run_directory=run.base,
        replica_index=replica_index,
        replica_count=config.execution.replicas,
        selected_items=len(items),
        expected_episodes=len(episodes),
        expected_confidence_cells=len(cells),
        raw_created=raw_created,
        raw_reused=raw_reused,
        parsed_created=parsed_created,
        parsed_reused=parsed_reused,
        unsuccessful_episodes=sum(not score["valid"] for score in unique_scores),
        invalid_tool_outputs=sum(
            record.invalid_tool_outputs for record in episode_records.values()
        ),
        invalid_confidence_outputs=invalid_confidence,
    )


def derive_agent_records(
    config_path: Path,
    repository_root: Path,
    replica_index: int = 0,
    *,
    preflight: bool = False,
) -> AgentDerivationResult:
    """Score and parse complete saved BFCL raw stages without loading a model.

    This is deliberately separate from generation so an evaluator or parser
    repair can be versioned and rerun against immutable raw records.  It never
    constructs an inference backend.
    """

    config_path = config_path.resolve()
    repository_root = repository_root.resolve()
    config = load_agent_config(config_path)
    if not 0 <= replica_index < config.execution.replicas:
        raise ValueError("agent replica index is outside the configured range")
    validate_agent_integrations(config, repository_root, preflight=preflight)
    storage = StorageLayout.from_spec(config.storage, repository_root)
    storage.configure_process_caches()
    evaluator_root = (
        storage.dataset_cache / config.bfcl.evaluator_root_relative_to_dataset_cache
    )
    docs_root = evaluator_root / config.bfcl.function_docs_relative_to_evaluator_root
    if function_docs_bundle_sha256(docs_root) != config.bfcl.function_docs_bundle_sha256:
        raise RuntimeError("BFCL function-document bundle differs from the pinned config")
    if str(evaluator_root) not in sys.path:
        sys.path.insert(0, str(evaluator_root))

    run = storage.run(config.run_id)
    manifest_path = run.manifests / (
        f"replica-{replica_index:03d}-of-{config.execution.replicas:03d}.json"
    )
    if not manifest_path.is_file():
        raise RuntimeError("cannot derive agent records without a generation manifest")
    manifest = read_json(manifest_path)
    if (
        manifest.get("run_id") != config.run_id
        or manifest.get("config_sha256") != config.config_sha256
        or manifest.get("replica_index") != replica_index
    ):
        raise RuntimeError("agent generation manifest does not match this derivation")

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
    if {item.partition for item in all_items} != {config.dataset.partition}:
        raise RuntimeError("agent derivation requires a sealed single-partition file")
    items = [
        item
        for item in partition_items(all_items, config.dataset)
        if assign_replica(item.item_id, config.execution.replicas) == replica_index
    ]
    episodes, cells = _build_observations(config, items)

    episode_records: dict[str, AgentEpisodeRecord] = {}
    for observation in episodes:
        path = run.raw_answer / f"{observation.record_id}.json"
        if not path.is_file():
            raise RuntimeError(f"missing raw agent episode: {path}")
        record = AgentEpisodeRecord.model_validate(read_json(path))
        if not _episode_record_matches(
            record, config, observation, _prepared(observation.item, docs_root)
        ):
            raise RuntimeError(f"saved agent episode mismatch: {path}")
        episode_records[observation.observation_id] = record

    confidence_records: dict[str, GenerationRecord] = {}
    for cell in cells:
        episode_raw = episode_records[cell.episode.observation_id]
        transcript = json.dumps(
            episode_raw.final_messages,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        prompt = build_confidence_prompt(
            cell.episode.item, transcript, config.confidence
        )
        path = run.raw_confidence / f"{cell.record_id}.json"
        if not path.is_file():
            raise RuntimeError(f"missing raw agent confidence: {path}")
        record = GenerationRecord.model_validate(read_json(path))
        confidence_seed = derive_agent_step_seed(
            cell.episode.seed, cell.episode.item.item_id, 2**16 - 1, 0
        )
        condition = InferenceCondition(
            condition_id=(
                f"a-{cell.episode.condition.condition_id}--r-{cell.readout.readout_id}"
            ),
            answer=cell.episode.condition.sampling,
            confidence=cell.readout.sampling,
        )
        if (
            record.record_id != cell.record_id
            or record.observation_id != cell.observation_id
            or record.parent_answer_record_id != episode_raw.record_id
            or record.prompt_sha256 != prompt.sha256
            or record.condition != condition
            or record.sampling != cell.readout.sampling
            or record.sampling_seed != confidence_seed
        ):
            raise RuntimeError(f"saved agent confidence mismatch: {path}")
        confidence_records[cell.observation_id] = record

    scores: dict[str, dict[str, Any]] = {}
    for observation in episodes:
        episode_raw = episode_records[observation.observation_id]
        scores[observation.observation_id] = score_episode(
            episode=_episode_result(episode_raw),
            prepared=_prepared(observation.item, docs_root),
            evaluation_namespace=_stable_id(
                "cpis-bfcl-score-v1", episode_raw.sha256
            ),
        )

    parsed_created = parsed_reused = invalid_confidence = 0
    for cell in cells:
        episode_raw = episode_records[cell.episode.observation_id]
        confidence_raw = confidence_records[cell.observation_id]
        confidence = parse_confidence(
            confidence_raw.raw_text, config.confidence.confidence_parser_version
        )
        invalid_confidence += confidence.status == "invalid"
        score = scores[cell.episode.observation_id]
        path = run.parsed / f"{cell.observation_id}.json"
        if path.exists():
            record = AgentParsedRecord.model_validate(read_json(path))
            if (
                record.episode_raw_record_sha256 != episode_raw.sha256
                or record.confidence_raw_record_sha256 != confidence_raw.sha256
                or record.design_roles != cell.roles
            ):
                raise RuntimeError(f"resumed agent parsed record mismatch: {path}")
            parsed_reused += 1
            continue
        if cell.episode.item.group_id is None:
            raise RuntimeError("BFCL item lacks group ID")
        record = AgentParsedRecord(
            record_id=cell.observation_id,
            episode_observation_id=cell.episode.observation_id,
            episode_raw_record_id=episode_raw.record_id,
            episode_raw_record_sha256=episode_raw.sha256,
            confidence_raw_record_id=confidence_raw.record_id,
            confidence_raw_record_sha256=confidence_raw.sha256,
            run_id=config.run_id,
            experiment_id=config.experiment_id,
            dataset_id=config.dataset.dataset_id,
            dataset_revision=config.dataset.revision,
            dataset_item_id=cell.episode.item.item_id,
            dataset_group_id=cell.episode.item.group_id,
            dataset_partition=config.dataset.partition,
            answer_condition_id=cell.episode.condition.condition_id,
            confidence_readout_id=cell.readout.readout_id,
            design_roles=cell.roles,
            seed=cell.episode.seed,
            scorer="bfcl_v3_official_multi_turn_v1",
            score_valid=bool(score["valid"]),
            score_error_type=score.get("error_type"),
            score_details=score,
            confidence_parse_status=confidence.status,
            confidence=confidence.confidence,
            confidence_parse_error=confidence.error,
            correct=bool(score["valid"]),
        )
        write_immutable_json(path, record.model_dump(mode="json"))
        parsed_created += 1

    return AgentDerivationResult(
        run_id=config.run_id,
        run_directory=run.base,
        replica_index=replica_index,
        expected_episodes=len(episodes),
        expected_confidence_cells=len(cells),
        parsed_created=parsed_created,
        parsed_reused=parsed_reused,
        unsuccessful_episodes=sum(not score["valid"] for score in scores.values()),
        invalid_confidence_outputs=invalid_confidence,
    )
