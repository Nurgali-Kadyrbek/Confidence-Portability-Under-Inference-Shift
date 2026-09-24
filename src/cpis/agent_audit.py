"""Read-only integrity audit for immutable BFCL agent episodes and readouts."""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cpis.agent_config import load_agent_config
from cpis.agent_pipeline import _build_observations, derive_agent_pairing_id
from cpis.agent_records import AgentEpisodeRecord, AgentParsedRecord
from cpis.analysis.matrix import _load_replica_manifests
from cpis.datasets import load_dataset, partition_items
from cpis.manifest import git_metadata
from cpis.matrix_audit import _is_writable, _sha256_bytes, _stage_digest
from cpis.phase_gate import (
    authorize_phase_config,
    record_dataset_access,
    require_phase_gate,
)
from cpis.records import GenerationRecord
from cpis.storage import StorageLayout, canonical_json_bytes, read_json, write_immutable_json


def audit_agent(
    config_path: Path, repository_root: Path
) -> tuple[Path, dict[str, Any]]:
    """Verify a complete BFCL run without constructing an inference backend."""

    config_path = config_path.resolve()
    repository_root = repository_root.resolve()
    config = load_agent_config(config_path)
    storage = StorageLayout.from_spec(config.storage, repository_root)
    run = storage.run(config.run_id)
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
        replica_index=0,
        gate=gate,
    )
    manifests = _load_replica_manifests(  # type: ignore[arg-type]
        run, config, repository_root
    )
    items = partition_items(
        load_dataset(config.dataset, config_path.parent, storage.dataset_cache),
        config.dataset,
    )
    episodes, cells = _build_observations(config, items)
    expected_episodes = {row.record_id: row for row in episodes}
    expected_cells = {row.record_id: row for row in cells}
    expected_parsed = {row.observation_id: row for row in cells}
    if len(expected_episodes) != len(episodes) or len(expected_cells) != len(cells):
        raise RuntimeError("agent design generated duplicate record identities")

    episode_paths = sorted(run.raw_answer.glob("*.json"))
    confidence_paths = sorted(run.raw_confidence.glob("*.json"))
    parsed_paths = sorted(run.parsed.glob("*.json"))
    expected_names = {
        "episode": {f"{key}.json" for key in expected_episodes},
        "confidence": {f"{key}.json" for key in expected_cells},
        "parsed": {f"{row.observation_id}.json" for row in expected_cells.values()},
    }
    for label, paths in (
        ("episode", episode_paths),
        ("confidence", confidence_paths),
        ("parsed", parsed_paths),
    ):
        discovered = {path.name for path in paths}
        expected = expected_names[label]
        if discovered != expected:
            raise RuntimeError(
                f"agent {label} record mismatch: missing={len(expected-discovered)}, "
                f"unexpected={len(discovered-expected)}"
            )

    episode_records: dict[str, AgentEpisodeRecord] = {}
    maximum_step_tokens = 0
    finish_reasons: Counter[str] = Counter()
    forced_terminations = invalid_tool_outputs = 0
    for path in episode_paths:
        if _is_writable(path):
            raise RuntimeError(f"agent episode is writable: {path}")
        record = AgentEpisodeRecord.model_validate(read_json(path))
        expected = expected_episodes.get(record.record_id)
        if (
            path.stem != record.record_id
            or expected is None
            or record.run_id != config.run_id
            or record.observation_id != expected.observation_id
            or record.dataset_item_id != expected.item.item_id
            or record.answer_condition_id != expected.condition.condition_id
            or record.replicate_seed != expected.seed
            or record.answer_sampling != expected.condition.sampling
            or (
                record.tool_call_pairing_id is not None
                and record.tool_call_pairing_id
                != derive_agent_pairing_id(
                    config.model.model_id,
                    config.model.revision,
                    config.model.model_mode,
                    expected.seed,
                    expected.item.item_id,
                )
            )
        ):
            raise RuntimeError(f"agent episode/design mismatch: {path}")
        episode_records[record.record_id] = record
        forced_terminations += record.force_terminated
        invalid_tool_outputs += record.invalid_tool_outputs
        for step in record.steps:
            finish_reasons[f"answer_step:{step.finish_reason}"] += 1
            maximum_step_tokens = max(maximum_step_tokens, step.completion_tokens or 0)

    confidence_records: dict[str, GenerationRecord] = {}
    maximum_confidence_tokens = 0
    for path in confidence_paths:
        if _is_writable(path):
            raise RuntimeError(f"agent confidence is writable: {path}")
        record = GenerationRecord.model_validate(read_json(path))
        expected = expected_cells.get(record.record_id)
        if (
            path.stem != record.record_id
            or expected is None
            or record.run_id != config.run_id
            or record.generation_stage != "confidence"
            or record.observation_id != expected.observation_id
            or record.parent_answer_record_id != expected.episode.record_id
            or record.dataset_item_id != expected.episode.item.item_id
            or record.sampling != expected.readout.sampling
        ):
            raise RuntimeError(f"agent confidence/design mismatch: {path}")
        confidence_records[record.record_id] = record
        finish_reasons[f"confidence:{record.finish_reason}"] += 1
        maximum_confidence_tokens = max(
            maximum_confidence_tokens, record.completion_tokens or 0
        )

    invalid_confidences = unsuccessful_cells = 0
    parsed_hashes: list[str] = []
    for path in parsed_paths:
        if _is_writable(path):
            raise RuntimeError(f"agent parsed record is writable: {path}")
        record = AgentParsedRecord.model_validate(read_json(path))
        expected = expected_parsed.get(record.record_id)
        if expected is None:
            raise RuntimeError(f"agent parsed record has unknown identity: {path}")
        try:
            episode = episode_records[record.episode_raw_record_id]
            confidence = confidence_records[record.confidence_raw_record_id]
        except KeyError as exc:
            raise RuntimeError(f"agent parsed record lacks raw provenance: {path}") from exc
        if (
            path.stem != record.record_id
            or record.run_id != config.run_id
            or record.episode_observation_id != expected.episode.observation_id
            or record.episode_raw_record_sha256 != episode.sha256
            or record.confidence_raw_record_sha256 != confidence.sha256
            or record.answer_condition_id != expected.episode.condition.condition_id
            or record.confidence_readout_id != expected.readout.readout_id
            or record.design_roles != expected.roles
            or record.seed != expected.episode.seed
        ):
            raise RuntimeError(f"agent parsed/raw/design provenance mismatch: {path}")
        invalid_confidences += record.confidence_parse_status == "invalid"
        unsuccessful_cells += not record.correct
        parsed_hashes.append(_sha256_bytes(path))

    access_paths = sorted(run.manifests.glob("dataset-access-replica-*.json"))
    if len(access_paths) != config.execution.replicas:
        raise RuntimeError("agent run lacks one dataset-access event per replica")
    for index, path in enumerate(access_paths):
        event = read_json(path)
        if (
            event.get("run_id") != config.run_id
            or event.get("partition") != config.dataset.partition
            or event.get("replica_index") != index
        ):
            raise RuntimeError(f"agent dataset-access event mismatch: {path}")

    result: dict[str, Any] = {
        "schema_version": "1.0",
        "audit_id": f"{config.run_id}-agent-integrity-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": config.run_id,
        "config_sha256": config.config_sha256,
        "generation_git": manifests[0]["git"],
        "generation_git_commits": [manifest["git"]["commit"] for manifest in manifests],
        "audit_git": git_metadata(repository_root),
        "replicas": config.execution.replicas,
        "dataset_rows": len(items),
        "independent_scenarios": len({item.group_id for item in items}),
        "episode_records": len(episode_paths),
        "confidence_records": len(confidence_paths),
        "parsed_records": len(parsed_paths),
        "forced_terminations": forced_terminations,
        "invalid_tool_outputs": invalid_tool_outputs,
        "invalid_confidence_cells": invalid_confidences,
        "unsuccessful_cells": unsuccessful_cells,
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "maximum_completion_tokens": {
            "answer_step": maximum_step_tokens,
            "confidence": maximum_confidence_tokens,
        },
        "writable_raw_files": 0,
        "writable_parsed_files": 0,
        "dataset_access_events": len(access_paths),
        "stage_sha256": {
            "raw_episode": _stage_digest(episode_paths),
            "raw_confidence": _stage_digest(confidence_paths),
            "parsed": hashlib.sha256(
                "\n".join(sorted(parsed_hashes)).encode("utf-8")
            ).hexdigest(),
        },
    }
    output = run.manifests / "agent-integrity-audit-v1.json"
    if output.exists():
        existing = read_json(output)
        comparable = dict(result)
        comparable["created_at_utc"] = existing.get("created_at_utc")
        comparable["audit_git"] = existing.get("audit_git")
        if canonical_json_bytes(existing) != canonical_json_bytes(comparable):
            raise RuntimeError(f"agent integrity audit collision at {output}")
        return output, existing
    write_immutable_json(output, result)
    return output, result
