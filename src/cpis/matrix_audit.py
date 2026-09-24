"""Read-only integrity audit for an immutable answer/readout matrix."""

from __future__ import annotations

import hashlib
import json
import stat
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cpis.analysis.matrix import _load_replica_manifests
from cpis.datasets import load_dataset, partition_items
from cpis.manifest import git_metadata
from cpis.matrix_config import load_matrix_config
from cpis.matrix_pipeline import _build_observations
from cpis.matrix_records import MatrixParsedRecord
from cpis.phase_gate import (
    authorize_phase_config,
    record_dataset_access,
    require_phase_gate,
)
from cpis.records import GenerationRecord
from cpis.storage import StorageLayout, canonical_json_bytes, read_json, write_immutable_json


def _sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_digest(paths: list[Path]) -> str:
    entries = [f"{path.name}\0{_sha256_bytes(path)}" for path in sorted(paths)]
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def _is_writable(path: Path) -> bool:
    return bool(path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def audit_matrix(
    config_path: Path, repository_root: Path
) -> tuple[Path, dict[str, Any]]:
    """Verify every expected immutable record without constructing a backend."""
    config_path = config_path.resolve()
    repository_root = repository_root.resolve()
    config = load_matrix_config(config_path)
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
    manifests = _load_replica_manifests(run, config, repository_root)

    items = partition_items(
        load_dataset(config.dataset, config_path.parent, storage.dataset_cache),
        config.dataset,
    )
    answers, cells = _build_observations(config, items)
    expected_answers = {answer.record_id: answer for answer in answers}
    expected_cells = {cell.record_id: cell for cell in cells}
    if len(expected_answers) != len(answers) or len(expected_cells) != len(cells):
        raise RuntimeError("matrix design generated duplicate record identities")

    answer_paths = sorted(run.raw_answer.glob("*.json"))
    confidence_paths = sorted(run.raw_confidence.glob("*.json"))
    parsed_paths = sorted(run.parsed.glob("*.json"))
    expected_answer_names = {f"{record_id}.json" for record_id in expected_answers}
    expected_confidence_names = {f"{record_id}.json" for record_id in expected_cells}
    expected_parsed_names = {
        f"{cell.observation_id}.json" for cell in expected_cells.values()
    }
    for label, paths, expected in (
        ("answer", answer_paths, expected_answer_names),
        ("confidence", confidence_paths, expected_confidence_names),
        ("parsed", parsed_paths, expected_parsed_names),
    ):
        discovered = {path.name for path in paths}
        if discovered != expected:
            raise RuntimeError(
                f"{label} record set mismatch: missing={len(expected-discovered)}, "
                f"unexpected={len(discovered-expected)}"
            )

    answer_records: dict[str, GenerationRecord] = {}
    confidence_records: dict[str, GenerationRecord] = {}
    finish_reasons: Counter[str] = Counter()
    maximum_tokens = {"answer": 0, "confidence": 0}
    for stage, paths, destination in (
        ("answer", answer_paths, answer_records),
        ("confidence", confidence_paths, confidence_records),
    ):
        for path in paths:
            if _is_writable(path):
                raise RuntimeError(f"raw record is writable: {path}")
            record = GenerationRecord.model_validate(read_json(path))
            if path.stem != record.record_id or record.generation_stage != stage:
                raise RuntimeError(f"raw record filename/stage mismatch: {path}")
            if record.run_id != config.run_id:
                raise RuntimeError(f"raw record belongs to another run: {path}")
            destination[record.record_id] = record
            finish_reasons[f"{stage}:{record.finish_reason}"] += 1
            maximum_tokens[stage] = max(
                maximum_tokens[stage], record.completion_tokens or 0
            )

    invalid_answers = invalid_confidences = 0
    parsed_record_hashes: list[str] = []
    for path in parsed_paths:
        if _is_writable(path):
            raise RuntimeError(f"parsed record is writable: {path}")
        parsed = MatrixParsedRecord.model_validate(read_json(path))
        if path.stem != parsed.record_id or parsed.run_id != config.run_id:
            raise RuntimeError(f"parsed record filename/run mismatch: {path}")
        try:
            answer = answer_records[parsed.answer_raw_record_id]
            confidence = confidence_records[parsed.confidence_raw_record_id]
            expected_cell = expected_cells[parsed.confidence_raw_record_id]
        except KeyError as exc:
            raise RuntimeError(f"parsed record has an unknown raw identity: {path}") from exc
        expected_answer = expected_answers[parsed.answer_raw_record_id]
        if (
            parsed.answer_observation_id != answer.observation_id
            or confidence.parent_answer_record_id != answer.record_id
            or parsed.answer_raw_record_sha256 != answer.sha256
            or parsed.confidence_raw_record_sha256 != confidence.sha256
            or parsed.dataset_item_id != expected_answer.item.item_id
            or parsed.dataset_partition != config.dataset.partition
            or parsed.answer_condition_id != expected_answer.condition.condition_id
            or parsed.confidence_readout_id != expected_cell.readout.readout_id
            or parsed.design_roles != expected_cell.roles
            or parsed.seed != expected_answer.seed
            or parsed.answer_sampling_seed != answer.sampling_seed
            or parsed.confidence_sampling_seed != confidence.sampling_seed
        ):
            raise RuntimeError(f"parsed/raw/design provenance mismatch: {path}")
        invalid_answers += parsed.answer_parse_status == "invalid"
        invalid_confidences += parsed.confidence_parse_status == "invalid"
        parsed_record_hashes.append(_sha256_bytes(path))

    access_paths = sorted(run.manifests.glob("dataset-access-replica-*.json"))
    if len(access_paths) != config.execution.replicas:
        raise RuntimeError("matrix lacks one dataset-access event per replica")
    for index, path in enumerate(access_paths):
        event = read_json(path)
        if (
            event.get("run_id") != config.run_id
            or event.get("partition") != config.dataset.partition
            or event.get("replica_index") != index
        ):
            raise RuntimeError(f"dataset-access event mismatch: {path}")

    result: dict[str, Any] = {
        "schema_version": "1.0",
        "audit_id": f"{config.run_id}-integrity-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": config.run_id,
        "config_sha256": config.config_sha256,
        "generation_git": manifests[0]["git"],
        "generation_git_commits": [manifest["git"]["commit"] for manifest in manifests],
        "audit_git": git_metadata(repository_root),
        "replicas": config.execution.replicas,
        "dataset_items": len(items),
        "expected_answer_records": len(expected_answers),
        "expected_confidence_records": len(expected_cells),
        "expected_parsed_records": len(expected_cells),
        "answer_records": len(answer_paths),
        "confidence_records": len(confidence_paths),
        "parsed_records": len(parsed_paths),
        "invalid_answer_cells": invalid_answers,
        "invalid_confidence_cells": invalid_confidences,
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "maximum_completion_tokens": maximum_tokens,
        "writable_raw_files": 0,
        "writable_parsed_files": 0,
        "dataset_access_events": len(access_paths),
        "stage_sha256": {
            "raw_answer": _stage_digest(answer_paths),
            "raw_confidence": _stage_digest(confidence_paths),
            "parsed": hashlib.sha256(
                "\n".join(sorted(parsed_record_hashes)).encode("utf-8")
            ).hexdigest(),
        },
    }
    output = run.manifests / "integrity-audit-v1.json"
    if output.exists():
        existing = read_json(output)
        comparable = dict(result)
        comparable["created_at_utc"] = existing.get("created_at_utc")
        comparable["audit_git"] = existing.get("audit_git")
        if canonical_json_bytes(existing) != canonical_json_bytes(comparable):
            raise RuntimeError(f"matrix integrity audit collision at {output}")
        return output, existing
    write_immutable_json(output, result)
    return output, result
