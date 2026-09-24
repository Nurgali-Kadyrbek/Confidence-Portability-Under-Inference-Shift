"""Regenerate compact BFCL integration/development evidence from saved outputs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import fmean

from cpis.agent_audit import audit_agent
from cpis.agent_config import load_agent_config
from cpis.agent_records import AgentEpisodeRecord, AgentParsedRecord
from cpis.datasets import load_dataset, partition_items
from cpis.manifest import git_metadata
from cpis.storage import StorageLayout, read_json


def _write_reproducible(path: Path, text: str) -> None:
    data = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"refusing to replace different agent evidence: {path}")
        return
    path.write_bytes(data)


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def write_agent_evidence(
    config_path: Path,
    repository_root: Path,
    destination: Path,
) -> dict[str, str]:
    repository_root = repository_root.resolve()
    config_path = config_path.resolve()
    config = load_agent_config(config_path)
    storage = StorageLayout.from_spec(config.storage, repository_root)
    run = storage.run(config.run_id)
    audit_path, audit = audit_agent(config_path, repository_root)
    items = partition_items(
        load_dataset(config.dataset, config_path.parent, storage.dataset_cache),
        config.dataset,
    )
    episodes = [
        AgentEpisodeRecord.model_validate(read_json(path))
        for path in sorted(run.raw_answer.glob("*.json"))
    ]
    parsed = [
        AgentParsedRecord.model_validate(read_json(path))
        for path in sorted(run.parsed.glob("*.json"))
    ]
    episode_success: dict[str, bool] = {}
    for record in parsed:
        previous = episode_success.setdefault(record.episode_observation_id, record.correct)
        if previous != record.correct:
            raise RuntimeError("BFCL score differs across confidence readouts")
    conditions = [row.condition_id for row in config.inference.answer_conditions]
    condition_summaries = {}
    for condition in conditions:
        condition_episodes = [row for row in episodes if row.answer_condition_id == condition]
        coupled = [
            row
            for row in parsed
            if row.answer_condition_id == condition and "coupled" in row.design_roles
        ]
        standardized = [
            row
            for row in parsed
            if row.answer_condition_id == condition and "standardized" in row.design_roles
        ]

        def readout_summary(rows: list[AgentParsedRecord]) -> dict:
            valid = [float(row.confidence) for row in rows if row.confidence is not None]
            return {
                "cells": len(rows),
                "valid_confidence_cells": len(valid),
                "invalid_confidence_rate": _rate(len(rows) - len(valid), len(rows)),
                "mean_confidence_valid_only": fmean(valid) if valid else None,
            }

        condition_summaries[condition] = {
            "episodes": len(condition_episodes),
            "successful_episodes": sum(
                episode_success[row.observation_id] for row in condition_episodes
            ),
            "success_rate": _rate(
                sum(episode_success[row.observation_id] for row in condition_episodes),
                len(condition_episodes),
            ),
            "forced_terminations": sum(row.force_terminated for row in condition_episodes),
            "invalid_tool_outputs": sum(
                row.invalid_tool_outputs for row in condition_episodes
            ),
            "coupled": readout_summary(coupled),
            "standardized_deterministic": readout_summary(standardized),
        }
    payload = {
        "schema_version": "1.0",
        "evidence_id": f"{config.experiment_id}-evidence-v1",
        "run_id": config.run_id,
        "experiment_id": config.experiment_id,
        "config_sha256": config.config_sha256,
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "model_mode": config.model.model_mode,
        "dataset_id": config.dataset.dataset_id,
        "dataset_revision": config.dataset.revision,
        "phase": config.phase,
        "dataset_rows": audit["dataset_rows"],
        "independent_scenarios": audit["independent_scenarios"],
        "replicate_seeds": list(config.inference.seeds),
        "audit_path": str(audit_path),
        "audit_id": audit["audit_id"],
        "stage_sha256": audit["stage_sha256"],
        "record_counts": {
            "episodes": audit["episode_records"],
            "confidence": audit["confidence_records"],
            "parsed": audit["parsed_records"],
        },
        "forced_terminations": audit["forced_terminations"],
        "invalid_tool_outputs": audit["invalid_tool_outputs"],
        "invalid_confidence_cells": audit["invalid_confidence_cells"],
        "maximum_completion_tokens": audit["maximum_completion_tokens"],
        "finish_reasons": audit["finish_reasons"],
        "conditions": condition_summaries,
        "category_rows": dict(
            sorted(Counter(str(item.metadata["category"]) for item in items).items())
        ),
        "generation_git": audit["generation_git"],
        "report_git": git_metadata(repository_root),
    }
    base = destination.resolve() / f"{config.experiment_id}-evidence-v1"
    json_path = base.with_suffix(".json")
    markdown_path = base.with_suffix(".md")
    lines = [
        f"# {config.experiment_id} evidence",
        "",
        f"- Run ID: `{config.run_id}`",
        f"- Model: `{config.model.model_id}` at `{config.model.revision}` ({config.model.model_mode})",
        f"- Dataset rows / independent scenarios: {audit['dataset_rows']} / {audit['independent_scenarios']}",
        f"- Immutable episodes / confidence / parsed: {audit['episode_records']} / {audit['confidence_records']} / {audit['parsed_records']}",
        f"- Forced terminations / invalid tool outputs / invalid confidence: {audit['forced_terminations']} / {audit['invalid_tool_outputs']} / {audit['invalid_confidence_cells']}",
        "",
        "| Condition | Episodes | Success | Coupled mean confidence | Coupled invalid |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        row = condition_summaries[condition]
        confidence = row["coupled"]["mean_confidence_valid_only"]
        lines.append(
            f"| {condition} | {row['episodes']} | {row['success_rate']:.3f} | "
            f"{confidence if confidence is not None else 'NA'} | "
            f"{row['coupled']['invalid_confidence_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "All success values use the pinned official BFCL V3 evaluator. Repeated",
            "category rows, seeds, decoder conditions, and confidence readouts are not",
            "treated as independent scenarios. Raw records are immutable and the saved",
            "stage digests above reproduce this report without another model call.",
            "",
        ]
    )
    _write_reproducible(json_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_reproducible(markdown_path, "\n".join(lines))
    return {"json": str(json_path), "markdown": str(markdown_path)}
