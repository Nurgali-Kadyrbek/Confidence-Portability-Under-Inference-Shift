"""Model-free exploratory analysis of a saved development readout matrix."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Callable, Iterable

from cpis.analysis.metrics import (
    confidence_accepts,
    has_valid_confidence,
    select_fixed_coverage_threshold,
    select_risk_contract_threshold,
    transport_fixed_coverage_threshold,
)
from cpis.analysis.units import deterministic_balanced_group_anchors
from cpis.analysis_plan import load_analysis_plan
from cpis.datasets import DatasetItem, load_dataset, partition_items
from cpis.manifest import git_metadata
from cpis.matrix_config import MatrixExperimentConfig, load_matrix_config
from cpis.matrix_records import MatrixParsedRecord
from cpis.official_scoring import OfficialScoreRecord
from cpis.provenance import validate_replica_git_equivalence
from cpis.records import ParsedRecord
from cpis.storage import StorageLayout, read_json, write_immutable_json


def _load_replica_manifests(
    run,
    config: MatrixExperimentConfig,
    repository_root: Path | None = None,
) -> list[dict]:
    """Require one mutually consistent generation manifest per configured replica."""
    expected_paths = [
        run.manifests
        / f"replica-{index:03d}-of-{config.execution.replicas:03d}.json"
        for index in range(config.execution.replicas)
    ]
    discovered_paths = sorted(run.manifests.glob("replica-*-of-*.json"))
    if discovered_paths != expected_paths:
        missing = sorted(set(expected_paths) - set(discovered_paths))
        unexpected = sorted(set(discovered_paths) - set(expected_paths))
        raise RuntimeError(
            "matrix replica manifests are incomplete or inconsistent: "
            f"missing={[path.name for path in missing]}, "
            f"unexpected={[path.name for path in unexpected]}"
        )

    manifests = [read_json(path) for path in expected_paths]
    for index, manifest in enumerate(manifests):
        if manifest.get("config_sha256") != config.config_sha256:
            raise RuntimeError(f"matrix manifest/config mismatch for replica {index}")
        if manifest.get("run_id") != config.run_id:
            raise RuntimeError(f"matrix manifest/run mismatch for replica {index}")
        if manifest.get("replica_index") != index:
            raise RuntimeError(f"matrix manifest index mismatch for replica {index}")
        if manifest.get("replica_count") != config.execution.replicas:
            raise RuntimeError(f"matrix manifest count mismatch for replica {index}")
    generation_git = manifests[0].get("git")
    if any(manifest.get("git") != generation_git for manifest in manifests[1:]):
        if repository_root is None:
            raise RuntimeError("matrix replicas were generated from different git states")
        validate_replica_git_equivalence(
            run_id=config.run_id,
            config_sha256=config.config_sha256,
            manifests=manifests,
            repository_root=repository_root.resolve(),
        )
    return manifests


def _as_parsed(record: MatrixParsedRecord) -> ParsedRecord:
    return ParsedRecord(
        record_id=record.record_id,
        answer_raw_record_id=record.answer_raw_record_id,
        answer_raw_record_sha256=record.answer_raw_record_sha256,
        confidence_raw_record_id=record.confidence_raw_record_id,
        confidence_raw_record_sha256=record.confidence_raw_record_sha256,
        run_id=record.run_id,
        experiment_id=record.experiment_id,
        dataset_id=record.dataset_id,
        dataset_revision=record.dataset_revision,
        dataset_item_id=record.dataset_item_id,
        dataset_partition=record.dataset_partition,
        condition_id=record.answer_condition_id,
        seed=record.seed,
        answer_parser_version=record.answer_parser_version,
        confidence_parser_version=record.confidence_parser_version,
        scorer=record.scorer,
        answer_parse_status=record.answer_parse_status,
        confidence_parse_status=record.confidence_parse_status,
        parse_status=record.parse_status,
        parsed_answer=record.parsed_answer,
        confidence=record.confidence,
        scoring_status=record.scoring_status,
        correct=record.correct,
        answer_parse_error=record.answer_parse_error,
        confidence_parse_error=record.confidence_parse_error,
    )


def _resolve_official_scores(
    run,
    records: list[MatrixParsedRecord],
    parsed_paths: list[Path],
) -> list[MatrixParsedRecord]:
    pending = [record for record in records if record.scoring_status != "scored"]
    if not pending:
        return records
    if len(pending) != len(records):
        raise RuntimeError("matrix mixes internally and externally scored records")

    score_paths = sorted((run.derived / "official-scores").glob("*.json"))
    if len(score_paths) != len(records):
        raise RuntimeError(
            "matrix analysis requires one official score per parsed record: "
            f"expected {len(records)}, got {len(score_paths)}"
        )
    by_parsed_id: dict[str, OfficialScoreRecord] = {}
    for score_path in score_paths:
        score = OfficialScoreRecord.model_validate(read_json(score_path))
        if score.parsed_record_id in by_parsed_id:
            raise RuntimeError("duplicate official score for a parsed record")
        by_parsed_id[score.parsed_record_id] = score

    resolved: list[MatrixParsedRecord] = []
    for parsed_path, record in zip(parsed_paths, records, strict=True):
        try:
            score = by_parsed_id.pop(record.record_id)
        except KeyError as exc:
            raise RuntimeError("official score is missing for a parsed record") from exc
        if (
            score.run_id != record.run_id
            or score.experiment_id != record.experiment_id
            or score.dataset_id != record.dataset_id
            or score.dataset_revision != record.dataset_revision
            or score.dataset_item_id != record.dataset_item_id
            or score.scorer != record.scorer
            or score.answer_raw_record_id != record.answer_raw_record_id
            or score.parsed_record_sha256 != _file_sha256(parsed_path)
            or score.answer_raw_record_sha256 != record.answer_raw_record_sha256
        ):
            raise RuntimeError("official score provenance does not match parsed record")
        resolved.append(
            record.model_copy(
                update={"scoring_status": "scored", "correct": score.primary_success}
            )
        )
    if by_parsed_id:
        raise RuntimeError("official score set contains records outside this matrix")
    return resolved


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _auroc(records: Iterable[MatrixParsedRecord]) -> float | None:
    valid = [record for record in records if has_valid_confidence(record)]
    groups: dict[float, list[int]] = {}
    for record in valid:
        counts = groups.setdefault(float(record.confidence), [0, 0])
        counts[0 if record.correct else 1] += 1
    positives = sum(counts[0] for counts in groups.values())
    negatives = sum(counts[1] for counts in groups.values())
    if not positives or not negatives:
        return None
    lower_negatives = 0
    wins = 0.0
    for confidence in sorted(groups):
        positive, negative = groups[confidence]
        wins += positive * lower_negatives + 0.5 * positive * negative
        lower_negatives += negative
    return wins / (positives * negatives)


def _summary(
    records: list[MatrixParsedRecord], cluster_by_item: dict[str, str]
) -> dict:
    if not records:
        raise ValueError("matrix summary requires observations")
    valid = [record for record in records if has_valid_confidence(record)]
    confidences = [record.confidence for record in valid]
    brier = (
        fmean((float(record.confidence) - int(record.correct)) ** 2 for record in valid)
        if valid
        else None
    )
    return {
        "observations": len(records),
        "experimental_units": len(
            {cluster_by_item[record.dataset_item_id] for record in records}
        ),
        "replicate_seeds": sorted({record.seed for record in records}),
        "accuracy": fmean(record.correct for record in records),
        "valid_confidence_observations": len(valid),
        "invalid_confidence_observations": len(records) - len(valid),
        "invalid_confidence_rate": (len(records) - len(valid)) / len(records),
        "mean_confidence_valid_only": fmean(confidences) if confidences else None,
        "median_confidence_valid_only": median(confidences) if confidences else None,
        "brier_valid_only": brier,
        "auroc_valid_only": _auroc(valid),
    }


def _resampling_design(
    items: list[DatasetItem], dataset_id: str
) -> tuple[dict[str, str], dict[str, str] | None, dict[str, int]]:
    """Resolve independent clusters and optional prespecified strata.

    Derived benchmark variants share ``group_id`` and must travel together.
    SuperGPQA has no derived variants, but its full partitions are resampled
    within discipline/subfield strata as prespecified.
    """
    cluster_by_item = {item.item_id: item.group_id or item.item_id for item in items}
    if len(cluster_by_item) != len(items):
        raise RuntimeError("selected dataset contains duplicate item IDs")
    members: dict[str, list[DatasetItem]] = {}
    for item in items:
        members.setdefault(cluster_by_item[item.item_id], []).append(item)

    strata_by_cluster: dict[str, str] | None = None
    stratum_counts: dict[str, int] = {}
    if dataset_id == "m-a-p/SuperGPQA":
        strata_by_cluster = {}
        for cluster_id, cluster_items in members.items():
            labels = {
                f"{item.metadata.get('discipline')}\0{item.metadata.get('subfield')}"
                for item in cluster_items
            }
            if len(labels) != 1 or "None\0None" in labels:
                raise RuntimeError("SuperGPQA item is missing its resampling stratum")
            strata_by_cluster[cluster_id] = next(iter(labels))
        stratum_counts = dict(Counter(strata_by_cluster.values()))
        if min(stratum_counts.values()) < 2:
            strata_by_cluster = None
    return cluster_by_item, strata_by_cluster, stratum_counts


def _sample_clusters(
    cluster_ids: list[str],
    strata_by_cluster: dict[str, str] | None,
    rng: random.Random,
) -> list[str]:
    if strata_by_cluster is None:
        return rng.choices(cluster_ids, k=len(cluster_ids))
    by_stratum: dict[str, list[str]] = {}
    for cluster_id in cluster_ids:
        by_stratum.setdefault(strata_by_cluster[cluster_id], []).append(cluster_id)
    return [
        sampled
        for stratum in sorted(by_stratum)
        for sampled in rng.choices(
            sorted(by_stratum[stratum]), k=len(by_stratum[stratum])
        )
    ]


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    low = math.floor(location)
    high = math.ceil(location)
    if low == high:
        return ordered[low]
    fraction = location - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def _interval(values: list[float]) -> dict:
    return {
        "lower_95": _quantile(values, 0.025),
        "upper_95": _quantile(values, 0.975),
        "successful_replicates": len(values),
    }


def _bootstrap_paired(
    reference: list[MatrixParsedRecord],
    shifted: list[MatrixParsedRecord],
    metric: Callable[[list[MatrixParsedRecord]], float | None],
    replicates: int,
    seed: int,
    cluster_by_item: dict[str, str],
    strata_by_cluster: dict[str, str] | None,
) -> list[float]:
    by_ref: dict[str, list[MatrixParsedRecord]] = {}
    by_shift: dict[str, list[MatrixParsedRecord]] = {}
    for record in reference:
        by_ref.setdefault(cluster_by_item[record.dataset_item_id], []).append(record)
    for record in shifted:
        by_shift.setdefault(cluster_by_item[record.dataset_item_id], []).append(record)
    if set(by_ref) != set(by_shift):
        raise ValueError("bootstrap requires paired benchmark clusters")
    cluster_ids = sorted(by_ref)
    rng = random.Random(seed)
    values: list[float] = []
    metric_kind = getattr(metric, "_cpis_metric_kind", None)
    metric_threshold = getattr(metric, "_cpis_metric_threshold", None)
    sufficient: dict[str, tuple[tuple[float, int], tuple[float, int]]] = {}
    if metric_kind is not None:
        def summarize(rows: list[MatrixParsedRecord]) -> tuple[float, int]:
            if metric_kind == "accuracy":
                return float(sum(bool(row.correct) for row in rows)), len(rows)
            if metric_kind == "mean_confidence":
                valid = [
                    float(row.confidence) for row in rows if has_valid_confidence(row)
                ]
                return math.fsum(valid), len(valid)
            if metric_kind == "brier":
                valid = [row for row in rows if has_valid_confidence(row)]
                return (
                    math.fsum(
                        (float(row.confidence) - int(bool(row.correct))) ** 2
                        for row in valid
                    ),
                    len(valid),
                )
            if metric_kind == "invalid_confidence_rate":
                return (
                    float(sum(not has_valid_confidence(row) for row in rows)),
                    len(rows),
                )
            accepted = [
                row
                for row in rows
                if confidence_accepts(row, float(metric_threshold))
            ]
            if metric_kind == "risk":
                return float(sum(not row.correct for row in accepted)), len(accepted)
            if metric_kind == "coverage":
                return float(len(accepted)), len(rows)
            raise RuntimeError(f"unknown optimized bootstrap metric: {metric_kind}")

        sufficient = {
            cluster: (summarize(by_ref[cluster]), summarize(by_shift[cluster]))
            for cluster in cluster_ids
        }
    for _ in range(replicates):
        sample = _sample_clusters(cluster_ids, strata_by_cluster, rng)
        if sufficient:
            ref_numerator = math.fsum(sufficient[cluster][0][0] for cluster in sample)
            ref_denominator = sum(sufficient[cluster][0][1] for cluster in sample)
            shift_numerator = math.fsum(
                sufficient[cluster][1][0] for cluster in sample
            )
            shift_denominator = sum(sufficient[cluster][1][1] for cluster in sample)
            ref_value = (
                ref_numerator / ref_denominator if ref_denominator else None
            )
            shift_value = (
                shift_numerator / shift_denominator if shift_denominator else None
            )
        else:
            ref_sample = [record for cluster in sample for record in by_ref[cluster]]
            shift_sample = [record for cluster in sample for record in by_shift[cluster]]
            ref_value = metric(ref_sample)
            shift_value = metric(shift_sample)
        if ref_value is not None and shift_value is not None:
            values.append(shift_value - ref_value)
    return values


def _risk_metric(threshold: float) -> Callable[[list[MatrixParsedRecord]], float | None]:
    def metric(records: list[MatrixParsedRecord]) -> float | None:
        accepted = [
            record
            for record in records
            if confidence_accepts(record, threshold)
        ]
        if not accepted:
            return None
        return fmean(not record.correct for record in accepted)

    setattr(metric, "_cpis_metric_kind", "risk")
    setattr(metric, "_cpis_metric_threshold", threshold)
    return metric


def _coverage_metric(
    threshold: float,
) -> Callable[[list[MatrixParsedRecord]], float | None]:
    def metric(records: list[MatrixParsedRecord]) -> float:
        return sum(confidence_accepts(record, threshold) for record in records) / len(
            records
        )

    setattr(metric, "_cpis_metric_kind", "coverage")
    setattr(metric, "_cpis_metric_threshold", threshold)
    return metric


def _mean_confidence(records: list[MatrixParsedRecord]) -> float | None:
    values = [record.confidence for record in records if has_valid_confidence(record)]
    return fmean(values) if values else None


def _accuracy(records: list[MatrixParsedRecord]) -> float:
    return fmean(record.correct for record in records)


setattr(_mean_confidence, "_cpis_metric_kind", "mean_confidence")
setattr(_accuracy, "_cpis_metric_kind", "accuracy")


def _fixed_coverage_analysis(
    reference: list[MatrixParsedRecord],
    conditions: dict[str, list[MatrixParsedRecord]],
    targets: tuple[float, ...],
    bootstrap_replicates: int,
    bootstrap_seed: int,
    cluster_by_item: dict[str, str],
    strata_by_cluster: dict[str, str] | None,
) -> dict:
    parsed_reference = [_as_parsed(record) for record in reference]
    result: dict[str, dict] = {}
    for target_index, target in enumerate(targets):
        selected = select_fixed_coverage_threshold(parsed_reference, target)
        target_result: dict = {"selection": asdict(selected), "transport": {}}
        if selected.status == "selected":
            assert selected.threshold is not None
            for condition_index, (condition_id, records) in enumerate(conditions.items()):
                transport = transport_fixed_coverage_threshold(
                    parsed_reference,
                    [_as_parsed(record) for record in records],
                    selected,
                )
                pair_seed = (
                    bootstrap_seed + 10_000 * target_index + condition_index
                )
                risk_bootstrap = _bootstrap_paired(
                    reference,
                    records,
                    _risk_metric(selected.threshold),
                    bootstrap_replicates,
                    pair_seed,
                    cluster_by_item,
                    strata_by_cluster,
                )
                coverage_bootstrap = _bootstrap_paired(
                    reference,
                    records,
                    _coverage_metric(selected.threshold),
                    bootstrap_replicates,
                    pair_seed + 5_000,
                    cluster_by_item,
                    strata_by_cluster,
                )
                target_result["transport"][condition_id] = {
                    **asdict(transport),
                    "delta_risk_item_bootstrap_95": _interval(risk_bootstrap),
                    "delta_coverage_from_reference_item_bootstrap_95": _interval(
                        coverage_bootstrap
                    ),
                }
        result[f"{target:.2f}"] = target_result
    return result


def _decomposition(
    records: list[MatrixParsedRecord],
    config: MatrixExperimentConfig,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    cluster_by_item: dict[str, str],
    strata_by_cluster: dict[str, str] | None,
) -> dict:
    design = config.inference.design
    indexed = {
        (
            record.answer_condition_id,
            record.confidence_readout_id,
            record.dataset_item_id,
            record.seed,
        ): record
        for record in records
    }
    result: dict[str, dict] = {}
    shifts = [
        condition.condition_id
        for condition in config.inference.answer_conditions
        if condition.condition_id != design.reference_answer_condition_id
    ]
    for shift_index, shift in enumerate(shifts):
        shift_readout = design.coupled_readout_by_answer[shift]
        rows: list[tuple[str, float, float, float, float]] = []
        total_pairs = 0
        for item_id in sorted({record.dataset_item_id for record in records}):
            for seed in config.inference.seeds:
                total_pairs += 1
                keys = (
                    (design.reference_answer_condition_id, design.reference_confidence_readout_id),
                    (shift, design.reference_confidence_readout_id),
                    (design.reference_answer_condition_id, shift_readout),
                    (shift, shift_readout),
                )
                cells = [indexed.get((*key, item_id, seed)) for key in keys]
                if any(cell is None or not has_valid_confidence(cell) for cell in cells):
                    continue
                c00, c10, c01, c11 = (float(cell.confidence) for cell in cells if cell)
                rows.append((item_id, c00, c10, c01, c11))
        by_cluster: dict[str, list[tuple[float, float, float]]] = {}
        for item_id, c00, c10, c01, c11 in rows:
            by_cluster.setdefault(cluster_by_item[item_id], []).append(
                (c10 - c00, c01 - c00, c11 - c10 - c01 + c00)
            )
        cluster_ids = sorted(by_cluster)
        rng = random.Random(bootstrap_seed + shift_index)
        boot_answer: list[float] = []
        boot_readout: list[float] = []
        boot_interaction: list[float] = []
        for _ in range(bootstrap_replicates):
            sample = _sample_clusters(cluster_ids, strata_by_cluster, rng)
            effects = [effect for cluster in sample for effect in by_cluster[cluster]]
            boot_answer.append(fmean(effect[0] for effect in effects))
            boot_readout.append(fmean(effect[1] for effect in effects))
            boot_interaction.append(fmean(effect[2] for effect in effects))
        answer_effect = fmean(c10 - c00 for _, c00, c10, _, _ in rows)
        readout_effect = fmean(c01 - c00 for _, c00, _, c01, _ in rows)
        interaction = fmean(
            c11 - c10 - c01 + c00 for _, c00, c10, c01, c11 in rows
        )
        result[shift] = {
            "complete_item_seed_rows": len(rows),
            "total_item_seed_rows": total_pairs,
            "valid_only_denominator": len(rows),
            "answer_path_effect": answer_effect,
            "confidence_readout_effect": readout_effect,
            "interaction_effect": interaction,
            "total_coupled_confidence_change": answer_effect
            + readout_effect
            + interaction,
            "answer_path_item_bootstrap_95": _interval(boot_answer),
            "confidence_readout_item_bootstrap_95": _interval(boot_readout),
            "interaction_item_bootstrap_95": _interval(boot_interaction),
        }
    return result


def analyze_matrix(
    config_path: Path,
    analysis_plan_path: Path,
    repository_root: Path,
) -> tuple[Path, dict]:
    """Analyze immutable saved outputs; this module never constructs a backend."""
    config_path = config_path.resolve()
    repository_root = repository_root.resolve()
    config = load_matrix_config(config_path)
    if config.study != "development_matrix":
        raise ValueError("matrix analysis currently accepts development_matrix only")
    plan = load_analysis_plan(analysis_plan_path.resolve())
    storage = StorageLayout.from_spec(config.storage, repository_root)
    run = storage.run(config.run_id)
    manifests = _load_replica_manifests(run, config, repository_root)
    manifest = manifests[0]
    paths = sorted(run.parsed.glob("*.json"))
    records = [MatrixParsedRecord.model_validate(read_json(path)) for path in paths]
    records = _resolve_official_scores(run, records, paths)
    if any(record.correct is None for record in records):
        raise RuntimeError("matrix analysis requires completed benchmark scoring")
    design = config.inference.design
    cells_per_item_seed = 0
    for condition in config.inference.answer_conditions:
        readout_ids = {
            design.coupled_readout_by_answer[condition.condition_id],
            design.standardized_readout_id,
            design.reference_confidence_readout_id,
        }
        if condition.condition_id == design.reference_answer_condition_id:
            readout_ids.update(design.report_only_readout_ids)
        cells_per_item_seed += len(readout_ids)
    all_items = load_dataset(config.dataset, config_path.parent, storage.dataset_cache)
    selected_items = partition_items(all_items, config.dataset)
    expected_cells = (
        cells_per_item_seed * len(selected_items) * len(config.inference.seeds)
    )
    if len(records) != expected_cells:
        raise RuntimeError(
            f"matrix is incomplete: expected {expected_cells} cells, got {len(records)}"
        )
    if any(record.run_id != config.run_id for record in records):
        raise RuntimeError("parsed matrix contains another run ID")

    cluster_by_item, strata_by_cluster, stratum_counts = _resampling_design(
        selected_items, config.dataset.dataset_id
    )
    certification_units = None
    if plan.grouped_certification is not None:
        certification_units = deterministic_balanced_group_anchors(
            selected_items, plan.grouped_certification.selector_salt
        )
    cluster_count = len(set(cluster_by_item.values()))
    condition_ids = [condition.condition_id for condition in config.inference.answer_conditions]
    coupled: dict[str, list[MatrixParsedRecord]] = {
        condition_id: [
            record
            for record in records
            if record.answer_condition_id == condition_id and "coupled" in record.design_roles
        ]
        for condition_id in condition_ids
    }
    standardized: dict[str, list[MatrixParsedRecord]] = {
        condition_id: [
            record
            for record in records
            if record.answer_condition_id == condition_id
            and "standardized" in record.design_roles
        ]
        for condition_id in condition_ids
    }
    reference_id = design.reference_answer_condition_id
    reference_coupled = coupled[reference_id]
    shifted_coupled = {key: value for key, value in coupled.items() if key != reference_id}
    reference_standardized = standardized[reference_id]
    shifted_standardized = {
        key: value for key, value in standardized.items() if key != reference_id
    }
    risk_plan = plan.risk_contract
    risk_reference_coupled = reference_coupled
    risk_reference_standardized = reference_standardized
    if certification_units is not None:
        anchors = certification_units.selected_item_ids
        risk_reference_coupled = [
            record for record in reference_coupled if record.dataset_item_id in anchors
        ]
        risk_reference_standardized = [
            record
            for record in reference_standardized
            if record.dataset_item_id in anchors
        ]
    risk_selection_coupled = select_risk_contract_threshold(
        [_as_parsed(record) for record in risk_reference_coupled],
        risk_plan.target_risk,
        risk_plan.minimum_coverage,
    )
    risk_selection_standardized = select_risk_contract_threshold(
        [_as_parsed(record) for record in risk_reference_standardized],
        risk_plan.target_risk,
        risk_plan.minimum_coverage,
    )
    condition_effects: dict[str, dict] = {}
    for index, condition_id in enumerate(condition_ids):
        if condition_id == reference_id:
            continue
        reference_mean = _mean_confidence(reference_coupled)
        shifted_mean = _mean_confidence(coupled[condition_id])
        condition_effects[condition_id] = {
            "delta_accuracy": _accuracy(coupled[condition_id])
            - _accuracy(reference_coupled),
            "delta_accuracy_item_bootstrap_95": _interval(
                _bootstrap_paired(
                    reference_coupled,
                    coupled[condition_id],
                    _accuracy,
                    plan.bootstrap.replicates,
                    plan.bootstrap.seed + index,
                    cluster_by_item,
                    strata_by_cluster,
                )
            ),
            "delta_mean_confidence_valid_only": (
                shifted_mean - reference_mean
                if shifted_mean is not None and reference_mean is not None
                else None
            ),
            "delta_mean_confidence_item_bootstrap_95": _interval(
                _bootstrap_paired(
                    reference_coupled,
                    coupled[condition_id],
                    _mean_confidence,
                    plan.bootstrap.replicates,
                    plan.bootstrap.seed + 100 + index,
                    cluster_by_item,
                    strata_by_cluster,
                )
            ),
        }
    result = {
        "schema_version": "2.0",
        "analysis_id": f"{config.experiment_id}-analysis-v2",
        "exploratory": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": config.run_id,
        "generation_git_commit": manifest["git"]["commit"],
        "analysis_git": git_metadata(repository_root),
        "config_sha256": config.config_sha256,
        "analysis_plan_id": plan.plan_id,
        "analysis_plan_sha256": plan.sha256,
        "experimental_units": cluster_count,
        "replicate_seeds": list(config.inference.seeds),
        "parsed_cells": len(records),
        "invalid_confidence_cells": sum(record.confidence is None for record in records),
        "bootstrap": {
            "replicates": plan.bootstrap.replicates,
            "unit": (
                "dataset_group_id"
                if any(item.group_id is not None for item in selected_items)
                else "original_benchmark_item"
            ),
            "stratified": strata_by_cluster is not None,
            "selected_subfields": len(stratum_counts),
            "singleton_subfields": sum(count == 1 for count in stratum_counts.values()),
            "rationale": (
                "discipline/subfield stratification"
                if strata_by_cluster is not None
                else (
                    "unstratified grouped resampling; all variants of each source group travel together"
                    if any(item.group_id is not None for item in selected_items)
                    else (
                        "unstratified in this 64-item pilot because singleton subfields make stratified resampling degenerate"
                        if config.dataset.dataset_id == "m-a-p/SuperGPQA"
                        and len(selected_items) == 64
                        else "unstratified item resampling because strata are absent or contain singleton items"
                    )
                )
            ),
        },
        "risk_contract_unit_selection": (
            {
                "method": certification_units.method,
                "selector_salt": certification_units.selector_salt,
                "source_groups": certification_units.source_groups,
                "selected_rows": len(certification_units.selected_item_ids),
                "variant_counts": certification_units.variant_counts,
            }
            if certification_units is not None
            else {
                "method": "legacy_all_rows",
                "source_groups": cluster_count,
                "selected_rows": len(selected_items),
            }
        ),
        "coupled": {
            "summaries": {
                key: _summary(value, cluster_by_item) for key, value in coupled.items()
            },
            "reference_risk_contract_selection": asdict(risk_selection_coupled),
            "condition_effects": condition_effects,
            "fixed_coverage": _fixed_coverage_analysis(
                reference_coupled,
                shifted_coupled,
                plan.fixed_coverage.targets,
                plan.bootstrap.replicates,
                plan.bootstrap.seed + 1_000,
                cluster_by_item,
                strata_by_cluster,
            ),
        },
        "standardized_deterministic": {
            "summaries": {
                key: _summary(value, cluster_by_item)
                for key, value in standardized.items()
            },
            "reference_risk_contract_selection": asdict(risk_selection_standardized),
            "fixed_coverage": _fixed_coverage_analysis(
                reference_standardized,
                shifted_standardized,
                plan.fixed_coverage.targets,
                plan.bootstrap.replicates,
                plan.bootstrap.seed + 2_000,
                cluster_by_item,
                strata_by_cluster,
            ),
        },
        "decomposition": _decomposition(
            records,
            config,
            plan.bootstrap.replicates,
            plan.bootstrap.seed + 3_000,
            cluster_by_item,
            strata_by_cluster,
        ),
    }
    output_path = run.statistics / "development-matrix-analysis-v2.json"
    if output_path.exists():
        existing = read_json(output_path)
        # Creation time and analysis-tree dirtiness are provenance, so a rerun is
        # allowed only when the substantive payload is identical.
        comparable = dict(result)
        comparable["created_at_utc"] = existing.get("created_at_utc")
        comparable["analysis_git"] = existing.get("analysis_git")
        if existing != comparable:
            raise RuntimeError(f"analysis result collision at {output_path}")
        return output_path, existing
    write_immutable_json(output_path, result)
    return output_path, result
