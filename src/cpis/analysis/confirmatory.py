"""Confirmatory test analysis using only development-frozen policy coordinates."""

from __future__ import annotations

import math
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Callable

from cpis.analysis.matrix import (
    _accuracy,
    _as_parsed,
    _auroc,
    _bootstrap_paired,
    _coverage_metric,
    _decomposition,
    _interval,
    _load_replica_manifests,
    _mean_confidence,
    _quantile,
    _resampling_design,
    _resolve_official_scores,
    _risk_metric,
    _summary,
)
from cpis.analysis.metrics import (
    FixedCoverageThreshold,
    confidence_accepts,
    has_valid_confidence,
    holm_adjust,
    risk_contract_violation,
    transport_fixed_coverage_threshold,
)
from cpis.analysis_plan import load_analysis_plan
from cpis.datasets import load_dataset, partition_items
from cpis.manifest import git_metadata
from cpis.matrix_config import expected_confidence_cells, MatrixExperimentConfig, load_matrix_config
from cpis.matrix_records import MatrixParsedRecord
from cpis.policy import (
    FrozenCoverageThreshold,
    validate_certification_decision_reference,
    validate_policy_reference,
)
from cpis.phase_gate import (
    authorize_phase_analysis_plan,
    authorize_phase_config,
    record_dataset_access,
    require_phase_gate,
)
from cpis.storage import StorageLayout, read_json, write_immutable_json


def _calibration_logistic(
    records: list[MatrixParsedRecord], epsilon: float
) -> dict:
    valid = [record for record in records if has_valid_confidence(record)]
    if not valid or len({bool(record.correct) for record in valid}) < 2:
        return {
            "status": "undefined_no_outcome_variation",
            "intercept": None,
            "slope": None,
            "iterations": 0,
            "epsilon": epsilon,
        }
    x = [
        math.log(
            min(1 - epsilon, max(epsilon, float(record.confidence)))
            / (1 - min(1 - epsilon, max(epsilon, float(record.confidence))))
        )
        for record in valid
    ]
    y = [float(record.correct) for record in valid]
    prevalence = min(1 - epsilon, max(epsilon, fmean(y)))
    intercept = math.log(prevalence / (1 - prevalence))
    slope = 1.0
    for iteration in range(1, 101):
        linear = [max(-40.0, min(40.0, intercept + slope * value)) for value in x]
        probabilities = [1 / (1 + math.exp(-value)) for value in linear]
        weights = [probability * (1 - probability) for probability in probabilities]
        g0 = math.fsum(target - probability for target, probability in zip(y, probabilities))
        g1 = math.fsum(
            (target - probability) * value
            for target, probability, value in zip(y, probabilities, x)
        )
        h00 = math.fsum(weights)
        h01 = math.fsum(weight * value for weight, value in zip(weights, x))
        h11 = math.fsum(weight * value * value for weight, value in zip(weights, x))
        determinant = h00 * h11 - h01 * h01
        if determinant <= 1e-12:
            return {
                "status": "undefined_singular_or_separated",
                "intercept": None,
                "slope": None,
                "iterations": iteration,
                "epsilon": epsilon,
            }
        delta_intercept = (h11 * g0 - h01 * g1) / determinant
        delta_slope = (-h01 * g0 + h00 * g1) / determinant
        step_scale = min(
            1.0,
            5.0 / max(abs(delta_intercept), abs(delta_slope), 5.0),
        )
        intercept += step_scale * delta_intercept
        slope += step_scale * delta_slope
        if max(abs(delta_intercept), abs(delta_slope)) < 1e-8:
            return {
                "status": "estimated",
                "intercept": intercept,
                "slope": slope,
                "iterations": iteration,
                "epsilon": epsilon,
            }
        if max(abs(intercept), abs(slope)) > 100:
            break
    return {
        "status": "undefined_nonconvergent_or_separated",
        "intercept": None,
        "slope": None,
        "iterations": 100,
        "epsilon": epsilon,
    }


def _confidence_diagnostics(
    records: list[MatrixParsedRecord], bins: int, calibration_epsilon: float
) -> dict:
    valid = [record for record in records if has_valid_confidence(record)]
    if not valid:
        return {
            "ece_valid_only": None,
            "ece_bins": bins,
            "aurc_operational_to_max_valid_coverage": None,
            "aurc_valid_only_normalized": None,
            "maximum_valid_coverage": 0.0,
            "calibration_logistic": _calibration_logistic(records, calibration_epsilon),
        }
    bin_rows: list[list[MatrixParsedRecord]] = [[] for _ in range(bins)]
    for record in valid:
        confidence = float(record.confidence)
        index = min(bins - 1, int(confidence * bins))
        bin_rows[index].append(record)
    ece = sum(
        len(rows)
        / len(valid)
        * abs(fmean(float(row.confidence) for row in rows) - fmean(row.correct for row in rows))
        for rows in bin_rows
        if rows
    )
    normalized_aurc = _aurc_valid_only(valid)
    assert normalized_aurc is not None
    return {
        "ece_valid_only": ece,
        "ece_bins": bins,
        "aurc_operational_to_max_valid_coverage": normalized_aurc
        * len(valid)
        / len(records),
        "aurc_valid_only_normalized": normalized_aurc,
        "maximum_valid_coverage": len(valid) / len(records),
        "calibration_logistic": _calibration_logistic(records, calibration_epsilon),
    }


def _aurc_valid_only(records: list[MatrixParsedRecord]) -> float | None:
    """Tie-averaged AURC among numeric confidences.

    Confidence reports are discrete and often heavily tied. Within each tied
    confidence group, this computes the expected cumulative risk under a
    uniformly random ordering rather than allowing item IDs to determine AURC.
    """

    groups: dict[float, list[int]] = {}
    for record in records:
        if not has_valid_confidence(record):
            continue
        counts = groups.setdefault(float(record.confidence), [0, 0])
        counts[0] += 1
        counts[1] += int(not record.correct)
    total_valid = sum(counts[0] for counts in groups.values())
    if not total_valid:
        return None
    cumulative_count = 0
    cumulative_errors = 0
    risk_sum = 0.0
    for confidence in sorted(groups, reverse=True):
        count, errors = groups[confidence]
        error_rate = errors / count
        for offset in range(1, count + 1):
            risk_sum += (
                cumulative_errors + offset * error_rate
            ) / (cumulative_count + offset)
        cumulative_count += count
        cumulative_errors += errors
    return risk_sum / total_valid


def _brier_valid_only(records: list[MatrixParsedRecord]) -> float | None:
    valid = [record for record in records if has_valid_confidence(record)]
    if not valid:
        return None
    return fmean(
        (float(record.confidence) - int(bool(record.correct))) ** 2
        for record in valid
    )


def _invalid_confidence_rate(records: list[MatrixParsedRecord]) -> float:
    return fmean(not has_valid_confidence(record) for record in records)


setattr(_brier_valid_only, "_cpis_metric_kind", "brier")
setattr(_invalid_confidence_rate, "_cpis_metric_kind", "invalid_confidence_rate")


def _confidence_portability(
    reference: list[MatrixParsedRecord],
    shifted: dict[str, list[MatrixParsedRecord]],
    plan,
    cluster_by_item: dict[str, str],
    strata_by_cluster: dict[str, str] | None,
    seed_offset: int,
) -> dict:
    """Estimate paired ranking/risk-coverage/calibration transfer effects."""

    metrics = {
        "auroc_valid_only": _auroc,
        "aurc_valid_only_normalized": _aurc_valid_only,
        "brier_valid_only": _brier_valid_only,
        "invalid_confidence_rate": _invalid_confidence_rate,
    }
    result: dict[str, dict] = {}
    for condition_index, (condition_id, rows) in enumerate(shifted.items()):
        effects: dict[str, dict] = {}
        for metric_index, (metric_name, metric) in enumerate(metrics.items()):
            reference_value = metric(reference)
            shifted_value = metric(rows)
            values = _bootstrap_paired(
                reference,
                rows,
                metric,
                plan.bootstrap.replicates,
                plan.bootstrap.seed
                + seed_offset
                + 10_000 * condition_index
                + 1_000 * metric_index,
                cluster_by_item,
                strata_by_cluster,
            )
            effects[metric_name] = {
                "reference": reference_value,
                "shifted": shifted_value,
                "delta": (
                    shifted_value - reference_value
                    if shifted_value is not None and reference_value is not None
                    else None
                ),
                "delta_item_bootstrap_95": _interval(values),
                "delta_item_bootstrap_variance": _sample_variance(values),
            }
        result[condition_id] = effects
    return result


def _endpoint(records: list[MatrixParsedRecord], threshold: float) -> dict:
    accepted = [
        record
        for record in records
        if confidence_accepts(record, threshold)
    ]
    risk = fmean(not record.correct for record in accepted) if accepted else None
    return {
        "threshold": threshold,
        "coverage": len(accepted) / len(records),
        "risk": risk,
        "accepted_observations": len(accepted),
        "total_observations": len(records),
    }


def _interval_at(values: list[float], level: float) -> dict:
    tail = (1 - level) / 2
    return {
        "level": level,
        "lower": _quantile(values, tail),
        "upper": _quantile(values, 1 - tail),
        "successful_replicates": len(values),
    }


def _sample_variance(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    center = fmean(values)
    return math.fsum((value - center) ** 2 for value in values) / (len(values) - 1)


def _randomization_p_value(
    reference: list[MatrixParsedRecord],
    shifted: list[MatrixParsedRecord],
    threshold: float,
    cluster_by_item: dict[str, str],
    replicates: int,
    seed: int,
    two_sided: bool,
) -> dict:
    def aggregate(records: list[MatrixParsedRecord]) -> tuple[int, int, int]:
        accepted = [
            record
            for record in records
            if confidence_accepts(record, threshold)
        ]
        return len(accepted), sum(not record.correct for record in accepted), len(records)

    by_ref: dict[str, list[MatrixParsedRecord]] = {}
    by_shift: dict[str, list[MatrixParsedRecord]] = {}
    for record in reference:
        by_ref.setdefault(cluster_by_item[record.dataset_item_id], []).append(record)
    for record in shifted:
        by_shift.setdefault(cluster_by_item[record.dataset_item_id], []).append(record)
    if set(by_ref) != set(by_shift):
        raise RuntimeError("randomization test requires paired source clusters")
    clusters = sorted(by_ref)
    sufficient = {cluster: (aggregate(by_ref[cluster]), aggregate(by_shift[cluster])) for cluster in clusters}

    def delta(left: list[tuple[int, int, int]], right: list[tuple[int, int, int]]) -> float | None:
        left_accepted = sum(value[0] for value in left)
        right_accepted = sum(value[0] for value in right)
        if not left_accepted or not right_accepted:
            return None
        return (
            sum(value[1] for value in right) / right_accepted
            - sum(value[1] for value in left) / left_accepted
        )

    observed = delta(
        [sufficient[cluster][0] for cluster in clusters],
        [sufficient[cluster][1] for cluster in clusters],
    )
    if observed is None:
        return {
            "p_value": 1.0,
            "observed_delta_risk": None,
            "valid_randomizations": 0,
            "reason": "zero accepted observations in at least one arm",
        }
    rng = random.Random(seed)
    extreme = 0
    valid = 0
    for _ in range(replicates):
        left: list[tuple[int, int, int]] = []
        right: list[tuple[int, int, int]] = []
        for cluster in clusters:
            ref_value, shift_value = sufficient[cluster]
            if rng.getrandbits(1):
                left.append(shift_value)
                right.append(ref_value)
            else:
                left.append(ref_value)
                right.append(shift_value)
        permuted = delta(left, right)
        if permuted is None:
            continue
        valid += 1
        if (abs(permuted) >= abs(observed)) if two_sided else (permuted >= observed):
            extreme += 1
    return {
        "p_value": (extreme + 1) / (valid + 1),
        "observed_delta_risk": observed,
        "valid_randomizations": valid,
        "alternative": "two_sided_change" if two_sided else "one_sided_increase",
        "reason": None,
    }


def _frozen_coverage_analysis(
    reference: list[MatrixParsedRecord],
    shifted: dict[str, list[MatrixParsedRecord]],
    policies: tuple[FrozenCoverageThreshold, ...],
    config: MatrixExperimentConfig,
    plan,
    cluster_by_item: dict[str, str],
    strata_by_cluster: dict[str, str] | None,
) -> dict:
    result: dict[str, dict] = {}
    for target_index, frozen in enumerate(policies):
        target = f"{frozen.target_coverage:.2f}"
        result[target] = {
            "frozen_development_policy": frozen.model_dump(mode="json"),
            "reference": None,
            "transport": {},
        }
        if frozen.status != "selected" or frozen.threshold is None:
            continue
        selected = FixedCoverageThreshold(
            status="selected",
            target_coverage=frozen.target_coverage,
            threshold=frozen.threshold,
            achieved_coverage=frozen.achieved_development_coverage,
            reference_risk=frozen.development_risk,
        )
        result[target]["reference"] = _endpoint(reference, frozen.threshold)
        for condition_index, (condition_id, records) in enumerate(shifted.items()):
            point = transport_fixed_coverage_threshold(
                [_as_parsed(record) for record in reference],
                [_as_parsed(record) for record in records],
                selected,
            )
            pair_seed = plan.bootstrap.seed + 10_000 * target_index + condition_index
            risk_values = _bootstrap_paired(
                reference,
                records,
                _risk_metric(frozen.threshold),
                plan.bootstrap.replicates,
                pair_seed,
                cluster_by_item,
                strata_by_cluster,
            )
            coverage_values = _bootstrap_paired(
                reference,
                records,
                _coverage_metric(frozen.threshold),
                plan.bootstrap.replicates,
                pair_seed + 5_000,
                cluster_by_item,
                strata_by_cluster,
            )
            risk_90 = _interval_at(risk_values, plan.equivalence_testing.interval_level)
            coverage_90 = _interval_at(
                coverage_values, plan.equivalence_testing.interval_level
            )
            result[target]["transport"][condition_id] = {
                **asdict(point),
                "delta_risk_item_bootstrap_95": _interval(risk_values),
                "delta_risk_item_bootstrap_90": risk_90,
                "delta_risk_item_bootstrap_variance": _sample_variance(risk_values),
                "delta_coverage_item_bootstrap_95": _interval(coverage_values),
                "delta_coverage_item_bootstrap_90": coverage_90,
                "risk_equivalent": (
                    risk_90["lower"] is not None
                    and risk_90["upper"] is not None
                    and risk_90["lower"] > -plan.equivalence_margins.selective_risk
                    and risk_90["upper"] < plan.equivalence_margins.selective_risk
                ),
                "coverage_equivalent": (
                    coverage_90["lower"] is not None
                    and coverage_90["upper"] is not None
                    and coverage_90["lower"] > -plan.equivalence_margins.coverage
                    and coverage_90["upper"] < plan.equivalence_margins.coverage
                ),
            }
    return result


def _certified_risk_analysis(
    reference: list[MatrixParsedRecord],
    shifted: dict[str, list[MatrixParsedRecord]],
    decision,
    target_risk: float,
    plan,
    cluster_by_item: dict[str, str],
    strata_by_cluster: dict[str, str] | None,
) -> dict:
    if decision.status != "certified" or decision.threshold is None:
        return {
            "status": "risk-contract infeasible",
            "reason": decision.reason,
            "reference": None,
            "transport": {},
        }
    threshold = decision.threshold
    reference_point = _endpoint(reference, threshold)
    transport: dict[str, dict] = {}
    for index, (condition_id, records) in enumerate(shifted.items()):
        shifted_point = _endpoint(records, threshold)
        risk_values = _bootstrap_paired(
            reference,
            records,
            _risk_metric(threshold),
            plan.bootstrap.replicates,
            plan.bootstrap.seed + 40_000 + index,
            cluster_by_item,
            strata_by_cluster,
        )
        coverage_values = _bootstrap_paired(
            reference,
            records,
            _coverage_metric(threshold),
            plan.bootstrap.replicates,
            plan.bootstrap.seed + 45_000 + index,
            cluster_by_item,
            strata_by_cluster,
        )
        shifted_risk = shifted_point["risk"]
        reference_risk = reference_point["risk"]
        risk_90 = _interval_at(risk_values, plan.equivalence_testing.interval_level)
        risk_95 = _interval(risk_values)
        transport[condition_id] = {
            "shifted": shifted_point,
            "delta_risk": (
                shifted_risk - reference_risk
                if shifted_risk is not None and reference_risk is not None
                else None
            ),
            "risk_violation": (
                risk_contract_violation(shifted_risk, target_risk)
                if shifted_risk is not None
                else None
            ),
            "delta_coverage": shifted_point["coverage"] - reference_point["coverage"],
            "delta_risk_item_bootstrap_95": risk_95,
            "delta_risk_item_bootstrap_90": risk_90,
            "delta_risk_item_bootstrap_variance": _sample_variance(risk_values),
            "delta_coverage_item_bootstrap_95": _interval(coverage_values),
            "risk_equivalent": (
                risk_90["lower"] is not None
                and risk_90["upper"] is not None
                and risk_90["lower"] > -plan.equivalence_margins.selective_risk
                and risk_90["upper"] < plan.equivalence_margins.selective_risk
            ),
            # The upper endpoint of a two-sided 90% interval is the one-sided
            # 95% bound used for the prespecified noninferiority decision.
            "risk_noninferior": (
                risk_90["upper"] is not None
                and risk_90["upper"] <= plan.equivalence_margins.selective_risk
            ),
        }
    return {
        "status": "certified",
        "target_risk": target_risk,
        "threshold": threshold,
        "reference": reference_point,
        "transport": transport,
    }


def _same_answer_analysis(
    reference: list[MatrixParsedRecord],
    shifted: dict[str, list[MatrixParsedRecord]],
) -> dict:
    reference_by_key = {(record.dataset_item_id, record.seed): record for record in reference}
    result = {}
    for condition_id, records in shifted.items():
        shifted_by_key = {(record.dataset_item_id, record.seed): record for record in records}
        if set(reference_by_key) != set(shifted_by_key):
            raise RuntimeError("same-answer analysis requires paired item/seed rows")
        pairs = [
            (reference_by_key[key], shifted_by_key[key]) for key in sorted(reference_by_key)
        ]
        same = [
            pair
            for pair in pairs
            if pair[0].answer_parse_status == "valid"
            and pair[1].answer_parse_status == "valid"
            and pair[0].parsed_answer == pair[1].parsed_answer
        ]
        confidence_differences = [
            float(shift.confidence) - float(ref.confidence)
            for ref, shift in same
            if has_valid_confidence(ref) and has_valid_confidence(shift)
        ]
        result[condition_id] = {
            "paired_rows": len(pairs),
            "same_answer_rows": len(same),
            "same_answer_rate": len(same) / len(pairs),
            "reference_accuracy_same_answer": (
                fmean(ref.correct for ref, _ in same) if same else None
            ),
            "shifted_accuracy_same_answer": (
                fmean(shift.correct for _, shift in same) if same else None
            ),
            "delta_mean_confidence_valid_same_answer": (
                fmean(confidence_differences) if confidence_differences else None
            ),
            "valid_confidence_pairs": len(confidence_differences),
        }
    return result


def _expected_cells(config: MatrixExperimentConfig, item_count: int) -> int:
    return expected_confidence_cells(config, item_count)


def analyze_confirmatory_test(
    config_path: Path, analysis_plan_path: Path, repository_root: Path
) -> tuple[Path, dict]:
    repository_root = repository_root.resolve()
    config_path = config_path.resolve()
    config = load_matrix_config(config_path)
    if config.study not in {"broad", "factorial", "decomposition"}:
        raise ValueError("confirmatory test analysis requires a test matrix")
    if config.dataset.partition != "test" or not config.confirmatory:
        raise ValueError("confirmatory test analysis cannot read non-test data")
    if config.frozen_policy is None or config.certification_decision is None:
        raise ValueError("test analysis requires frozen policy and certification decision")
    gate = require_phase_gate("test", repository_root)
    authorize_phase_config(gate, config_path, repository_root)
    plan = load_analysis_plan(analysis_plan_path.resolve())
    authorize_phase_analysis_plan(gate, plan.sha256)
    if plan.schema_version not in {"3.0", "4.0"} or plan.robustness is None:
        raise ValueError("confirmatory test analysis requires analysis plan v3+")
    policy = validate_policy_reference(config.frozen_policy, repository_root)
    decision = validate_certification_decision_reference(
        config.certification_decision, repository_root
    )
    if (
        policy.analysis_plan_sha256 != plan.sha256
        or decision.analysis_plan_sha256 != plan.sha256
        or decision.frozen_policy_sha256 != policy.canonical_sha256
    ):
        raise RuntimeError("test artifacts do not share one frozen analysis plan")

    storage = StorageLayout.from_spec(config.storage, repository_root)
    run = storage.run(config.run_id)
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
    paths = sorted(run.parsed.glob("*.json"))
    records = [MatrixParsedRecord.model_validate(read_json(path)) for path in paths]
    records = _resolve_official_scores(run, records, paths)
    if any(record.correct is None for record in records):
        raise RuntimeError("test analysis requires completed deterministic scoring")
    all_items = load_dataset(config.dataset, config_path.parent, storage.dataset_cache)
    items = partition_items(all_items, config.dataset)
    expected = _expected_cells(config, len(items))
    if len(records) != expected:
        raise RuntimeError(f"test matrix is incomplete: expected {expected}, got {len(records)}")
    cluster_by_item, strata_by_cluster, stratum_counts = _resampling_design(
        items, config.dataset.dataset_id
    )
    design = config.inference.design
    condition_ids = [condition.condition_id for condition in config.inference.answer_conditions]
    expected_conditions = {
        "reference",
        "low-temperature",
        "high-temperature",
        "low-top-p",
        "high-diversity",
    }
    if set(condition_ids) != expected_conditions:
        raise RuntimeError("primary test matrix does not contain the four frozen contrasts")
    coupled = {
        condition_id: [
            record
            for record in records
            if record.answer_condition_id == condition_id and "coupled" in record.design_roles
        ]
        for condition_id in condition_ids
    }
    standardized = {
        condition_id: [
            record
            for record in records
            if record.answer_condition_id == condition_id
            and "standardized" in record.design_roles
        ]
        for condition_id in condition_ids
    }
    reference_id = design.reference_answer_condition_id
    shifted_coupled = {key: value for key, value in coupled.items() if key != reference_id}
    shifted_standardized = {
        key: value for key, value in standardized.items() if key != reference_id
    }
    coupled_risk = _certified_risk_analysis(
        coupled[reference_id],
        shifted_coupled,
        decision.coupled,
        plan.risk_contract.target_risk,
        plan,
        cluster_by_item,
        strata_by_cluster,
    )
    standardized_risk = _certified_risk_analysis(
        standardized[reference_id],
        shifted_standardized,
        decision.standardized_deterministic,
        plan.risk_contract.target_risk,
        plan,
        cluster_by_item,
        strata_by_cluster,
    )
    coupled_fixed = _frozen_coverage_analysis(
        coupled[reference_id],
        shifted_coupled,
        policy.coupled.fixed_coverage,
        config,
        plan,
        cluster_by_item,
        strata_by_cluster,
    )
    standardized_fixed = _frozen_coverage_analysis(
        standardized[reference_id],
        shifted_standardized,
        policy.standardized_deterministic.fixed_coverage,
        config,
        plan,
        cluster_by_item,
        strata_by_cluster,
    )

    certified = decision.coupled.status == "certified"
    if certified:
        assert decision.coupled.threshold is not None
        test_threshold = decision.coupled.threshold
    else:
        target = f"{plan.fixed_coverage.universal_confirmatory_target:.2f}"
        frozen = next(
            entry
            for entry in policy.coupled.fixed_coverage
            if math.isclose(
                entry.target_coverage,
                plan.fixed_coverage.universal_confirmatory_target,
            )
        )
        test_threshold = frozen.threshold
        if test_threshold is None:
            raise RuntimeError("universal confirmatory fixed-coverage target is unattainable")
    randomization: dict[str, dict] = {}
    for index, condition_id in enumerate(
        ("low-temperature", "high-temperature", "low-top-p", "high-diversity")
    ):
        randomization[condition_id] = _randomization_p_value(
            coupled[reference_id],
            coupled[condition_id],
            test_threshold,
            cluster_by_item,
            plan.multiplicity.randomization_replicates,
            plan.multiplicity.randomization_seed + index,
            two_sided=not certified,
        )
    holm = holm_adjust(
        {condition: value["p_value"] for condition, value in randomization.items()},
        plan.multiplicity.familywise_alpha,
    )

    summaries = {
        condition: {
            **_summary(rows, cluster_by_item),
            **_confidence_diagnostics(
                rows,
                plan.robustness.ece_bins,
                plan.robustness.calibration_sensitivity_epsilon,
            ),
        }
        for condition, rows in coupled.items()
    }
    accuracy_matched = {
        condition: {
            "delta_accuracy": _accuracy(rows) - _accuracy(coupled[reference_id]),
            "within_prespecified_tolerance": abs(
                _accuracy(rows) - _accuracy(coupled[reference_id])
            )
            <= plan.robustness.accuracy_match_absolute_tolerance,
            "absolute_tolerance": plan.robustness.accuracy_match_absolute_tolerance,
        }
        for condition, rows in shifted_coupled.items()
    }
    result = {
        "schema_version": "1.0",
        "analysis_id": f"{config.experiment_id}-confirmatory-test-analysis-v1",
        "confirmatory": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": config.run_id,
        "generation_git_commit": manifests[0]["git"]["commit"],
        "analysis_git": git_metadata(repository_root),
        "config_sha256": config.config_sha256,
        "analysis_plan_id": plan.plan_id,
        "analysis_plan_sha256": plan.sha256,
        "frozen_policy_id": policy.policy_id,
        "frozen_policy_sha256": policy.canonical_sha256,
        "certification_decision_id": decision.decision_id,
        "certification_decision_sha256": decision.canonical_sha256,
        "experimental_units": len(set(cluster_by_item.values())),
        "replicate_seeds": list(config.inference.seeds),
        "parsed_cells": len(records),
        "bootstrap": {
            "replicates": plan.bootstrap.replicates,
            "unit": "source_group" if any(item.group_id for item in items) else "item",
            "stratified": strata_by_cluster is not None,
            "strata": len(stratum_counts),
        },
        "coupled": {
            "summaries": summaries,
            "risk_contract_transport": coupled_risk,
            "fixed_coverage_transport": coupled_fixed,
            "confidence_portability": _confidence_portability(
                coupled[reference_id],
                shifted_coupled,
                plan,
                cluster_by_item,
                strata_by_cluster,
                60_000,
            ),
        },
        "standardized_deterministic": {
            "summaries": {
                condition: {
                    **_summary(rows, cluster_by_item),
                    **_confidence_diagnostics(
                        rows,
                        plan.robustness.ece_bins,
                        plan.robustness.calibration_sensitivity_epsilon,
                    ),
                }
                for condition, rows in standardized.items()
            },
            "risk_contract_transport": standardized_risk,
            "fixed_coverage_transport": standardized_fixed,
            "confidence_portability": _confidence_portability(
                standardized[reference_id],
                shifted_standardized,
                plan,
                cluster_by_item,
                strata_by_cluster,
                160_000,
            ),
        },
        "multiplicity": {
            "endpoint": (
                "certified_risk_threshold"
                if certified
                else f"fixed_coverage_{plan.fixed_coverage.universal_confirmatory_target:.2f}"
            ),
            "threshold": test_threshold,
            "randomization": randomization,
            "holm": holm,
        },
        "robustness": {
            "same_answer": _same_answer_analysis(
                coupled[reference_id], shifted_coupled
            ),
            "accuracy_matched": accuracy_matched,
        },
        "decomposition": _decomposition(
            records,
            config,
            plan.bootstrap.replicates,
            plan.bootstrap.seed + 50_000,
            cluster_by_item,
            strata_by_cluster,
        ),
    }
    output_path = run.statistics / "confirmatory-test-analysis-v1.json"
    if output_path.exists():
        existing = read_json(output_path)
        comparable = dict(result)
        comparable["created_at_utc"] = existing.get("created_at_utc")
        comparable["analysis_git"] = existing.get("analysis_git")
        if existing != comparable:
            raise RuntimeError(f"test analysis collision at {output_path}")
        return output_path, existing
    write_immutable_json(output_path, result)
    return output_path, result
