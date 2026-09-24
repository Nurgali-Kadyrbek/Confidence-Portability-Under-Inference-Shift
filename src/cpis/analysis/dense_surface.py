"""Cluster-robust fractional-logit/logistic dense response surfaces."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping

from cpis.analysis.linalg import inverse, matvec, weighted_crossproduct, weighted_rhs


@dataclass(frozen=True)
class DenseSurfaceRow:
    outcome: float
    temperature: float
    top_p: float
    cluster_id: str
    categories: Mapping[str, str]


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def _logit(value: float, epsilon: float) -> float:
    bounded = min(1 - epsilon, max(epsilon, value))
    return math.log(bounded / (1 - bounded))


def _design(
    rows: list[DenseSurfaceRow],
    reference_temperature: float,
    reference_top_p: float,
    categorical_predictors: tuple[str, ...],
    top_p_logit_epsilon: float,
) -> tuple[list[str], list[list[float]], dict[str, str]]:
    names = [
        "intercept",
        "centered_temperature",
        "centered_logit_top_p",
        "temperature_squared",
        "logit_top_p_squared",
        "temperature_by_logit_top_p",
    ]
    references: dict[str, str] = {}
    levels_by_name: dict[str, list[str]] = {}
    for name in categorical_predictors:
        levels = sorted({row.categories[name] for row in rows})
        references[name] = levels[0]
        levels_by_name[name] = levels
        names.extend(f"{name}[{level}]" for level in levels[1:])
    ref_p = _logit(reference_top_p, top_p_logit_epsilon)
    design: list[list[float]] = []
    for row in rows:
        centered_t = row.temperature - reference_temperature
        centered_p = _logit(row.top_p, top_p_logit_epsilon) - ref_p
        values = [
            1.0,
            centered_t,
            centered_p,
            centered_t * centered_t,
            centered_p * centered_p,
            centered_t * centered_p,
        ]
        for name in categorical_predictors:
            values.extend(
                float(row.categories[name] == level)
                for level in levels_by_name[name][1:]
            )
        design.append(values)
    return names, design, references


def fit_dense_surface(
    rows: list[DenseSurfaceRow],
    *,
    reference_temperature: float,
    reference_top_p: float,
    categorical_predictors: tuple[str, ...] = (),
    top_p_logit_epsilon: float = 0.000001,
) -> dict:
    """Fit a logit-link mean surface with item-clustered sandwich uncertainty."""

    if not rows:
        raise ValueError("dense response surface requires observations")
    if not 0 < reference_top_p <= 1:
        raise ValueError("reference top_p must lie in (0, 1]")
    if not 0 < top_p_logit_epsilon < 0.01:
        raise ValueError("top_p logit epsilon must lie in (0, 0.01)")
    if any(not 0 <= row.outcome <= 1 for row in rows):
        raise ValueError("fractional-logit outcomes must lie in [0, 1]")
    if any(not 0 < row.top_p <= 1 for row in rows):
        raise ValueError("dense surface top_p coordinates must lie in (0, 1]")
    if any(name not in row.categories for row in rows for name in categorical_predictors):
        raise ValueError("every row must declare every categorical predictor")
    names, design, references = _design(
        rows,
        reference_temperature,
        reference_top_p,
        categorical_predictors,
        top_p_logit_epsilon,
    )
    outcomes = [row.outcome for row in rows]
    if len(rows) <= len(names):
        raise ValueError("dense surface has insufficient observations for its design")
    mean = min(1 - 1e-6, max(1e-6, sum(outcomes) / len(outcomes)))
    beta = [math.log(mean / (1 - mean)), *([0.0] * (len(names) - 1))]
    information: list[list[float]] | None = None
    probabilities: list[float] = []
    converged = False
    iterations = 0
    for iterations in range(1, 101):
        eta = [max(-30.0, min(30.0, value)) for value in matvec(design, beta)]
        probabilities = [1 / (1 + math.exp(-value)) for value in eta]
        weights = [max(1e-9, value * (1 - value)) for value in probabilities]
        working = [
            linear + (outcome - probability) / weight
            for linear, outcome, probability, weight in zip(
                eta, outcomes, probabilities, weights, strict=True
            )
        ]
        information = weighted_crossproduct(design, weights)
        covariance_model = inverse(information)
        updated = matvec(
            covariance_model, weighted_rhs(design, weights, working)
        )
        delta = max(abs(new - old) for new, old in zip(updated, beta, strict=True))
        beta = updated
        if delta < 1e-9:
            converged = True
            break
    if not converged or information is None:
        return {"status": "nonconvergent", "iterations": iterations}

    # Recompute at the converged coefficients before forming cluster scores.
    eta = [max(-30.0, min(30.0, value)) for value in matvec(design, beta)]
    probabilities = [1 / (1 + math.exp(-value)) for value in eta]
    weights = [max(1e-9, value * (1 - value)) for value in probabilities]
    bread = inverse(weighted_crossproduct(design, weights))
    cluster_scores: dict[str, list[float]] = {}
    for row, x, outcome, probability in zip(rows, design, outcomes, probabilities, strict=True):
        score = cluster_scores.setdefault(row.cluster_id, [0.0] * len(names))
        residual = outcome - probability
        for index, value in enumerate(x):
            score[index] += value * residual
    meat = [
        [
            sum(score[left] * score[right] for score in cluster_scores.values())
            for right in range(len(names))
        ]
        for left in range(len(names))
    ]
    robust = [
        [
            sum(
                bread[i][a] * meat[a][b] * bread[b][j]
                for a in range(len(names))
                for b in range(len(names))
            )
            for j in range(len(names))
        ]
        for i in range(len(names))
    ]
    clusters = len(cluster_scores)
    if clusters > 1:
        correction = clusters / (clusters - 1) * (len(rows) - 1) / (len(rows) - len(names))
        robust = [[value * correction for value in row] for row in robust]
    coefficients = {}
    for index, (name, estimate) in enumerate(zip(names, beta, strict=True)):
        standard_error = math.sqrt(max(0.0, robust[index][index]))
        coefficients[name] = {
            "estimate": estimate,
            "cluster_robust_standard_error": standard_error,
            "lower_95": estimate - 1.959963984540054 * standard_error,
            "upper_95": estimate + 1.959963984540054 * standard_error,
        }
    return {
        "status": "estimated",
        "link": "logit",
        "outcome_family": "fractional_or_binary",
        "observations": len(rows),
        "clusters": clusters,
        "iterations": iterations,
        "reference_temperature": reference_temperature,
        "reference_top_p": reference_top_p,
        "top_p_logit_epsilon": top_p_logit_epsilon,
        "reference_categories": references,
        "coefficients": coefficients,
    }


def bootstrap_dense_surface(
    rows: list[DenseSurfaceRow],
    *,
    reference_temperature: float,
    reference_top_p: float,
    replicates: int,
    seed: int,
    categorical_predictors: tuple[str, ...] = (),
    top_p_logit_epsilon: float = 0.000001,
) -> dict:
    """Add percentile uncertainty from resampling complete source clusters."""

    if replicates < 1:
        raise ValueError("dense surface bootstrap requires positive replicates")
    base = fit_dense_surface(
        rows,
        reference_temperature=reference_temperature,
        reference_top_p=reference_top_p,
        categorical_predictors=categorical_predictors,
        top_p_logit_epsilon=top_p_logit_epsilon,
    )
    if base["status"] != "estimated":
        return {**base, "item_clustered_bootstrap": {"successful_replicates": 0}}
    by_cluster: dict[str, list[DenseSurfaceRow]] = {}
    for row in rows:
        by_cluster.setdefault(row.cluster_id, []).append(row)
    cluster_ids = sorted(by_cluster)
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {
        name: [] for name in base["coefficients"]
    }
    for _ in range(replicates):
        sample = rng.choices(cluster_ids, k=len(cluster_ids))
        sampled_rows: list[DenseSurfaceRow] = []
        for draw_index, cluster_id in enumerate(sample):
            sampled_rows.extend(
                DenseSurfaceRow(
                    outcome=row.outcome,
                    temperature=row.temperature,
                    top_p=row.top_p,
                    cluster_id=f"draw-{draw_index}:{cluster_id}",
                    categories=row.categories,
                )
                for row in by_cluster[cluster_id]
            )
        try:
            fitted = fit_dense_surface(
                sampled_rows,
                reference_temperature=reference_temperature,
                reference_top_p=reference_top_p,
                categorical_predictors=categorical_predictors,
                top_p_logit_epsilon=top_p_logit_epsilon,
            )
        except ValueError:
            # A cluster resample can be rank deficient even when the observed
            # complete grid is estimable. It is a failed replicate, not a
            # reason to change the observed design matrix.
            continue
        if fitted["status"] != "estimated":
            continue
        for name in draws:
            draws[name].append(fitted["coefficients"][name]["estimate"])
    return {
        **base,
        "uncertainty": "item_clustered_bootstrap",
        "item_clustered_bootstrap": {
            "requested_replicates": replicates,
            "successful_replicates": min(len(values) for values in draws.values()),
            "seed": seed,
            "coefficient_intervals": {
                name: {
                    "lower_95": _quantile(values, 0.025),
                    "upper_95": _quantile(values, 0.975),
                }
                for name, values in draws.items()
            },
        },
    }
