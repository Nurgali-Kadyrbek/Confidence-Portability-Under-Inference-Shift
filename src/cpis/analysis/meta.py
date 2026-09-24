"""Two-stage REML random-effects meta-analysis for portability effects."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from cpis.analysis.linalg import inverse, matvec, weighted_crossproduct, weighted_rhs


@dataclass(frozen=True)
class MetaEffect:
    effect_id: str
    estimate: float
    variance: float
    moderators: Mapping[str, str]


def _fit_at_tau(
    effects: list[MetaEffect], design: list[list[float]], tau_squared: float
) -> tuple[list[float], list[list[float]], list[float], list[float]]:
    weights = [1 / (effect.variance + tau_squared) for effect in effects]
    information = weighted_crossproduct(design, weights)
    covariance = inverse(information)
    beta = matvec(
        covariance,
        weighted_rhs(design, weights, [effect.estimate for effect in effects]),
    )
    residuals = [
        effect.estimate - sum(value * coefficient for value, coefficient in zip(row, beta, strict=True))
        for effect, row in zip(effects, design, strict=True)
    ]
    return beta, covariance, residuals, weights


def _reml_score(
    effects: list[MetaEffect], design: list[list[float]], tau_squared: float
) -> float:
    _, covariance, residuals, weights = _fit_at_tau(effects, design, tau_squared)
    columns = len(design[0])
    xtw2x = [
        [
            sum(
                weight * weight * row[left] * row[right]
                for row, weight in zip(design, weights, strict=True)
            )
            for right in range(columns)
        ]
        for left in range(columns)
    ]
    leverage_trace = sum(
        covariance[row][column] * xtw2x[column][row]
        for row in range(columns)
        for column in range(columns)
    )
    return sum(
        weight * weight * residual * residual
        for weight, residual in zip(weights, residuals, strict=True)
    ) - (sum(weights) - leverage_trace)


def _reml_tau_squared(
    effects: list[MetaEffect], design: list[list[float]]
) -> tuple[float, int]:
    if _reml_score(effects, design, 0.0) <= 0:
        return 0.0, 0
    high = max(effect.estimate * effect.estimate + effect.variance for effect in effects)
    high = max(high, 1e-8)
    expansions = 0
    while _reml_score(effects, design, high) > 0 and expansions < 60:
        high *= 2
        expansions += 1
    low = 0.0
    iterations = 0
    for iterations in range(1, 101):
        middle = (low + high) / 2
        if _reml_score(effects, design, middle) > 0:
            low = middle
        else:
            high = middle
        if high - low <= 1e-12 * max(1.0, high):
            break
    return (low + high) / 2, iterations


def _design(
    effects: list[MetaEffect], moderator_names: tuple[str, ...]
) -> tuple[list[str], list[list[float]], dict[str, str]]:
    columns = ["intercept"]
    references: dict[str, str] = {}
    levels_by_moderator: dict[str, list[str]] = {}
    for moderator in moderator_names:
        levels = sorted({effect.moderators[moderator] for effect in effects})
        if not levels:
            raise ValueError(f"moderator {moderator!r} has no levels")
        references[moderator] = levels[0]
        levels_by_moderator[moderator] = levels
        columns.extend(f"{moderator}[{level}]" for level in levels[1:])
    rows: list[list[float]] = []
    for effect in effects:
        row = [1.0]
        for moderator in moderator_names:
            levels = levels_by_moderator[moderator]
            row.extend(float(effect.moderators[moderator] == level) for level in levels[1:])
        rows.append(row)
    return columns, rows, references


def reml_meta_analysis(
    effects: list[MetaEffect],
    moderator_names: tuple[str, ...] = (),
    minimum_effects_per_coefficient: int = 10,
) -> dict:
    """Fit the prespecified REML model or fail closed on sparse moderators."""

    if len(effects) < 2:
        raise ValueError("meta-analysis requires at least two effects")
    if any(effect.variance <= 0 or not math.isfinite(effect.variance) for effect in effects):
        raise ValueError("every meta-analytic sampling variance must be positive")
    if any(not math.isfinite(effect.estimate) for effect in effects):
        raise ValueError("every meta-analytic estimate must be finite")
    if len({effect.effect_id for effect in effects}) != len(effects):
        raise ValueError("meta-analytic effect IDs must be unique")
    if any(name not in effect.moderators for effect in effects for name in moderator_names):
        raise ValueError("every effect must declare every requested moderator")

    columns, design, references = _design(effects, moderator_names)
    if moderator_names and len(effects) < minimum_effects_per_coefficient * len(columns):
        stratified: dict[str, dict[str, dict]] = {}
        for moderator in moderator_names:
            by_level: dict[str, list[MetaEffect]] = {}
            for effect in effects:
                by_level.setdefault(effect.moderators[moderator], []).append(effect)
            stratified[moderator] = {}
            for level, members in sorted(by_level.items()):
                if len(members) >= 2:
                    stratified[moderator][level] = reml_meta_analysis(members)
                else:
                    stratified[moderator][level] = {
                        "status": "descriptive_single_effect",
                        "effects": 1,
                        "estimate": members[0].estimate,
                        "variance": members[0].variance,
                    }
        return {
            "status": "sparse_moderator_fallback_required",
            "effects": len(effects),
            "coefficients_requested": len(columns),
            "minimum_effects_per_coefficient": minimum_effects_per_coefficient,
            "moderators": list(moderator_names),
            "stratified_effects": stratified,
        }
    tau_squared, iterations = _reml_tau_squared(effects, design)
    beta, covariance, residuals, weights = _fit_at_tau(effects, design, tau_squared)
    estimates = {
        name: {
            "estimate": estimate,
            "standard_error": math.sqrt(max(0.0, covariance[index][index])),
            "lower_95": estimate - 1.959963984540054 * math.sqrt(max(0.0, covariance[index][index])),
            "upper_95": estimate + 1.959963984540054 * math.sqrt(max(0.0, covariance[index][index])),
        }
        for index, (name, estimate) in enumerate(zip(columns, beta, strict=True))
    }
    fixed_weights = [1 / effect.variance for effect in effects]
    fixed_mean = sum(
        weight * effect.estimate for weight, effect in zip(fixed_weights, effects, strict=True)
    ) / sum(fixed_weights)
    cochran_q = sum(
        weight * (effect.estimate - fixed_mean) ** 2
        for weight, effect in zip(fixed_weights, effects, strict=True)
    )
    degrees_freedom = len(effects) - 1
    i_squared = (
        max(0.0, (cochran_q - degrees_freedom) / cochran_q)
        if cochran_q > 0
        else 0.0
    )
    return {
        "status": "estimated",
        "method": "two_stage_reml_random_effects_meta_analysis",
        "effects": len(effects),
        "coefficient_names": columns,
        "reference_levels": references,
        "coefficients": estimates,
        "tau_squared": tau_squared,
        "i_squared": i_squared,
        "cochran_q_fixed_effect": cochran_q,
        "heterogeneity_degrees_freedom": degrees_freedom,
        "reml_iterations": iterations,
        "residual_sum_weighted_squares": sum(
            weight * residual * residual
            for weight, residual in zip(weights, residuals, strict=True)
        ),
    }
