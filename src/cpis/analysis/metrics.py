"""Prespecified analysis primitives that preserve the item as experimental unit."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Protocol

from cpis.records import ParsedRecord


class ConfidenceObservation(Protocol):
    answer_parse_status: str
    confidence_parse_status: str
    confidence: float | None


def has_valid_answer(record: ConfidenceObservation) -> bool:
    """Whether an answer was actually produced and parsed."""

    return record.answer_parse_status == "valid"


def has_valid_confidence(record: ConfidenceObservation) -> bool:
    """Return whether a numeric confidence is valid, independent of answer validity."""

    return record.confidence_parse_status == "valid" and record.confidence is not None


def confidence_accepts(record: ConfidenceObservation, threshold: float) -> bool:
    """Apply the operational policy: accept only a produced answer above tau.

    An answer that never terminated has nothing to accept, so it cannot enter
    coverage however confidently the readout then scores its own truncated text.
    Answer failure and confidence abstention are different channels and are
    reported separately by `operational_endpoints`; neither is ever imputed.
    """

    return (
        has_valid_answer(record)
        and has_valid_confidence(record)
        and float(record.confidence) >= threshold
    )


@dataclass(frozen=True)
class OperationalEndpoints:
    threshold: float
    total_observations: int
    answer_failure_rate: float
    coverage: float
    selective_risk: float | None
    operational_success: float | None
    accepted_observations: int
    confidence_invalid_rate: float


def operational_endpoints(
    records: Iterable[ParsedRecord], threshold: float
) -> OperationalEndpoints:
    """Separate answer failure, coverage, selective risk and operational success.

    A decoder shift can lower selective risk merely by crashing the generations
    it would otherwise have got wrong, so risk is never reported without the
    coverage and answer-failure rates that explain it.
    """

    values = list(records)
    if not values:
        raise ValueError("operational endpoints require at least one observation")
    invalid_answers = sum(1 for r in values if not has_valid_answer(r))
    accepted = [r for r in values if confidence_accepts(r, threshold)]
    valid_answers = [r for r in values if has_valid_answer(r)]
    risk = (
        sum(1 for r in accepted if not r.correct) / len(accepted) if accepted else None
    )
    coverage = len(accepted) / len(values)
    return OperationalEndpoints(
        threshold=threshold,
        total_observations=len(values),
        answer_failure_rate=invalid_answers / len(values),
        coverage=coverage,
        selective_risk=risk,
        operational_success=None if risk is None else coverage * (1.0 - risk),
        accepted_observations=len(accepted),
        confidence_invalid_rate=(
            sum(1 for r in valid_answers if not has_valid_confidence(r))
            / len(valid_answers)
            if valid_answers
            else 0.0
        ),
    )


@dataclass(frozen=True)
class OutcomePartition:
    """Every item lands in exactly one of five outcomes, and they sum to one.

    Selective risk is a conditional quantity, so it cannot by itself say what a
    decoder shift did to a deployed system: two systems with identical
    R = E/(S+E) can differ enormously in how much work they accept at all. The
    partition makes the alternatives explicit.

        1 = S + E + q + a_C + a_tau

    where S and E are accepted-correct and accepted-wrong, and the remaining
    three are the disjoint reasons an item was not accepted: the answer did not
    parse, the answer parsed but the confidence did not, or both parsed and the
    confidence fell below the frozen threshold.
    """

    threshold: float
    total_observations: int
    accepted_success: float
    accepted_error: float
    answer_failure: float
    confidence_failure: float
    low_confidence_abstention: float

    @property
    def coverage(self) -> float:
        """K = S + E."""
        return self.accepted_success + self.accepted_error

    @property
    def non_acceptance(self) -> float:
        """A = q + a_C + a_tau = 1 - K."""
        return (
            self.answer_failure
            + self.confidence_failure
            + self.low_confidence_abstention
        )


def outcome_partition(
    records: Iterable[ParsedRecord], threshold: float
) -> OutcomePartition:
    """Assign every observation to exactly one outcome at the frozen threshold."""
    values = list(records)
    if not values:
        raise ValueError("an outcome partition requires at least one observation")
    total = len(values)
    counts = {
        "answer_failure": 0,
        "confidence_failure": 0,
        "low_confidence_abstention": 0,
        "accepted_error": 0,
        "accepted_success": 0,
    }
    for record in values:
        if not has_valid_answer(record):
            counts["answer_failure"] += 1
        elif not has_valid_confidence(record):
            counts["confidence_failure"] += 1
        elif float(record.confidence) < threshold:
            counts["low_confidence_abstention"] += 1
        elif record.correct:
            counts["accepted_success"] += 1
        else:
            counts["accepted_error"] += 1
    return OutcomePartition(
        threshold=threshold,
        total_observations=total,
        **{key: value / total for key, value in counts.items()},
    )


@dataclass(frozen=True)
class SelectiveRisk:
    threshold: float
    risk: float
    coverage: float
    accepted_observations: int
    total_observations: int
    experimental_units: int


@dataclass(frozen=True)
class FrozenRiskDelta:
    reference: SelectiveRisk
    shifted: SelectiveRisk
    delta_risk: float


@dataclass(frozen=True)
class RiskContractSelection:
    status: Literal["candidate", "risk-contract infeasible"]
    target_risk: float
    minimum_coverage: float
    estimate: SelectiveRisk | None
    reason: str | None


@dataclass(frozen=True)
class RiskContractCertification:
    status: Literal["certified", "risk-contract infeasible"]
    threshold: float
    target_risk: float
    minimum_coverage: float
    coverage: float
    accepted_items: int
    total_items: int
    errors: int
    empirical_risk: float | None
    upper_confidence_bound: float | None
    confidence_level: float
    certification_replicate_seed: int
    reason: str | None


@dataclass(frozen=True)
class FixedCoverageThreshold:
    status: Literal["selected", "target unattainable"]
    target_coverage: float
    threshold: float | None
    achieved_coverage: float
    reference_risk: float | None


@dataclass(frozen=True)
class FixedCoverageTransport:
    target_coverage: float
    threshold: float
    reference_risk: float
    shifted_risk: float | None
    reference_coverage: float
    shifted_coverage: float
    delta_risk: float | None
    delta_coverage_from_target: float
    delta_coverage_from_reference: float


@dataclass(frozen=True)
class InvalidConfidenceSummary:
    invalid_observations: int
    total_observations: int
    invalid_rate: float
    valid_observations: int


def _single_condition(records: list[ParsedRecord], context: str) -> str:
    conditions = {record.condition_id for record in records}
    if len(conditions) != 1:
        raise ValueError(f"{context} requires exactly one inference condition")
    return next(iter(conditions))


def selective_risk(records: Iterable[ParsedRecord], threshold: float) -> SelectiveRisk:
    """Compute a point estimate; uncertainty must cluster by dataset item."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    values = list(records)
    if not values:
        raise ValueError("selective risk requires at least one observation")
    accepted = [
        record
        for record in values
        if confidence_accepts(record, threshold)
    ]
    if not accepted:
        raise ValueError("threshold accepts no observations")
    errors = sum(not record.correct for record in accepted)
    return SelectiveRisk(
        threshold=threshold,
        risk=errors / len(accepted),
        coverage=len(accepted) / len(values),
        accepted_observations=len(accepted),
        total_observations=len(values),
        experimental_units=len({record.dataset_item_id for record in values}),
    )


def invalid_confidence_summary(
    records: Iterable[ParsedRecord],
) -> InvalidConfidenceSummary:
    """Report invalid confidence explicitly; never impute a numeric value."""
    values = list(records)
    if not values:
        raise ValueError("invalid-confidence summary requires observations")
    invalid = sum(
        record.confidence_parse_status != "valid" or record.confidence is None
        for record in values
    )
    return InvalidConfidenceSummary(
        invalid_observations=invalid,
        total_observations=len(values),
        invalid_rate=invalid / len(values),
        valid_observations=len(values) - invalid,
    )


def select_empirical_threshold(
    development_records: Iterable[ParsedRecord], target_risk: float
) -> SelectiveRisk:
    """Choose maximum empirical coverage on development data only.

    This is not a certification bound. The final one-sided certification method
    and its error rate must be prespecified before a real confirmatory run.
    """
    if not 0 <= target_risk <= 1:
        raise ValueError("target_risk must lie in [0, 1]")
    records = list(development_records)
    if any(record.dataset_partition != "development" for record in records):
        raise ValueError("threshold selection accepts development records only")
    _single_condition(records, "threshold selection")
    candidates = sorted(
        {
            record.confidence
            for record in records
            if has_valid_confidence(record)
        }
    )
    eligible = [
        estimate
        for threshold in candidates
        if (estimate := selective_risk(records, threshold)).risk <= target_risk
    ]
    if not eligible:
        raise ValueError("no observed threshold satisfies the empirical risk target")
    return max(eligible, key=lambda estimate: (estimate.coverage, -estimate.threshold))


def select_risk_contract_threshold(
    development_records: Iterable[ParsedRecord],
    target_risk: float,
    minimum_coverage: float,
) -> RiskContractSelection:
    """Select the maximum-coverage development candidate without relaxing alpha."""
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must lie in (0, 1]")
    try:
        estimate = select_empirical_threshold(development_records, target_risk)
    except ValueError as exc:
        if str(exc) != "no observed threshold satisfies the empirical risk target":
            raise
        return RiskContractSelection(
            status="risk-contract infeasible",
            target_risk=target_risk,
            minimum_coverage=minimum_coverage,
            estimate=None,
            reason="no development threshold satisfies the empirical risk target",
        )
    if estimate.coverage < minimum_coverage:
        return RiskContractSelection(
            status="risk-contract infeasible",
            target_risk=target_risk,
            minimum_coverage=minimum_coverage,
            estimate=estimate,
            reason="development coverage is below the prespecified minimum",
        )
    return RiskContractSelection(
        status="candidate",
        target_risk=target_risk,
        minimum_coverage=minimum_coverage,
        estimate=estimate,
        reason=None,
    )


def _binomial_cdf(errors: int, trials: int, probability: float) -> float:
    """Compute P(X <= errors) stably without adding a statistics dependency."""
    if probability <= 0:
        return 1.0
    if probability >= 1:
        return 1.0 if errors == trials else 0.0
    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    terms = [
        math.lgamma(trials + 1)
        - math.lgamma(k + 1)
        - math.lgamma(trials - k + 1)
        + k * log_p
        + (trials - k) * log_q
        for k in range(errors + 1)
    ]
    maximum = max(terms)
    return math.exp(maximum) * math.fsum(math.exp(term - maximum) for term in terms)


def clopper_pearson_upper(
    errors: int, trials: int, tail_probability: float
) -> float:
    """One-sided exact Clopper--Pearson upper bound for a binomial rate."""
    if trials <= 0:
        raise ValueError("Clopper-Pearson requires at least one trial")
    if not 0 <= errors <= trials:
        raise ValueError("errors must lie between zero and trials")
    if not 0 < tail_probability < 1:
        raise ValueError("tail_probability must lie in (0, 1)")
    if errors == trials:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(80):
        midpoint = (low + high) / 2
        if _binomial_cdf(errors, trials, midpoint) > tail_probability:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2


def certify_risk_contract(
    certification_records: Iterable[ParsedRecord],
    threshold: float,
    target_risk: float,
    minimum_coverage: float,
    tail_probability: float,
    certification_replicate_seed: int,
) -> RiskContractCertification:
    """Certify using exactly one prespecified observation per benchmark item."""
    records = list(certification_records)
    if not records:
        raise ValueError("certification requires observations")
    if any(record.dataset_partition != "certification" for record in records):
        raise ValueError("risk-contract certification accepts certification records only")
    _single_condition(records, "risk-contract certification")
    if any(record.seed != certification_replicate_seed for record in records):
        raise ValueError("certification records must use only the prespecified replicate seed")
    item_ids = [record.dataset_item_id for record in records]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("certification requires exactly one observation per dataset item")
    accepted = [
        record
        for record in records
        if confidence_accepts(record, threshold)
    ]
    accepted_count = len(accepted)
    total_count = len(records)
    coverage = accepted_count / total_count
    errors = sum(not record.correct for record in accepted)
    empirical_risk = errors / accepted_count if accepted_count else None
    upper = (
        clopper_pearson_upper(errors, accepted_count, tail_probability)
        if accepted_count
        else None
    )
    if coverage < minimum_coverage:
        status: Literal["certified", "risk-contract infeasible"] = (
            "risk-contract infeasible"
        )
        reason = "certification coverage is below the prespecified minimum"
    elif upper is None or upper > target_risk:
        status = "risk-contract infeasible"
        reason = "one-sided exact upper bound exceeds the risk target"
    else:
        status = "certified"
        reason = None
    return RiskContractCertification(
        status=status,
        threshold=threshold,
        target_risk=target_risk,
        minimum_coverage=minimum_coverage,
        coverage=coverage,
        accepted_items=accepted_count,
        total_items=total_count,
        errors=errors,
        empirical_risk=empirical_risk,
        upper_confidence_bound=upper,
        confidence_level=1 - tail_probability,
        certification_replicate_seed=certification_replicate_seed,
        reason=reason,
    )


def risk_contract_violation(risk: float, target_risk: float) -> float:
    if not 0 <= risk <= 1 or not 0 <= target_risk <= 1:
        raise ValueError("risk and target_risk must lie in [0, 1]")
    return max(0.0, risk - target_risk)


def select_fixed_coverage_threshold(
    reference_records: Iterable[ParsedRecord], target_coverage: float
) -> FixedCoverageThreshold:
    """Use the smallest attainable coverage not below the prespecified target."""
    if not 0 < target_coverage <= 1:
        raise ValueError("target_coverage must lie in (0, 1]")
    records = list(reference_records)
    if not records:
        raise ValueError("fixed-coverage selection requires observations")
    _single_condition(records, "fixed-coverage selection")
    candidates = sorted(
        {
            record.confidence
            for record in records
            if has_valid_confidence(record)
        }
    )
    estimates = [selective_risk(records, threshold) for threshold in candidates]
    feasible = [e for e in estimates if e.coverage >= target_coverage]
    if not feasible:
        highest = max((e.coverage for e in estimates), default=0.0)
        return FixedCoverageThreshold(
            status="target unattainable",
            target_coverage=target_coverage,
            threshold=None,
            achieved_coverage=highest,
            reference_risk=None,
        )
    selected = min(feasible, key=lambda e: (e.coverage, -e.threshold))
    return FixedCoverageThreshold(
        status="selected",
        target_coverage=target_coverage,
        threshold=selected.threshold,
        achieved_coverage=selected.coverage,
        reference_risk=selected.risk,
    )


def transport_fixed_coverage_threshold(
    reference_records: Iterable[ParsedRecord],
    shifted_records: Iterable[ParsedRecord],
    selected: FixedCoverageThreshold,
) -> FixedCoverageTransport:
    """Transport the numeric reference threshold unchanged to a paired condition."""
    if selected.status != "selected" or selected.threshold is None:
        raise ValueError("fixed-coverage transport requires a selected threshold")
    reference = list(reference_records)
    shifted = list(shifted_records)
    _single_condition(reference, "fixed-coverage reference")
    _single_condition(shifted, "fixed-coverage shift")
    ref_keys = {(r.dataset_item_id, r.seed) for r in reference}
    shift_keys = {(r.dataset_item_id, r.seed) for r in shifted}
    if ref_keys != shift_keys:
        raise ValueError("fixed-coverage transport requires paired item/seed observations")
    ref = selective_risk(reference, selected.threshold)
    accepted_shift = [
        record
        for record in shifted
        if confidence_accepts(record, selected.threshold)
    ]
    shifted_coverage = len(accepted_shift) / len(shifted)
    shifted_risk = (
        sum(not record.correct for record in accepted_shift) / len(accepted_shift)
        if accepted_shift
        else None
    )
    return FixedCoverageTransport(
        target_coverage=selected.target_coverage,
        threshold=selected.threshold,
        reference_risk=ref.risk,
        shifted_risk=shifted_risk,
        reference_coverage=ref.coverage,
        shifted_coverage=shifted_coverage,
        delta_risk=(shifted_risk - ref.risk if shifted_risk is not None else None),
        delta_coverage_from_target=shifted_coverage - selected.target_coverage,
        delta_coverage_from_reference=shifted_coverage - ref.coverage,
    )


def holm_adjust(p_values: Mapping[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Return Holm adjusted p-values and step-down rejection decisions."""
    if not p_values:
        raise ValueError("Holm adjustment requires at least one p-value")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if any(not 0 <= value <= 1 for value in p_values.values()):
        raise ValueError("p-values must lie in [0, 1]")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return {
        name: {
            "raw_p_value": p_values[name],
            "adjusted_p_value": adjusted[name],
            "reject": adjusted[name] <= alpha,
        }
        for name in p_values
    }


def frozen_threshold_delta(
    reference_records: Iterable[ParsedRecord],
    shifted_records: Iterable[ParsedRecord],
    threshold: float,
) -> FrozenRiskDelta:
    reference = list(reference_records)
    shifted = list(shifted_records)
    _single_condition(reference, "reference risk")
    _single_condition(shifted, "shifted risk")
    ref_partitions = {record.dataset_partition for record in reference}
    shifted_partitions = {record.dataset_partition for record in shifted}
    if len(ref_partitions) != 1 or ref_partitions != shifted_partitions:
        raise ValueError("frozen-risk comparison cannot mix dataset partitions")
    ref_keys = {(r.dataset_item_id, r.seed) for r in reference}
    shifted_keys = {(r.dataset_item_id, r.seed) for r in shifted}
    if ref_keys != shifted_keys:
        raise ValueError("frozen-risk comparison requires paired item/seed observations")
    ref_risk = selective_risk(reference, threshold)
    shifted_risk = selective_risk(shifted, threshold)
    return FrozenRiskDelta(
        reference=ref_risk,
        shifted=shifted_risk,
        delta_risk=shifted_risk.risk - ref_risk.risk,
    )


def item_clustered_bootstrap_deltas(
    reference_records: Iterable[ParsedRecord],
    shifted_records: Iterable[ParsedRecord],
    threshold: float,
    replicates: int,
    seed: int,
) -> list[float]:
    """Resample items, retaining every seed/config observation in each cluster."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    reference = list(reference_records)
    shifted = list(shifted_records)
    by_item_ref: dict[str, list[ParsedRecord]] = {}
    by_item_shift: dict[str, list[ParsedRecord]] = {}
    for record in reference:
        by_item_ref.setdefault(record.dataset_item_id, []).append(record)
    for record in shifted:
        by_item_shift.setdefault(record.dataset_item_id, []).append(record)
    if set(by_item_ref) != set(by_item_shift):
        raise ValueError("clustered bootstrap requires paired dataset items")
    item_ids = sorted(by_item_ref)
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(replicates):
        sampled = rng.choices(item_ids, k=len(item_ids))
        ref_sample = [record for item_id in sampled for record in by_item_ref[item_id]]
        shift_sample = [record for item_id in sampled for record in by_item_shift[item_id]]
        try:
            deltas.append(
                frozen_threshold_delta(ref_sample, shift_sample, threshold).delta_risk
            )
        except ValueError:
            # Some resamples can have zero accepted observations in one arm.
            continue
    return deltas
