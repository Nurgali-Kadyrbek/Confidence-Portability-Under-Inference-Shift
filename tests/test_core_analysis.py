"""The held-out estimator must be paired, item-clustered and honest about nulls."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from analyze_core_test import (  # noqa: E402
    classify,
    equivalence_realisation_separation,
    paired_bootstrap,
    randomisation_p,
)

from cpis.records import ParsedRecord


def _record(item: str, confidence: float | None, correct: bool, valid_answer=True):
    return ParsedRecord(
        record_id=f"{item}-{confidence}-{correct}",
        answer_raw_record_id="a", answer_raw_record_sha256="0" * 64,
        confidence_raw_record_id="c", confidence_raw_record_sha256="1" * 64,
        run_id="run", experiment_id="exp",
        dataset_id="d", dataset_revision="r", dataset_item_id=item,
        dataset_partition="test", condition_id="cond", seed=1729,
        answer_parser_version="final_json_answer_v1",
        confidence_parser_version="final_json_confidence_v1",
        scorer="exact_match_casefold_v1",
        answer_parse_status="valid" if valid_answer else "invalid",
        confidence_parse_status="valid" if confidence is not None else "invalid",
        parse_status="valid" if valid_answer and confidence is not None else "invalid",
        parsed_answer="x" if valid_answer else None,
        confidence=confidence, correct=correct,
    )


def _grouped(records):
    out = {}
    for r in records:
        out.setdefault(r.dataset_item_id, []).append(r)
    return out


def test_identical_conditions_give_a_zero_effect_and_a_null_p() -> None:
    records = [_record(f"i{n}", 0.9, n % 3 == 0) for n in range(40)]
    a = _grouped(records)
    b = _grouped([_record(r.dataset_item_id, 0.9, r.correct) for r in records])
    deltas = paired_bootstrap(a, b, 0.8, 300, seed=1)
    for key in ("q", "K", "R", "S"):
        assert deltas[key]["delta"] == pytest.approx(0.0, abs=1e-12)
    p = randomisation_p(a, b, 0.8, "R", 400, seed=2)
    assert p == pytest.approx(1.0, abs=1e-9)


def test_a_degenerate_shift_lowers_coverage_not_just_risk() -> None:
    """Crashing the wrong items must show as coverage loss, not a safety win."""
    reference = [_record(f"i{n}", 0.9, n % 2 == 0) for n in range(40)]
    shifted = [
        _record(r.dataset_item_id, 0.9, True)
        if r.correct
        else _record(r.dataset_item_id, 0.9, False, valid_answer=False)
        for r in reference
    ]
    a, b = _grouped(reference), _grouped(shifted)
    deltas = paired_bootstrap(a, b, 0.8, 300, seed=3)
    assert deltas["R"]["delta"] < 0          # risk falls
    assert deltas["q"]["delta"] > 0          # because answers failed
    assert deltas["K"]["delta"] < 0          # coverage falls with them
    assert deltas["S"]["delta"] == pytest.approx(0.0, abs=1e-9)  # success unchanged


def test_bootstrap_is_paired_and_reproducible() -> None:
    reference = [_record(f"i{n}", 0.9, n % 2 == 0) for n in range(40)]
    shifted = [_record(f"i{n}", 0.9, n % 4 == 0) for n in range(40)]
    a, b = _grouped(reference), _grouped(shifted)
    first = paired_bootstrap(a, b, 0.8, 300, seed=7)
    second = paired_bootstrap(a, b, 0.8, 300, seed=7)
    assert first == second
    r = first["R"]
    assert r["ci_low"] <= r["delta"] <= r["ci_high"]
    assert r["replicates"] == 300


MARGINS = {"R": 0.025, "K": 0.05}


def test_classification_is_statistical_only_and_applies_no_realisation_gate() -> None:
    """A tight interval on an inert coordinate is still classified equivalent.

    No realisation floor was prespecified, so the classifier must not invent
    one. Whether the classification is informative is reported separately.
    """
    inert = {
        "deltas": {"R": {"ci_low": -0.004, "ci_high": 0.011}, "K": {"ci_low": -0.01, "ci_high": 0.02}},
        "reject_at_familywise_alpha": False,
        "realisation": 0.01,
    }
    assert classify(inert, MARGINS)["statistical_classification"] == "equivalent_within_margin"


def test_classification_labels_change_equivalence_and_inconclusive() -> None:
    def row(low, high, reject):
        return {
            "deltas": {"R": {"ci_low": low, "ci_high": high}, "K": {"ci_low": low, "ci_high": high}},
            "reject_at_familywise_alpha": reject,
        }

    assert classify(row(0.007, 0.026, True), MARGINS)["statistical_classification"] == "change_detected"
    assert classify(row(-0.004, 0.011, False), MARGINS)["statistical_classification"] == "equivalent_within_margin"
    # a CI reaching the margin is not equivalence
    assert classify(row(-0.030, 0.015, False), MARGINS)["statistical_classification"] == "inconclusive"
    assert classify(row(None, None, None), MARGINS)["statistical_classification"] == "not_estimable"


def test_equivalence_separation_reports_order_statistics_not_a_verdict() -> None:
    results = [
        {
            "experiment_id": "e",
            "contrasts": {
                "inert": {"equivalent_R_at_margin": True, "realisation": 0.015},
                "strong": {"equivalent_R_at_margin": False, "realisation": 1.0},
                "middling": {"equivalent_R_at_margin": False, "realisation": 0.115},
            },
        }
    ]
    out = equivalence_realisation_separation(results)
    assert out["equivalent_contrasts"] == ["e::inert"]
    assert out["max_realisation_among_equivalent"] == 0.015
    assert out["min_realisation_among_non_equivalent"] == 0.115
    assert out["separated"] is True


def test_equivalence_separation_is_false_when_a_realised_contrast_is_equivalent() -> None:
    results = [
        {
            "experiment_id": "e",
            "contrasts": {
                "realised": {"equivalent_R_at_margin": True, "realisation": 0.9},
                "other": {"equivalent_R_at_margin": False, "realisation": 0.4},
            },
        }
    ]
    assert equivalence_realisation_separation(results)["separated"] is False


def test_precedence_is_declared_when_significance_and_equivalence_overlap() -> None:
    """A change can be real and still sit inside the practical margin.

    That is a coherent state, not a contradiction, so both flags survive and the
    overlap is reported rather than erased by whichever branch runs first.
    """
    both = {
        "deltas": {"R": {"ci_low": 0.001, "ci_high": 0.02}, "K": {"ci_low": 0.0, "ci_high": 0.01}},
        "reject_at_familywise_alpha": True,
    }
    out = classify(both, MARGINS)
    assert out["statistical_classification"] == "change_detected"
    assert out["equivalent_R_at_margin"] is True
    assert out["significant_and_within_margin"] is True


def test_non_overlapping_contrasts_do_not_claim_overlap() -> None:
    row = {
        "deltas": {"R": {"ci_low": 0.007, "ci_high": 0.026}, "K": {"ci_low": 0.0, "ci_high": 0.01}},
        "reject_at_familywise_alpha": True,
    }
    assert classify(row, MARGINS)["significant_and_within_margin"] is False


def test_realisation_table_marks_undefined_rather_than_zero() -> None:
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    from report_core_study import realisation_table  # noqa: E402

    analysis = {
        "pairs": [
            {
                "status": "estimated",
                "model_id": "org/M",
                "model_mode": "instruct",
                "dataset_id": "org/SuperGPQA",
                "contrasts": {
                    "reference_to_high_temperature": {
                        "answer_changed": 1.0,
                        "confidence_changed": 1.0,
                        "confidence_changed_given_answer_identical": None,
                        "items_with_identical_answer": 0,
                    },
                    "reference_to_low_top_p": {
                        "answer_changed": 0.36,
                        "confidence_changed": 0.74,
                        "confidence_changed_given_answer_identical": 0.723,
                        "items_with_identical_answer": 382,
                    },
                },
            }
        ]
    }
    table = realisation_table(analysis)
    assert "undefined" in table
    assert "0.723" in table
    assert "1 of these contrasts changed every compared answer" in table
    assert "not a causal decomposition" in table


def test_outcome_partition_sums_to_one_and_matches_endpoints() -> None:
    """The five outcomes are disjoint and exhaustive, and K = S + E.

    If the partition and operational_endpoints ever disagree on coverage, one
    of them is miscounting an abstention as an acceptance.
    """
    from cpis.analysis.metrics import operational_endpoints, outcome_partition

    records = [
        _record("a", 0.9, True),
        _record("b", 0.9, False),
        _record("c", 0.1, True),
        _record("d", None, False),
        _record("e", 0.95, False, valid_answer=False),
    ]
    part = outcome_partition(records, 0.8)
    assert part.accepted_success == pytest.approx(0.2)
    assert part.accepted_error == pytest.approx(0.2)
    assert part.low_confidence_abstention == pytest.approx(0.2)
    assert part.confidence_failure == pytest.approx(0.2)
    assert part.answer_failure == pytest.approx(0.2)
    assert part.coverage + part.non_acceptance == pytest.approx(1.0)
    assert part.coverage == pytest.approx(
        operational_endpoints(records, 0.8).coverage
    )


def test_partition_separates_equal_risk_systems_with_different_coverage() -> None:
    """Identical R, very different amounts of accepted work.

    This is the case the conditional endpoint cannot distinguish and the whole
    reason the partition is reported.
    """
    from cpis.analysis.metrics import operational_endpoints, outcome_partition

    wide = [_record(f"w{n}", 0.9, n % 2 == 0) for n in range(20)]
    narrow = [
        _record(f"n{n}", 0.9 if n < 4 else 0.1, n % 2 == 0) for n in range(20)
    ]
    assert operational_endpoints(wide, 0.8).selective_risk == pytest.approx(
        operational_endpoints(narrow, 0.8).selective_risk
    )
    assert outcome_partition(wide, 0.8).coverage == pytest.approx(1.0)
    assert outcome_partition(narrow, 0.8).coverage == pytest.approx(0.2)


def _corp_fixture():
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    from reference_confidence_quality import _corp, _pav  # noqa: E402

    return _corp, _pav


def test_pav_returns_a_monotone_fit() -> None:
    _, _pav = _corp_fixture()
    x = [0.1, 0.9, 0.5, 0.3, 0.7]
    y = [1.0, 0.0, 1.0, 0.0, 1.0]
    fitted = _pav(x, y)
    ordered = [f for _, f in sorted(zip(x, fitted))]
    assert all(a <= b + 1e-12 for a, b in zip(ordered, ordered[1:]))


def test_corp_identity_holds() -> None:
    """BS = MCB - DSC + UNC is an identity, not an approximation."""
    _corp, _ = _corp_fixture()
    confidences = [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95]
    outcomes = [0, 0, 1, 0, 1, 1, 1, 0]
    d = _corp(confidences, outcomes)
    assert d["brier"] == pytest.approx(d["mcb"] - d["dsc"] + d["unc"], abs=1e-12)


def test_corp_separates_scaling_failure_from_uninformativeness() -> None:
    """A well-ordered but badly scaled forecast is not the same failure.

    Both score poorly on the mean gap; only one of them carries information.
    """
    _corp, _ = _corp_fixture()
    outcomes = [0] * 8 + [1] * 8
    # perfectly ordered, wildly overconfident
    ordered = [0.90] * 8 + [0.99] * 8
    # perfectly scaled at the base rate, no ordering at all
    flat = [0.5] * 16
    a, b = _corp(ordered, outcomes), _corp(flat, outcomes)
    assert a["dsc"] > b["dsc"]          # ordering carries information
    assert a["mcb"] > b["mcb"]          # and is badly scaled
    assert b["dsc"] == pytest.approx(0.0, abs=1e-12)


def test_pav_pools_tied_forecasts_instead_of_ordering_them_by_outcome() -> None:
    """Every observation sharing a forecast value gets the same fitted value.

    Breaking ties by the outcome would let a constant forecast look perfectly
    ordered, inventing discrimination where there is none. These confidences
    are heavily tied, so this is the failure mode that matters.
    """
    _, _pav = _corp_fixture()
    fitted = _pav([0.5] * 6, [0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    assert len(set(fitted)) == 1
    assert fitted[0] == pytest.approx(0.5)


def _seed_module():
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    import analyze_seed_robustness  # noqa: E402

    return analyze_seed_robustness


def test_law_of_total_variance_identity() -> None:
    """Var(D) = E[Var(D|i)] + Var(E[D|i]) must hold on the computed pieces."""
    from statistics import fmean, pvariance

    cells = {"a": [0.0, 1.0, 1.0], "b": [1.0, 1.0, 1.0], "c": [0.0, 0.0, 1.0]}
    within = fmean(pvariance(v) for v in cells.values())
    between = pvariance([fmean(v) for v in cells.values()])
    flat = [v for values in cells.values() for v in values]
    assert within + between == pytest.approx(pvariance(flat), abs=1e-12)


def test_seed_analysis_excludes_ratio_endpoint_from_decomposition() -> None:
    """R has no per-item value, so it must not get a variance split.

    Decomposing it would restrict to items accepted at two or more seeds,
    which is a subset selected by the outcome being measured.
    """
    module = _seed_module()
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert 'for k in (e for e in ENDPOINTS if e != "R")' in source
