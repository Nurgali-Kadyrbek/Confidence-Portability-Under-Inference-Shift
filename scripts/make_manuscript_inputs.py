#!/usr/bin/env python3
"""Generate every manuscript number, table and figure from the frozen artifacts.

The manuscript never writes a result as a literal. It uses macros defined here,
and its tables and figures are \\input from files written here, so a number in
the paper cannot drift from the artifact that produced it. If an artifact
changes, the paper changes with it or the build fails.

Nothing here recomputes a result. It reads the immutable analysis artifacts and
formats them.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CORE = Path("evidence/core")
PROTOCOL = Path("evidence/protocol")

MODEL_LABEL = {
    ("Qwen/Qwen3.5-9B", "non_thinking"): ("Qwen3.5-9B", "non-thinking"),
    ("Qwen/Qwen3.5-9B", "thinking"): ("Qwen3.5-9B", "thinking"),
    ("mistralai/Ministral-3-8B-Instruct-2512-BF16", "instruct"): (
        "Ministral-3-8B", "instruct"),
    ("mistralai/Ministral-3-8B-Reasoning-2512", "reasoning"): (
        "Ministral-3-8B", "reasoning"),
}
SHIFT_LABEL = {
    "reference_to_high_temperature": "high-$T$",
    "reference_to_low_top_p": "low-$p$",
}
CLASS_LABEL = {
    "change_detected": "change",
    "equivalent_within_margin": "equivalent",
    "inconclusive": "inconclusive",
    "not_estimable": "n/a",
}


def dataset_label(dataset_id: str) -> str:
    return "ManyIFEval" if "Instruct" in dataset_id else "SuperGPQA"


def pair_label(entry: dict) -> tuple[str, str, str]:
    model, mode = MODEL_LABEL[(entry["model_id"], entry["model_mode"])]
    return model, mode, dataset_label(entry["dataset_id"])


def num(value, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "--"
    return f"{value:{'+' if signed else ''}.{digits}f}"


def ci(entry: dict, digits: int = 3, signed: bool = False) -> str:
    """A point estimate with its interval, or a dash when not estimable."""
    point = entry.get("delta", entry.get("estimate"))
    if point is None:
        return "--"
    low, high = entry.get("ci_low"), entry.get("ci_high")
    if low is None:
        return num(point, digits, signed)
    return (
        f"{num(point, digits, signed)} "
        f"[{num(low, digits, signed)}, {num(high, digits, signed)}]"
    )



def table(
    *, label: str, caption: str, spec: str, header: str, rows: str,
    face: str = "small",
) -> str:
    """A complete MDPI table float.

    The whole float is emitted here rather than only its rows. A partial table
    cannot be \\input from inside an alignment: LaTeX's \\input does not nest
    cleanly inside a tabular, and the following \\bottomrule then executes
    outside a row. Owning the column specification here also means the header
    and the data can never disagree on how many columns there are.
    """
    columns = len(re.findall(r"[lcrXLCR]", spec))
    for line in rows.splitlines():
        found = line.count("&") + 1
        if found != columns:
            raise SystemExit(
                f"{label}: row has {found} cells but the spec declares {columns}"
            )
    return (
        "\\begin{table}[H]\n"
        "\\begin{adjustwidth}{-\\extralength}{0cm}\n"
        f"\\{face}\n"
        f"\\caption{{{caption}\\label{{{label}}}}}\n"
        f"\\begin{{tabular}}{{{spec}}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        f"{rows}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{adjustwidth}\n"
        "\\end{table}\n"
    )

# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------


def table_reference_quality(quality: dict) -> str:
    rows = []
    for pair in quality["pairs"]:
        model, mode, data = pair_label(pair)
        m = pair["metrics"]
        rows.append(
            " & ".join(
                [
                    model,
                    mode,
                    data,
                    num(m["accuracy"]["estimate"]),
                    num(m["mean_confidence"]["estimate"]),
                    num(m["confidence_accuracy_gap"]["estimate"], signed=True),
                    ci(m["auroc"]),
                    ci(m["brier_skill_score"], signed=True),
                    num(m["miscalibration_ratio"]["estimate"]),
                    num(m["discrimination_skill"]["estimate"]),
                ]
            )
            + r" \tabularnewline"
        )
    return "\n".join(rows)


def table_thresholds(thresholds: dict, quantisation: dict) -> str:
    q_by_key = {
        (p["model_id"], p["model_mode"], p["dataset_id"]): p["confidence_quantisation"]
        for p in quantisation["pairs"]
    }
    rows = []
    for pair in thresholds["pairs"]:
        model, mode, data = pair_label(pair)
        q = q_by_key[(pair["model_id"], pair["model_mode"], pair["dataset_id"])]
        endpoint = (
            "risk target"
            if pair.get("endpoint") == "certified_risk_threshold"
            else "coverage 0.50"
        )
        rows.append(
            " & ".join(
                [
                    model,
                    mode,
                    data,
                    endpoint,
                    num(pair["threshold"], 2),
                    num(pair["reference_coverage"]),
                    num(pair["reference_risk"]),
                    str(q["distinct_reported_values"]),
                    f"{num(q['largest_atom_value'], 2)} ({num(q['largest_atom_mass'], 2)})",
                ]
            )
            + r" \tabularnewline"
        )
    return "\n".join(rows)


def table_contrasts(analysis: dict) -> str:
    rows = []
    for pair in analysis["pairs"]:
        if pair["status"] != "estimated":
            continue
        model, mode, data = pair_label(pair)
        for contrast, payload in sorted(pair["contrasts"].items()):
            d = payload["deltas"]
            holm = payload.get("holm_adjusted_p")
            rows.append(
                " & ".join(
                    [
                        model,
                        mode,
                        data,
                        SHIFT_LABEL[contrast],
                        num(payload.get("answer_changed"), 2),
                        num(payload.get("confidence_changed"), 2),
                        num(d["q"]["delta"], 3, signed=True),
                        num(d["K"]["delta"], 3, signed=True),
                        ci(d["R"], signed=True),
                        num(d["S"]["delta"], 3, signed=True),
                        "--" if holm is None else num(holm),
                        CLASS_LABEL.get(payload.get("statistical_classification"), "--"),
                    ]
                )
                + r" \tabularnewline"
            )
    return "\n".join(rows)


def table_partition(analysis: dict) -> str:
    rows = []
    for pair in analysis["pairs"]:
        if pair["status"] != "estimated":
            continue
        model, mode, data = pair_label(pair)
        ref = pair["reference_endpoints"]
        for contrast, payload in sorted(pair["contrasts"].items()):
            d = payload["deltas"]
            rows.append(
                " & ".join(
                    [
                        model,
                        mode,
                        data,
                        SHIFT_LABEL[contrast],
                        num(ref["S"]),
                        num(ref["E"]),
                        num(d["S"]["delta"], 3, signed=True),
                        num(d["E"]["delta"], 3, signed=True),
                        num(d["q"]["delta"], 3, signed=True),
                        num(d["aC"]["delta"], 3, signed=True),
                        num(d["aTau"]["delta"], 3, signed=True),
                    ]
                )
                + r" \tabularnewline"
            )
    return "\n".join(rows)


def table_precision(properties: dict) -> str:
    rows = []
    for pair in properties["pairs"]:
        model, mode, data = pair_label(pair)
        p = pair["selective_risk_precision"]
        for contrast, row in sorted(p["contrasts"].items()):
            rows.append(
                " & ".join(
                    [
                        model,
                        mode,
                        data,
                        SHIFT_LABEL[contrast],
                        str(p["implied_accepted_at_held_out"]),
                        num(p["asymptotic_se_at_held_out"]),
                        num(row["bootstrap_ci_width"]),
                        f"{num(row['width_over_equivalence_band'], 1)}$\\times$",
                        "yes" if row["band_could_contain_interval"] else r"\textbf{no}",
                    ]
                )
                + r" \tabularnewline"
            )
    return "\n".join(rows)


def table_seeds(seeds: dict) -> str:
    rows = []
    for pair in seeds["pairs"]:
        if pair.get("status") != "estimated":
            continue
        model, mode, data = pair_label(pair)
        for contrast, payload in sorted(pair["contrasts"].items()):
            across = payload["across_seed_summary"]
            decomposition = payload["variance_decomposition"]
            for endpoint in ("R", "K", "S"):
                row = across.get(endpoint)
                if row is None:
                    continue
                share = (decomposition.get(endpoint) or {}).get(
                    "seed_share_of_variance"
                )
                values = ", ".join(
                    num(v, 3, signed=True)
                    for v in row["per_seed"].values()
                    if v is not None
                )
                rows.append(
                    " & ".join(
                        [
                            model,
                            mode,
                            data,
                            SHIFT_LABEL[contrast],
                            f"$\\Delta {endpoint}$",
                            values,
                            num(row["sd"]),
                            "yes" if row["sign_consistent"] else r"\textbf{no}",
                            "--" if share is None else num(share, 2),
                        ]
                    )
                    + r" \tabularnewline"
                )
    return "\n".join(rows)


# --------------------------------------------------------------------------
# macros
# --------------------------------------------------------------------------


def macros(
    analysis: dict, quality: dict, quality_cal: dict, thresholds: dict,
    properties: dict, seeds: dict, counts: dict,
) -> str:
    out: list[str] = []

    def define(name: str, value) -> None:
        out.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    # ---- scale
    for key, value in counts.items():
        define(key, f"{value:,}" if isinstance(value, int) and value >= 1000 else value)

    # ---- classification counts
    classes = analysis["classification_counts"]
    total = sum(classes.values())
    define("NumContrasts", total)
    define("NumChangeDetected", classes.get("change_detected", 0))
    define("NumEquivalent", classes.get("equivalent_within_margin", 0))
    define("NumInconclusive", classes.get("inconclusive", 0))
    define("MarginRisk", analysis["equivalence_margins"]["selective_risk"])
    define("MarginCoverage", analysis["equivalence_margins"]["coverage"])
    define("BootstrapReplicates", f"{analysis['bootstrap_replicates']:,}")

    separation = analysis["equivalence_realisation_separation"]
    define("MaxRealisationEquivalent", num(separation["max_realisation_among_equivalent"]))
    define(
        "MinRealisationNonEquivalent",
        num(separation["min_realisation_among_non_equivalent"]),
    )
    define(
        "NumAllAnswersChanged",
        sum(
            1
            for pair in analysis["pairs"]
            if pair["status"] == "estimated"
            for payload in pair["contrasts"].values()
            if payload["answer_changed"] == 1.0
        ),
    )

    # ---- the single detected change
    for pair in analysis["pairs"]:
        for contrast, payload in (pair.get("contrasts") or {}).items():
            if payload.get("reject_at_familywise_alpha"):
                model, mode, data = pair_label(pair)
                d = payload["deltas"]["R"]
                define("DetectedModel", f"{model} {mode}")
                define("DetectedDataset", data)
                define("DetectedShift", SHIFT_LABEL[contrast])
                define("DetectedDeltaR", num(d["delta"], 3, signed=True))
                define("DetectedCILow", num(d["ci_low"], 3, signed=True))
                define("DetectedCIHigh", num(d["ci_high"], 3, signed=True))
                define("DetectedHolmP", num(payload["holm_adjusted_p"]))

    # ---- extremes of the operational response
    widest_k = max(
        (
            (abs(p["contrasts"][c]["deltas"]["K"]["delta"]), p, c)
            for p in analysis["pairs"]
            if p["status"] == "estimated"
            for c in p["contrasts"]
        ),
        key=lambda t: t[0],
    )
    _, pair, contrast = widest_k
    model, mode, data = pair_label(pair)
    define("MaxCoverageModel", f"{model} {mode}")
    define("MaxCoverageDataset", data)
    define("MaxCoverageShift", SHIFT_LABEL[contrast])
    define("MaxCoverageDeltaK", num(pair["contrasts"][contrast]["deltas"]["K"]["delta"], 3, signed=True))
    define("MaxCoverageDeltaR", num(pair["contrasts"][contrast]["deltas"]["R"]["delta"], 3, signed=True))

    widest_q = max(
        (
            (p["contrasts"][c]["deltas"]["q"]["delta"], p, c)
            for p in analysis["pairs"]
            if p["status"] == "estimated"
            for c in p["contrasts"]
        ),
        key=lambda t: t[0],
    )
    _, pair, contrast = widest_q
    model, mode, data = pair_label(pair)
    define("MaxAnswerFailModel", f"{model} {mode}")
    define("MaxAnswerFailDataset", data)
    define("MaxAnswerFailShift", SHIFT_LABEL[contrast])
    define("MaxAnswerFailDeltaQ", num(pair["contrasts"][contrast]["deltas"]["q"]["delta"], 3, signed=True))

    # ---- calibration
    define("NumRiskTargetAttained", thresholds["risk_contract_attained"])
    define("NumRiskTargetInfeasible", thresholds["risk_contract_infeasible"])
    define("TargetRisk", thresholds["target_risk"] if "target_risk" in thresholds
           else thresholds["pairs"][0]["target_risk"])
    define("FixedCoverageTarget", thresholds["pairs"][0].get("fixed_coverage_target") or "0.5")
    worst_overshoot = max(
        (p for p in thresholds["pairs"] if p.get("endpoint", "").startswith("fixed_coverage")),
        key=lambda p: p["reference_coverage"],
    )
    model, mode, data = pair_label(worst_overshoot)
    define("OvershootModel", f"{model} {mode}")
    define("OvershootDataset", data)
    define("OvershootCoverage", num(worst_overshoot["reference_coverage"]))

    # ---- reference probability quality (held-out)
    bss = [p["metrics"]["brier_skill_score"] for p in quality["pairs"]]
    define("NumBSSBelowZero", sum(1 for b in bss if b["ci_high"] < 0))
    define("NumQualityPairs", len(bss))
    aurocs = [p["metrics"]["auroc"] for p in quality["pairs"]]
    define("NumAurocContainsChance", sum(1 for a in aurocs if a["ci_low"] <= 0.5 <= a["ci_high"]))
    define("NumAurocAboveChance", sum(1 for a in aurocs if a["ci_low"] > 0.5))
    define("NumAurocBelowChance", sum(1 for a in aurocs if a["ci_high"] < 0.5))
    define("MinAuroc", num(min(a["estimate"] for a in aurocs)))
    define("MaxAuroc", num(max(a["estimate"] for a in aurocs)))
    gaps = [p["metrics"]["confidence_accuracy_gap"]["estimate"] for p in quality["pairs"]]
    define("MinGap", num(min(gaps), 3, signed=True))
    define("MaxGap", num(max(gaps), 3, signed=True))

    # the sharpest ranking-versus-probability case: highest AUROC
    best = max(quality["pairs"], key=lambda p: p["metrics"]["auroc"]["estimate"])
    model, mode, data = pair_label(best)
    m = best["metrics"]
    define("RankingCaseModel", f"{model} {mode}")
    define("RankingCaseDataset", data)
    define("RankingCaseAuroc", num(m["auroc"]["estimate"]))
    define("RankingCaseBSS", num(m["brier_skill_score"]["estimate"], 3, signed=True))
    define("RankingCaseMCB", num(m["miscalibration_ratio"]["estimate"]))
    define("RankingCaseDSC", num(m["discrimination_skill"]["estimate"]))
    define("RankingCaseAccuracy", num(m["accuracy"]["estimate"]))
    define("RankingCaseMeanC", num(m["mean_confidence"]["estimate"]))

    # ---- quantisation
    atoms = [p["confidence_quantisation"] for p in properties["pairs"]]
    define("MinDistinctConfidence", min(a["distinct_reported_values"] for a in atoms))
    define("MaxDistinctConfidence", max(a["distinct_reported_values"] for a in atoms))
    define("MaxAtomMass", num(max(a["largest_atom_mass"] for a in atoms), 2))

    # ---- precision
    fits = [
        row["band_could_contain_interval"]
        for p in properties["pairs"]
        for row in p["selective_risk_precision"]["contrasts"].values()
    ]
    define("NumCannotFitBand", sum(1 for f in fits if not f))
    define("NumCanFitBand", sum(1 for f in fits if f))
    widths = [
        row["width_over_equivalence_band"]
        for p in properties["pairs"]
        for row in p["selective_risk_precision"]["contrasts"].values()
    ]
    define("MaxWidthRatio", num(max(widths), 1))
    accepted = [p["selective_risk_precision"]["implied_accepted_at_held_out"] for p in properties["pairs"]]
    define("MinAcceptedN", min(accepted))
    define("MaxAcceptedN", max(accepted))

    # ---- seeds
    signs = [
        row["sign_consistent"]
        for p in seeds["pairs"]
        if p.get("status") == "estimated"
        for payload in p["contrasts"].values()
        for endpoint, row in payload["across_seed_summary"].items()
        if endpoint == "R"
    ]
    define("NumSeedSignFlipRisk", sum(1 for s in signs if not s))
    define("NumSeedContrasts", len(signs))
    all_signs = [
        row["sign_consistent"]
        for p in seeds["pairs"]
        if p.get("status") == "estimated"
        for payload in p["contrasts"].values()
        for endpoint, row in payload["across_seed_summary"].items()
        if endpoint in ("R", "K", "S")
    ]
    define("NumSeedSignFlipAll", sum(1 for s in all_signs if not s))
    define("NumSeedEndpointCombos", len(all_signs))
    shares = sorted(
        share
        for p in seeds["pairs"]
        if p.get("status") == "estimated"
        for payload in p["contrasts"].values()
        for dec in payload["variance_decomposition"].values()
        if (share := dec["seed_share_of_variance"]) is not None
    )
    define("NumVarianceSplits", len(shares))
    define("NumSeedShareAboveHalf", sum(1 for s in shares if s > 0.5))
    define("MedianSeedShare", num(shares[len(shares) // 2]))
    define("MinSeedShare", num(min(shares)))
    define("MaxSeedShare", num(max(shares)))
    sds = [
        p["contrasts"][c]["across_seed_summary"]["R"]["sd"]
        for p in seeds["pairs"]
        if p.get("status") == "estimated"
        for c in p["contrasts"]
        if p["contrasts"][c]["across_seed_summary"].get("R")
    ]
    margin = analysis["equivalence_margins"]["selective_risk"]
    define("NumSeedSDExceedsMargin", sum(1 for s in sds if s > margin))
    define("MaxSeedSD", num(max(sds)))
    define("SeedSubsetItems", seeds["pairs"][0]["items"])
    define("SeedList", ", ".join(str(s) for s in seeds["pairs"][0]["seeds"]))
    define("NumSeeds", len(seeds["pairs"][0]["seeds"]))

    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path("paper/generated"))
    args = parser.parse_args()

    root = args.repository_root.resolve()
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    def load(path: Path) -> dict:
        return json.loads((root / path).read_text(encoding="utf-8"))

    analysis = load(CORE / "held-out-analysis.json")
    quality = load(CORE / "reference-confidence-quality-heldout.json")
    quality_cal = load(CORE / "reference-confidence-quality.json")
    thresholds = load(PROTOCOL / "calibration_thresholds_v1.json")
    properties = load(CORE / "measurement-properties.json")
    seeds = load(CORE / "seed-robustness.json")

    # Record counts come from a frozen artifact, not from a live scan of the
    # record store, so the manuscript regenerates correctly on a machine that
    # has the published artifacts but not the 53,568 raw generations.
    counts_path = root / PROTOCOL / "record_counts_v1.json"
    counts = dict(json.loads(counts_path.read_text(encoding="utf-8"))["counts"])

    counts["NumTotalRecords"] = (
        counts["NumHeldOutRecords"]
        + counts["NumCalibrationRecords"]
        + counts["NumSeedRecords"]
        + counts["NumSmokeRecords"]
    )

    (out / "numbers.tex").write_text(
        "% Generated by scripts/make_manuscript_inputs.py from frozen artifacts.\n"
        "% Do not edit. Every result in the manuscript resolves through these.\n"
        + macros(analysis, quality, quality_cal, thresholds, properties, seeds, counts),
        encoding="utf-8",
    )
    specs = {
        "table-reference-quality": dict(
            label="tab:refquality", spec="lllccccccc", face="footnotesize",
            caption=(
                "Reference-configuration confidence quality on held-out data "
                "(600 items per pair). Intervals are item-bootstrap percentiles. "
                "BSS is the Brier skill score against the base-rate forecast; "
                "MCB/UNC and DSC/UNC are the normalised miscalibration and "
                "discrimination components of the Brier score. These are "
                "secondary descriptive comparisons and carry no multiplicity "
                "correction."),
            header=(
                r"\textbf{Model} & \textbf{Regime} & \textbf{Data} & \textbf{Acc.} & "
                r"\textbf{Mean $C$} & \textbf{Gap} & \textbf{AUROC (95\% CI)} & "
                r"\textbf{BSS (95\% CI)} & \textbf{MCB/UNC} & \textbf{DSC/UNC}"
                r" \\"),
            rows=table_reference_quality(quality)),
        "table-thresholds": dict(
            label="tab:thresholds", spec="llllcccrl",
            caption=(
                "Frozen thresholds and the granularity of the confidence "
                "channel. The endpoint column records which branch of the "
                "prespecified rule applied. The last two columns describe the "
                "reported-confidence distribution at calibration."),
            header=(
                r"\textbf{Model} & \textbf{Regime} & \textbf{Data} & \textbf{Endpoint} & "
                r"$\boldsymbol{\tau_0}$ & \textbf{Coverage} & \textbf{Risk} & "
                r"\textbf{Values} & \textbf{Top atom (mass)} \\"),
            rows=table_thresholds(thresholds, properties)),
        "table-contrasts": dict(
            label="tab:contrasts", spec="llllcccccccl", face="footnotesize",
            caption=(
                "The 16 prespecified held-out contrasts. $I_A$ and $I_C$ are the "
                "fractions of answer and confidence generations the shift "
                "changed. Intervals are 95\\% item-clustered paired bootstrap; "
                "$p$ is Holm-adjusted within each model--dataset family."),
            header=(
                r"\textbf{Model} & \textbf{Regime} & \textbf{Data} & \textbf{Shift} & "
                r"$\boldsymbol{I_A}$ & $\boldsymbol{I_C}$ & $\boldsymbol{\Delta q}$ & "
                r"$\boldsymbol{\Delta K}$ & $\boldsymbol{\Delta R}$ \textbf{(95\% CI)} & "
                r"$\boldsymbol{\Delta S}$ & \textbf{Holm} $\boldsymbol{p}$ & "
                r"\textbf{Class} \\"),
            rows=table_contrasts(analysis)),
        "table-partition": dict(
            label="tab:partition", spec="llllccccccc", face="footnotesize",
            caption=(
                "The outcome partition. $S$ and $E$ at the reference configuration, "
                "then the change in each of the five disjoint components under "
                "each shift. By construction $S+E+q+a_C+a_\\tau=1$ and $K=S+E$."),
            header=(
                r"\textbf{Model} & \textbf{Regime} & \textbf{Data} & \textbf{Shift} & "
                r"$\boldsymbol{S_0}$ & $\boldsymbol{E_0}$ & $\boldsymbol{\Delta S}$ & "
                r"$\boldsymbol{\Delta E}$ & $\boldsymbol{\Delta q}$ & "
                r"$\boldsymbol{\Delta a_C}$ & $\boldsymbol{\Delta a_\tau}$ \\"),
            rows=table_partition(analysis)),
        "table-precision": dict(
            label="tab:precision", spec="llllcrccl",
            caption=(
                "Precision of the selective-risk contrasts against the "
                "equivalence region. The accepted count is $NK$ at the reference "
                "configuration. The last column records whether the 95\\% "
                "interval could have fitted inside the region at all."),
            header=(
                r"\textbf{Model} & \textbf{Regime} & \textbf{Data} & \textbf{Shift} & "
                r"\textbf{Accepted} $\boldsymbol{n}$ & \textbf{SE} & "
                r"\textbf{CI width} & \textbf{vs band} & \textbf{Could fit} \\"),
            rows=table_precision(properties)),
        "table-seeds": dict(
            label="tab:seeds", spec="llllclccr", face="footnotesize",
            caption=(
                "Seed robustness on the 128-item held-out subset. The variance "
                "share is the fraction of per-item contrast variability "
                "attributable to resampling the decoder; selective risk is a "
                "ratio with no per-item value and carries no split."),
            header=(
                r"\textbf{Model} & \textbf{Regime} & \textbf{Data} & \textbf{Shift} & "
                r"\textbf{Endpoint} & \textbf{Per-seed values} & \textbf{SD} & "
                r"\textbf{Sign stable} & \textbf{Seed share} \\"),
            rows=table_seeds(seeds)),
    }
    for name, kwargs in specs.items():
        (out / f"{name}.tex").write_text(table(**kwargs), encoding="utf-8")

    print(
        json.dumps(
            {
                "out": str(out),
                "macros": (out / "numbers.tex").read_text().count("newcommand"),
                "tables": 6,
                "total_records": counts["NumTotalRecords"],
            }
        )
    )


if __name__ == "__main__":
    main()
