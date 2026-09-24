#!/usr/bin/env python3
"""Render the frozen core's tables, figures and provenance from saved outputs.

Every number here is read from an immutable artifact; nothing is recomputed
from raw generations and no model is invoked. The figures are plain SVG so the
report has no plotting dependency and each one is regenerable from the same
JSON that produced the tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ENDPOINT_LABEL = {
    "q": "answer failure",
    "K": "coverage",
    "R": "selective risk",
    "S": "operational success",
}
SHIFT_LABEL = {
    "reference_to_high_temperature": "high-temperature",
    "reference_to_low_top_p": "low-top-p",
}
CLASS_LABEL = {
    "change_detected": "change",
    "equivalent_within_margin": "equivalent",
    "inconclusive": "inconclusive",
    "not_estimable": "not estimable",
}


def _rate(value) -> str:
    return "-" if value is None else f"{value:.2f}"


def _short(model_id: str) -> str:
    return model_id.split("/")[-1]


DATASET_LABEL = {"SuperGPQA": "SuperGPQA"}


def _dataset(dataset_id: str) -> str:
    tail = dataset_id.split("/")[-1]
    if "ManyIFEval" in tail or "Instruction" in tail:
        return "ManyIFEval"
    return DATASET_LABEL.get(tail, tail[:16])


def calibration_table(thresholds: dict) -> str:
    lines = [
        "| model | mode | dataset | status | tau0 | coverage | risk | answer failure |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    def num(entry: dict, key: str) -> str:
        value = entry.get(key)
        return "-" if value is None else f"{value:.3f}"

    for p in thresholds["pairs"]:
        if p["status"] == "selected":
            lines.append(
                f"| {_short(p['model_id'])} | {p['model_mode']} | {_dataset(p['dataset_id'])} "
                f"| selected | {num(p, 'threshold')} | {num(p, 'reference_coverage')} "
                f"| {num(p, 'reference_risk')} | {num(p, 'reference_answer_failure_rate')} |"
            )
        else:
            lines.append(
                f"| {_short(p['model_id'])} | {p['model_mode']} | {_dataset(p['dataset_id'])} "
                f"| **infeasible** | - | - | - | - |"
            )
    return "\n".join(lines)


def reference_quality_table(quality: dict) -> str:
    """Secondary descriptive metrics at theta_0, each with its interval.

    AUROC is shown with the interval because a point estimate on either side of
    0.5 cannot say whether the ranking is reversed or merely noisy, and these
    are eight uncorrected comparisons.
    """
    lines = [
        "| model | mode | dataset | accuracy | mean C | gap | AUROC (95% CI) | excludes 0.5 | BSS (95% CI) | MCB/UNC | DSC/UNC |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |",
    ]

    def ci(metric: dict) -> str:
        if metric["estimate"] is None:
            return "-"
        if metric["ci_low"] is None:
            return f"{metric['estimate']:.3f}"
        return f"{metric['estimate']:.3f} [{metric['ci_low']:.3f}, {metric['ci_high']:.3f}]"

    for pair in quality["pairs"]:
        m = pair["metrics"]
        excludes = pair["auroc_excludes_chance"]
        lines.append(
            f"| {_short(pair['model_id'])} | {pair['model_mode']} "
            f"| {_dataset(pair['dataset_id'])} "
            f"| {m['accuracy']['estimate']:.3f} | {m['mean_confidence']['estimate']:.3f} "
            f"| {m['confidence_accuracy_gap']['estimate']:+.3f} | {ci(m['auroc'])} "
            f"| {'-' if excludes is None else ('yes' if excludes else 'no')} "
            f"| {ci(m['brier_skill_score'])} "
            f"| {m['miscalibration_ratio']['estimate']:.3f} "
            f"| {m['discrimination_skill']['estimate']:.3f} |"
        )
    lines.append("")
    lines.append(
        f"Reference coordinate, {quality.get('partition', 'calibration')} "
        f"partition. Secondary and descriptive: these eight comparisons are not "
        f"in the confirmatory family and carry no multiplicity correction, so "
        f"an interval that excludes 0.5 by a narrow margin is weak evidence of "
        f"reversed ranking on its own."
    )
    lines.append("")
    lines.append(
        "BSS = (DSC - MCB)/UNC against the base-rate forecast, so a negative "
        "skill score says miscalibration outweighs whatever discrimination the "
        "forecast carries. Both ratios are shown because only their difference "
        "appears in the skill score and they are separate failures: a forecast "
        "that ranks well can still be worthless as a probability."
    )
    return "\n".join(lines)


def endpoint_table(analysis: dict) -> str:
    lines = [
        "| model | mode | dataset | shift | I_ans | I_conf | dq | dK | dR (95% CI) | dS | Holm p | class |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for pair in analysis["pairs"]:
        if pair["status"] != "estimated":
            lines.append(
                f"| {_short(pair['model_id'])} | {pair['model_mode']} "
                f"| {_dataset(pair['dataset_id'])} | - | - | - | - | - | "
                f"no primary estimate; calibration infeasible | - | - |"
            )
            continue
        for contrast, payload in sorted(pair["contrasts"].items()):
            d = payload["deltas"]
            realisation = (pair.get("generations_changed_by_shift") or {}).get(
                SHIFT_LABEL[contrast]
            ) or {}
            def fmt(key: str) -> str:
                v = d[key]["delta"]
                return "-" if v is None else f"{v:+.3f}"
            r = d["R"]
            ci = (
                "-"
                if r["delta"] is None or r["ci_low"] is None
                else f"{r['delta']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
            )
            holm = payload.get("holm_adjusted_p")
            lines.append(
                f"| {_short(pair['model_id'])} | {pair['model_mode']} "
                f"| {_dataset(pair['dataset_id'])} | {SHIFT_LABEL[contrast]} "
                f"| {_rate(realisation.get('answer_changed'))} "
                f"| {_rate(realisation.get('confidence_changed'))} "
                f"| {fmt('q')} | {fmt('K')} | {ci} | {fmt('S')} "
                f"| {'-' if holm is None else f'{holm:.3f}'} "
                f"| {CLASS_LABEL.get(payload.get('statistical_classification'), '-')} |"
            )
    return "\n".join(lines)


def partition_table(analysis: dict) -> str:
    """The five disjoint outcomes at the frozen threshold, and their changes.

    Selective risk is conditional on acceptance, so a flat R says nothing about
    how much work the policy accepted at all. The partition sums to one and
    shows which channel absorbed each shift.
    """
    lines = [
        "| model | mode | dataset | shift | dS | dE | dq | da_C | da_tau | dK |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair in analysis["pairs"]:
        if pair["status"] != "estimated":
            continue
        for contrast, payload in sorted(pair["contrasts"].items()):
            d = payload["deltas"]

            def fmt(key: str) -> str:
                v = d.get(key, {}).get("delta")
                return "-" if v is None else f"{v:+.3f}"

            lines.append(
                f"| {_short(pair['model_id'])} | {pair['model_mode']} "
                f"| {_dataset(pair['dataset_id'])} | {SHIFT_LABEL[contrast]} "
                f"| {fmt('S')} | {fmt('E')} | {fmt('q')} | {fmt('aC')} "
                f"| {fmt('aTau')} | {fmt('K')} |"
            )
    lines.append("")
    lines.append(
        "S accepted-correct, E accepted-wrong, q answer failure, a_C confidence "
        "failure, a_tau low-confidence abstention. S + E + q + a_C + a_tau = 1 "
        "and K = S + E."
    )
    return "\n".join(lines)


def precision_table(properties: dict) -> str:
    """Whether each contrast could have established equivalence at all.

    An interval wider than the whole equivalence band cannot fall inside it
    whatever the true effect, so calling such a contrast inconclusive describes
    the measurement rather than the phenomenon.
    """
    margin = properties["equivalence_margin_selective_risk"]
    lines = [
        f"| model | mode | dataset | shift | accepted n | CI width | band (+-{margin}) | could fit |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    attainable = total = 0
    for pair in properties["pairs"]:
        p = pair["selective_risk_precision"]
        for contrast, row in sorted(p["contrasts"].items()):
            total += 1
            fits = row["band_could_contain_interval"]
            attainable += bool(fits)
            lines.append(
                f"| {_short(pair['model_id'])} | {pair['model_mode']} "
                f"| {_dataset(pair['dataset_id'])} "
                f"| {contrast.replace('reference_to_', '').replace('_', '-')} "
                f"| {p['implied_accepted_at_held_out']} "
                f"| {row['bootstrap_ci_width']:.3f} "
                f"| {row['width_over_equivalence_band']:.1f}x | "
                f"{'yes' if fits else '**no**'} |"
            )
    lines.append("")
    lines.append(
        f"For {total - attainable} of {total} contrasts the realised 95% "
        f"interval was wider than the entire prespecified equivalence region, "
        f"so it could not have fallen inside that region wherever it was "
        f"centred. This describes the precision achieved here, not a bound on "
        f"what a sample of this size could achieve: a different true risk, "
        f"with the same N, would give a different width. The reason the "
        f"achieved precision is what it is: selective risk is estimated on "
        f"accepted items, so its effective sample size is NK rather than N."
    )
    return "\n".join(lines)


def quantisation_table(properties: dict) -> str:
    """Why a nominal coverage target is not generally reachable."""
    lines = [
        "| model | mode | dataset | distinct values | entropy (bits) | largest atom | its mass | coverage at tau0 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair in properties["pairs"]:
        q = pair["confidence_quantisation"]
        p = pair["selective_risk_precision"]
        lines.append(
            f"| {_short(pair['model_id'])} | {pair['model_mode']} "
            f"| {_dataset(pair['dataset_id'])} | {q['distinct_reported_values']} "
            f"| {q['entropy_bits']:.2f} | {q['largest_atom_value']:.2f} "
            f"| {q['largest_atom_mass']:.2f} | {p['reference_coverage']:.3f} |"
        )
    lines.append("")
    lines.append(
        "Coverage is a step function of the threshold whose jumps are the "
        "confidence atoms. Where one value carries more than half the mass, a "
        "nominal 0.50 coverage target is not reachable and the achieved "
        "coverage overshoots it. That is a property of the elicited "
        "confidence channel, not a failure to optimise the threshold."
    )
    return "\n".join(lines)


def seed_table(seeds: dict) -> str:
    """Per-seed contrast estimates, their spread, and the variance split.

    The broad study's intervals resample items at one draw from the decoder.
    These columns say how much the same contrast moves when the decoder is
    resampled, on a 128-item subset of the same held-out items.
    """
    lines = [
        "| model | mode | dataset | shift | endpoint | per-seed | sd | range | sign consistent | seed share of variance |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | ---: |",
    ]
    inconsistent = 0
    total = 0
    for pair in seeds["pairs"]:
        if pair.get("status") != "estimated":
            continue
        for contrast, payload in sorted(pair["contrasts"].items()):
            across = payload["across_seed_summary"]
            decomposition = payload["variance_decomposition"]
            for endpoint in ("R", "K", "S"):
                row = across.get(endpoint)
                if row is None:
                    continue
                total += 1
                if not row["sign_consistent"]:
                    inconsistent += 1
                share = (decomposition.get(endpoint) or {}).get(
                    "seed_share_of_variance"
                )
                values = "/".join(
                    f"{v:+.3f}" for v in row["per_seed"].values() if v is not None
                )
                lines.append(
                    f"| {_short(pair['model_id'])} | {pair['model_mode']} "
                    f"| {_dataset(pair['dataset_id'])} | {SHIFT_LABEL[contrast]} "
                    f"| d{endpoint} | {values} | {row['sd']:.3f} "
                    f"| {row['range']:.3f} | {'yes' if row['sign_consistent'] else '**no**'} "
                    f"| {'-' if share is None else f'{share:.2f}'} |"
                )
    shares = sorted(
        share
        for pair in seeds["pairs"]
        if pair.get("status") == "estimated"
        for payload in pair["contrasts"].values()
        for decomposition in payload["variance_decomposition"].values()
        if (share := decomposition["seed_share_of_variance"]) is not None
    )
    lines.append("")
    lines.append(
        f"{inconsistent} of {total} reported contrast-endpoint combinations "
        f"changed sign across seeds. Selective risk carries no variance split: "
        f"it is a ratio with no per-item value, so decomposing it would "
        f"restrict to items accepted at two or more seeds."
    )
    if shares:
        above = sum(1 for s in shares if s > 0.5)
        middle = shares[len(shares) // 2]
        lines.append("")
        lines.append(
            f"Across the {len(shares)} combinations that admit a variance "
            f"split, the seed share exceeded one half in {above} of them "
            f"(median {middle:.3f}, range {min(shares):.3f} to "
            f"{max(shares):.3f}). This is a panel-level description, not a "
            f"property of every contrast: {len(shares) - above} combinations "
            f"are dominated by item heterogeneity instead. Three seeds on 128 "
            f"items cannot support a stronger claim than that within-item "
            f"variation across decoder realisations was frequently the larger "
            f"of the two components."
        )
    lines.append("")
    lines.append(
        "This is a 128-item subset, so its per-seed point estimates are noisier "
        "than the 600-item held-out estimates and do not replace them. All "
        "three seeds were generated in one run because generation is not "
        "invariant to batch composition, so what is measured is the "
        "variability of a rerun under one batching regime."
    )
    return "\n".join(lines)


def realisation_table(analysis: dict) -> str:
    """Where the shift's observable effect appeared in the two-stage pipeline.

    I_ans and I_conf are a manipulation diagnostic, not pathway estimates: the
    readout is decoded after the answer, so a changed confidence output can
    follow a changed answer or arise in the readout itself. The last column
    conditions on items whose answer the shift left byte-identical, which
    removes the first route by construction but only on that subset.
    """
    lines = [
        "| model | mode | dataset | shift | I_ans | I_conf | I_conf given answer identical | n |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    undefined = 0
    for pair in analysis["pairs"]:
        if pair["status"] != "estimated":
            continue
        for contrast, payload in sorted(pair["contrasts"].items()):
            stable = payload.get("confidence_changed_given_answer_identical")
            n = payload.get("items_with_identical_answer") or 0
            if stable is None:
                undefined += 1
            lines.append(
                f"| {_short(pair['model_id'])} | {pair['model_mode']} "
                f"| {_dataset(pair['dataset_id'])} | {SHIFT_LABEL[contrast]} "
                f"| {_rate(payload.get('answer_changed'))} "
                f"| {_rate(payload.get('confidence_changed'))} "
                f"| {'undefined' if stable is None else f'{stable:.3f}'} | {n} |"
            )
    lines.append("")
    lines.append(
        "A manipulation diagnostic, not a causal decomposition. The conditional "
        "column is identified only on items the shift left alone, and that "
        "subset is enriched for items the model answers stably rather than "
        "being a random sample."
    )
    if undefined:
        lines.append("")
        lines.append(
            f"{undefined} of these contrasts changed every compared answer, "
            f"leaving no items on which to condition; the question cannot be "
            f"asked of them in this design."
        )
    return "\n".join(lines)


def classification_summary(analysis: dict) -> str:
    """Descriptive tally of the per-contrast classifications.

    A count of independently classified contrasts, not a pooled estimate: no
    weighting, no interval and no test across the panel. The realisation
    separation is printed with it because an equivalence classification earned
    by an intervention that did not happen is not evidence that a policy ported.
    """
    counts = analysis.get("classification_counts") or {}
    separation = analysis.get("equivalence_realisation_separation") or {}
    total = sum(counts.values())
    lines = [
        "| classification | contrasts |",
        "| --- | ---: |",
    ]
    for key in ("change_detected", "equivalent_within_margin", "inconclusive", "not_estimable"):
        if key in counts:
            lines.append(f"| {CLASS_LABEL[key]} | {counts[key]} of {total} |")
    strongest = separation.get("max_realisation_among_equivalent")
    weakest = separation.get("min_realisation_among_non_equivalent")
    lines.append("")
    lines.append(
        "These are counts of contrasts classified one at a time. No pooled or "
        "weighted decoder effect is estimated here or anywhere in the analysis; "
        "the tally describes this panel and does not generalise to decoder "
        "shifts at large."
    )
    if separation.get("separated") and strongest is not None and weakest is not None:
        lines.append("")
        lines.append(
            f"Every contrast classified equivalent has realisation at most "
            f"{strongest:.3f}, while every other contrast has realisation at "
            f"least {weakest:.3f}. The equivalence classifications therefore sit "
            f"entirely in the weakly realised tail of the panel, which limits "
            f"how far they support a portability claim. This is a statement "
            f"about where those contrasts fall in the observed realisation "
            f"distribution; it does not reclassify a realisation of "
            f"{strongest:.3f} as an intervention that did not occur."
        )
    elif counts.get("equivalent_within_margin"):
        lines.append("")
        lines.append(
            "Equivalence classifications are not confined to weakly realised "
            "contrasts; read each one against its own realisation."
        )
    return "\n".join(lines)


def forest_svg(analysis: dict, endpoint: str) -> str:
    """One row per model-dataset-shift: point estimate and interval."""
    rows = []
    for pair in analysis["pairs"]:
        if pair["status"] != "estimated":
            continue
        for contrast, payload in sorted(pair["contrasts"].items()):
            d = payload["deltas"][endpoint]
            if d["delta"] is None:
                continue
            rows.append(
                (
                    f"{_short(pair['model_id'])[:18]} {pair['model_mode'][:9]} "
                    f"{_dataset(pair['dataset_id'])[:10]} {SHIFT_LABEL[contrast][:8]}",
                    d["delta"],
                    d["ci_low"],
                    d["ci_high"],
                )
            )
    if not rows:
        return "<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10'></svg>"
    lo = min(min(r[2] if r[2] is not None else r[1] for r in rows), -0.05)
    hi = max(max(r[3] if r[3] is not None else r[1] for r in rows), 0.05)
    span = hi - lo or 1.0
    left, width, row_h, top = 330, 420, 22, 54
    height = top + row_h * len(rows) + 34

    def x(value: float) -> float:
        return left + (value - lo) / span * width

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{left + width + 40}' "
        f"height='{height}' font-family='system-ui,sans-serif' font-size='12'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='16' y='24' font-size='14' font-weight='600'>Change in "
        f"{ENDPOINT_LABEL[endpoint]} at the frozen threshold</text>",
        f"<text x='16' y='40' fill='#555'>held-out, paired item-clustered bootstrap</text>",
        f"<line x1='{x(0):.1f}' y1='{top - 8}' x2='{x(0):.1f}' y2='{height - 26}' "
        "stroke='#999' stroke-dasharray='3,3'/>",
    ]
    for index, (label, delta, ci_low, ci_high) in enumerate(rows):
        y = top + row_h * index + 10
        parts.append(
            f"<text x='16' y='{y + 4}' fill='#222'>{label}</text>"
        )
        if ci_low is not None and ci_high is not None:
            parts.append(
                f"<line x1='{x(ci_low):.1f}' y1='{y}' x2='{x(ci_high):.1f}' y2='{y}' "
                "stroke='#4a6fa5' stroke-width='2'/>"
            )
        colour = "#b23a48" if delta > 0 else "#2e7d5b"
        parts.append(
            f"<circle cx='{x(delta):.1f}' cy='{y}' r='4' fill='{colour}'/>"
        )
    parts.append(
        f"<text x='{x(lo):.1f}' y='{height - 10}' fill='#555'>{lo:+.2f}</text>"
        f"<text x='{x(hi):.1f}' y='{height - 10}' fill='#555' text-anchor='end'>{hi:+.2f}</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def provenance(root: Path, artifacts: dict[str, Path]) -> dict:
    # A released copy is not a Git checkout, and the report must still build
    # there; the commit is provenance, not an input to any number.
    def git(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else None

    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    dirty = None if status is None else bool(status)
    return {
        "git_commit": head,
        "git_dirty": dirty,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            name: {
                "path": (
                    str(path.relative_to(root))
                    if path.is_relative_to(root)
                    else str(path)
                ),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in artifacts.items()
            if path.is_file()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, default=Path("evidence/protocol/core_study_freeze_v1.json"))
    parser.add_argument(
        "--seed-robustness",
        type=Path,
        default=Path("evidence/core/seed-robustness.json"),
    )
    parser.add_argument(
        "--measurement-properties",
        type=Path,
        default=Path("evidence/core/measurement-properties.json"),
    )
    parser.add_argument(
        "--reference-quality",
        type=Path,
        # The held-out reference records, not the calibration ones: the
        # strongest probability-quality statements should not rest on the split
        # that chose tau_0. The calibration artifact is kept for comparison.
        default=Path("evidence/core/reference-confidence-quality-heldout.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("evidence/core"))
    args = parser.parse_args()

    root = args.repository_root.resolve()
    out = (root / args.out_dir).resolve()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    thresholds = json.loads((root / args.thresholds).read_text(encoding="utf-8"))
    analysis = json.loads((root / args.analysis).read_text(encoding="utf-8"))

    (out / "table-calibration-thresholds.md").write_text(
        calibration_table(thresholds) + "\n", encoding="utf-8"
    )
    (out / "table-held-out-endpoints.md").write_text(
        endpoint_table(analysis) + "\n", encoding="utf-8"
    )
    properties_path = root / args.measurement_properties
    if properties_path.is_file():
        properties = json.loads(properties_path.read_text(encoding="utf-8"))
        (out / "table-risk-precision.md").write_text(
            precision_table(properties) + "\n", encoding="utf-8"
        )
        (out / "table-confidence-quantisation.md").write_text(
            quantisation_table(properties) + "\n", encoding="utf-8"
        )
    seeds_path = root / args.seed_robustness
    if seeds_path.is_file():
        (out / "table-seed-robustness.md").write_text(
            seed_table(json.loads(seeds_path.read_text(encoding="utf-8"))) + "\n",
            encoding="utf-8",
        )
    (out / "table-outcome-partition.md").write_text(
        partition_table(analysis) + "\n", encoding="utf-8"
    )
    (out / "table-realisation.md").write_text(
        realisation_table(analysis) + "\n", encoding="utf-8"
    )
    (out / "table-classification-summary.md").write_text(
        classification_summary(analysis) + "\n", encoding="utf-8"
    )
    quality_path = root / args.reference_quality
    if quality_path.is_file():
        (out / "table-reference-confidence-quality.md").write_text(
            reference_quality_table(
                json.loads(quality_path.read_text(encoding="utf-8"))
            )
            + "\n",
            encoding="utf-8",
        )
    for endpoint in ("R", "S", "K"):
        (out / "figures" / f"forest-{endpoint}.svg").write_text(
            forest_svg(analysis, endpoint), encoding="utf-8"
        )
    record = provenance(
        root,
        {
            "freeze": root / args.freeze,
            "calibration_thresholds": root / args.thresholds,
            "held_out_analysis": root / args.analysis,
            "table_calibration": out / "table-calibration-thresholds.md",
            "table_endpoints": out / "table-held-out-endpoints.md",
            "table_classification": out / "table-classification-summary.md",
            "table_realisation": out / "table-realisation.md",
            "measurement_properties": properties_path,
            "table_outcome_partition": out / "table-outcome-partition.md",
            "seed_robustness": seeds_path,
            "table_seed_robustness": out / "table-seed-robustness.md",
            "table_risk_precision": out / "table-risk-precision.md",
            "table_confidence_quantisation": out / "table-confidence-quantisation.md",
            "reference_confidence_quality": quality_path,
            "table_reference_quality": out / "table-reference-confidence-quality.md",
            "figure_risk": out / "figures" / "forest-R.svg",
            "figure_success": out / "figures" / "forest-S.svg",
            "figure_coverage": out / "figures" / "forest-K.svg",
        },
    )
    (out / "provenance.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"out_dir": str(out), "artifacts": len(record["artifacts"])}))


if __name__ == "__main__":
    main()
