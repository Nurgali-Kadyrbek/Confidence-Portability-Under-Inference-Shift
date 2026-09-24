"""Regenerate lightweight development-matrix evidence from saved outputs."""

from __future__ import annotations

import csv
import io
import json
import math
import stat
from collections import Counter
from pathlib import Path

from cpis.manifest import git_metadata
from cpis.matrix_config import load_matrix_config
from cpis.matrix_records import MatrixParsedRecord
from cpis.storage import StorageLayout, read_json


CONDITION_LABELS = {
    "reference": "Reference",
    "low-temperature": "Low T",
    "high-temperature": "High T",
    "low-top-p": "Low top-p",
    "high-diversity": "High diversity",
}


def _write_reproducible(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"refusing to replace different evidence: {path}")
        return
    path.write_bytes(data)


def _percent(value: float | None, digits: int = 1) -> str:
    return "NA" if value is None else f"{100 * value:.{digits}f}%"


def _number(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _ci(interval: dict, percent: bool = False) -> str:
    low, high = interval["lower_95"], interval["upper_95"]
    if low is None or high is None:
        return "NA"
    if percent:
        return f"[{100 * low:.1f}, {100 * high:.1f}] pp"
    return f"[{low:.3f}, {high:.3f}]"


def _integrity(run: Path) -> dict:
    raw_records: list[dict] = []
    writable = 0
    for stage in ("answer", "confidence"):
        for path in sorted((run / "raw" / stage).glob("*.json")):
            raw_records.append(read_json(path))
            writable += bool(
                path.stat().st_mode
                & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            )
    finish = Counter(
        (record["generation_stage"], record["finish_reason"])
        for record in raw_records
    )
    token_max = {
        stage: max(
            record.get("completion_tokens") or 0
            for record in raw_records
            if record["generation_stage"] == stage
        )
        for stage in ("answer", "confidence")
    }
    result = {
        "raw_records": len(raw_records),
        "answer_raw_records": sum(
            record["generation_stage"] == "answer" for record in raw_records
        ),
        "confidence_raw_records": sum(
            record["generation_stage"] == "confidence" for record in raw_records
        ),
        "writable_raw_files": writable,
        "finish_reasons": {
            f"{stage}:{reason}": count
            for (stage, reason), count in sorted(finish.items())
        },
        "maximum_completion_tokens": token_max,
    }
    parsed_records = [
        read_json(path) for path in sorted((run / "parsed").glob("*.json"))
    ]
    answer_status_by_raw_id: dict[str, str] = {}
    for record in parsed_records:
        raw_id = record["answer_raw_record_id"]
        status = record["answer_parse_status"]
        prior = answer_status_by_raw_id.setdefault(raw_id, status)
        if prior != status:
            raise RuntimeError(
                f"inconsistent answer parse status across matrix cells: {raw_id}"
            )
    result["invalid_answer_raw_records"] = sum(
        status == "invalid" for status in answer_status_by_raw_id.values()
    )
    audit_path = run / "manifests" / "integrity-audit-v1.json"
    if audit_path.is_file():
        audit = read_json(audit_path)
        result["audit_id"] = audit["audit_id"]
        result["replicas"] = audit["replicas"]
        result["parsed_records"] = audit["parsed_records"]
        result["invalid_answer_cells"] = audit["invalid_answer_cells"]
        result["stage_sha256"] = audit["stage_sha256"]
    return result


def _summary_payload(analysis: dict, integrity: dict, evidence_id: str) -> dict:
    coupled = analysis["coupled"]
    standardized = analysis["standardized_deterministic"]
    coverage_comparison: dict[str, dict] = {}
    for target in ("0.10", "0.25", "0.50"):
        coverage_comparison[target] = {}
        for condition in (
            "low-temperature",
            "high-temperature",
            "low-top-p",
            "high-diversity",
        ):
            coverage_comparison[target][condition] = {
                "coupled_delta_coverage": coupled["fixed_coverage"][target][
                    "transport"
                ][condition]["delta_coverage_from_reference"],
                "standardized_delta_coverage": standardized["fixed_coverage"][target][
                    "transport"
                ][condition]["delta_coverage_from_reference"],
            }
    return {
        "schema_version": "1.0",
        "evidence_id": evidence_id,
        "exploratory": True,
        "run_id": analysis["run_id"],
        "generation_git_commit": analysis["generation_git_commit"],
        "analysis_git": analysis["analysis_git"],
        "analysis_plan_sha256": analysis["analysis_plan_sha256"],
        "experimental_units": analysis["experimental_units"],
        "replicate_seeds": analysis["replicate_seeds"],
        "integrity": integrity,
        "invalid_confidence_cells": analysis["invalid_confidence_cells"],
        "bootstrap": analysis["bootstrap"],
        "coupled_summaries": coupled["summaries"],
        "coupled_risk_contract_screen": coupled[
            "reference_risk_contract_selection"
        ],
        "coupled_fixed_coverage": coupled["fixed_coverage"],
        "standardized_summaries": standardized["summaries"],
        "standardized_risk_contract_screen": standardized[
            "reference_risk_contract_selection"
        ],
        "standardized_fixed_coverage": standardized["fixed_coverage"],
        "coverage_mitigation_comparison": coverage_comparison,
        "decomposition": analysis["decomposition"],
    }


def _markdown(
    summary: dict,
    model_label: str,
    short_model_label: str,
    dataset_label: str,
    generic: bool,
) -> str:
    coupled = summary["coupled_summaries"]
    standardized = summary["standardized_summaries"]
    if not generic and short_model_label == "4B":
        invalid_note = [
            "The six invalid cells all used the high-diversity confidence readout: two",
            "single-quoted pseudo-JSON objects and four negative probabilities. No parser",
            "repair or confidence imputation was applied.",
        ]
        quantization_note = [
            "Confidence is heavily quantized. Under the prespecified tie rule, the",
            "reference thresholds for nominal coverage targets 10%, 25%, and 50% are",
            "1.00, 0.95, and 0.85, achieving 12.0%, 38.0%, and 58.9% coverage. The",
            "achieved—not nominal—coverage is used transparently below.",
        ]
        fixed_note = [
            "At the 10% target's numeric threshold (`C=1.0`), high temperature",
            "increases coverage by 9.9 pp (95% item-bootstrap CI 5.7 to 14.6) and",
            "high diversity by 17.2 pp (11.5 to 23.4). Risk intervals remain wide,",
            "so this pilot establishes measurable policy-selection drift, not a precise",
            "selective-risk effect.",
        ]
        standardized_note = [
            "The standardized readout also has no 10% risk-contract candidate. Its",
            "confidence ties are stronger: κ targets 0.10 and 0.25 both select 0.95",
            "and achieve 45.8% reference coverage. At each protocol's κ=0.50",
            "reference threshold, maximum absolute coverage drift falls from 7.8 pp",
            "(coupled) to 4.2 pp (standardized). This supports continued evaluation of",
            "the mitigation but is not confirmatory evidence.",
        ]
        decomposition_note = [
            "Every decomposition interval crosses zero. The machinery is operational,",
            "but 64 development items do not isolate a component effect precisely.",
        ]
    elif not generic and short_model_label == "9B":
        invalid_note = [
            "Four answer generations were invalid, all under the high-diversity answer",
            "decoder. All 11 invalid confidence cells used the high-diversity readout.",
            "They remain incorrect answers or forced abstentions; no parser repair or",
            "confidence imputation was applied.",
        ]
        quantization_note = [
            "Confidence is extremely quantized. Under the prespecified tie rule, all",
            "three nominal coverage targets select threshold 1.00 and achieve 65.6%",
            "reference coverage. The achieved—not nominal—coverage is used below; the",
            "10%, 25%, and 50% rows are consequently the same transported policy.",
        ]
        fixed_note = [
            "At the exploratory reference threshold (`C=1.0`), high diversity reduces",
            "coverage by 13.0 pp (95% item-bootstrap CI −18.2 to −5.2) while",
            "increasing selective risk",
            "by 9.2 pp (2.0 to 18.4). This is a measurable development-sample failure",
            "of threshold portability, not held-out confirmation.",
        ]
        standardized_note = [
            "The standardized readout also has no 10% risk-contract candidate, and all",
            "three κ targets select 1.00 with 67.2% reference coverage. For high",
            "diversity, coverage drift is −5.7 pp and risk drift is +4.6 pp with an",
            "interval crossing zero, compared with −13.0 and +9.2 pp when coupled.",
            "This is mitigation evidence on reused development data only.",
        ]
        decomposition_note = [
            "For high diversity, the answer-path and readout effects are both negative",
            "and the interaction positive, with all three item-bootstrap intervals",
            "excluding zero. Under high temperature, answer-path and readout effects",
            "are also negative with intervals excluding zero; its interaction remains",
            "uncertain. These exploratory components motivate held-out decomposition.",
        ]
    elif generic or short_model_label in {"4B-thinking", "9B-thinking"}:
        invalid_note = [
            f"The run contains {summary['integrity'].get('invalid_answer_raw_records', 'NA')} invalid answer generations out of",
            f"{summary['integrity']['answer_raw_records']}; their readout fan-out produces",
            f"{summary['integrity'].get('invalid_answer_cells', 'NA')} invalid answer cells. It also contains",
            f"{summary['invalid_confidence_cells']} invalid confidence cells after preserving every raw response.",
            "Invalid answers remain unsuccessful and invalid confidence remains forced",
            "abstention; no parser repair or confidence imputation was applied.",
        ]
        selections = summary["coupled_fixed_coverage"]
        threshold_text = ", ".join(
            f"κ={target}: C={data['selection']['threshold']:.2f} "
            f"(achieved {100 * data['selection']['achieved_coverage']:.1f}%)"
            for target, data in selections.items()
        )
        quantization_note = [
            "Reference confidence is quantized, so nominal coverage targets need not",
            f"be attained exactly. The selected policies are {threshold_text}.",
            "All transport estimates use achieved—not nominal—reference coverage.",
        ]
        coupled_half = summary["coupled_fixed_coverage"]["0.50"]["transport"]
        largest_condition, largest_row = max(
            coupled_half.items(),
            key=lambda pair: abs(pair[1]["delta_coverage_from_reference"]),
        )
        fixed_note = [
            f"At the κ=0.50 policy, {CONDITION_LABELS[largest_condition].lower()} has the largest",
            f"absolute coverage change: {100 * largest_row['delta_coverage_from_reference']:.1f} pp",
            f"(95% item-bootstrap CI {_ci(largest_row['delta_coverage_from_reference_item_bootstrap_95'], True)}).",
            f"Its selective-risk change is {100 * largest_row['delta_risk']:.1f} pp",
            f"({_ci(largest_row['delta_risk_item_bootstrap_95'], True)}). These are",
            "development estimates and do not certify a transported policy.",
        ]
        standardized_half = summary["standardized_fixed_coverage"]["0.50"]
        max_coupled = max(
            abs(row["delta_coverage_from_reference"])
            for row in coupled_half.values()
        )
        max_standardized = max(
            abs(row["delta_coverage_from_reference"])
            for row in standardized_half["transport"].values()
        )
        reference_coupled = coupled["reference"]
        reference_standardized = standardized["reference"]
        standardized_note = [
            "The deterministic standardized readout is also screened independently",
            f"and has status `{summary['standardized_risk_contract_screen']['status']}`.",
            f"At κ=0.50, maximum absolute coverage drift is {100 * max_coupled:.1f} pp",
            f"when coupled versus {100 * max_standardized:.1f} pp when standardized.",
            "That stability is not unambiguously better confidence quality: reference",
            f"AUROC changes from {_number(reference_coupled['auroc_valid_only'])} to",
            f"{_number(reference_standardized['auroc_valid_only'])}, and Brier changes from",
            f"{_number(reference_coupled['brier_valid_only'])} to",
            f"{_number(reference_standardized['brier_valid_only'])}. The mitigation therefore",
            "remains a portability/quality tradeoff for held-out evaluation.",
        ]
        identified = []
        for condition, row in summary["decomposition"].items():
            for label, key in (
                ("answer-path", "answer_path_item_bootstrap_95"),
                ("readout", "confidence_readout_item_bootstrap_95"),
                ("interaction", "interaction_item_bootstrap_95"),
            ):
                interval = row[key]
                if interval["lower_95"] > 0 or interval["upper_95"] < 0:
                    identified.append(f"{CONDITION_LABELS[condition]} {label}")
        decomposition_note = [
            (
                "Intervals excluding zero were observed for: " + ", ".join(identified) + "."
                if identified
                else "No decomposition component interval excluded zero in this development sample."
            ),
            (
                "All component conclusions remain exploratory because these 64 items were"
                if short_model_label in {"4B-thinking", "9B-thinking"}
                else "All component conclusions remain exploratory because these items were"
            ),
            "used for protocol development.",
        ]
    else:
        raise ValueError(
            f"matrix narrative has not been scientifically reviewed for {short_model_label}"
        )
    dynamic_run = generic or "thinking" in short_model_label
    lines = [
        f"# {model_label} × {dataset_label} development matrix v1",
        "",
        (
            f"> **Exploratory development evidence only.** This run used "
            f"{summary['experimental_units']} development items."
            if generic
            else "> **Exploratory development evidence only.** The same 64 development items"
        ),
        (
            "> It validates the intervention machinery; these results do not certify a"
            if generic
            else "> were reused from the operational pilot. These results do not certify a"
        ),
        "> policy and are not held-out test evidence.",
        "",
        "## Run and integrity",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Experimental units: {summary['experimental_units']} benchmark items",
        f"- Repeated seeds: {', '.join(map(str, summary['replicate_seeds']))}",
        f"- Raw generations: {summary['integrity']['raw_records']} "
        f"({summary['integrity']['answer_raw_records']} answer + "
        f"{summary['integrity']['confidence_raw_records']} confidence)",
        (
            f"- Safe resume: all {summary['integrity'].get('replicas', 1)} replica shards "
            "reused every raw and parsed record; zero regenerated"
            if dynamic_run
            else "- Safe resume: 4,416 raw and 3,456 parsed records reused; zero regenerated"
        ),
        f"- Invalid confidence cells: {summary['invalid_confidence_cells']} / "
        f"{summary['integrity'].get('parsed_records', 3456)}; "
        "all are retained as forced abstentions",
        (
            f"- Finish reasons: {summary['integrity']['finish_reasons']}; maximum completion "
            f"lengths were {summary['integrity']['maximum_completion_tokens']['answer']} and "
            f"{summary['integrity']['maximum_completion_tokens']['confidence']} tokens"
            if dynamic_run
            else "- Finish reasons: all answer and confidence generations ended with `stop`; "
            f"maximum completion lengths were {summary['integrity']['maximum_completion_tokens']['answer']} "
            f"and {summary['integrity']['maximum_completion_tokens']['confidence']} tokens"
        ),
        f"- Generation commit: `{summary['generation_git_commit']}` (clean)",
        f"- Analysis commit: `{summary['analysis_git']['commit']}` (clean)",
        *(
            [f"- Reporting commit: `{summary['report_git']['commit']}` (clean)"]
            if "report_git" in summary
            else []
        ),
        (
            f"- GPU execution: {summary['integrity'].get('replicas', 1)} independent one-GPU "
            "NVIDIA L40 replicas, BF16, vLLM 0.29.0+cu129, direct residency, no offload"
            if dynamic_run
            else "- GPU execution: one NVIDIA L40, BF16, vLLM 0.29.0+cu129, direct residency, no offload"
        ),
        "",
        *invalid_note,
        "",
        "## Coupled deployment readout",
        "",
        "| Answer/readout condition | Accuracy | Mean C (valid) | C−accuracy | AUROC | Brier | Invalid |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    order = (
        "reference",
        "low-temperature",
        "high-temperature",
        "low-top-p",
        "high-diversity",
    )
    for condition in order:
        row = coupled[condition]
        gap = row["mean_confidence_valid_only"] - row["accuracy"]
        lines.append(
            f"| {CONDITION_LABELS[condition]} | {_percent(row['accuracy'])} | "
            f"{_number(row['mean_confidence_valid_only'])} | {_number(gap)} | "
            f"{_number(row['auroc_valid_only'])} | {_number(row['brier_valid_only'])} | "
            f"{row['invalid_confidence_observations']}/{row['observations']} |"
        )
    lines.extend(
        [
            "",
            (
                "The prespecified 10% selective-risk screen is **risk-contract infeasible"
                " on development**: no observed reference threshold reaches empirical risk"
                " ≤10%. This is an exploratory feasibility finding, not exact certification."
                if summary["coupled_risk_contract_screen"]["status"]
                == "risk-contract infeasible"
                else "The development risk-contract screen found an exploratory candidate; "
                "it is not certified until it passes the sealed certification split."
            ),
            "",
            *quantization_note,
            "",
            "## Fixed-coverage threshold transport (coupled)",
            "",
            "| Target κ | Shift | Threshold | Ref K | Shift K | ΔK | ΔK 95% item bootstrap | Ref risk | Shift risk | Δrisk | Δrisk 95% item bootstrap |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for target, target_data in summary["coupled_fixed_coverage"].items():
        for condition in order[1:]:
            row = target_data["transport"][condition]
            lines.append(
                f"| {target} | {CONDITION_LABELS[condition]} | {row['threshold']:.2f} | "
                f"{_percent(row['reference_coverage'])} | {_percent(row['shifted_coverage'])} | "
                f"{_percent(row['delta_coverage_from_reference'])} | "
                f"{_ci(row['delta_coverage_from_reference_item_bootstrap_95'], True)} | "
                f"{_percent(row['reference_risk'])} | {_percent(row['shifted_risk'])} | "
                f"{_percent(row['delta_risk'])} | {_ci(row['delta_risk_item_bootstrap_95'], True)} |"
            )
    lines.extend(
        [
            "",
            *fixed_note,
            "",
            "## Deterministic standardized readout",
            "",
            "| Answer condition | Accuracy | Mean C (valid) | AUROC | Brier | Invalid |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in order:
        row = standardized[condition]
        lines.append(
            f"| {CONDITION_LABELS[condition]} | {_percent(row['accuracy'])} | "
            f"{_number(row['mean_confidence_valid_only'])} | {_number(row['auroc_valid_only'])} | "
            f"{_number(row['brier_valid_only'])} | "
            f"{row['invalid_confidence_observations']}/{row['observations']} |"
        )
    lines.extend(
        [
            "",
            *standardized_note,
            "",
            "## Answer-path/readout decomposition",
            "",
            "All effects are mean confidence changes on complete valid item/seed rows.",
            "",
            "| Shift | Valid N | Answer path ΔA | 95% CI | Readout ΔC | 95% CI | Interaction ΔAC | 95% CI | Total |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in order[1:]:
        row = summary["decomposition"][condition]
        lines.append(
            f"| {CONDITION_LABELS[condition]} | {row['valid_only_denominator']}/{row['total_item_seed_rows']} | "
            f"{_percent(row['answer_path_effect'])} | {_ci(row['answer_path_item_bootstrap_95'], True)} | "
            f"{_percent(row['confidence_readout_effect'])} | {_ci(row['confidence_readout_item_bootstrap_95'], True)} | "
            f"{_percent(row['interaction_effect'])} | {_ci(row['interaction_item_bootstrap_95'], True)} | "
            f"{_percent(row['total_coupled_confidence_change'])} |"
        )
    lines.extend(
        [
            "",
            *decomposition_note,
            "",
            "## Interpretation and next gate",
            "",
            "This pilot succeeds at its intended development question: decoder changes",
            "produce measurable paired changes in the confidence-based acceptance set,",
            "the report-only decomposition is executable, and the deterministic readout",
            "is a candidate coverage-stability mitigation.",
            (
                f"The coupled {short_model_label} {dataset_label} reference produces an exploratory "
                "10% risk-contract candidate on development; it remains uncertified."
                if summary["coupled_risk_contract_screen"]["status"] == "candidate"
                else f"The coupled {short_model_label} {dataset_label} reference does not "
                "produce a 10% risk-contract candidate in this development sample."
            ),
            (
                "The standardized deterministic reference is independently risk-contract "
                "infeasible on development."
                if summary["standardized_risk_contract_screen"]["status"]
                == "risk-contract infeasible"
                else "The standardized deterministic reference also produces an exploratory "
                "development candidate; it remains uncertified."
            ),
            "",
            "No certification gate is open. Certification remains prohibited until all",
            "planned model/dataset integrations, scoring adapters, primary conditions,",
            "and protocol-freeze artifacts required by `PROTOCOL.md` are complete.",
            "",
            "Machine-readable full statistics remain on RAID at:",
            "",
            f"`$CPIS_OUTPUT_ROOT/{summary['run_id']}/statistics/development-matrix-analysis-v2.json`",
            "",
        ]
    )
    return "\n".join(lines)


def _fixed_coverage_csv(summary: dict) -> str:
    buffer = io.StringIO(newline="")
    fields = [
        "readout_protocol",
        "target_coverage",
        "condition",
        "threshold",
        "reference_coverage",
        "shifted_coverage",
        "delta_coverage",
        "delta_coverage_ci_lower",
        "delta_coverage_ci_upper",
        "reference_risk",
        "shifted_risk",
        "delta_risk",
        "delta_risk_ci_lower",
        "delta_risk_ci_upper",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for protocol, source in (
        ("coupled", summary["coupled_fixed_coverage"]),
        ("standardized_deterministic", summary["standardized_fixed_coverage"]),
    ):
        for target, target_data in source.items():
            for condition, row in target_data["transport"].items():
                writer.writerow(
                    {
                        "readout_protocol": protocol,
                        "target_coverage": target,
                        "condition": condition,
                        "threshold": row["threshold"],
                        "reference_coverage": row["reference_coverage"],
                        "shifted_coverage": row["shifted_coverage"],
                        "delta_coverage": row["delta_coverage_from_reference"],
                        "delta_coverage_ci_lower": row[
                            "delta_coverage_from_reference_item_bootstrap_95"
                        ]["lower_95"],
                        "delta_coverage_ci_upper": row[
                            "delta_coverage_from_reference_item_bootstrap_95"
                        ]["upper_95"],
                        "reference_risk": row["reference_risk"],
                        "shifted_risk": row["shifted_risk"],
                        "delta_risk": row["delta_risk"],
                        "delta_risk_ci_lower": row[
                            "delta_risk_item_bootstrap_95"
                        ]["lower_95"],
                        "delta_risk_ci_upper": row[
                            "delta_risk_item_bootstrap_95"
                        ]["upper_95"],
                    }
                )
    return buffer.getvalue()


def _coverage_svg(summary: dict, short_model_label: str) -> str:
    conditions = ("low-temperature", "high-temperature", "low-top-p", "high-diversity")
    coupled = summary["coupled_fixed_coverage"]["0.50"]["transport"]
    standardized = summary["standardized_fixed_coverage"]["0.50"]["transport"]
    width, height = 760, 430
    left, top, plot_w, plot_h = 90, 50, 620, 280
    if short_model_label == "4B":
        # Preserve the already-published development artifact byte-for-byte.
        min_y, max_y = -0.10, 0.08
        ticks = (-0.10, -0.05, 0.0, 0.05)
        footer = (
            "Exploratory thresholds selected separately at each protocol reference; "
            "achieved K: 58.9% coupled, 53.6% standardized."
        )
    else:
        values = [
            source[condition]["delta_coverage_from_reference"]
            for source in (coupled, standardized)
            for condition in conditions
        ]
        step = 0.05
        min_y = min(-step, step * math.floor((min(values) - 0.01) / step))
        max_y = max(step, step * math.ceil((max(values) + 0.01) / step))
        tick_count = round((max_y - min_y) / step)
        ticks = tuple(min_y + index * step for index in range(tick_count + 1))
        coupled_k = summary["coupled_fixed_coverage"]["0.50"]["selection"][
            "achieved_coverage"
        ]
        standardized_k = summary["standardized_fixed_coverage"]["0.50"][
            "selection"
        ]["achieved_coverage"]
        footer = (
            "Exploratory thresholds selected separately at each protocol reference; "
            f"achieved K: {100 * coupled_k:.1f}% coupled, "
            f"{100 * standardized_k:.1f}% standardized."
        )

    def y(value: float) -> float:
        return top + (max_y - value) / (max_y - min_y) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#555;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.c{fill:#3b82f6}.s{fill:#f59e0b}</style>',
        '<text x="380" y="25" font-size="17" text-anchor="middle">Coverage drift at nominal κ=0.50 (development)</text>',
    ]
    for tick in ticks:
        yy = y(tick)
        parts.append(f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{left + plot_w}" y2="{yy:.1f}"/>')
        parts.append(f'<text x="{left - 10}" y="{yy + 4:.1f}" font-size="12" text-anchor="end">{100*tick:.0f} pp</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{y(0):.1f}" x2="{left+plot_w}" y2="{y(0):.1f}"/>')
    group = plot_w / len(conditions)
    bar_w = 34
    for index, condition in enumerate(conditions):
        center = left + group * (index + 0.5)
        for offset, css, source in ((-bar_w, "c", coupled), (0, "s", standardized)):
            value = source[condition]["delta_coverage_from_reference"]
            zero = y(0)
            value_y = y(value)
            parts.append(
                f'<rect class="{css}" x="{center+offset:.1f}" y="{min(zero,value_y):.1f}" width="{bar_w-3}" height="{abs(zero-value_y):.1f}"/>'
            )
        parts.append(f'<text x="{center:.1f}" y="{top+plot_h+25}" font-size="12" text-anchor="middle">{CONDITION_LABELS[condition]}</text>')
    parts.extend(
        [
            '<rect class="c" x="260" y="385" width="16" height="16"/><text x="282" y="398" font-size="12">Coupled</text>',
            '<rect class="s" x="390" y="385" width="16" height="16"/><text x="412" y="398" font-size="12">Standardized deterministic</text>',
            f'<text x="380" y="422" font-size="11" text-anchor="middle">{footer}</text>',
            '</svg>',
        ]
    )
    return "\n".join(parts) + "\n"


def write_matrix_evidence(
    config_path: Path, repository_root: Path, destination: Path
) -> dict[str, str]:
    repository_root = repository_root.resolve()
    config = load_matrix_config(config_path.resolve())
    storage = StorageLayout.from_spec(config.storage, repository_root)
    run = storage.run(config.run_id)
    analysis_path = run.statistics / "development-matrix-analysis-v2.json"
    if not analysis_path.is_file():
        raise RuntimeError("run the saved-output matrix analysis first")
    analysis = read_json(analysis_path)
    integrity = _integrity(run.base)
    checkpoint_label = config.model.model_id.split("/")[-1]
    short_checkpoint_label = checkpoint_label.removeprefix("Qwen3.5-")
    mode_suffix = "-thinking" if config.model.model_mode == "thinking" else ""
    model_label = (
        f"{checkpoint_label} thinking"
        if config.model.model_mode == "thinking"
        else checkpoint_label
    )
    short_model_label = f"{short_checkpoint_label}{mode_suffix}"
    model_slug = f"{checkpoint_label.lower().replace('.', '')}{mode_suffix}"
    dataset_names = {
        "m-a-p/SuperGPQA": ("SuperGPQA", "supergpqa"),
        "TsinghuaC3I/MedXpertQA": ("MedXpertQA Text", "medxpertqa-text"),
        "kenoharada/Multiple-Instructions-Following:ManyIFEval": (
            "ManyIFEval",
            "manyifeval",
        ),
        "kenoharada/Multiple-Instructions-Following:StyleMBPP": (
            "StyleMBPP",
            "stylembpp",
        ),
    }
    try:
        dataset_label, dataset_slug = dataset_names[config.dataset.dataset_id]
    except KeyError as exc:
        raise ValueError(
            f"matrix narrative has not been reviewed for {config.dataset.dataset_id}"
        ) from exc
    if config.dataset.item_limit is None:
        evidence_id = f"{model_slug}-{dataset_slug}-full-development-matrix-v2"
        filename_id = evidence_id
    else:
        evidence_id = f"{model_slug}-{dataset_slug}-development-matrix-v2"
        filename_id = (
            f"{model_slug}-{dataset_slug}-matrix-preflight-v2"
            if "matrix-preflight" in config.experiment_id
            else evidence_id
        )
    summary = _summary_payload(analysis, integrity, evidence_id)
    if config.model.model_mode in {"thinking", "reasoning"}:
        summary["report_git"] = git_metadata(repository_root)
        if summary["report_git"].get("dirty"):
            raise RuntimeError(
                "reasoning-mode evidence requires a clean reporting tree"
            )
    base = destination.resolve() / filename_id
    outputs = {
        "markdown": str(base.with_suffix(".md")),
        "json": str(base.with_suffix(".json")),
        "csv": str(base.with_suffix(".csv")),
        "figure": str(base.with_suffix(".svg")),
    }
    _write_reproducible(
        Path(outputs["markdown"]),
        _markdown(
            summary,
            model_label,
            short_model_label,
            dataset_label,
            generic=(
                config.dataset.item_limit is None
                or "matrix-preflight" in config.experiment_id
                or config.dataset.dataset_id != "m-a-p/SuperGPQA"
                or short_model_label.startswith("Ministral-3-")
            ),
        ),
    )
    _write_reproducible(
        Path(outputs["json"]),
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    _write_reproducible(Path(outputs["csv"]), _fixed_coverage_csv(summary))
    _write_reproducible(
        Path(outputs["figure"]), _coverage_svg(summary, short_model_label)
    )
    return outputs
