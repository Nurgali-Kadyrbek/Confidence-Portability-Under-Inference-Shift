"""Manuscript-ready tables and forest plot from the frozen panel result."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path

from cpis.storage import read_json


def _write(path: Path, text: str) -> None:
    data = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"refusing to replace different final evidence: {path}")
        return
    path.write_bytes(data)


def _csv(rows: list[dict], columns: list[str]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows({column: row.get(column) for column in columns} for row in rows)
    return stream.getvalue()


def _pooled_rows(panel: dict) -> list[dict]:
    rows = []
    for key, result in sorted(panel["pooled_analyses"].items()):
        meta = result.get("unmoderated_random_effects")
        if result.get("status") != "estimated" or not meta:
            rows.append(
                {
                    "analysis_key": key,
                    "status": result.get("status"),
                    "effects": result.get("eligible_effects", 0),
                }
            )
            continue
        intercept = meta["coefficients"]["intercept"]
        rows.append(
            {
                "analysis_key": key,
                "status": "estimated",
                "effects": meta["effects"],
                "estimate": intercept["estimate"],
                "standard_error": intercept["standard_error"],
                "lower_95": intercept["lower_95"],
                "upper_95": intercept["upper_95"],
                "tau_squared": meta["tau_squared"],
                "i_squared": meta["i_squared"],
            }
        )
    return rows


def _forest_svg(rows: list[dict]) -> str:
    estimated = [row for row in rows if row.get("status") == "estimated"]
    width = 1100
    row_height = 34
    top = 70
    bottom = 55
    height = top + bottom + row_height * max(1, len(estimated))
    left = 520
    right = 40
    plot_width = width - left - right
    endpoints = [
        value
        for row in estimated
        for value in (row["lower_95"], row["upper_95"])
        if math.isfinite(float(value))
    ]
    limit = max([0.05, *(abs(float(value)) for value in endpoints)])
    limit = math.ceil(limit * 100) / 100

    def x(value: float) -> float:
        return left + (float(value) + limit) / (2 * limit) * plot_width

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.axis{stroke:#374151;stroke-width:1}.ci{stroke:#2563eb;stroke-width:3}.point{fill:#1d4ed8}.zero{stroke:#9ca3af;stroke-dasharray:4 4}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="30" font-size="20" font-weight="bold">CPIS pooled portability effects</text>',
        '<text x="20" y="51" font-size="12">Random-effects mean within-model portability effects with 95% CI; endpoint-specific direction is retained in each label.</text>',
        f'<line class="zero" x1="{x(0):.1f}" y1="{top-15}" x2="{x(0):.1f}" y2="{height-bottom+5}"/>',
    ]
    for index, row in enumerate(estimated):
        y = top + index * row_height
        label = row["analysis_key"].replace("&", "&amp;")
        parts.extend(
            [
                f'<text x="20" y="{y+5}" font-size="11">{label}</text>',
                f'<line class="ci" x1="{x(row["lower_95"]):.1f}" y1="{y}" x2="{x(row["upper_95"]):.1f}" y2="{y}"/>',
                f'<circle class="point" cx="{x(row["estimate"]):.1f}" cy="{y}" r="5"/>',
                f'<text x="{width-right}" y="{y+5}" font-size="11" text-anchor="end">{row["estimate"]:+.3f} [{row["lower_95"]:+.3f}, {row["upper_95"]:+.3f}]</text>',
            ]
        )
    axis_y = height - bottom + 15
    parts.append(
        f'<line class="axis" x1="{left}" y1="{axis_y}" x2="{width-right}" y2="{axis_y}"/>'
    )
    for value in (-limit, -limit / 2, 0, limit / 2, limit):
        parts.append(
            f'<text x="{x(value):.1f}" y="{axis_y+22}" font-size="11" text-anchor="middle">{value:+.2f}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_panel_evidence(
    panel_result_path: Path, repository_root: Path, destination: Path
) -> dict[str, str]:
    """Serialize final evidence; never reads raw generations or invokes models."""

    panel_result_path = panel_result_path.resolve()
    panel = read_json(panel_result_path)
    if panel.get("confirmatory") is not True:
        raise ValueError("final panel evidence requires a confirmatory panel result")
    effects = []
    for row in panel["effect_inventory"]:
        variance = row.get("variance")
        standard_error = math.sqrt(variance) if variance and variance > 0 else None
        effects.append(
            {
                **row,
                "standard_error": standard_error,
                "lower_95": (
                    row["estimate"] - 1.959963984540054 * standard_error
                    if standard_error is not None and row.get("estimate") is not None
                    else None
                ),
                "upper_95": (
                    row["estimate"] + 1.959963984540054 * standard_error
                    if standard_error is not None and row.get("estimate") is not None
                    else None
                ),
                **{f"moderator_{key}": value for key, value in row["moderators"].items()},
            }
        )
    pooled = _pooled_rows(panel)
    destination = destination.resolve()
    outputs = {
        "effects_csv": destination / "confirmatory-effect-inventory.csv",
        "pooled_csv": destination / "confirmatory-pooled-effects.csv",
        "markdown": destination / "confirmatory-results.md",
        "figure": destination / "confirmatory-pooled-forest.svg",
        "provenance": destination / "confirmatory-provenance.json",
    }
    effect_columns = [
        "effect_id",
        "endpoint",
        "readout_role",
        "contrast",
        "model_condition_id",
        "dataset_panel_id",
        "estimate",
        "variance",
        "standard_error",
        "lower_95",
        "upper_95",
        "meta_analysis_eligible",
        "exclusion_reason",
        "moderator_model_family",
        "moderator_reasoning_regime",
        "moderator_scale_band",
        "moderator_task_domain",
    ]
    pooled_columns = [
        "analysis_key",
        "status",
        "effects",
        "estimate",
        "standard_error",
        "lower_95",
        "upper_95",
        "tau_squared",
        "i_squared",
    ]
    _write(outputs["effects_csv"], _csv(effects, effect_columns))
    _write(outputs["pooled_csv"], _csv(pooled, pooled_columns))
    lines = [
        "# CPIS confirmatory results",
        "",
        f"- Analysis: `{panel['analysis_id']}`",
        f"- Analysis plan: `{panel['analysis_plan_id']}` (`{panel['analysis_plan_sha256']}`)",
        f"- Study plan: `{panel['study_plan_id']}` (`{panel['study_plan_sha256']}`)",
        f"- Model–dataset inputs: {len(panel['input_provenance'])}",
        f"- Prespecified effect rows: {len(effects)}",
        "",
        "| Endpoint / readout / contrast | k | Pooled effect | 95% CI | τ² | I² |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in pooled:
        if row["status"] != "estimated":
            lines.append(
                f"| {row['analysis_key']} | {row.get('effects', 0)} | NA | NA | NA | NA |"
            )
        else:
            lines.append(
                f"| {row['analysis_key']} | {row['effects']} | {row['estimate']:+.3f} | "
                f"[{row['lower_95']:+.3f}, {row['upper_95']:+.3f}] | "
                f"{row['tau_squared']:.4f} | {row['i_squared']:.1%} |"
            )
    lines.extend(
        [
            "",
            "Pooled estimates summarize heterogeneous within-model, within-dataset",
            "contrasts. They do not imply that every model exhibits the pooled effect;",
            "the effect inventory preserves every model/dataset estimate and moderator.",
            "Positive changes mean degradation for risk, AURC, Brier, and invalid-rate",
            "endpoints, but improvement for AUROC; endpoint labels are therefore never",
            "collapsed into a direction-agnostic overall effect.",
            "",
        ]
    )
    _write(outputs["markdown"], "\n".join(lines))
    _write(outputs["figure"], _forest_svg(pooled))
    source_sha256 = hashlib.sha256(panel_result_path.read_bytes()).hexdigest()
    provenance = {
        "schema_version": "1.0",
        "panel_result_name": panel_result_path.name,
        "panel_result_sha256": source_sha256,
        "analysis_id": panel["analysis_id"],
        "analysis_plan_sha256": panel["analysis_plan_sha256"],
        "study_plan_sha256": panel["study_plan_sha256"],
        "input_provenance": panel["input_provenance"],
        "claim_guard": panel["claim_guard"],
    }
    _write(outputs["provenance"], json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return {key: str(path) for key, path in outputs.items()}
