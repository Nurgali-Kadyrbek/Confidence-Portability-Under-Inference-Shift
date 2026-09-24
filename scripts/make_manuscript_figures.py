#!/usr/bin/env python3
"""Emit the manuscript's figures as LaTeX-native pgfplots/TikZ from artifacts.

Vector output with no plotting dependency and no raster step: the figure is
drawn by the same compiler that sets the text, so fonts and sizes match the
document and nothing is a screenshot. Every coordinate is read from a frozen
analysis artifact.

The figures deliberately encode condition by marker as well as by colour, so
they survive greyscale printing and common colour-vision deficiencies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CORE = Path("evidence/core")
PROTOCOL = Path("evidence/protocol")

PAIR_ORDER = [
    ("Qwen/Qwen3.5-9B", "non_thinking"),
    ("Qwen/Qwen3.5-9B", "thinking"),
    ("mistralai/Ministral-3-8B-Instruct-2512-BF16", "instruct"),
    ("mistralai/Ministral-3-8B-Reasoning-2512", "reasoning"),
]
SHORT = {
    ("Qwen/Qwen3.5-9B", "non_thinking"): "Qwen non-think",
    ("Qwen/Qwen3.5-9B", "thinking"): "Qwen thinking",
    ("mistralai/Ministral-3-8B-Instruct-2512-BF16", "instruct"): "Min. instruct",
    ("mistralai/Ministral-3-8B-Reasoning-2512", "reasoning"): "Min. reasoning",
}
SHIFT_SHORT = {
    "reference_to_high_temperature": "high-$T$",
    "reference_to_low_top_p": "low-$p$",
}


def data_label(dataset_id: str) -> str:
    return "ManyIFEval" if "Instruct" in dataset_id else "SuperGPQA"


def row_label(entry: dict) -> str:
    return f"{SHORT[(entry['model_id'], entry['model_mode'])]}, {data_label(entry['dataset_id'])}"


def sort_key(entry: dict) -> tuple:
    return (
        PAIR_ORDER.index((entry["model_id"], entry["model_mode"])),
        data_label(entry["dataset_id"]),
    )


# --------------------------------------------------------------------------


def fig_reference_quality(quality: dict) -> str:
    """AUROC against Brier skill: ranking quality versus probability quality."""
    pairs = sorted(quality["pairs"], key=sort_key)
    auroc_rows, bss_rows, mcb_rows, dsc_rows, labels = [], [], [], [], []
    for index, pair in enumerate(pairs):
        m = pair["metrics"]
        labels.append(row_label(pair))
        a = m["auroc"]
        auroc_rows.append(
            f"({a['estimate']:.4f},{index}) += ({a['ci_high'] - a['estimate']:.4f},0) "
            f"-= ({a['estimate'] - a['ci_low']:.4f},0)"
        )
        b = m["brier_skill_score"]
        bss_rows.append(
            f"({b['estimate']:.4f},{index}) += ({b['ci_high'] - b['estimate']:.4f},0) "
            f"-= ({b['estimate'] - b['ci_low']:.4f},0)"
        )
        mcb_rows.append(f"({m['miscalibration_ratio']['estimate']:.4f},{index})")
        dsc_rows.append(f"({m['discrimination_skill']['estimate']:.4f},{index})")
    ticks = ",".join(str(i) for i in range(len(labels)))
    names = ",".join(f"{{{l}}}" for l in labels)
    return rf"""
\begin{{tikzpicture}}
\pgfplotsset{{
  cpisrow/.style={{
    width=0.47\textwidth, height=5.4cm,
    ytick={{{ticks}}}, yticklabels={{{names}}},
    y dir=reverse, ymin=-0.7, ymax={len(labels) - 0.3},
    tick label style={{font=\scriptsize}}, label style={{font=\scriptsize}},
    title style={{font=\scriptsize\bfseries}},
    grid=major, grid style={{gray!22,very thin}},
  }},
}}
\begin{{axis}}[cpisrow, name=left, xlabel={{AUROC}}, title={{(a) Ranking quality}},
  xmin=0.35, xmax=0.95, extra x ticks={{0.5}}, extra x tick style={{grid=major,
  grid style={{black!55,dashed}}, xticklabel=\empty}}]
\addplot[only marks, mark=*, mark size=1.7pt, color=black,
  error bars/.cd, x dir=both, x explicit, error bar style={{black}}]
  coordinates {{{' '.join(auroc_rows)}}};
\end{{axis}}
\begin{{axis}}[cpisrow, at={{(left.south east)}}, xshift=1.45cm, anchor=south west,
  yticklabels=\empty, xlabel={{Brier skill score}}, title={{(b) Probability quality}},
  xmin=-9.6, xmax=1.2, extra x ticks={{0}}, extra x tick style={{grid=major,
  grid style={{black!55,dashed}}, xticklabel=\empty}}]
\addplot[only marks, mark=square*, mark size=1.7pt, color=black,
  error bars/.cd, x dir=both, x explicit, error bar style={{black}}]
  coordinates {{{' '.join(bss_rows)}}};
\end{{axis}}
\end{{tikzpicture}}

\vspace{{1.5ex}}

\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.94\textwidth, height=5.0cm,
  ytick={{{ticks}}}, yticklabels={{{names}}}, y dir=reverse,
  ymin=-0.7, ymax={len(labels) - 0.3},
  xmode=log, xmin=0.004, xmax=9,
  xlabel={{component relative to base-rate uncertainty (log scale)}},
  title style={{font=\scriptsize\bfseries}},
  title={{(c) Miscalibration and discrimination, normalised by UNC}},
  tick label style={{font=\scriptsize}}, label style={{font=\scriptsize}},
  legend style={{font=\scriptsize, at={{(0.5,-0.32)}}, anchor=north, legend columns=2, draw=none}},
  grid=major, grid style={{gray!22,very thin}}]
\addplot[only marks, mark=triangle*, mark size=2.2pt, color=black] coordinates {{{' '.join(mcb_rows)}}};
\addlegendentry{{MCB/UNC (miscalibration)}}
\addplot[only marks, mark=o, mark size=2.2pt, color=black] coordinates {{{' '.join(dsc_rows)}}};
\addlegendentry{{DSC/UNC (discrimination)}}
\end{{axis}}
\end{{tikzpicture}}
"""


def fig_forest(analysis: dict) -> str:
    """Per-contrast change in selective risk, never pooled."""
    rows_high, rows_low, labels = [], [], []
    pairs = sorted(
        (p for p in analysis["pairs"] if p["status"] == "estimated"), key=sort_key
    )
    index = 0
    for pair in pairs:
        labels.append(row_label(pair))
        for contrast, payload in sorted(pair["contrasts"].items()):
            d = payload["deltas"]["R"]
            if d["delta"] is None:
                continue
            offset = -0.16 if contrast == "reference_to_high_temperature" else 0.16
            entry = (
                f"({d['delta']:.4f},{index + offset}) "
                f"+= ({d['ci_high'] - d['delta']:.4f},0) -= ({d['delta'] - d['ci_low']:.4f},0)"
            )
            (rows_high if contrast == "reference_to_high_temperature" else rows_low).append(entry)
        index += 1
    ticks = ",".join(str(i) for i in range(len(labels)))
    names = ",".join(f"{{{l}}}" for l in labels)
    margin = analysis["equivalence_margins"]["selective_risk"]
    return rf"""
\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.88\textwidth, height=7.2cm,
  ytick={{{ticks}}}, yticklabels={{{names}}}, y dir=reverse,
  ymin=-0.7, ymax={len(labels) - 0.3},
  xmin=-0.20, xmax=0.20,
  xlabel={{$\Delta R$ at the frozen threshold (95\% item-clustered bootstrap)}},
  tick label style={{font=\scriptsize}}, label style={{font=\scriptsize}},
  legend style={{font=\scriptsize, at={{(0.5,-0.20)}}, anchor=north, legend columns=3, draw=none}},
  grid=major, grid style={{gray!22,very thin}}]
\fill[black!7] (axis cs:{-margin},-0.7) rectangle (axis cs:{margin},{len(labels) - 0.3});
\draw[black!55,dashed] (axis cs:0,-0.7) -- (axis cs:0,{len(labels) - 0.3});
\addplot[only marks, mark=*, mark size=1.7pt, color=black,
  error bars/.cd, x dir=both, x explicit, error bar style={{black}}]
  coordinates {{{' '.join(rows_high)}}};
\addlegendentry{{high-$T$}}
\addplot[only marks, mark=square*, mark size=1.7pt, color=black!55,
  error bars/.cd, x dir=both, x explicit, error bar style={{black!55}}]
  coordinates {{{' '.join(rows_low)}}};
\addlegendentry{{low-$p$}}
\addlegendimage{{area legend, fill=black!7, draw=none}}
\addlegendentry{{equivalence region $\pm{margin}$}}
\end{{axis}}
\end{{tikzpicture}}
"""


def fig_endpoints(analysis: dict) -> str:
    """Coverage against selective risk: the conditional endpoint is not the system."""
    points, labels = [], []
    for pair in sorted(
        (p for p in analysis["pairs"] if p["status"] == "estimated"), key=sort_key
    ):
        for contrast, payload in sorted(pair["contrasts"].items()):
            d = payload["deltas"]
            points.append(
                (d["K"]["delta"], d["R"]["delta"], d["q"]["delta"],
                 f"{row_label(pair)}, {SHIFT_SHORT[contrast]}")
            )
    marks = ["*", "square*", "triangle*", "diamond*"]
    series = []
    for i, family in enumerate(PAIR_ORDER):
        coords = [
            f"({k:.4f},{r:.4f})"
            for k, r, q, lab in points
            if SHORT[family].split(",")[0] in lab
        ]
        if coords:
            series.append(
                rf"\addplot[only marks, mark={marks[i]}, mark size=2.0pt, color=black!{95 - 20 * i}] "
                rf"coordinates {{{' '.join(coords)}}};" + "\n"
                rf"\addlegendentry{{{SHORT[family]}}}"
            )
    margin = analysis["equivalence_margins"]["selective_risk"]
    cmargin = analysis["equivalence_margins"]["coverage"]
    return rf"""
\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.80\textwidth, height=6.6cm,
  xlabel={{$\Delta K$ (coverage over all items)}},
  ylabel={{$\Delta R$ (selective risk among accepted)}},
  xmin=-0.42, xmax=0.16, ymin=-0.09, ymax=0.09,
  tick label style={{font=\scriptsize}}, label style={{font=\scriptsize}},
  legend style={{font=\scriptsize, at={{(0.5,-0.22)}}, anchor=north, legend columns=4, draw=none}},
  grid=major, grid style={{gray!22,very thin}}]
\fill[black!7] (axis cs:{-cmargin},{-margin}) rectangle (axis cs:{cmargin},{margin});
\draw[black!55,dashed] (axis cs:-0.42,0) -- (axis cs:0.16,0);
\draw[black!55,dashed] (axis cs:0,-0.09) -- (axis cs:0,0.09);
{chr(10).join(series)}
\end{{axis}}
\end{{tikzpicture}}
"""


def fig_realisation(analysis: dict) -> str:
    """Answer-channel against confidence-channel realisation, per contrast."""
    coords, undefined = [], []
    for pair in sorted(
        (p for p in analysis["pairs"] if p["status"] == "estimated"), key=sort_key
    ):
        for contrast, payload in sorted(pair["contrasts"].items()):
            a = payload.get("answer_changed")
            c = payload.get("confidence_changed")
            if a is None or c is None:
                continue
            stable = payload.get("confidence_changed_given_answer_identical")
            (undefined if stable is None else coords).append((a, c, stable))
    defined_pts = " ".join(f"({a:.4f},{c:.4f})" for a, c, _ in coords)
    undef_pts = " ".join(f"({a:.4f},{c:.4f})" for a, c, _ in undefined)
    return rf"""
\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.72\textwidth, height=6.0cm,
  xlabel={{$I_A$ (answer output changed)}},
  ylabel={{$I_C$ (confidence output changed)}},
  xmin=-0.05, xmax=1.08, ymin=-0.05, ymax=1.08,
  tick label style={{font=\scriptsize}}, label style={{font=\scriptsize}},
  legend style={{font=\scriptsize, at={{(0.5,-0.24)}}, anchor=north, legend columns=2, draw=none}},
  grid=major, grid style={{gray!22,very thin}}]
\addplot[domain=0:1, samples=2, black!45, dashed, forget plot] {{x}};
\addplot[only marks, mark=*, mark size=2.0pt, color=black] coordinates {{{defined_pts}}};
\addlegendentry{{conditional rate estimable}}
\addplot[only marks, mark=x, mark size=3.0pt, color=black!60] coordinates {{{undef_pts}}};
\addlegendentry{{undefined ($I_A=1$, no stable items)}}
\end{{axis}}
\end{{tikzpicture}}
"""


def fig_seeds(seeds: dict) -> str:
    """Per-seed risk changes on the robustness subset."""
    rows, labels = [], []
    index = 0
    marks = ["*", "square*", "triangle*"]
    series: dict[int, list[str]] = {0: [], 1: [], 2: []}
    for pair in sorted(
        (p for p in seeds["pairs"] if p.get("status") == "estimated"), key=sort_key
    ):
        for contrast, payload in sorted(pair["contrasts"].items()):
            row = payload["across_seed_summary"].get("R")
            if row is None:
                continue
            labels.append(f"{row_label(pair)}, {SHIFT_SHORT[contrast]}")
            for s, value in enumerate(
                v for v in row["per_seed"].values() if v is not None
            ):
                series[s].append(f"({value:.4f},{index})")
            index += 1
    ticks = ",".join(str(i) for i in range(len(labels)))
    names = ",".join(f"{{{l}}}" for l in labels)
    seed_ids = seeds["pairs"][0]["seeds"]
    plots = "\n".join(
        rf"\addplot[only marks, mark={marks[s]}, mark size=1.7pt, color=black!{90 - 25 * s}] "
        rf"coordinates {{{' '.join(series[s])}}};" + "\n"
        rf"\addlegendentry{{seed {seed_ids[s]}}}"
        for s in range(3)
        if series[s]
    )
    return rf"""
\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.86\textwidth, height=9.2cm,
  ytick={{{ticks}}}, yticklabels={{{names}}}, y dir=reverse,
  ymin=-0.7, ymax={len(labels) - 0.3},
  xmin=-0.20, xmax=0.22,
  xlabel={{$\Delta R$ per sampling seed, 128-item held-out subset}},
  tick label style={{font=\scriptsize}}, label style={{font=\scriptsize}},
  legend style={{font=\scriptsize, at={{(0.5,-0.15)}}, anchor=north, legend columns=3, draw=none}},
  grid=major, grid style={{gray!22,very thin}}]
\draw[black!55,dashed] (axis cs:0,-0.7) -- (axis cs:0,{len(labels) - 0.3});
{plots}
\end{{axis}}
\end{{tikzpicture}}
"""


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
    seeds = load(CORE / "seed-robustness.json")

    written = []
    for name, body in (
        ("fig-reference-quality", fig_reference_quality(quality)),
        ("fig-forest-risk", fig_forest(analysis)),
        ("fig-endpoints", fig_endpoints(analysis)),
        ("fig-realisation", fig_realisation(analysis)),
        ("fig-seeds", fig_seeds(seeds)),
    ):
        (out / f"{name}.tex").write_text(body, encoding="utf-8")
        written.append(name)
    print(json.dumps({"figures": written, "out": str(out)}))


if __name__ == "__main__":
    main()
