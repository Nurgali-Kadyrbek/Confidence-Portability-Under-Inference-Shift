#!/usr/bin/env python3
"""Decide the smallest non-binding confidence budget from development records.

A budget is only sound if it does not truncate native generation. A raw
exceedance rate is not enough to establish that: "0.12% observed" can mean
1 of 833 or 12 of 10,000, which carry very different certainty. This reports a
one-sided Clopper-Pearson upper bound on P(L_C > B) and accepts a budget only
when that bound clears the margin, so the decision survives the sample size it
was made on.

Two quantities are reported separately, because they are operationally
different: exceedance of a counterfactual budget B, and actual truncation of
the records as generated (finish_reason == "length" at the configured budget).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

CANDIDATES = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)


def _log_binom_cdf(k: int, n: int, p: float) -> float:
    """log P(X <= k) for X ~ Binomial(n, p), stable for small k."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return -math.inf
    terms = []
    for i in range(k + 1):
        log_c = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        terms.append(log_c + i * math.log(p) + (n - i) * math.log1p(-p))
    peak = max(terms)
    return peak + math.log(sum(math.exp(t - peak) for t in terms))


def clopper_pearson_upper(k: int, n: int, confidence: float = 0.95) -> float:
    """Smallest p whose P(X <= k) equals 1 - confidence; 1.0 when k == n."""
    if k >= n:
        return 1.0
    target = math.log(1.0 - confidence)
    low, high = 0.0, 1.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if _log_binom_cdf(k, n, mid) > target:
            low = mid
        else:
            high = mid
    return high


def collect(directory: Path, limit: int | None, seed: int = 3) -> tuple[list[int], int]:
    paths = sorted(directory.glob("*.json"))
    if limit is not None and len(paths) > limit:
        random.seed(seed)
        paths = random.sample(paths, limit)
    lengths: list[int] = []
    truncated = 0
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("completion_tokens") is not None:
            lengths.append(record["completion_tokens"])
        if record.get("finish_reason") == "length":
            truncated += 1
    return lengths, truncated


def evaluate(lengths: list[int], truncated: int, margin: float, confidence: float) -> dict:
    n = len(lengths)
    ordered = sorted(lengths)
    rows = {}
    smallest = None
    for budget in CANDIDATES:
        exceed = sum(1 for value in ordered if value > budget)
        upper = clopper_pearson_upper(exceed, n, confidence)
        acceptable = upper < margin
        rows[str(budget)] = {
            "exceeding": exceed,
            "point_estimate": round(exceed / n, 6),
            "upper_bound": round(upper, 6),
            "non_binding": acceptable,
        }
        if acceptable and smallest is None:
            smallest = budget
    return {
        "n": n,
        "median": ordered[n // 2],
        "p99": ordered[min(n - 1, int(0.99 * n))],
        "max": ordered[-1],
        "observed_truncations_at_configured_budget": truncated,
        "observed_truncation_rate": round(truncated / n, 6),
        "margin": margin,
        "confidence": confidence,
        "smallest_non_binding_budget": smallest,
        "candidates": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--stage", default="confidence", choices=["confidence", "answer"])
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    directory = args.run_directory / "raw" / args.stage
    lengths, truncated = collect(directory, args.limit)
    if not lengths:
        raise SystemExit(f"no completion-token data under {directory}")
    result = evaluate(lengths, truncated, args.margin, args.confidence)
    result |= {"run": args.run_directory.name, "stage": args.stage, "label": args.label}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
