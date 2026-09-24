#!/usr/bin/env python3
"""Prespecified held-out analysis for the frozen core.

For each model condition, dataset and prespecified shift this estimates the
paired change in the four operational endpoints at the frozen threshold:

  q  answer-generation failure
  K  coverage over all items
  R  selective risk among accepted
  S  operational success, K(1-R)

Uncertainty is an item-clustered paired bootstrap, because the item is the
experimental unit and the same items are seen under every coordinate. p-values
come from a paired cluster randomisation of the coordinate label within item,
and the Holm correction is applied over the declared family.

Nothing here chooses a threshold. Thresholds are read from the immutable
calibration artifact; a pair that calibration found infeasible is reported as
such and contributes no primary estimate.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from cpis.analysis.matrix import _as_parsed, _resolve_official_scores
from cpis.analysis.metrics import holm_adjust, operational_endpoints, outcome_partition
from cpis.analysis_plan import load_analysis_plan
from cpis.matrix_config import load_matrix_config
from cpis.matrix_records import MatrixParsedRecord
from cpis.storage import StorageLayout, read_json

# S, E, q, aC and aTau are the five disjoint outcomes and sum to one; K and R
# are the derived coverage and conditional risk reported alongside them. A
# shift that leaves R flat has to show up somewhere in the partition, which is
# what makes the partition worth carrying through the bootstrap.
ENDPOINTS = ("q", "K", "R", "S", "E", "aC", "aTau")
CONTRAST_CONDITION = {
    "reference_to_high_temperature": "high-temperature",
    "reference_to_low_top_p": "low-top-p",
}


def _endpoint_vector(records) -> dict[str, float | None]:
    e = operational_endpoints(records, _endpoint_vector.threshold)
    p = outcome_partition(records, _endpoint_vector.threshold)
    return {
        "q": e.answer_failure_rate,
        "K": e.coverage,
        "R": e.selective_risk,
        "S": e.operational_success,
        "E": p.accepted_error,
        "aC": p.confidence_failure,
        "aTau": p.low_confidence_abstention,
    }


def _by_item(records) -> dict[str, list]:
    grouped = defaultdict(list)
    for r in records:
        grouped[r.dataset_item_id].append(r)
    return grouped


def paired_bootstrap(
    reference: dict[str, list],
    shifted: dict[str, list],
    threshold: float,
    replicates: int,
    seed: int,
) -> dict[str, dict]:
    """Resample items, not observations, keeping both coordinates paired."""
    items = sorted(set(reference) & set(shifted))
    rng = random.Random(seed)
    _endpoint_vector.threshold = threshold

    def delta(sample: list[str]) -> dict[str, float | None]:
        ref = [r for i in sample for r in reference[i]]
        shf = [r for i in sample for r in shifted[i]]
        a, b = _endpoint_vector(ref), _endpoint_vector(shf)
        return {
            k: (None if a[k] is None or b[k] is None else b[k] - a[k]) for k in ENDPOINTS
        }

    point = delta(items)
    draws = {k: [] for k in ENDPOINTS}
    for _ in range(replicates):
        sample = [items[rng.randrange(len(items))] for _ in items]
        d = delta(sample)
        for k in ENDPOINTS:
            if d[k] is not None:
                draws[k].append(d[k])
    out = {}
    for k in ENDPOINTS:
        values = sorted(draws[k])
        if point[k] is None or len(values) < 100:
            out[k] = {"delta": point[k], "ci_low": None, "ci_high": None, "replicates": len(values)}
            continue
        out[k] = {
            "delta": round(point[k], 6),
            "ci_low": round(values[int(0.025 * len(values))], 6),
            "ci_high": round(values[min(len(values) - 1, int(0.975 * len(values)))], 6),
            "replicates": len(values),
        }
    return out


def randomisation_p(
    reference: dict[str, list],
    shifted: dict[str, list],
    threshold: float,
    endpoint: str,
    replicates: int,
    seed: int,
) -> float | None:
    """Permute the coordinate label within each item, preserving pairing."""
    items = sorted(set(reference) & set(shifted))
    _endpoint_vector.threshold = threshold
    a = _endpoint_vector([r for i in items for r in reference[i]])
    b = _endpoint_vector([r for i in items for r in shifted[i]])
    if a[endpoint] is None or b[endpoint] is None:
        return None
    observed = abs(b[endpoint] - a[endpoint])
    rng = random.Random(seed)
    extreme = 0
    for _ in range(replicates):
        left, right = [], []
        for i in items:
            if rng.random() < 0.5:
                left.extend(reference[i]); right.extend(shifted[i])
            else:
                left.extend(shifted[i]); right.extend(reference[i])
        x, y = _endpoint_vector(left), _endpoint_vector(right)
        if x[endpoint] is None or y[endpoint] is None:
            continue
        if abs(y[endpoint] - x[endpoint]) >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (replicates + 1)


def intervention_strength(run: Path, conditions: tuple[str, ...]) -> dict[str, dict]:
    """How often a shifted coordinate actually changed what was generated.

    Reported separately for the answer and for the native confidence readout.
    The coupled readout is decoded at the shifted coordinate too, so an answer
    that happens to come out identical can still hide a real readout-side
    intervention; collapsing the two would let that pass as inert.

    A null effect at a coordinate with near-zero realisation is uninformative
    about portability rather than evidence for it.
    """
    from cpis.records import GenerationRecord

    answers: dict[str, dict[str, str]] = {}
    confidences: dict[str, dict[str, str]] = {}
    for stage, sink in (("answer", answers), ("confidence", confidences)):
        for path in (run / "raw" / stage).glob("*.json"):
            record = GenerationRecord.model_validate(read_json(path))
            condition = record.condition.condition_id
            if stage == "confidence":
                # cells are identified as a-<answer condition>--r-<readout>
                if not condition.startswith("a-") or "--r-" not in condition:
                    continue
                condition = condition[2:].split("--r-", 1)[0]
            sink.setdefault(condition, {})[record.dataset_item_id] = record.raw_text

    def fraction_changed(sink: dict[str, dict[str, str]], condition: str):
        reference, shifted = sink.get("reference", {}), sink.get(condition, {})
        shared = set(reference) & set(shifted)
        if not shared:
            return None, 0
        changed = sum(1 for i in shared if reference[i] != shifted[i])
        return round(changed / len(shared), 6), len(shared)

    def confidence_changed_given_answer_identical(condition: str):
        """Confidence changes on the items whose answer the shift left alone.

        The readout is decoded after the answer, so an unconditional confidence
        change rate mixes two routes: the answer moved and the readout followed,
        or the readout itself moved. Restricting to items where the reference
        and shifted answers are byte-identical removes the first route by
        construction, because there was no answer change to propagate.

        This is a conditional observation, not a causal decomposition. It is
        identified only on the subset the shift happened to leave alone, and
        that subset is not a random sample of items - it is enriched for items
        the model answers stably. Nothing here estimates what the readout would
        have done on the items whose answers did change; that is the crossed
        design's question and it is not run in the broad study.
        """
        reference_a, shifted_a = answers.get("reference", {}), answers.get(condition, {})
        reference_c, shifted_c = confidences.get("reference", {}), confidences.get(condition, {})
        shared = (
            set(reference_a) & set(shifted_a) & set(reference_c) & set(shifted_c)
        )
        stable = [i for i in shared if reference_a[i] == shifted_a[i]]
        if not stable:
            return None, 0
        changed = sum(1 for i in stable if reference_c[i] != shifted_c[i])
        return round(changed / len(stable), 6), len(stable)

    out: dict[str, dict] = {}
    for condition in conditions:
        answer_rate, answer_n = fraction_changed(answers, condition)
        confidence_rate, confidence_n = fraction_changed(confidences, condition)
        stable_rate, stable_n = confidence_changed_given_answer_identical(condition)
        realised = [r for r in (answer_rate, confidence_rate) if r is not None]
        out[condition] = {
            "answer_changed": answer_rate,
            "confidence_changed": confidence_rate,
            "confidence_changed_given_answer_identical": stable_rate,
            "items_with_identical_answer": stable_n,
            "compared_answers": answer_n,
            "compared_confidences": confidence_n,
            "realisation": max(realised) if realised else None,
        }
    return out


def classify(payload: dict, margins: dict[str, float]) -> dict:
    """Label a contrast from its own interval and its own p-value.

    This is the statistical classification and nothing more. It applies no
    realisation gate, because no realisation floor was prespecified and a floor
    chosen now would be chosen after seeing which contrasts realised. Whether a
    classification is informative is a separate, explicitly reported judgement;
    see equivalence_realisation_separation in the artifact header.

    The rule, in precedence order:

        change_detected          <=> holm adjusted p < familywise alpha
        equivalent_within_margin <=> CI(dR) strictly inside +-margin
        not_estimable            <=> CI(dR) undefined
        inconclusive             otherwise

    The two leading categories are not disjoint by construction: a contrast can
    be significantly different from zero and still lie wholly inside a practical
    margin, which is a real state rather than a contradiction. Both flags are
    always recorded, change_detected takes the label, and the overlap is
    reported rather than hidden by the order of an if.
    """
    out: dict[str, object] = {}
    for endpoint, margin in margins.items():
        d = payload["deltas"].get(endpoint) or {}
        low, high = d.get("ci_low"), d.get("ci_high")
        out[f"equivalent_{endpoint}_at_margin"] = (
            None
            if low is None or high is None
            else bool(low > -margin and high < margin)
        )
    reject = payload.get("reject_at_familywise_alpha")
    equivalent = out.get("equivalent_R_at_margin")
    if reject:
        label = "change_detected"
    elif equivalent:
        label = "equivalent_within_margin"
    elif equivalent is None:
        label = "not_estimable"
    else:
        label = "inconclusive"
    out["statistical_classification"] = label
    out["significant_and_within_margin"] = bool(reject and equivalent)
    return out


def equivalence_realisation_separation(results: list[dict]) -> dict:
    """Is every equivalence classification confined to weakly realised contrasts?

    Reported as the two order statistics that decide it rather than as a verdict
    against an invented floor: when the strongest realisation among equivalent
    contrasts falls below the weakest among the others, the reading does not
    depend on where a floor would have been drawn anywhere in that gap.
    """
    equivalent, other = [], []
    for pair in results:
        for contrast, payload in (pair.get("contrasts") or {}).items():
            realisation = payload.get("realisation")
            if realisation is None:
                continue
            row = (realisation, f"{pair['experiment_id']}::{contrast}")
            target = equivalent if payload.get("equivalent_R_at_margin") else other
            target.append(row)
    strongest = max(equivalent, default=None)
    weakest = min(other, default=None)
    return {
        "equivalent_contrasts": sorted(name for _, name in equivalent),
        "max_realisation_among_equivalent": None if strongest is None else strongest[0],
        "min_realisation_among_non_equivalent": None if weakest is None else weakest[0],
        "separated": (
            None
            if strongest is None or weakest is None
            else bool(strongest[0] < weakest[0])
        ),
    }


def load_records(config_path: Path, repository_root: Path):
    config = load_matrix_config(config_path)
    storage = StorageLayout.from_spec(config.storage, repository_root)
    run = storage.output_root / config.run_id
    paths = sorted((run / "parsed").glob("*.json"))
    records = [MatrixParsedRecord.model_validate(read_json(p)) for p in paths]
    records = _resolve_official_scores(
        type("R", (), {"derived": run / "derived"})(), records, paths
    )
    unscored = [r for r in records if r.correct is None]
    if unscored:
        raise RuntimeError(f"{config.run_id}: {len(unscored)} unscored records")
    coupled = [r for r in records if "coupled" in r.design_roles]
    by_condition = defaultdict(list)
    for r in coupled:
        by_condition[r.answer_condition_id].append(_as_parsed(r))
    return config, by_condition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--configs", type=Path, default=Path("configs/core/test"))
    parser.add_argument("--analysis-plan", type=Path, default=Path("configs/analysis/core-v5.yaml"))
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan = load_analysis_plan((root / args.analysis_plan).resolve())
    thresholds = json.loads((root / args.thresholds).read_text(encoding="utf-8"))
    tau = {
        (p["model_id"], p["model_mode"], p["dataset_id"]): p
        for p in thresholds["pairs"]
    }
    replicates = args.bootstrap_replicates or plan.bootstrap.replicates

    results, pvals = [], {}
    for path in sorted((root / args.configs).glob("*.yaml")):
        config, by_condition = load_records(path, root)
        key = (config.model.model_id, config.model.model_mode, config.dataset.dataset_id)
        pair = tau.get(key)
        if pair is None:
            raise RuntimeError(f"no calibration entry for {key}")
        base = {
            "experiment_id": config.experiment_id,
            "run_id": config.run_id,
            "config_sha256": config.config_sha256,
            "model_id": config.model.model_id,
            "model_mode": config.model.model_mode,
            "dataset_id": config.dataset.dataset_id,
            "calibration_run_id": pair["run_id"],
            "threshold": pair["threshold"],
            "calibration_status": pair["status"],
        }
        if pair["status"] != "selected":
            results.append(base | {"status": "no primary estimate; calibration infeasible"})
            continue
        threshold = float(pair["threshold"])
        _endpoint_vector.threshold = threshold
        reference = _by_item(by_condition["reference"])
        storage = StorageLayout.from_spec(config.storage, root)
        entry = base | {
            "status": "estimated",
            "items": len(reference),
            "generations_changed_by_shift": intervention_strength(
                storage.output_root / config.run_id,
                tuple(CONTRAST_CONDITION[c] for c in plan.multiplicity.confirmatory_contrasts),
            ),
            "reference_endpoints": {
                k: (None if v is None else round(v, 6))
                for k, v in _endpoint_vector(by_condition["reference"]).items()
            },
            "contrasts": {},
        }
        for contrast in plan.multiplicity.confirmatory_contrasts:
            condition = CONTRAST_CONDITION[contrast]
            shifted = _by_item(by_condition[condition])
            entry["contrasts"][contrast] = {
                "shifted_endpoints": {
                    k: (None if v is None else round(v, 6))
                    for k, v in _endpoint_vector(by_condition[condition]).items()
                },
                "deltas": paired_bootstrap(
                    reference, shifted, threshold, replicates, plan.bootstrap.seed
                ),
            }
            p = randomisation_p(
                reference,
                shifted,
                threshold,
                "R",
                plan.multiplicity.randomization_replicates,
                plan.multiplicity.randomization_seed,
            )
            strength = (entry["generations_changed_by_shift"] or {}).get(condition) or {}
            entry["contrasts"][contrast]["answer_changed"] = strength.get("answer_changed")
            entry["contrasts"][contrast]["confidence_changed"] = strength.get(
                "confidence_changed"
            )
            entry["contrasts"][contrast]["realisation"] = strength.get("realisation")
            entry["contrasts"][contrast][
                "confidence_changed_given_answer_identical"
            ] = strength.get("confidence_changed_given_answer_identical")
            entry["contrasts"][contrast]["items_with_identical_answer"] = strength.get(
                "items_with_identical_answer"
            )
            entry["contrasts"][contrast]["raw_p_selective_risk"] = p
            if p is not None:
                pvals[f"{config.experiment_id}::{contrast}"] = p
        results.append(entry)

    adjusted = holm_adjust(pvals, plan.multiplicity.familywise_alpha) if pvals else {}
    margins = {
        "R": plan.equivalence_margins.selective_risk,
        "K": plan.equivalence_margins.coverage,
    }
    counts: Counter[str] = Counter()
    for entry in results:
        for contrast, payload in entry.get("contrasts", {}).items():
            row = adjusted.get(f"{entry['experiment_id']}::{contrast}")
            payload["holm_adjusted_p"] = None if row is None else row["adjusted_p_value"]
            payload["reject_at_familywise_alpha"] = None if row is None else row["reject"]
            payload.update(classify(payload, margins))
            counts[payload["statistical_classification"]] += 1
    separation = equivalence_realisation_separation(results)

    destination = (root / args.destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "artifact": "cpis-core-test-analysis-v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "analysis_plan_id": plan.plan_id,
                "analysis_plan_sha256": plan.sha256,
                "threshold_artifact_sha256": thresholds.get("artifact_sha256"),
                "endpoints": list(ENDPOINTS),
                "bootstrap_replicates": replicates,
                "multiplicity": plan.multiplicity.method,
                "family": plan.multiplicity.family_definition,
                "equivalence_margins": {
                    "selective_risk": plan.equivalence_margins.selective_risk,
                    "coverage": plan.equivalence_margins.coverage,
                },
                "classification_counts": dict(sorted(counts.items())),
                "classification_note": (
                    "counts are per contrast and are a tally of independently "
                    "classified contrasts, not a pooled or averaged effect. A "
                    "contrast classified equivalent_within_margin has met the "
                    "prespecified statistical criterion only; see "
                    "equivalence_realisation_separation before reading it as "
                    "evidence that a policy ported"
                ),
                "equivalence_realisation_separation": separation,
                "pairs": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"pairs": len(results), "destination": str(destination)}))


if __name__ == "__main__":
    main()
