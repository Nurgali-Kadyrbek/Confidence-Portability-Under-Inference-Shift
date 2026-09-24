"""Final panel aggregation from immutable confirmatory test statistics."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from cpis.analysis.meta import MetaEffect, reml_meta_analysis
from cpis.analysis_plan import load_analysis_plan
from cpis.manifest import git_metadata
from cpis.matrix_config import load_matrix_config
from cpis.panel_config import load_panel_analysis_config
from cpis.phase_gate import (
    authorize_phase_analysis_plan,
    authorize_phase_config,
    require_phase_gate,
)
from cpis.storage import StorageLayout, read_json, write_immutable_json
from cpis.study_plan import load_study_plan


CONTRAST_TO_CONDITION = {
    "reference_to_low_temperature": "low-temperature",
    "reference_to_high_temperature": "high-temperature",
    "reference_to_low_top_p": "low-top-p",
    "reference_to_high_diversity": "high-diversity",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _effect(
    *,
    effect_id: str,
    endpoint: str,
    readout_role: str,
    contrast: str,
    model_condition_id: str,
    dataset_panel_id: str,
    moderators: dict[str, str],
    row: dict,
    estimate_key: str = "delta_risk",
    variance_key: str = "delta_risk_item_bootstrap_variance",
    undefined_reason: str = "effect is undefined for this model/dataset contrast",
) -> dict:
    estimate = row.get(estimate_key)
    variance = row.get(variance_key)
    reason = None
    if estimate is None:
        reason = undefined_reason
    elif variance is None:
        reason = "fewer than two valid item-bootstrap replicates"
    elif variance <= 0:
        reason = "item-bootstrap sampling variance is zero"
    return {
        "effect_id": effect_id,
        "endpoint": endpoint,
        "readout_role": readout_role,
        "contrast": contrast,
        "model_condition_id": model_condition_id,
        "dataset_panel_id": dataset_panel_id,
        "estimate": estimate,
        "variance": variance,
        "moderators": moderators,
        "meta_analysis_eligible": reason is None,
        "exclusion_reason": reason,
    }


def extract_panel_effects(
    analysis: dict,
    *,
    model_condition_id: str,
    dataset_panel_id: str,
    moderators: dict[str, str],
    contrasts: tuple[str, ...],
    readout_roles: tuple[str, ...],
    fixed_coverage_target: float,
) -> list[dict]:
    """Extract prespecified model/dataset/contrast effects without recomputation."""

    effects: list[dict] = []
    target = f"{fixed_coverage_target:.2f}"
    for readout_role in readout_roles:
        source = analysis[readout_role]
        fixed = source["fixed_coverage_transport"][target]["transport"]
        certified = source["risk_contract_transport"]
        confidence = source["confidence_portability"]
        for contrast in contrasts:
            condition = CONTRAST_TO_CONDITION[contrast]
            prefix = f"{model_condition_id}:{dataset_panel_id}:{contrast}:{readout_role}"
            effects.append(
                _effect(
                    effect_id=f"{prefix}:fixed-coverage-{target}",
                    endpoint=f"fixed_coverage_{target}",
                    readout_role=readout_role,
                    contrast=contrast,
                    model_condition_id=model_condition_id,
                    dataset_panel_id=dataset_panel_id,
                    moderators=moderators,
                    row=fixed[condition],
                )
            )
            if certified["status"] == "certified":
                effects.append(
                    _effect(
                        effect_id=f"{prefix}:certified-risk-contract",
                        endpoint="certified_risk_contract",
                        readout_role=readout_role,
                        contrast=contrast,
                        model_condition_id=model_condition_id,
                        dataset_panel_id=dataset_panel_id,
                        moderators=moderators,
                        row=certified["transport"][condition],
                    )
                )
            for metric_name, metric_row in confidence[condition].items():
                effects.append(
                    _effect(
                        effect_id=f"{prefix}:delta-{metric_name}",
                        endpoint=f"delta_{metric_name}",
                        readout_role=readout_role,
                        contrast=contrast,
                        model_condition_id=model_condition_id,
                        dataset_panel_id=dataset_panel_id,
                        moderators=moderators,
                        row=metric_row,
                        estimate_key="delta",
                        variance_key="delta_item_bootstrap_variance",
                        undefined_reason=(
                            f"{metric_name} is undefined because the required "
                            "valid confidence or outcome variation is absent"
                        ),
                    )
                )
    return effects


def _meta(effect_rows: list[dict], moderator_names: tuple[str, ...], minimum: int) -> dict:
    eligible = [row for row in effect_rows if row["meta_analysis_eligible"]]
    excluded = [row for row in effect_rows if not row["meta_analysis_eligible"]]
    payload: dict = {
        "effect_inventory": len(effect_rows),
        "eligible_effects": len(eligible),
        "excluded_nonestimable": [
            {"effect_id": row["effect_id"], "reason": row["exclusion_reason"]}
            for row in excluded
        ],
    }
    if len(eligible) < 2:
        payload["status"] = "insufficient_estimable_effects"
        return payload
    effects = [
        MetaEffect(
            effect_id=row["effect_id"],
            estimate=float(row["estimate"]),
            variance=float(row["variance"]),
            moderators=row["moderators"],
        )
        for row in eligible
    ]
    payload["unmoderated_random_effects"] = reml_meta_analysis(effects)
    payload["moderator_analysis"] = reml_meta_analysis(
        effects,
        moderator_names=moderator_names,
        minimum_effects_per_coefficient=minimum,
    )
    by_family: dict[str, list[MetaEffect]] = defaultdict(list)
    for effect in effects:
        by_family[effect.moderators["model_family"]].append(effect)
    payload["family_heterogeneity"] = {}
    for family, members in sorted(by_family.items()):
        if len(members) >= 2:
            payload["family_heterogeneity"][family] = reml_meta_analysis(members)
        else:
            payload["family_heterogeneity"][family] = {
                "status": "descriptive_single_effect",
                "estimate": members[0].estimate,
                "variance": members[0].variance,
            }
    payload["status"] = "estimated"
    return payload


def analyze_confirmatory_panel(
    panel_config_path: Path, repository_root: Path, destination: Path
) -> tuple[Path, dict]:
    """Aggregate all declared broad test results and fail closed on missing cells."""

    repository_root = repository_root.resolve()
    panel_config_path = panel_config_path.resolve()
    gate = require_phase_gate("test", repository_root)
    authorize_phase_config(gate, panel_config_path, repository_root)
    panel = load_panel_analysis_config(panel_config_path)
    analysis_plan_path = repository_root / panel.analysis_plan_path
    study_plan_path = repository_root / panel.study_plan_path
    plan = load_analysis_plan(analysis_plan_path)
    authorize_phase_analysis_plan(gate, plan.sha256)
    study = load_study_plan(study_plan_path)
    if plan.sha256 != panel.analysis_plan_sha256:
        raise RuntimeError("panel config analysis-plan hash mismatch")
    if study.canonical_sha256 != panel.study_plan_sha256:
        raise RuntimeError("panel config study-plan hash mismatch")
    if plan.pooled_analysis is None:
        raise RuntimeError("analysis plan has no pooled-analysis contract")

    expected = {
        (model, dataset)
        for model in study.broad_model_conditions
        for dataset in study.broad_dataset_panels
    }
    declared = {
        (entry.model_condition_id, entry.dataset_panel_id)
        for entry in panel.results
    }
    if declared != expected:
        raise RuntimeError(
            f"panel result set mismatch: missing={sorted(expected-declared)}, "
            f"unexpected={sorted(declared-expected)}"
        )
    models = {entry.condition_id: entry for entry in study.model_conditions}
    datasets = {entry.panel_id: entry for entry in study.core_datasets}
    inventory: list[dict] = []
    input_provenance: list[dict] = []
    for entry in panel.results:
        config_path = (repository_root / entry.config_path).resolve()
        try:
            config_path.relative_to(repository_root)
        except ValueError as exc:
            raise RuntimeError("panel test config escapes repository") from exc
        config = load_matrix_config(config_path)
        model = models[entry.model_condition_id]
        dataset = datasets[entry.dataset_panel_id]
        if (
            config.study != "broad"
            or not config.confirmatory
            or config.dataset.partition != "test"
            or config.model_integration != model.integration
            or config.dataset_integration != dataset.integration
        ):
            raise RuntimeError(f"panel entry does not match frozen study plan: {config_path}")
        storage = StorageLayout.from_spec(config.storage, repository_root)
        result_path = (
            storage.run(config.run_id).statistics / "confirmatory-test-analysis-v1.json"
        )
        if not result_path.is_file():
            raise RuntimeError(f"missing confirmatory test result: {result_path}")
        analysis = read_json(result_path)
        if (
            analysis.get("confirmatory") is not True
            or analysis.get("run_id") != config.run_id
            or analysis.get("config_sha256") != config.config_sha256
            or analysis.get("analysis_plan_sha256") != plan.sha256
        ):
            raise RuntimeError(f"test analysis provenance mismatch: {result_path}")
        moderators = {
            "model_family": model.family,
            "reasoning_regime": model.reasoning_regime,
            "scale_band": model.scale_band,
            "task_domain": dataset.task_domain,
        }
        inventory.extend(
            extract_panel_effects(
                analysis,
                model_condition_id=entry.model_condition_id,
                dataset_panel_id=entry.dataset_panel_id,
                moderators=moderators,
                contrasts=study.decoder_contrasts,
                readout_roles=panel.readout_roles,
                fixed_coverage_target=panel.fixed_coverage_target,
            )
        )
        input_provenance.append(
            {
                "model_condition_id": entry.model_condition_id,
                "dataset_panel_id": entry.dataset_panel_id,
                "config_path": entry.config_path,
                "config_sha256": config.config_sha256,
                "run_id": config.run_id,
                "analysis_path": str(result_path),
                "analysis_sha256": _file_sha256(result_path),
            }
        )

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for effect in inventory:
        grouped[
            (effect["endpoint"], effect["readout_role"], effect["contrast"])
        ].append(effect)
    analyses: dict[str, dict] = {}
    for (endpoint, readout, contrast), rows in sorted(grouped.items()):
        analyses[f"{endpoint}:{readout}:{contrast}"] = _meta(
            rows,
            tuple(plan.pooled_analysis.moderators),
            plan.pooled_analysis.minimum_effects_per_fitted_coefficient,
        )
    result = {
        "schema_version": "1.0",
        "analysis_id": panel.panel_analysis_id,
        "confirmatory": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "panel_config_sha256": panel.canonical_sha256,
        "analysis_plan_id": plan.plan_id,
        "analysis_plan_sha256": plan.sha256,
        "study_plan_id": study.plan_id,
        "study_plan_sha256": study.canonical_sha256,
        "analysis_git": git_metadata(repository_root),
        "input_provenance": input_provenance,
        "effect_inventory": inventory,
        "pooled_analyses": analyses,
        "claim_guard": (
            "Pooled estimates summarize heterogeneous model/dataset contrasts and "
            "must never be reported as evidence that every model shares the effect."
        ),
    }
    destination = destination.resolve()
    if destination.exists():
        existing = read_json(destination)
        comparable = dict(result)
        comparable["created_at_utc"] = existing.get("created_at_utc")
        comparable["analysis_git"] = existing.get("analysis_git")
        if existing != comparable:
            raise RuntimeError(f"confirmatory panel analysis collision at {destination}")
        return destination, existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_immutable_json(destination, result)
    return destination, result
