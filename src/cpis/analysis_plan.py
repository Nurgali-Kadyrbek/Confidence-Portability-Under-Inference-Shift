"""Strict configuration for confirmatory statistics, separate from generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from cpis.config import Slug, StrictModel


class RiskContractPlan(StrictModel):
    target_risk: float = Field(gt=0, lt=1)
    certification_tail_probability: float = Field(gt=0, lt=1)
    minimum_coverage: float = Field(gt=0, le=1)
    threshold_selection: Literal[
        "max_coverage_subject_to_empirical_risk"
    ]
    certification_method: Literal["one_sided_clopper_pearson"]
    infeasible_label: Literal["risk-contract infeasible"]
    certification_replicate_seed: int = Field(ge=0)


class GroupedCertificationPlan(StrictModel):
    method: Literal["deterministic_balanced_one_variant_per_source_group"]
    selector_salt: str = Field(min_length=16)
    variant_order_key: Literal["instruction_count"]
    rationale: str = Field(min_length=40)


class AgentGroupedCertificationPlan(StrictModel):
    method: Literal["deterministic_balanced_one_category_per_source_group"]
    selector_salt: str = Field(min_length=16)
    variant_key: Literal["category"]
    categories: tuple[
        Literal["base", "long_context", "miss_func", "miss_param"], ...
    ]
    rationale: str = Field(min_length=40)

    @model_validator(mode="after")
    def all_categories_once(self) -> "AgentGroupedCertificationPlan":
        expected = {"base", "long_context", "miss_func", "miss_param"}
        if set(self.categories) != expected or len(self.categories) != len(expected):
            raise ValueError("agent certification must declare every BFCL category once")
        return self


class FixedCoveragePlan(StrictModel):
    targets: tuple[float, ...] = Field(min_length=1)
    tie_rule: Literal["smallest_coverage_not_below_target"]
    universal_confirmatory_target: float = Field(default=0.50, gt=0, le=1)

    @model_validator(mode="after")
    def validate_targets(self) -> "FixedCoveragePlan":
        if any(not 0 < target <= 1 for target in self.targets):
            raise ValueError("fixed coverage targets must lie in (0, 1]")
        if tuple(sorted(set(self.targets))) != self.targets:
            raise ValueError("fixed coverage targets must be unique and increasing")
        if self.universal_confirmatory_target not in self.targets:
            raise ValueError("universal confirmatory target must be a fixed-coverage target")
        return self


class BootstrapPlan(StrictModel):
    replicates: Literal[5000]
    seed: int = Field(ge=0)
    unit: Literal["original_benchmark_item"]
    strata: tuple[str, ...] = Field(min_length=1)


class MultiplicityPlan(StrictModel):
    method: Literal["holm_step_down"]
    familywise_alpha: Literal[0.05]
    confirmatory_contrasts: tuple[
        Literal[
            "reference_to_low_temperature",
            "reference_to_high_temperature",
            "reference_to_low_top_p",
            "reference_to_high_diversity",
        ],
        ...,
    ]
    # The frozen core runs two single-mechanism contrasts, not the original
    # four; the family a Holm correction is applied over must match what is
    # actually tested, so the declared family and the contrast list are checked
    # against each other rather than assumed.
    family_definition: Literal[
        "four_decoder_contrasts_within_model_dataset",
        "two_decoder_contrasts_within_model_dataset",
    ] = "four_decoder_contrasts_within_model_dataset"
    raw_p_value_method: Literal["paired_cluster_randomization"] = (
        "paired_cluster_randomization"
    )
    randomization_replicates: int = Field(default=9999, ge=999)
    randomization_seed: int = Field(default=20260921, ge=0)
    endpoint_rule: Literal[
        "certified_risk_threshold_else_fixed_coverage_target"
    ] = "certified_risk_threshold_else_fixed_coverage_target"
    alternative_rule: Literal[
        "one_sided_risk_increase_when_certified_else_two_sided_change"
    ] = "one_sided_risk_increase_when_certified_else_two_sided_change"

    @model_validator(mode="after")
    def unique_contrasts(self) -> "MultiplicityPlan":
        if len(set(self.confirmatory_contrasts)) != len(self.confirmatory_contrasts):
            raise ValueError("confirmatory contrasts must be unique")
        declared = {
            "four_decoder_contrasts_within_model_dataset": 4,
            "two_decoder_contrasts_within_model_dataset": 2,
        }[self.family_definition]
        if len(self.confirmatory_contrasts) != declared:
            raise ValueError(
                f"{self.family_definition} requires exactly {declared} confirmatory "
                f"contrasts, got {len(self.confirmatory_contrasts)}"
            )
        return self


class MarginPlan(StrictModel):
    selective_risk: Literal[0.025]
    coverage: Literal[0.05]
    rationale: str = Field(min_length=20)


class InvalidConfidencePlan(StrictModel):
    operational: Literal["forced_abstention"]
    calibration: Literal["valid_only_with_denominator_and_invalid_rate"]
    imputation: Literal["prohibited"]


class EquivalenceTestingPlan(StrictModel):
    method: Literal["paired_cluster_bootstrap_tost"]
    interval_level: Literal[0.90]
    noninferiority_interval_level: Literal[0.95]


class PooledAnalysisPlan(StrictModel):
    method: Literal["two_stage_reml_random_effects_meta_analysis"]
    input_unit: Literal["model_dataset_contrast"]
    variance_source: Literal["paired_item_cluster_bootstrap"]
    heterogeneity_statistics: tuple[Literal["tau_squared", "i_squared"], ...]
    moderators: tuple[
        Literal["model_family", "reasoning_regime", "scale_band", "task_domain"],
        ...,
    ]
    minimum_effects_per_fitted_coefficient: int = Field(ge=5)
    sparse_moderator_fallback: Literal["stratified_effects_without_meta_regression"]
    claim_rule: Literal["pooled_effect_never_implies_every_model"]


class DenseSurfacePlan(StrictModel):
    role: Literal["secondary_model_based_description"]
    model: Literal["clustered_fractional_logit_and_logistic_response_surfaces"]
    cluster: Literal["original_benchmark_item_or_source_group"]
    predictors: tuple[
        Literal[
            "centered_temperature",
            "centered_logit_top_p",
            "temperature_squared",
            "logit_top_p_squared",
            "temperature_by_logit_top_p",
            "model_condition",
            "dataset",
        ],
        ...,
    ]
    uncertainty: Literal["item_clustered_bootstrap"]
    pooling: Literal["model_specific_surfaces_then_reml_meta_analysis"]
    top_p_logit_epsilon: Literal[0.000001] | None = None


class RobustnessPlan(StrictModel):
    same_answer_definition: Literal["exact_parsed_answer_identity_within_item_seed"]
    accuracy_match_absolute_tolerance: float = Field(gt=0, le=0.05)
    confidence_metrics: tuple[Literal["auroc", "aurc", "brier", "ece"], ...]
    ece_bins: Literal[10]
    calibration_sensitivity_epsilon: float = Field(gt=0, lt=0.01)


class AnalysisPlan(StrictModel):
    schema_version: Literal["1.0", "2.0", "3.0", "4.0"]
    plan_id: Slug
    risk_contract: RiskContractPlan
    fixed_coverage: FixedCoveragePlan
    invalid_confidence: InvalidConfidencePlan
    bootstrap: BootstrapPlan
    multiplicity: MultiplicityPlan
    equivalence_margins: MarginPlan
    grouped_certification: GroupedCertificationPlan | None = None
    agent_grouped_certification: AgentGroupedCertificationPlan | None = None
    equivalence_testing: EquivalenceTestingPlan | None = None
    pooled_analysis: PooledAnalysisPlan | None = None
    dense_surface: DenseSurfacePlan | None = None
    robustness: RobustnessPlan | None = None

    @model_validator(mode="after")
    def versioned_requirements(self) -> "AnalysisPlan":
        additions = (
            self.grouped_certification,
            self.equivalence_testing,
            self.pooled_analysis,
            self.dense_surface,
            self.robustness,
        )
        if self.schema_version == "1.0" and any(value is not None for value in additions):
            raise ValueError("analysis plan v1 cannot declare v2 analysis sections")
        if self.schema_version in {"1.0", "2.0", "3.0"} and (
            self.agent_grouped_certification is not None
        ):
            raise ValueError("agent grouped certification was introduced in plan v4")
        if self.schema_version in {"2.0", "3.0"} and any(
            value is None for value in additions
        ):
            raise ValueError("analysis plan v2+ requires all confirmatory analysis sections")
        if self.schema_version == "4.0" and (
            any(value is None for value in additions)
            or self.agent_grouped_certification is None
        ):
            raise ValueError("analysis plan v4 requires controlled and agent sections")
        if self.schema_version == "2.0" and self.dense_surface is not None:
            if self.dense_surface.top_p_logit_epsilon is not None:
                raise ValueError("analysis plan v2 predates the boundary-logit rule")
        if self.schema_version == "3.0" and self.dense_surface is not None:
            if self.dense_surface.top_p_logit_epsilon is None:
                raise ValueError("analysis plan v3 requires the top-p boundary-logit rule")
        if self.schema_version == "4.0" and self.dense_surface is not None:
            if self.dense_surface.top_p_logit_epsilon is None:
                raise ValueError("analysis plan v4 requires the top-p boundary-logit rule")
        return self

    def canonical_json(self) -> str:
        payload = self.model_dump(mode="json")
        # Preserve the already-published v1 digest after extending the loader.
        # Pilot artifacts refer to that exact hash and must remain reproducible.
        if self.schema_version == "1.0":
            for key in (
                "grouped_certification",
                "equivalence_testing",
                "pooled_analysis",
                "dense_surface",
                "robustness",
                "agent_grouped_certification",
            ):
                payload.pop(key, None)
            for key in (
                "family_definition",
                "raw_p_value_method",
                "randomization_replicates",
                "randomization_seed",
                "endpoint_rule",
                "alternative_rule",
            ):
                payload["multiplicity"].pop(key, None)
            payload["fixed_coverage"].pop("universal_confirmatory_target", None)
        if self.schema_version == "2.0":
            payload["dense_surface"].pop("top_p_logit_epsilon", None)
        if self.schema_version == "3.0":
            payload.pop("agent_grouped_certification", None)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_analysis_plan(path: Path) -> AnalysisPlan:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("analysis plan must be a YAML mapping")
    return AnalysisPlan.model_validate(payload)
