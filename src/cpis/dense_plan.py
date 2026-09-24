"""Prespecified common-coordinate dense decoder grid."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from cpis.config import Slug, StrictModel


class DenseModelPlan(StrictModel):
    model_condition_id: Slug
    reference_temperature: float = Field(ge=0)
    reference_top_p: float = Field(gt=0, le=1)
    held_top_k: int | None = Field(default=None, ge=1)
    top_k_handling: Literal["hold_model_native"]
    interpretation: str = Field(min_length=30)


class DenseQualificationPlan(StrictModel):
    completeness_rule: Literal["all_planned_preflight_records_required"]
    sampling_identity_rule: Literal["exact_requested_sampling_metadata_required"]
    maximum_answer_length_finish_rate: Literal[0.0]
    maximum_confidence_length_finish_rate: Literal[0.0]
    invalid_output_rule: Literal["retain_as_outcome_never_exclude"]
    low_performance_rule: Literal["retain_as_outcome_never_exclude"]
    unsupported_cell_rule: Literal[
        "missing_by_design_only_after_prespecified_technical_failure"
    ]


class DenseGridPlan(StrictModel):
    schema_version: Literal["1.0"]
    grid_id: Slug
    role: Literal["secondary_common_coordinate_response_surface"]
    model_conditions: tuple[DenseModelPlan, ...] = Field(min_length=4, max_length=4)
    dataset_panels: tuple[Slug, ...] = Field(min_length=1)
    temperatures: tuple[float, ...] = Field(min_length=5)
    top_ps: tuple[float, ...] = Field(min_length=5)
    replicate_seeds: tuple[int, int, int]
    development_preflight_dataset_panel: Slug
    development_preflight_item_limit: int = Field(ge=4, le=64)
    development_preflight_seed: Literal[1729]
    development_preflight_selection_salt: str = Field(min_length=16)
    standardized_readout: Literal["model_specific_validated_fixed_readout"]
    unsupported_cell_rule: Literal["missing_by_design_never_coerced"]
    cross_model_claim_rule: Literal[
        "nominal_coordinates_do_not_imply_identical_sampling_mechanisms"
    ]
    qualification: DenseQualificationPlan

    @model_validator(mode="after")
    def complete_grid(self) -> "DenseGridPlan":
        if tuple(sorted(set(self.temperatures))) != self.temperatures:
            raise ValueError("dense temperatures must be sorted and unique")
        if tuple(sorted(set(self.top_ps))) != self.top_ps:
            raise ValueError("dense top-p coordinates must be sorted and unique")
        if any(value < 0 or value > 2 for value in self.temperatures):
            raise ValueError("dense temperatures must lie in [0, 2]")
        if any(value <= 0 or value > 1 for value in self.top_ps):
            raise ValueError("dense top-p coordinates must lie in (0, 1]")
        ids = [entry.model_condition_id for entry in self.model_conditions]
        if len(ids) != len(set(ids)):
            raise ValueError("dense model-condition IDs must be unique")
        if len(self.dataset_panels) != len(set(self.dataset_panels)):
            raise ValueError("dense dataset panels must be unique")
        if self.development_preflight_dataset_panel not in self.dataset_panels:
            raise ValueError("dense preflight dataset must be a declared dense panel")
        if self.replicate_seeds != (1729, 2718, 31415):
            raise ValueError("dense grid uses the three frozen paired seeds")
        for entry in self.model_conditions:
            if entry.reference_temperature not in self.temperatures:
                raise ValueError("every model reference temperature must be on the grid")
            if entry.reference_top_p not in self.top_ps:
                raise ValueError("every model reference top-p must be on the grid")
        return self

    @property
    def canonical_sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_dense_grid_plan(path: Path) -> DenseGridPlan:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dense grid plan must be a YAML mapping")
    return DenseGridPlan.model_validate(payload)
