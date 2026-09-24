"""Strict registry of the prespecified model, dataset, and contrast panel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from cpis.config import Slug, StrictModel
from cpis.integration import IntegrationReference


class ModelConditionPlan(StrictModel):
    condition_id: Slug
    family: Literal["qwen", "ministral", "gemma"]
    scale_band: Literal["small", "mid", "large"]
    reasoning_regime: Literal["non_thinking", "thinking", "instruct", "reasoning"]
    integration: IntegrationReference | None = None
    status: Literal["included", "excluded_external_access"]
    role: Literal["primary", "independent_family_replication"]
    native_reference: str = Field(min_length=20)
    exclusion_reason: str | None = None

    @model_validator(mode="after")
    def status_is_complete(self) -> "ModelConditionPlan":
        if self.status == "included" and self.integration is None:
            raise ValueError("included model condition requires an integration record")
        if self.status == "excluded_external_access" and not self.exclusion_reason:
            raise ValueError("excluded model condition requires a reason")
        return self


class DatasetPanelPlan(StrictModel):
    panel_id: Slug
    task_domain: Literal[
        "knowledge", "high_stakes_medical", "instruction_following", "executable_code"
    ]
    integration: IntegrationReference
    status: Literal["included"]
    success_event: str = Field(min_length=20)
    independent_unit: str = Field(min_length=10)


class ExternalStudyPlan(StrictModel):
    study_id: Slug
    role: Literal["freshness_replication", "agent_external_validation"]
    status: Literal[
        "integration_pending", "prepared", "excluded_external_access"
    ]
    source: str = Field(min_length=8)
    note: str = Field(min_length=20)


class StudyPlan(StrictModel):
    schema_version: Literal["1.0"]
    plan_id: Slug
    model_conditions: tuple[ModelConditionPlan, ...] = Field(min_length=1)
    core_datasets: tuple[DatasetPanelPlan, ...] = Field(min_length=4)
    broad_model_conditions: tuple[Slug, ...] = Field(min_length=1)
    broad_dataset_panels: tuple[Slug, ...] = Field(min_length=4)
    dense_model_conditions: tuple[Slug, ...] = Field(min_length=4, max_length=4)
    dense_dataset_panels: tuple[Slug, ...] = Field(min_length=1)
    agent_model_conditions: tuple[Slug, ...] = Field(min_length=3)
    decoder_contrasts: tuple[
        Literal[
            "reference_to_low_temperature",
            "reference_to_high_temperature",
            "reference_to_low_top_p",
            "reference_to_high_diversity",
        ],
        ...,
    ]
    replicate_seeds: tuple[int, int, int]
    external_studies: tuple[ExternalStudyPlan, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def declared_sets_are_valid(self) -> "StudyPlan":
        models = {entry.condition_id: entry for entry in self.model_conditions}
        datasets = {entry.panel_id: entry for entry in self.core_datasets}
        if len(models) != len(self.model_conditions) or len(datasets) != len(
            self.core_datasets
        ):
            raise ValueError("model condition and dataset panel IDs must be unique")
        included = {key for key, value in models.items() if value.status == "included"}
        for label, declared in (
            ("broad", self.broad_model_conditions),
            ("dense", self.dense_model_conditions),
            ("agent", self.agent_model_conditions),
        ):
            if len(set(declared)) != len(declared) or not set(declared) <= included:
                raise ValueError(f"{label} model conditions must be unique included models")
        if set(self.broad_dataset_panels) != set(datasets):
            raise ValueError("broad study must declare every core dataset panel")
        if not set(self.dense_dataset_panels) <= set(datasets):
            raise ValueError("dense study names an unknown dataset panel")
        if len(set(self.decoder_contrasts)) != 4:
            raise ValueError("exactly four unique decoder contrasts are required")
        if self.replicate_seeds != (1729, 2718, 31415):
            raise ValueError("the three paired replicate seeds are frozen")
        return self

    @property
    def canonical_sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_study_plan(path: Path) -> StudyPlan:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("study plan must be a YAML mapping")
    return StudyPlan.model_validate(payload)
