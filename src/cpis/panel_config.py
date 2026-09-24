"""Strict configuration for final saved-output panel aggregation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from cpis.config import Sha256, Slug, StrictModel


class PanelResultEntry(StrictModel):
    model_condition_id: Slug
    dataset_panel_id: Slug
    config_path: str = Field(min_length=1)

    @model_validator(mode="after")
    def safe_path(self) -> "PanelResultEntry":
        path = Path(self.config_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("panel test config paths must be repository-relative")
        return self


class PanelAnalysisConfig(StrictModel):
    schema_version: Literal["1.0"]
    panel_analysis_id: Slug
    scope: Literal["broad_confirmatory_test"]
    analysis_plan_path: str
    analysis_plan_sha256: Sha256
    study_plan_path: str
    study_plan_sha256: Sha256
    fixed_coverage_target: Literal[0.5]
    readout_roles: tuple[Literal["coupled", "standardized_deterministic"], ...]
    results: tuple[PanelResultEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def complete_and_safe(self) -> "PanelAnalysisConfig":
        for label, value in (
            ("analysis plan", self.analysis_plan_path),
            ("study plan", self.study_plan_path),
        ):
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{label} path must be repository-relative")
        identities = [
            (entry.model_condition_id, entry.dataset_panel_id)
            for entry in self.results
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("panel result model/dataset identities must be unique")
        if set(self.readout_roles) != {"coupled", "standardized_deterministic"}:
            raise ValueError("panel aggregation requires both prespecified readout roles")
        return self

    @property
    def canonical_sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_panel_analysis_config(path: Path) -> PanelAnalysisConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("panel analysis configuration must be a YAML mapping")
    return PanelAnalysisConfig.model_validate(payload)
