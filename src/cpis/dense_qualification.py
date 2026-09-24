"""Outcome-blind technical qualification of dense decoder-grid coordinates."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from cpis.config import Sha256, Slug, StrictModel
from cpis.dense_plan import DenseGridPlan, DenseQualificationPlan, load_dense_grid_plan
from cpis.integration import IntegrationReference
from cpis.manifest import git_metadata
from cpis.matrix_audit import audit_matrix
from cpis.matrix_config import load_matrix_config
from cpis.matrix_records import MatrixParsedRecord
from cpis.phase_gate import PhaseGate, authorize_phase_file, require_committed_file
from cpis.records import GenerationRecord
from cpis.storage import StorageLayout, read_json
from cpis.study_plan import StudyPlan, load_study_plan


class DenseCellQualification(StrictModel):
    condition_id: Slug
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    status: Literal["supported", "unsupported"]
    answer_records: int = Field(gt=0)
    confidence_records: int = Field(gt=0)
    answer_length_finishes: int = Field(ge=0)
    confidence_length_finishes: int = Field(ge=0)
    invalid_answers: int = Field(ge=0)
    invalid_confidences: int = Field(ge=0)
    sampling_metadata_exact: bool
    exclusion_basis: str | None = None

    @model_validator(mode="after")
    def exclusion_basis_matches_status(self) -> "DenseCellQualification":
        stated = bool(self.exclusion_basis and self.exclusion_basis.strip())
        if (self.status == "unsupported") != stated:
            raise ValueError(
                "an unsupported dense cell requires a technical exclusion basis and "
                "a supported dense cell must not carry one"
            )
        return self


class QualificationProvenance(StrictModel):
    """A qualification is reproducible only from a clean, committed checkout."""

    available: Literal[True]
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    dirty: Literal[False]


class DenseQualificationRecord(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    qualification_id: Slug
    created_at_utc: datetime
    preflight_run_id: str
    preflight_config_sha256: Sha256
    preflight_audit_sha256: Sha256
    dense_grid_id: Slug
    dense_grid_sha256: Sha256
    study_plan_id: Slug
    study_plan_sha256: Sha256
    model_condition_id: Slug
    dataset_panel_id: Slug
    preflight_items: int = Field(gt=0)
    preflight_seed: int = Field(ge=0)
    qualification_git: QualificationProvenance
    cells: tuple[DenseCellQualification, ...] = Field(min_length=1)
    invalid_output_exclusion_prohibited: Literal[True]
    low_performance_exclusion_prohibited: Literal[True]

    @model_validator(mode="after")
    def cells_are_distinct(self) -> "DenseQualificationRecord":
        ids = [cell.condition_id for cell in self.cells]
        coordinates = [(cell.temperature, cell.top_p) for cell in self.cells]
        if len(set(ids)) != len(ids):
            raise ValueError("dense qualification cell IDs must be unique")
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("dense qualification coordinates must be unique")
        return self

    @property
    def canonical_sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def coordinates(self, status: str) -> set[tuple[float, float]]:
        return {
            (cell.temperature, cell.top_p)
            for cell in self.cells
            if cell.status == status
        }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dense_qualification(path: Path) -> DenseQualificationRecord:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dense qualification must be a YAML mapping")
    return DenseQualificationRecord.model_validate(payload)


def classify_dense_cell(
    plan: DenseQualificationPlan,
    *,
    condition_id: str,
    temperature: float,
    top_p: float,
    expected_records: int,
    answer_records: int,
    confidence_records: int,
    answer_length_finishes: int,
    confidence_length_finishes: int,
    invalid_answers: int,
    invalid_confidences: int,
    sampling_metadata_exact: bool,
) -> DenseCellQualification:
    """Apply the prespecified technical rules without reading any outcome."""

    if expected_records <= 0:
        raise ValueError("dense qualification requires a positive expected record count")
    if answer_records != expected_records or confidence_records != expected_records:
        raise RuntimeError(f"dense coordinate {condition_id} is incomplete")
    reasons: list[str] = []
    if not sampling_metadata_exact:
        reasons.append("saved sampling metadata differs from the requested coordinate")
    if (
        answer_length_finishes / expected_records
        > plan.maximum_answer_length_finish_rate
    ):
        reasons.append("answer length-finish rate exceeds the prespecified maximum")
    if (
        confidence_length_finishes / expected_records
        > plan.maximum_confidence_length_finish_rate
    ):
        reasons.append("confidence length-finish rate exceeds the prespecified maximum")
    return DenseCellQualification(
        condition_id=condition_id,
        temperature=temperature,
        top_p=top_p,
        status="unsupported" if reasons else "supported",
        answer_records=answer_records,
        confidence_records=confidence_records,
        answer_length_finishes=answer_length_finishes,
        confidence_length_finishes=confidence_length_finishes,
        invalid_answers=invalid_answers,
        invalid_confidences=invalid_confidences,
        sampling_metadata_exact=sampling_metadata_exact,
        exclusion_basis="; ".join(reasons) if reasons else None,
    )


def validate_dense_qualification_reference(
    reference: IntegrationReference,
    repository_root: Path,
    gate: PhaseGate | None = None,
) -> DenseQualificationRecord:
    """Resolve a cited qualification that is committed, unmodified, and hash-bound."""

    repository_root = repository_root.resolve()
    path = (repository_root / reference.record_path).resolve()
    try:
        path.relative_to(repository_root)
    except ValueError as exc:
        raise RuntimeError("dense qualification reference escapes the repository") from exc
    if not path.is_file():
        raise RuntimeError(f"dense qualification record is missing: {reference.record_path}")
    require_committed_file(path, repository_root, "dense qualification record")
    authorize_phase_file(gate, path, repository_root, "dense qualification record")
    record = load_dense_qualification(path)
    if record.qualification_id != reference.record_id:
        raise RuntimeError("dense qualification ID does not match its reference")
    if record.canonical_sha256 != reference.canonical_sha256:
        raise RuntimeError("dense qualification hash does not match its reference")
    return record


def check_dense_qualification_covers(
    record: DenseQualificationRecord,
    *,
    dense: DenseGridPlan,
    study: StudyPlan,
    model_condition_id: str,
    planned: set[tuple[float, float]],
    observed: set[tuple[float, float]],
) -> dict:
    """Require every present and absent coordinate to match the frozen qualification."""

    if record.dense_grid_id != dense.grid_id or (
        record.dense_grid_sha256 != dense.canonical_sha256
    ):
        raise RuntimeError("dense qualification was made against a different frozen grid")
    if record.study_plan_id != study.plan_id or (
        record.study_plan_sha256 != study.canonical_sha256
    ):
        raise RuntimeError("dense qualification was made against a different study plan")
    if record.model_condition_id != model_condition_id:
        raise RuntimeError("dense qualification belongs to a different model condition")
    if record.dataset_panel_id != dense.development_preflight_dataset_panel:
        raise RuntimeError("dense qualification did not use the declared preflight panel")
    if record.preflight_seed != dense.development_preflight_seed:
        raise RuntimeError("dense qualification did not use the declared preflight seed")
    qualified = record.coordinates("supported") | record.coordinates("unsupported")
    if qualified != planned:
        raise RuntimeError("dense qualification does not classify every planned coordinate")
    supported = record.coordinates("supported")
    if observed != supported:
        missing = sorted(supported - observed)
        extra = sorted(observed - supported)
        raise RuntimeError(
            "dense matrix cells disagree with the frozen qualification; "
            f"qualified-but-absent={missing}, present-but-unqualified={extra}"
        )
    return {
        "dense_qualification_id": record.qualification_id,
        "dense_qualification_sha256": record.canonical_sha256,
        "preflight_run_id": record.preflight_run_id,
        "preflight_config_sha256": record.preflight_config_sha256,
        "exclusions": [
            {
                "temperature": cell.temperature,
                "top_p": cell.top_p,
                "exclusion_basis": cell.exclusion_basis,
            }
            for cell in sorted(
                (cell for cell in record.cells if cell.status == "unsupported"),
                key=lambda cell: (cell.temperature, cell.top_p),
            )
        ],
    }


def qualify_dense_preflight(
    config_path: Path,
    dense_plan_path: Path,
    study_plan_path: Path,
    repository_root: Path,
    destination: Path,
) -> DenseQualificationRecord:
    """Classify cells using technical criteria fixed before preflight execution."""

    repository_root = repository_root.resolve()
    config_path = config_path.resolve()
    destination = destination.resolve()
    existing = load_dense_qualification(destination) if destination.is_file() else None
    dense = load_dense_grid_plan(dense_plan_path.resolve())
    study = load_study_plan(study_plan_path.resolve())
    config = load_matrix_config(config_path)
    if (
        config.study != "development_matrix"
        or config.dataset.partition != "development"
        or config.dataset.item_limit != dense.development_preflight_item_limit
        or config.dataset.item_selection_salt
        != dense.development_preflight_selection_salt
        or config.inference.seeds != (dense.development_preflight_seed,)
    ):
        raise ValueError("dense qualification requires the declared development preflight")
    model_matches = [
        entry
        for entry in study.model_conditions
        if entry.integration == config.model_integration
        and entry.condition_id in study.dense_model_conditions
    ]
    dataset_matches = [
        entry
        for entry in study.core_datasets
        if entry.integration == config.dataset_integration
        and entry.panel_id == dense.development_preflight_dataset_panel
    ]
    if len(model_matches) != 1 or len(dataset_matches) != 1:
        raise RuntimeError("dense preflight is outside the declared model/dataset panel")
    model_id = model_matches[0].condition_id
    model_plan = next(
        entry for entry in dense.model_conditions if entry.model_condition_id == model_id
    )
    planned = {
        (temperature, top_p)
        for temperature in dense.temperatures
        for top_p in dense.top_ps
    }
    observed = {
        (condition.sampling.temperature, condition.sampling.top_p)
        for condition in config.inference.answer_conditions
    }
    if observed != planned or len(config.inference.answer_conditions) != len(planned):
        raise RuntimeError("dense preflight must execute every planned coordinate")
    if any(
        condition.sampling.top_k != model_plan.held_top_k
        for condition in config.inference.answer_conditions
    ):
        raise RuntimeError("dense preflight changed the model-native top-k setting")

    audit_path, audit = audit_matrix(config_path, repository_root)
    storage = StorageLayout.from_spec(config.storage, repository_root)
    run = storage.run(config.run_id)
    answers = {
        record.record_id: record
        for path in sorted(run.raw_answer.glob("*.json"))
        if (record := GenerationRecord.model_validate(read_json(path)))
    }
    confidences = {
        record.record_id: record
        for path in sorted(run.raw_confidence.glob("*.json"))
        if (record := GenerationRecord.model_validate(read_json(path)))
    }
    parsed = [
        MatrixParsedRecord.model_validate(read_json(path))
        for path in sorted(run.parsed.glob("*.json"))
    ]
    design = config.inference.design
    expected_per_cell = dense.development_preflight_item_limit
    cells: list[DenseCellQualification] = []
    for condition in config.inference.answer_conditions:
        coupled_readout_id = design.coupled_readout_by_answer[condition.condition_id]
        readout = next(
            row
            for row in config.inference.confidence_readouts
            if row.readout_id == coupled_readout_id
        )
        coupled = [
            row
            for row in parsed
            if row.answer_condition_id == condition.condition_id
            and "coupled" in row.design_roles
        ]
        answer_ids = {row.answer_raw_record_id for row in coupled}
        confidence_ids = {row.confidence_raw_record_id for row in coupled}
        if len(coupled) != len(answer_ids) or len(coupled) != len(confidence_ids):
            raise RuntimeError(
                f"dense coordinate {condition.condition_id} reuses a raw generation"
            )
        answer_rows = [answers[record_id] for record_id in answer_ids]
        confidence_rows = [confidences[record_id] for record_id in confidence_ids]
        cells.append(
            classify_dense_cell(
                dense.qualification,
                condition_id=condition.condition_id,
                temperature=condition.sampling.temperature,
                top_p=condition.sampling.top_p,
                expected_records=expected_per_cell,
                answer_records=len(answer_rows),
                confidence_records=len(confidence_rows),
                answer_length_finishes=sum(
                    row.finish_reason == "length" for row in answer_rows
                ),
                confidence_length_finishes=sum(
                    row.finish_reason == "length" for row in confidence_rows
                ),
                invalid_answers=sum(
                    row.answer_parse_status == "invalid" for row in coupled
                ),
                invalid_confidences=sum(
                    row.confidence_parse_status == "invalid" for row in coupled
                ),
                sampling_metadata_exact=all(
                    row.sampling == condition.sampling for row in answer_rows
                )
                and all(row.sampling == readout.sampling for row in confidence_rows),
            )
        )
    if existing is not None:
        # Re-deriving an already published record only checks it for collision;
        # the checkout that minted it is the one that had to be clean.
        provenance = existing.qualification_git
    else:
        state = git_metadata(repository_root)
        if not state.get("available") or state.get("dirty"):
            raise RuntimeError(
                "a dense qualification must be produced from a clean, committed checkout"
            )
        provenance = QualificationProvenance.model_validate(state)
    record = DenseQualificationRecord(
        qualification_id=f"{model_id}-dense-qualification-v1",
        created_at_utc=datetime.now(timezone.utc),
        preflight_run_id=config.run_id,
        preflight_config_sha256=config.config_sha256,
        preflight_audit_sha256=_file_sha256(audit_path),
        dense_grid_id=dense.grid_id,
        dense_grid_sha256=dense.canonical_sha256,
        study_plan_id=study.plan_id,
        study_plan_sha256=study.canonical_sha256,
        model_condition_id=model_id,
        dataset_panel_id=dataset_matches[0].panel_id,
        preflight_items=audit["dataset_items"],
        preflight_seed=dense.development_preflight_seed,
        qualification_git=provenance,
        cells=tuple(cells),
        invalid_output_exclusion_prohibited=True,
        low_performance_exclusion_prohibited=True,
    )
    if existing is not None:
        comparable = record.model_copy(
            update={"created_at_utc": existing.created_at_utc}
        )
        if comparable != existing:
            raise RuntimeError(f"dense qualification collision at {destination}")
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(record.model_dump(mode="json"), sort_keys=False, width=100),
        encoding="utf-8",
    )
    return record
