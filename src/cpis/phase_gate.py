"""Programmatic certification/test firewall and immutable dataset-access events."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from cpis.analysis_plan import load_analysis_plan
from cpis.config import Sha256, Slug, StrictModel
from cpis.storage import RunLayout, read_json, write_immutable_json


class PhaseGate(StrictModel):
    schema_version: Literal["1.0"]
    gate_id: Slug
    phase: Literal["certification", "test"]
    protocol_git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    study_design: Literal["legacy_certification_ladder", "frozen_two_phase_core"] = (
        "legacy_certification_ladder"
    )
    analysis_plan_sha256: Sha256
    prerequisites: dict[str, Literal[True]] = Field(min_length=1)
    frozen_files: dict[str, Sha256] = Field(min_length=1)

    @model_validator(mode="after")
    def safe_frozen_paths(self) -> "PhaseGate":
        for value in self.frozen_files:
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("phase-gate frozen files must be repository-relative")
        return self

    @property
    def canonical_sha256(self) -> str:
        import json

        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# The legacy ladder ran development, certification and test. The frozen core
# has no certification phase, so a test gate under it cannot honestly attest
# certification_complete; it attests the artifacts that actually gate held-out
# access instead. The firewall itself is unchanged: test data stay locked until
# a committed gate names them and the frozen artifacts exist.
REQUIRED_PREREQUISITES = {
    "certification": {
        "protocol_decisions_frozen",
        "model_integrations_complete",
        "dataset_integrations_complete",
        "scoring_adapters_verified",
        "primary_conditions_declared",
        "statistical_code_verified",
        "generation_configs_frozen",
    },
    "test": {
        "certification_complete",
        "primary_estimands_executable",
        "scoring_adapters_verified",
        "multiplicity_fixed",
        "equivalence_margins_fixed",
        "invalid_output_handling_fixed",
        "statistical_code_verified",
        "primary_conditions_declared",
    },
}

FROZEN_CORE_TEST_PREREQUISITES = {
    "calibration_complete",
    "thresholds_frozen",
    "primary_estimands_executable",
    "scoring_adapters_verified",
    "multiplicity_fixed",
    "equivalence_margins_fixed",
    "invalid_output_handling_fixed",
    "statistical_code_verified",
    "primary_conditions_declared",
}

REQUIRED_FROZEN_FILES = {
    "THESIS.md",
    "EXPERIMENT.md",
    "AGENTS.md",
    "PROTOCOL.md",
    "configs/analysis/confirmatory-v4.yaml",
    "configs/registry/primary-conditions-v1.yaml",
    "configs/registry/dense-grid-v1.yaml",
}

# The frozen core is governed by its own analysis plan and freeze record; the
# dense grid and the five-condition registry are outside it and are not what a
# held-out run must be pinned to.
FROZEN_CORE_REQUIRED_FILES = {
    "THESIS.md",
    "EXPERIMENT.md",
    "AGENTS.md",
    "PROTOCOL.md",
    "configs/analysis/core-v5.yaml",
    "evidence/protocol/core_study_freeze_v1.json",
}

ANALYSIS_PLAN_BY_DESIGN = {
    "legacy_certification_ladder": "configs/analysis/confirmatory-v4.yaml",
    "frozen_two_phase_core": "configs/analysis/core-v5.yaml",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_committed_file(
    path: Path, repository_root: Path, label: str = "phase gate"
) -> str:
    """Prove a decision artifact is tracked and byte-identical to its commit."""

    try:
        relative = str(path.resolve().relative_to(repository_root.resolve()))
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes the repository") from exc
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", relative],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0 or status.returncode != 0 or status.stdout.strip():
        raise RuntimeError(f"{label} must be committed and unmodified")
    return relative


def _load_and_validate_gate(path: Path, repository_root: Path) -> PhaseGate:
    require_committed_file(path, repository_root)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    gate = PhaseGate.model_validate(payload)
    if gate.phase == "test" and gate.study_design == "frozen_two_phase_core":
        required = FROZEN_CORE_TEST_PREREQUISITES
    else:
        required = REQUIRED_PREREQUISITES[gate.phase]
    if set(gate.prerequisites) != required:
        missing = sorted(required - set(gate.prerequisites))
        extra = sorted(set(gate.prerequisites) - required)
        raise RuntimeError(
            f"{gate.phase} gate prerequisite set mismatch; "
            f"missing={missing}, extra={extra}"
        )
    required_files = (
        FROZEN_CORE_REQUIRED_FILES
        if gate.study_design == "frozen_two_phase_core"
        else REQUIRED_FROZEN_FILES
    )
    if not required_files <= set(gate.frozen_files):
        missing = sorted(required_files - set(gate.frozen_files))
        raise RuntimeError(f"phase gate omits required frozen files: {missing}")
    analysis_path = repository_root / ANALYSIS_PLAN_BY_DESIGN[gate.study_design]
    if load_analysis_plan(analysis_path).sha256 != gate.analysis_plan_sha256:
        raise RuntimeError("phase-gate analysis-plan hash does not match the frozen file")
    for relative, expected in gate.frozen_files.items():
        frozen_path = (repository_root / relative).resolve()
        try:
            frozen_path.relative_to(repository_root.resolve())
        except ValueError as exc:
            raise RuntimeError("phase-gate frozen file escapes the repository") from exc
        if not frozen_path.is_file() or _file_sha256(frozen_path) != expected:
            raise RuntimeError(f"frozen protocol file differs from its gate: {relative}")
    try:
        exists = subprocess.run(
            ["git", "cat-file", "-e", f"{gate.protocol_git_commit}^{{commit}}"],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", gate.protocol_git_commit, "HEAD"],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("cannot validate phase-gate Git provenance") from exc
    if exists.returncode != 0:
        raise RuntimeError("phase gate references an unavailable protocol commit")
    if ancestor.returncode != 0:
        raise RuntimeError("phase-gate protocol commit is not an ancestor of HEAD")
    for relative, expected in gate.frozen_files.items():
        committed = subprocess.run(
            ["git", "show", f"{gate.protocol_git_commit}:{relative}"],
            cwd=repository_root,
            check=False,
            capture_output=True,
        )
        if (
            committed.returncode != 0
            or hashlib.sha256(committed.stdout).hexdigest() != expected
        ):
            raise RuntimeError(
                f"frozen file is absent or differs at the protocol commit: {relative}"
            )
    return gate


def require_phase_gate(partition: str, repository_root: Path) -> PhaseGate | None:
    if partition == "development":
        return None
    if partition not in {"certification", "test"}:
        raise ValueError(f"unknown dataset partition: {partition}")
    path = repository_root / "gates" / f"{partition}-open.yaml"
    if not path.is_file():
        raise RuntimeError(
            f"{partition} access is closed; missing versioned phase gate {path}"
        )
    gate = _load_and_validate_gate(path, repository_root)
    if gate.phase != partition:
        raise RuntimeError(f"phase gate {path} does not authorize {partition}")
    if partition == "test" and gate.study_design != "frozen_two_phase_core":
        certification_gate = repository_root / "gates/certification-open.yaml"
        if not certification_gate.is_file():
            raise RuntimeError("test access requires the certification phase gate")
        certified = _load_and_validate_gate(certification_gate, repository_root)
        if certified.phase != "certification":
            raise RuntimeError("test access requires a valid certification phase gate")
    return gate


def authorize_phase_file(
    gate: PhaseGate | None, path: Path, repository_root: Path, label: str
) -> None:
    """Require the exact non-development artifact to be frozen in its phase gate."""

    if gate is None:
        return
    resolved = path.resolve()
    try:
        relative = str(resolved.relative_to(repository_root.resolve()))
    except ValueError as exc:
        raise RuntimeError(f"phase {label} escapes the repository") from exc
    expected = gate.frozen_files.get(relative)
    if expected is None:
        raise RuntimeError(
            f"phase gate {gate.gate_id} does not authorize {label} {relative}"
        )
    if _file_sha256(resolved) != expected:
        raise RuntimeError(f"phase {label} differs from its gate: {relative}")


def authorize_phase_config(
    gate: PhaseGate | None, config_path: Path, repository_root: Path
) -> None:
    """Require the exact non-development config to be frozen in its phase gate."""

    authorize_phase_file(gate, config_path, repository_root, "config")


def authorize_phase_analysis_plan(
    gate: PhaseGate | None, analysis_plan_sha256: str
) -> None:
    """Bind non-development analyses to the plan declared by the phase gate."""

    if gate is not None and analysis_plan_sha256 != gate.analysis_plan_sha256:
        raise RuntimeError("analysis plan is not authorized by the phase gate")


def record_dataset_access(
    run: RunLayout,
    *,
    run_id: str,
    experiment_id: str,
    config_sha256: str,
    dataset_id: str,
    dataset_revision: str,
    partition: str,
    replica_index: int,
    gate: PhaseGate | None,
) -> None:
    """Write one immutable access event before the dataset file is opened."""
    path = run.manifests / f"dataset-access-replica-{replica_index:03d}.json"
    identity: dict[str, Any] = {
        "schema_version": "1.0",
        "event_type": "dataset_access",
        "run_id": run_id,
        "experiment_id": experiment_id,
        "config_sha256": config_sha256,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "partition": partition,
        "replica_index": replica_index,
        "phase_gate_id": gate.gate_id if gate else None,
        "phase_gate_sha256": gate.canonical_sha256 if gate else None,
        "protocol_git_commit": gate.protocol_git_commit if gate else None,
        "analysis_plan_sha256": gate.analysis_plan_sha256 if gate else None,
    }
    if path.exists():
        existing = read_json(path)
        if any(existing.get(key) != value for key, value in identity.items()):
            raise RuntimeError(f"dataset-access event collision at {path}")
        return
    write_immutable_json(
        path,
        {**identity, "accessed_at_utc": datetime.now(timezone.utc).isoformat()},
    )
