"""Explicit provenance review for replicas produced by different clean commits."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

from cpis.storage import canonical_json_bytes, read_json


def _git(repository_root: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repository_root,
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("cannot verify cross-commit replica provenance") from exc


def validate_replica_git_equivalence(
    *,
    run_id: str,
    config_sha256: str,
    manifests: list[dict[str, Any]],
    repository_root: Path,
) -> dict[str, Any]:
    """Validate an outcome-blind, byte-exact review of every generation commit."""

    commits = sorted({manifest.get("git", {}).get("commit") for manifest in manifests})
    if None in commits or any(
        not manifest.get("git", {}).get("available")
        or manifest.get("git", {}).get("dirty")
        for manifest in manifests
    ):
        raise RuntimeError("cross-commit replicas require available clean git states")
    path = (
        repository_root
        / "provenance"
        / "replica-git-equivalence"
        / f"{run_id}.json"
    )
    if not path.is_file():
        raise RuntimeError(
            "matrix replicas were generated from different git states without a "
            f"versioned equivalence review: {path}"
        )
    record = read_json(path)
    if (
        record.get("schema_version") != "1.0"
        or record.get("run_id") != run_id
        or record.get("config_sha256") != config_sha256
        or sorted(record.get("commits", [])) != commits
        or record.get("reviewed_before_outcome_analysis") is not True
        or record.get("generation_semantics_equivalent") is not True
        or not isinstance(record.get("rationale"), str)
        or not record["rationale"].strip()
    ):
        raise RuntimeError("invalid cross-commit replica equivalence review")
    base = record.get("base_commit")
    comparisons = record.get("comparisons")
    if base not in commits or not isinstance(comparisons, list):
        raise RuntimeError("invalid cross-commit replica comparison plan")
    by_commit = {entry.get("comparison_commit"): entry for entry in comparisons}
    if set(by_commit) != set(commits) - {base}:
        raise RuntimeError("equivalence review does not cover every generation commit")
    for commit in commits:
        _git(repository_root, "cat-file", "-e", f"{commit}^{{commit}}")
    for comparison_commit, entry in by_commit.items():
        if entry.get("base_commit") != base:
            raise RuntimeError("equivalence comparison has an inconsistent base commit")
        diff = _git(
            repository_root,
            "diff",
            "--binary",
            "--full-index",
            base,
            comparison_commit,
        )
        names = _git(
            repository_root,
            "diff",
            "--name-only",
            "-z",
            base,
            comparison_commit,
        )
        changed_paths = [raw.decode("utf-8") for raw in names.split(b"\0") if raw]
        if (
            entry.get("diff_sha256") != hashlib.sha256(diff).hexdigest()
            or entry.get("changed_paths") != changed_paths
        ):
            raise RuntimeError("generation-commit diff no longer matches its review")
    return {
        **record,
        "canonical_sha256": hashlib.sha256(canonical_json_bytes(record)).hexdigest(),
        "path": str(path.relative_to(repository_root)),
    }
