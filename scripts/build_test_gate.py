#!/usr/bin/env python3
"""Rebuild the test-partition phase gate over the currently committed files.

A gate is an attestation that a named set of files was committed and unmodified
when a test-partition run was authorized. It is therefore bound to a commit: the
held-out gate attested the protocol as it stood in September, and the protocol
has since been amended, so that gate no longer byte-matches and cannot authorize
anything new. That is the mechanism working, not a fault to route around.

This regenerates the gate over the current committed state and adds the
seed-robustness configs to the pinned set. Every listed file must be committed
and clean; the script refuses to attest a dirty tree, because a hash taken from
an uncommitted file attests nothing.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml

from cpis.analysis_plan import load_analysis_plan
from cpis.phase_gate import (
    FROZEN_CORE_REQUIRED_FILES,
    FROZEN_CORE_TEST_PREREQUISITES,
    _file_sha256,
    require_committed_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--gate-id", required=True)
    parser.add_argument(
        "--config-dirs",
        nargs="*",
        default=["configs/core/test", "configs/core/seeds"],
    )
    parser.add_argument("--destination", type=Path, default=Path("gates/test-open.yaml"))
    args = parser.parse_args()

    root = args.repository_root.resolve()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        raise SystemExit(
            "refusing to build a gate from a dirty tree; commit first:\n" + dirty
        )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()

    paths = sorted(FROZEN_CORE_REQUIRED_FILES)
    for directory in args.config_dirs:
        paths.extend(
            str(p.relative_to(root)) for p in sorted((root / directory).glob("*.yaml"))
        )

    frozen = {}
    for relative in sorted(set(paths)):
        path = root / relative
        require_committed_file(path, root, label=f"gate file {relative}")
        frozen[relative] = _file_sha256(path)

    plan = root / "configs/analysis/core-v5.yaml"
    gate = {
        "schema_version": "1.0",
        "gate_id": args.gate_id,
        "phase": "test",
        "study_design": "frozen_two_phase_core",
        "protocol_git_commit": head,
        # The plan's canonical hash, not the file digest: the validator compares
        # against load_analysis_plan(...).sha256, which is stable under
        # formatting that does not change the plan.
        "analysis_plan_sha256": load_analysis_plan(plan).sha256,
        "prerequisites": {k: True for k in sorted(FROZEN_CORE_TEST_PREREQUISITES)},
        "frozen_files": frozen,
    }
    destination = root / args.destination
    destination.write_text(yaml.safe_dump(gate, sort_keys=True), encoding="utf-8")
    print(f"{args.gate_id}: {len(frozen)} frozen files at {head[:12]}")


if __name__ == "__main__":
    main()
