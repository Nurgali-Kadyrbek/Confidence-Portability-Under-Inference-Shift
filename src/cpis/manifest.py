"""Immutable experiment manifest with code and runtime provenance."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from cpis.config import ExperimentConfig


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    experiment_id: str
    replica_index: int
    replica_count: int
    config_sha256: str
    config: dict[str, Any]
    source_config: str
    created_at_utc: datetime
    git: dict[str, Any]
    runtime: dict[str, Any]
    packages: dict[str, str | None]
    gpus: list[dict[str, str]]


def _command(args: list[str], cwd: Path) -> tuple[int, str]:
    try:
        result = subprocess.run(
            args, cwd=cwd, text=True, capture_output=True, check=False, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return 1, ""
    return result.returncode, result.stdout.strip()


def git_metadata(repository_root: Path) -> dict[str, Any]:
    code, commit = _command(["git", "rev-parse", "HEAD"], repository_root)
    if code != 0:
        return {"available": False, "commit": None, "dirty": None}
    dirty_code, status = _command(["git", "status", "--porcelain"], repository_root)
    return {
        "available": True,
        "commit": commit,
        "dirty": dirty_code != 0 or bool(status),
    }


def gpu_metadata(repository_root: Path) -> list[dict[str, str]]:
    code, output = _command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        repository_root,
    )
    if code != 0 or not output:
        return []
    keys = ("index", "name", "uuid", "memory_total_mib", "driver_version")
    return [
        dict(zip(keys, (value.strip() for value in line.split(",")), strict=True))
        for line in output.splitlines()
    ]


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in (
        "cpis",
        "pydantic",
        "PyYAML",
        "vllm",
        "torch",
        "transformers",
        "mistral-common",
        "flashinfer-python",
    ):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def runtime_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch
    except ImportError:
        metadata.update(
            {"torch_cuda_version": None, "cuda_available": None, "visible_gpu_count": None}
        )
    else:
        metadata.update(
            {
                "torch_cuda_version": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "visible_gpu_count": torch.cuda.device_count(),
            }
        )
    return metadata


def build_manifest(
    config: ExperimentConfig,
    config_path: Path,
    repository_root: Path,
    replica_index: int,
) -> Manifest:
    return Manifest(
        run_id=config.run_id,
        experiment_id=config.experiment_id,
        replica_index=replica_index,
        replica_count=config.execution.replicas,
        config_sha256=config.config_sha256,
        config=config.model_dump(mode="json"),
        source_config=str(config_path.resolve()),
        created_at_utc=datetime.now(timezone.utc),
        git=git_metadata(repository_root),
        runtime=runtime_metadata(),
        packages=package_versions(),
        gpus=gpu_metadata(repository_root),
    )
