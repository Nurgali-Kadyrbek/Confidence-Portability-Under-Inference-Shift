"""RAID-rooted paths and immutable JSON persistence."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cpis.config import StorageSpec


def _absolute_env_path(env_name: str, fallback: Path | None = None) -> Path:
    value = os.environ.get(env_name)
    if value is None:
        if fallback is None:
            raise ValueError(
                f"{env_name} is required; point it at the external RAID storage root"
            )
        path = fallback
    else:
        path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{env_name} must be an absolute path, got {path}")
    return path.resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class StorageLayout:
    root: Path
    model_cache: Path
    dataset_cache: Path
    output_root: Path
    log_root: Path

    @classmethod
    def from_spec(cls, spec: StorageSpec, repository_root: Path) -> "StorageLayout":
        root = _absolute_env_path(spec.root_env)
        layout = cls(
            root=root,
            model_cache=_absolute_env_path(spec.model_cache_env, root / "model-cache"),
            dataset_cache=_absolute_env_path(spec.dataset_cache_env, root / "dataset-cache"),
            output_root=_absolute_env_path(spec.output_root_env, root / "experiments"),
            log_root=_absolute_env_path(spec.log_root_env, root / "logs"),
        )
        repo = repository_root.resolve()
        for name, path in (
            ("storage root", layout.root),
            ("model cache", layout.model_cache),
            ("dataset cache", layout.dataset_cache),
            ("output root", layout.output_root),
            ("log root", layout.log_root),
        ):
            if _is_within(path, repo):
                raise ValueError(f"{name} must be outside the Git repository: {path}")
            path.mkdir(parents=True, exist_ok=True)
        return layout

    def configure_process_caches(self) -> None:
        """Route heavyweight library caches away from the system disk."""
        cache_paths = {
            "HF_HOME": self.model_cache / "huggingface",
            "HF_HUB_CACHE": self.model_cache / "huggingface" / "hub",
            "TRANSFORMERS_CACHE": self.model_cache / "huggingface" / "transformers",
            "TORCH_HOME": self.model_cache / "torch",
            "TRITON_CACHE_DIR": self.model_cache / "triton",
            "TORCHINDUCTOR_CACHE_DIR": self.model_cache / "torchinductor",
            "CUDA_CACHE_PATH": self.model_cache / "cuda",
            "FLASHINFER_WORKSPACE_BASE": self.model_cache / "flashinfer",
            "NUMBA_CACHE_DIR": self.model_cache / "numba",
            "VLLM_CACHE_ROOT": self.model_cache / "vllm",
            "XDG_CACHE_HOME": self.model_cache / "xdg",
            "HF_DATASETS_CACHE": self.dataset_cache / "huggingface",
        }
        for variable, path in cache_paths.items():
            path.mkdir(parents=True, exist_ok=True)
            os.environ[variable] = str(path)

    def run(self, run_id: str) -> "RunLayout":
        base = self.output_root / run_id
        result = RunLayout(
            base=base,
            manifests=base / "manifests",
            raw=base / "raw",
            raw_answer=base / "raw" / "answer",
            raw_confidence=base / "raw" / "confidence",
            parsed=base / "parsed",
            derived=base / "derived",
            statistics=base / "statistics",
            figures=base / "figures",
            logs=self.log_root / run_id,
        )
        result.create()
        return result


@dataclass(frozen=True)
class RunLayout:
    base: Path
    manifests: Path
    raw: Path
    raw_answer: Path
    raw_confidence: Path
    parsed: Path
    derived: Path
    statistics: Path
    figures: Path
    logs: Path

    def create(self) -> None:
        for path in (
            self.manifests,
            self.raw,
            self.raw_answer,
            self.raw_confidence,
            self.parsed,
            self.derived,
            self.statistics,
            self.figures,
            self.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def write_immutable_json(path: Path, payload: Any) -> None:
    """Publish a complete read-only file exactly once without replacing a peer."""
    data = canonical_json_bytes(payload)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o440)
        os.link(temporary, path)
    except FileExistsError:
        raise
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        raise
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload
