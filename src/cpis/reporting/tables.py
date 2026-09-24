"""Lightweight report serialization from saved statistical results only."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def write_summary_table(path: Path, rows: list[Any]) -> None:
    """Write a small table artifact; callers must target the reporting stage."""
    payload = [asdict(row) if is_dataclass(row) else row for row in rows]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

