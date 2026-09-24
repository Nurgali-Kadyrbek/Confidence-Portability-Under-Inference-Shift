#!/usr/bin/env python3
"""Test whether two saved runs that differ only in batch size generated the same text.

Batch size looks like an execution parameter, so it is tempting to treat it as
scientifically inert and vary it for throughput. That is only sound if the saved
generations are byte-identical for matched requests. This compares two completed
or partial runs on the join that actually identifies a generation — dataset item,
answer condition and replicate seed — and reports both the agreement rate and
every field that could otherwise explain a disagreement.

If the generation-relevant fields all match and the text does not, batch
composition has entered the effective stochastic treatment and batch size must
be held fixed wherever generations are compared.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from cpis.matrix_config import load_matrix_config
from cpis.records import GenerationRecord
from cpis.storage import StorageLayout, read_json

# Everything that determines a generation apart from how requests were grouped.
GENERATION_FIELDS = (
    "sampling",
    "model",
    "condition",
    "prompt_text",
    "prompt_sha256",
    "sampling_seed",
    "confidence_protocol_id",
    "prompt_template_version",
    "parser_version",
    "inference_backend",
    "inference_backend_version",
    "engine_seed",
    "generation_stage",
)


def _index(directory: Path) -> dict[tuple[str, str, int], GenerationRecord]:
    records: dict[tuple[str, str, int], GenerationRecord] = {}
    for path in sorted(directory.glob("*.json")):
        record = GenerationRecord.model_validate(read_json(path))
        records[(record.dataset_item_id, record.condition.condition_id, record.seed)] = record
    return records


def _shared_prefix(left: str, right: str) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def compare(a_dir: Path, b_dir: Path) -> dict:
    a, b = _index(a_dir), _index(b_dir)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise RuntimeError("the two runs share no comparable generation")
    identical = [key for key in shared if a[key].raw_text == b[key].raw_text]
    controls = {
        field: sum(
            1 for key in shared if getattr(a[key], field) == getattr(b[key], field)
        )
        for field in GENERATION_FIELDS
    }
    divergent = [key for key in shared if key not in set(identical)]
    prefixes = [
        _shared_prefix(a[key].raw_text, b[key].raw_text) for key in divergent[:200]
    ]
    return {
        "schema_version": "1.0",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "records_a": len(a),
        "records_b": len(b),
        "comparable": len(shared),
        "identical_raw_text": len(identical),
        "identical_fraction": round(len(identical) / len(shared), 4),
        "controls_identical": controls,
        "controls_all_identical": all(v == len(shared) for v in controls.values()),
        "identical_finish_reason": sum(
            1 for key in shared if a[key].finish_reason == b[key].finish_reason
        ),
        "median_identical_prefix_chars": (
            sorted(prefixes)[len(prefixes) // 2] if prefixes else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-a", type=Path, required=True)
    parser.add_argument("--config-b", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--stage", choices=["answer", "confidence"], default="answer")
    parser.add_argument("--destination", type=Path, default=None)
    args = parser.parse_args()

    root = args.repository_root.resolve()
    config_a = load_matrix_config(args.config_a.resolve())
    config_b = load_matrix_config(args.config_b.resolve())
    storage = StorageLayout.from_spec(config_a.storage, root)
    sub = "raw/answer" if args.stage == "answer" else "raw/confidence"
    result = compare(
        storage.output_root / config_a.run_id / sub,
        storage.output_root / config_b.run_id / sub,
    )
    result |= {
        "stage": args.stage,
        "run_a": config_a.run_id,
        "run_b": config_b.run_id,
        "batch_size_a": config_a.execution.batch_size,
        "batch_size_b": config_b.execution.batch_size,
        "config_sha256_a": config_a.config_sha256,
        "config_sha256_b": config_b.config_sha256,
        "batch_invariant": result["identical_fraction"] == 1.0,
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.destination:
        args.destination.parent.mkdir(parents=True, exist_ok=True)
        args.destination.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
