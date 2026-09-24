"""Immutable benchmark-specific scoring of saved model outputs."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, JsonValue

from cpis.datasets import DatasetItem, load_dataset, partition_items
from cpis.integration import (
    DatasetIntegrationRecord,
    EvaluatorIntegrationRecord,
    load_integration_record,
    validate_integration_reference,
)
from cpis.manifest import git_metadata, runtime_metadata
from cpis.matrix_config import expected_confidence_cells, load_matrix_config
from cpis.matrix_records import MatrixParsedRecord
from cpis.phase_gate import (
    authorize_phase_config,
    record_dataset_access,
    require_phase_gate,
)
from cpis.records import FrozenRecord
from cpis.storage import StorageLayout, read_json, write_immutable_json


EVALUATOR_RECORDS = {
    "manyifeval_official_v1": "integrations/evaluators/manyifeval-official-v1.yaml",
    "stylembpp_official_v1": "integrations/evaluators/stylembpp-official-v1.yaml",
}


class OfficialScoreRecord(FrozenRecord):
    schema_version: Literal["1.0"] = "1.0"
    score_id: str
    created_at_utc: datetime
    run_id: str
    experiment_id: str
    dataset_id: str
    dataset_revision: str
    dataset_item_id: str
    dataset_group_id: str
    parsed_record_id: str
    parsed_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    answer_raw_record_id: str
    answer_raw_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scorer: Literal["manyifeval_official_v1", "stylembpp_official_v1"]
    evaluator_record_id: str
    evaluator_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_source_revision: str
    answer_parse_valid: bool
    primary_success: bool
    component_results: dict[str, JsonValue]


@dataclass(frozen=True)
class OfficialScoringResult:
    run_id: str
    scorer: str
    expected_records: int
    created: int
    reused: int
    evaluator_calls: int
    successes: int
    output_directory: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_evaluator_record(
    scorer: str, repository_root: Path
) -> EvaluatorIntegrationRecord:
    try:
        relative = EVALUATOR_RECORDS[scorer]
    except KeyError as exc:
        raise ValueError(f"no official evaluator is registered for {scorer}") from exc
    record = load_integration_record(repository_root / relative)
    if not isinstance(record, EvaluatorIntegrationRecord):
        raise TypeError(f"{relative} is not an evaluator integration record")
    if record.scorer != scorer:
        raise RuntimeError("evaluator integration/scorer mismatch")
    return record


def _verify_evaluator(
    record: EvaluatorIntegrationRecord, dataset_cache: Path
) -> Path:
    source_root = dataset_cache / record.source_root_relative_to_dataset_cache
    if not source_root.is_dir():
        raise RuntimeError(f"pinned evaluator source is missing: {source_root}")
    archive = source_root / "source.tar.gz"
    if not archive.is_file() or _sha256(archive) != record.source_archive_sha256:
        raise RuntimeError("official evaluator source archive hash mismatch")
    for artifact in record.artifacts:
        path = source_root / artifact.relative_path
        if not path.is_file() or _sha256(path) != artifact.sha256:
            raise RuntimeError(
                f"official evaluator artifact hash mismatch: {artifact.relative_path}"
            )
    for asset in record.runtime_assets:
        path = dataset_cache / asset.relative_path
        if not path.is_file() or _sha256(path) != asset.sha256:
            raise RuntimeError(
                f"official evaluator runtime asset hash mismatch: {asset.relative_path}"
            )
    runtime = f"{sys.version_info.major}.{sys.version_info.minor}"
    if runtime != record.python_version:
        raise RuntimeError(
            f"official scorer requires Python {record.python_version}, got {runtime}"
        )
    for package, expected in record.package_versions.items():
        try:
            installed = version(package)
        except PackageNotFoundError as exc:
            raise RuntimeError(f"official scorer dependency is missing: {package}") from exc
        if installed != expected:
            raise RuntimeError(
                f"official scorer dependency mismatch: {package}={installed}, expected {expected}"
            )
    return source_root


def _score_manyifeval(
    response: str | None, payload: dict[str, Any], source_root: Path
) -> tuple[bool, dict[str, JsonValue]]:
    if response is None or not response.strip():
        return False, {
            "follow_instruction_list": [False] * len(payload["instruction_id_list"]),
            "instruction_id_list": payload["instruction_id_list"],
            "instruction_success_fraction": 0.0,
        }

    import langdetect
    import nltk

    langdetect.DetectorFactory.seed = 0
    random.seed(1234)
    nltk_root = source_root.parents[2] / "evaluator-assets" / "nltk"
    if str(nltk_root) not in nltk.data.path:
        nltk.data.path.insert(0, str(nltk_root))
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from instruction_following_eval import instructions_registry

    following: list[bool] = []
    instruction_ids = payload["instruction_id_list"]
    kwargs = payload["kwargs"]
    if len(instruction_ids) != len(kwargs):
        raise RuntimeError("ManyIFEval instruction/kwargs length mismatch")
    for index, instruction_id in enumerate(instruction_ids):
        instruction_class = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_class(instruction_id)
        instruction.build_description(**kwargs[index])
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=payload["task_prompt"])
        following.append(bool(instruction.check_following(response)))
    return all(following), {
        "follow_instruction_list": following,
        "instruction_id_list": instruction_ids,
        "instruction_success_fraction": sum(following) / len(following),
    }


def _verify_stylembpp_image(record: EvaluatorIntegrationRecord) -> str:
    sandbox = record.sandbox
    assert sandbox.image_reference is not None
    assert sandbox.image_id_sha256 is not None
    try:
        completed = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                sandbox.image_reference,
                "--format",
                "{{.Id}} {{.Config.User}}",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Docker is unavailable for StyleMBPP scoring") from exc
    if completed.returncode != 0:
        raise RuntimeError(
            "pinned StyleMBPP evaluator image is unavailable: "
            f"{completed.stderr.strip()}"
        )
    parts = completed.stdout.strip().split()
    if len(parts) != 2:
        raise RuntimeError("cannot inspect StyleMBPP evaluator image identity")
    image_id, image_user = parts
    if image_id != f"sha256:{sandbox.image_id_sha256}":
        raise RuntimeError("StyleMBPP evaluator image ID does not match its pin")
    if image_user not in {"65532", "65532:65532"}:
        raise RuntimeError("StyleMBPP evaluator image is not configured as non-root")
    return image_id


def _score_stylembpp(
    response: str | None,
    payload: dict[str, Any],
    evaluator: EvaluatorIntegrationRecord,
    container_suffix: str,
) -> tuple[bool, dict[str, JsonValue]]:
    sandbox = evaluator.sandbox
    assert sandbox.image_reference is not None
    assert sandbox.memory_limit_mb is not None
    assert sandbox.pids_limit is not None
    container_name = f"cpis-stylembpp-{container_suffix[:24]}"
    command = [
        "docker",
        "run",
        "--rm",
        "--interactive",
        "--name",
        container_name,
        "--network",
        "none",
        "--read-only",
        "--user",
        "65532:65532",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--memory",
        f"{sandbox.memory_limit_mb}m",
        "--pids-limit",
        str(sandbox.pids_limit),
        "--cpus",
        "1",
        "--ulimit",
        "nofile=128:128",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777",
        sandbox.image_reference,
    ]
    scorer_input = json.dumps(
        {"response": response or "", "scoring_payload": payload},
        sort_keys=True,
        separators=(",", ":"),
    )
    try:
        completed = subprocess.run(
            command,
            input=scorer_input,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        subprocess.run(
            ["docker", "rm", "--force", container_name],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return False, {"evaluation_status": "outer_container_timeout"}
    except OSError as exc:
        raise RuntimeError("Docker failed to start the StyleMBPP evaluator") from exc
    if completed.returncode in {125, 126, 127}:
        raise RuntimeError(
            "StyleMBPP evaluator infrastructure failure: "
            f"{completed.stderr.strip()}"
        )
    if completed.returncode != 0 and not completed.stdout.strip():
        return False, {
            "evaluation_status": "generated_program_terminated_evaluator",
            "container_exit_code": completed.returncode,
        }
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return False, {
            "evaluation_status": "generated_program_corrupted_evaluator_output",
            "container_exit_code": completed.returncode,
            "stdout_sha256": hashlib.sha256(
                completed.stdout.encode("utf-8")
            ).hexdigest(),
            "stderr_sha256": hashlib.sha256(
                completed.stderr.encode("utf-8")
            ).hexdigest(),
        }
    if (
        not isinstance(result, dict)
        or not isinstance(result.get("primary_success"), bool)
        or not isinstance(result.get("component_results"), dict)
    ):
        raise RuntimeError("StyleMBPP evaluator returned an invalid result schema")
    return result["primary_success"], result["component_results"]


def _score_id(parsed_record_id: str, evaluator_sha256: str) -> str:
    identity = f"cpis-official-score-v1\0{parsed_record_id}\0{evaluator_sha256}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _expected_matrix_cells(config, selected_item_count: int) -> int:
    return expected_confidence_cells(config, selected_item_count)


def score_saved_matrix(
    config_path: Path, repository_root: Path
) -> OfficialScoringResult:
    """Score a complete saved matrix without constructing an inference backend."""
    config_path = config_path.resolve()
    repository_root = repository_root.resolve()
    config = load_matrix_config(config_path)
    scorer = config.dataset.scorer
    if scorer not in EVALUATOR_RECORDS:
        raise ValueError(f"{scorer} does not require an official scoring pass")
    gate = require_phase_gate(config.dataset.partition, repository_root)
    authorize_phase_config(gate, config_path, repository_root)
    storage = StorageLayout.from_spec(config.storage, repository_root)
    run = storage.run(config.run_id)
    record_dataset_access(
        run,
        run_id=config.run_id,
        experiment_id=config.experiment_id,
        config_sha256=config.config_sha256,
        dataset_id=config.dataset.dataset_id,
        dataset_revision=config.dataset.revision,
        partition=config.dataset.partition,
        replica_index=0,
        gate=gate,
    )
    dataset_record = validate_integration_reference(
        config.dataset_integration,
        repository_root,
        "dataset_integration",
    )
    assert isinstance(dataset_record, DatasetIntegrationRecord)
    expected_evaluator_revision = {
        "manyifeval_official_v1": f"{config.dataset.revision}:strict",
        "stylembpp_official_v1": f"{config.dataset.revision}:code_bench",
    }[scorer]
    if dataset_record.evaluator_revision != expected_evaluator_revision:
        raise RuntimeError("dataset/evaluator revision mismatch")
    evaluator = _load_evaluator_record(scorer, repository_root)
    if evaluator.source_revision != config.dataset.revision:
        raise RuntimeError("configuration/evaluator source revision mismatch")
    source_root = _verify_evaluator(evaluator, storage.dataset_cache)
    evaluator_image_id = None
    if evaluator.sandbox.required:
        evaluator_image_id = _verify_stylembpp_image(evaluator)

    all_items = load_dataset(config.dataset, config_path.parent, storage.dataset_cache)
    items = partition_items(all_items, config.dataset)
    by_id: dict[str, DatasetItem] = {item.item_id: item for item in items}
    parsed_paths = sorted(run.parsed.glob("*.json"))
    expected_records = _expected_matrix_cells(config, len(items))
    if len(parsed_paths) != expected_records:
        raise RuntimeError(
            "saved matrix is incomplete for official scoring: "
            f"expected {expected_records} parsed records, got {len(parsed_paths)}"
        )
    output = run.derived / "official-scores"
    output.mkdir(parents=True, exist_ok=True)
    created = reused = successes = 0
    answer_scores: dict[str, tuple[bool, dict[str, JsonValue]]] = {}
    parsed_record_ids: set[str] = set()
    parsed_record_hashes: list[str] = []
    entries: list[
        tuple[Path, MatrixParsedRecord, str, DatasetItem, str, Path]
    ] = []
    existing_scores: dict[str, OfficialScoreRecord] = {}
    for parsed_path in parsed_paths:
        parsed = MatrixParsedRecord.model_validate(read_json(parsed_path))
        if parsed.record_id in parsed_record_ids:
            raise RuntimeError("saved matrix contains a duplicate parsed record ID")
        parsed_record_ids.add(parsed.record_id)
        parsed_sha256 = _sha256(parsed_path)
        parsed_record_hashes.append(parsed_sha256)
        if parsed.run_id != config.run_id or parsed.scorer != scorer:
            raise RuntimeError(f"parsed scoring identity mismatch: {parsed_path}")
        if parsed.scoring_status != "pending_official_evaluator":
            raise RuntimeError(f"official scorer received pre-scored record: {parsed_path}")
        try:
            item = by_id[parsed.dataset_item_id]
        except KeyError as exc:
            raise RuntimeError("parsed record names an unselected dataset item") from exc
        if item.group_id is None or item.scoring_payload is None:
            raise RuntimeError("official-scored dataset item lacks grouping/payload")
        score_id = _score_id(parsed.record_id, evaluator.canonical_sha256)
        destination = output / f"{score_id}.json"
        entries.append((parsed_path, parsed, parsed_sha256, item, score_id, destination))
        if destination.exists():
            existing = OfficialScoreRecord.model_validate(read_json(destination))
            if (
                existing.score_id != score_id
                or existing.run_id != config.run_id
                or existing.dataset_item_id != item.item_id
                or existing.dataset_group_id != item.group_id
                or existing.parsed_record_id != parsed.record_id
                or existing.parsed_record_sha256 != parsed_sha256
                or existing.answer_raw_record_id != parsed.answer_raw_record_id
                or existing.answer_raw_record_sha256
                != parsed.answer_raw_record_sha256
                or existing.scorer != scorer
                or existing.evaluator_record_id != evaluator.record_id
                or existing.evaluator_record_sha256 != evaluator.canonical_sha256
            ):
                raise RuntimeError(f"official score collision at {destination}")
            existing_scores[score_id] = existing
            cached = answer_scores.get(existing.answer_raw_record_id)
            existing_result = (
                existing.primary_success,
                existing.component_results,
            )
            if cached is not None and cached != existing_result:
                raise RuntimeError("official scores disagree for one raw answer")
            answer_scores[existing.answer_raw_record_id] = existing_result

    missing_answers: dict[
        str, tuple[MatrixParsedRecord, DatasetItem, str]
    ] = {}
    for _, parsed, _, item, score_id, destination in entries:
        if destination.exists() or parsed.answer_raw_record_id in answer_scores:
            continue
        missing_answers.setdefault(
            parsed.answer_raw_record_id, (parsed, item, score_id)
        )

    entries_by_answer: dict[
        str, list[tuple[Path, MatrixParsedRecord, str, DatasetItem, str, Path]]
    ] = {}
    for entry in entries:
        entries_by_answer.setdefault(entry[1].answer_raw_record_id, []).append(entry)

    publications_by_score: dict[str, tuple[bool, bool]] = {}

    def publish_score(
        entry: tuple[Path, MatrixParsedRecord, str, DatasetItem, str, Path],
    ) -> tuple[bool, bool]:
        _, parsed, parsed_sha256, item, score_id, destination = entry
        published = publications_by_score.get(score_id)
        if published is not None:
            return published
        existing = existing_scores.get(score_id)
        if existing is not None:
            result = (False, existing.primary_success)
            publications_by_score[score_id] = result
            return result
        success, components = answer_scores[parsed.answer_raw_record_id]
        record = OfficialScoreRecord(
            score_id=score_id,
            created_at_utc=datetime.now(timezone.utc),
            run_id=config.run_id,
            experiment_id=config.experiment_id,
            dataset_id=config.dataset.dataset_id,
            dataset_revision=config.dataset.revision,
            dataset_item_id=item.item_id,
            dataset_group_id=item.group_id,
            parsed_record_id=parsed.record_id,
            parsed_record_sha256=parsed_sha256,
            answer_raw_record_id=parsed.answer_raw_record_id,
            answer_raw_record_sha256=parsed.answer_raw_record_sha256,
            scorer=scorer,
            evaluator_record_id=evaluator.record_id,
            evaluator_record_sha256=evaluator.canonical_sha256,
            evaluator_source_revision=evaluator.source_revision,
            answer_parse_valid=parsed.answer_parse_status == "valid",
            primary_success=success,
            component_results=components,
        )
        write_immutable_json(destination, record.model_dump(mode="json"))
        result = (True, record.primary_success)
        publications_by_score[score_id] = result
        return result

    scoring_workers = 1
    if scorer == "stylembpp_official_v1":
        raw_workers = os.environ.get("CPIS_STYLEMBPP_SCORING_WORKERS", "8")
        try:
            scoring_workers = int(raw_workers)
        except ValueError as exc:
            raise ValueError("CPIS_STYLEMBPP_SCORING_WORKERS must be an integer") from exc
        if not 1 <= scoring_workers <= 64:
            raise ValueError("CPIS_STYLEMBPP_SCORING_WORKERS must lie in [1, 64]")

    def evaluate_missing(
        entry: tuple[str, tuple[MatrixParsedRecord, DatasetItem, str]],
    ) -> tuple[str, tuple[bool, dict[str, JsonValue]]]:
        answer_id, (parsed, item, score_id) = entry
        assert item.scoring_payload is not None
        if scorer == "manyifeval_official_v1":
            result = _score_manyifeval(
                parsed.parsed_answer, item.scoring_payload, source_root
            )
        elif scorer == "stylembpp_official_v1":
            result = _score_stylembpp(
                parsed.parsed_answer,
                item.scoring_payload,
                evaluator,
                score_id,
            )
        else:
            raise AssertionError(f"unhandled official scorer: {scorer}")
        return answer_id, result

    def retain_evaluation(
        evaluated: tuple[str, tuple[bool, dict[str, JsonValue]]],
    ) -> None:
        answer_id, result = evaluated
        answer_scores[answer_id] = result
        # Publish every parsed/readout fan-out immediately. Existing immutable
        # score records are the resume cache, so an interrupted executable
        # benchmark never needs to repeat a completed evaluator call.
        for entry in entries_by_answer[answer_id]:
            publish_score(entry)

    pending = sorted(missing_answers.items())
    if scorer == "stylembpp_official_v1" and scoring_workers > 1:
        with ThreadPoolExecutor(max_workers=scoring_workers) as executor:
            for evaluated in executor.map(evaluate_missing, pending):
                retain_evaluation(evaluated)
    else:
        for entry in pending:
            retain_evaluation(evaluate_missing(entry))
    evaluator_calls = len(pending)

    raw_write_workers = os.environ.get("CPIS_OFFICIAL_SCORE_WRITE_WORKERS", "8")
    try:
        score_write_workers = int(raw_write_workers)
    except ValueError as exc:
        raise ValueError(
            "CPIS_OFFICIAL_SCORE_WRITE_WORKERS must be an integer"
        ) from exc
    if not 1 <= score_write_workers <= 64:
        raise ValueError("CPIS_OFFICIAL_SCORE_WRITE_WORKERS must lie in [1, 64]")

    if score_write_workers > 1:
        with ThreadPoolExecutor(max_workers=score_write_workers) as executor:
            publications = list(executor.map(publish_score, entries))
    else:
        publications = [publish_score(entry) for entry in entries]
    created = sum(was_created for was_created, _ in publications)
    reused = len(publications) - created
    successes = sum(success for _, success in publications)

    manifest_path = run.manifests / "official-scoring.json"
    manifest = {
        "schema_version": "1.0",
        "run_id": config.run_id,
        "scorer": scorer,
        "evaluator_record_id": evaluator.record_id,
        "evaluator_record_sha256": evaluator.canonical_sha256,
        "evaluator_source_revision": evaluator.source_revision,
        "scored_records": len(parsed_paths),
        "unique_answer_records": len(answer_scores),
        "parsed_record_set_sha256": hashlib.sha256(
            "\n".join(sorted(parsed_record_hashes)).encode("ascii")
        ).hexdigest(),
        "git": git_metadata(repository_root),
        "runtime": runtime_metadata(),
        "package_versions": evaluator.package_versions,
        "sandbox": evaluator.sandbox.model_dump(mode="json"),
        "evaluator_image_id": evaluator_image_id,
        "scoring_workers": scoring_workers,
        "score_write_workers": score_write_workers,
        "incremental_score_publication": True,
    }
    if manifest_path.exists():
        existing = read_json(manifest_path)
        comparable = dict(manifest)
        comparable["git"] = existing.get("git")
        comparable["runtime"] = existing.get("runtime")
        if "scoring_workers" not in existing:
            comparable.pop("scoring_workers")
        if "score_write_workers" not in existing:
            comparable.pop("score_write_workers")
        if "incremental_score_publication" not in existing:
            comparable.pop("incremental_score_publication")
        if existing != comparable:
            raise RuntimeError(f"official scoring manifest collision at {manifest_path}")
    else:
        write_immutable_json(manifest_path, manifest)
    return OfficialScoringResult(
        run_id=config.run_id,
        scorer=scorer,
        expected_records=expected_records,
        created=created,
        reused=reused,
        evaluator_calls=evaluator_calls,
        successes=successes,
        output_directory=output,
    )
