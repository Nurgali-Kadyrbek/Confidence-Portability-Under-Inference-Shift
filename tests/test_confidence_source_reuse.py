"""A confidence-only run must reuse answers, and refuse the wrong ones."""


import stat
from pathlib import Path

import pytest
import yaml

from cpis.confidence_pipeline import (
    answer_source_provenance,
    load_answer_source,
    resolve_source_answers,
    verify_answer_identity,
)
from cpis.datasets import load_dataset, partition_items
from cpis.matrix_config import AnswerSourceReference, load_matrix_config
from cpis.matrix_pipeline import run_matrix_experiment
from cpis.storage import StorageLayout

ROOT = Path(__file__).parents[1]
SMOKE = ROOT / "configs/matrix-smoke.yaml"


def _write(path: Path, config) -> Path:
    path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    return path


def _source_reference(source) -> AnswerSourceReference:
    return AnswerSourceReference(
        run_id=source.run_id,
        config_path="configs/matrix-smoke.yaml",
        config_sha256=source.config_sha256,
    )


def _consumer(source, tmp_path: Path, **overrides):
    """A config differing from the source only in confidence-side fields."""
    readouts = tuple(
        r.model_copy(update={"sampling": r.sampling.model_copy(update={"max_tokens": 32})})
        for r in source.inference.confidence_readouts
    )
    config = source.model_copy(
        update={
            "experiment_id": "cpis-matrix-smoke-confidence",
            "answer_source": _source_reference(source),
            "inference": source.inference.model_copy(
                update={"confidence_readouts": readouts}
            ),
            **overrides,
        }
    )
    return load_matrix_config(_write(tmp_path / "consumer.yaml", config))


def _items(config):
    storage = StorageLayout.from_spec(config.storage, ROOT)
    return partition_items(
        load_dataset(config.dataset, SMOKE.parent, storage.dataset_cache), config.dataset
    )


def test_confidence_run_consumes_existing_answers_read_only(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("CPIS_STORAGE_ROOT", str(tmp_path / "storage"))
    produced = run_matrix_experiment(SMOKE, ROOT, answers_only=True)
    source = load_matrix_config(SMOKE)
    consumer = _consumer(source, tmp_path)

    # A readout-budget change must not move the run's answer identity.
    assert consumer.run_id != source.run_id
    resolved = resolve_source_answers(consumer, source, _items(consumer), ROOT)
    assert len(resolved) == produced.expected_answer_observations

    before = {
        path: (path.read_bytes(), path.stat().st_mode)
        for path in (produced.run_directory / "raw" / "answer").glob("*.json")
    }
    provenance = answer_source_provenance(source, resolved)
    assert provenance["source_answer_run_id"] == source.run_id
    assert provenance["source_answer_count"] == len(resolved)
    assert provenance["consumed_read_only"] is True

    # The source run is untouched, and its records remain read-only on disk.
    after = {
        path: (path.read_bytes(), path.stat().st_mode)
        for path in (produced.run_directory / "raw" / "answer").glob("*.json")
    }
    assert after == before
    for path, (_, mode) in after.items():
        assert not mode & stat.S_IWUSR, path


def test_answer_affecting_change_refuses_reuse(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CPIS_STORAGE_ROOT", str(tmp_path / "storage"))
    run_matrix_experiment(SMOKE, ROOT, answers_only=True)
    source = load_matrix_config(SMOKE)
    item = _items(source)[0].item_id
    condition = source.inference.answer_conditions[0].condition_id
    seed = source.inference.seeds[0]

    # Confidence-side change: reuse is allowed.
    consumer = _consumer(source, tmp_path)
    assert verify_answer_identity(
        consumer, source, item_id=item, condition_id=condition, seed=seed
    )

    # Answer-side changes: reuse must be refused, naming the difference.
    answer_side = source.inference.answer_conditions[0]
    retuned = source.model_copy(
        update={
            "inference": source.inference.model_copy(
                update={
                    "answer_conditions": (
                        answer_side.model_copy(
                            update={
                                "sampling": answer_side.sampling.model_copy(
                                    update={"max_tokens": 7}
                                )
                            }
                        ),
                    )
                    + source.inference.answer_conditions[1:]
                }
            )
        }
    )
    hostile = _consumer(retuned, tmp_path, answer_source=_source_reference(source))
    with pytest.raises(RuntimeError, match="different answer-affecting inputs"):
        verify_answer_identity(
            hostile, source, item_id=item, condition_id=condition, seed=seed
        )

    relabelled = source.model_copy(
        update={"model": source.model.model_copy(update={"model_mode": "instruct"})}
    )
    hostile = _consumer(relabelled, tmp_path, answer_source=_source_reference(source))
    with pytest.raises(RuntimeError, match="different answer-affecting inputs"):
        verify_answer_identity(
            hostile, source, item_id=item, condition_id=condition, seed=seed
        )


def test_answer_source_reference_must_match_the_run_that_produced_it(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("CPIS_STORAGE_ROOT", str(tmp_path / "storage"))
    source = load_matrix_config(SMOKE)

    consumer = _consumer(source, tmp_path)
    assert load_answer_source(consumer, ROOT).run_id == source.run_id

    for update, message in (
        ({"config_sha256": "0" * 64}, "hash does not match"),
        ({"run_id": "not-the-run-that-ran"}, "run ID does not match"),
    ):
        broken = consumer.model_copy(
            update={"answer_source": consumer.answer_source.model_copy(update=update)}
        )
        with pytest.raises(RuntimeError, match=message):
            load_answer_source(broken, ROOT)

    with pytest.raises(ValueError, match="must cite an answer source"):
        load_answer_source(source, ROOT)


def test_missing_source_answer_is_refused_not_regenerated(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("CPIS_STORAGE_ROOT", str(tmp_path / "storage"))
    produced = run_matrix_experiment(SMOKE, ROOT, answers_only=True)
    source = load_matrix_config(SMOKE)
    consumer = _consumer(source, tmp_path)

    victim = next((produced.run_directory / "raw" / "answer").glob("*.json"))
    victim.chmod(0o640)
    victim.unlink()
    with pytest.raises(RuntimeError, match="missing a generation"):
        resolve_source_answers(consumer, source, _items(consumer), ROOT)
