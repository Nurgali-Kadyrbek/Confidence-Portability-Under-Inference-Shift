"""The generator and every expectation of it must agree on the cell count.

The rule lived in three copies. Changing the design reached the generator and
left the official scorer expecting 256 cells where 96 were produced, which
surfaced only when a real matrix was scored.
"""

from pathlib import Path

import pytest

from cpis.analysis.confirmatory import _expected_cells
from cpis.datasets import load_dataset, partition_items
from cpis.matrix_config import expected_confidence_cells, load_matrix_config
from cpis.matrix_pipeline import _build_observations
from cpis.official_scoring import _expected_matrix_cells
from cpis.storage import StorageLayout

ROOT = Path(__file__).parents[1]
SMOKE = ROOT / "configs/matrix-smoke.yaml"


def _items(config):
    storage = StorageLayout.from_spec(config.storage, ROOT)
    return partition_items(
        load_dataset(config.dataset, SMOKE.parent, storage.dataset_cache), config.dataset
    )


@pytest.mark.parametrize("mode", ["coupled_standardized_reference", "coupled_only"])
def test_every_consumer_agrees_with_the_generator(mode, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CPIS_STORAGE_ROOT", str(tmp_path / "storage"))
    base = load_matrix_config(SMOKE)
    config = base.model_copy(
        update={
            "inference": base.inference.model_copy(
                update={
                    "design": base.inference.design.model_copy(
                        update={"confidence_cells": mode}
                    )
                }
            )
        }
    )
    items = _items(config)
    _, cells = _build_observations(config, items)

    generated = len(cells)
    assert expected_confidence_cells(config, len(items)) == generated
    assert _expected_cells(config, len(items)) == generated
    assert _expected_matrix_cells(config, len(items)) == generated
    if mode == "coupled_only":
        answers, _ = _build_observations(config, items)
        assert generated == len(answers)
