"""Replica placement is scheduling; the replica count is experimental design.

Running replica 0 and replica 1 on two GPUs instead of one is an execution
optimisation and must not change a single generated record. Changing the
replica count is not: it re-shards the items and therefore re-composes the
batches, and generation is measurably not batch-invariant.
"""

from pathlib import Path

from cpis.artifacts import answer_identity
from cpis.datasets import assign_replica, load_dataset, partition_items
from cpis.matrix_config import load_matrix_config
from cpis.matrix_pipeline import _answer_observation_id, _build_observations
from cpis.records import derive_sampling_seed, stable_generation_id
from cpis.storage import StorageLayout

ROOT = Path(__file__).parents[1]
SMOKE = ROOT / "configs/matrix-smoke.yaml"


def _items(config):
    storage = StorageLayout.from_spec(config.storage, ROOT)
    return partition_items(
        load_dataset(config.dataset, SMOKE.parent, storage.dataset_cache), config.dataset
    )


def test_shards_partition_the_items_by_item_id_alone(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CPIS_STORAGE_ROOT", str(tmp_path / "storage"))
    config = load_matrix_config(SMOKE)
    items = _items(config)
    replicas = max(config.execution.replicas, 2)
    shards = {
        r: [i for i in items if assign_replica(i.item_id, replicas) == r]
        for r in range(replicas)
    }
    flat = [i.item_id for shard in shards.values() for i in shard]
    assert sorted(flat) == sorted(i.item_id for i in items)
    assert len(flat) == len(set(flat))
    # placement depends on nothing but the item id
    for r in range(replicas):
        for item in shards[r]:
            assert assign_replica(item.item_id, replicas) == r


def test_record_identity_does_not_depend_on_placement(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CPIS_STORAGE_ROOT", str(tmp_path / "storage"))
    config = load_matrix_config(SMOKE)
    items = _items(config)
    whole, _ = _build_observations(config, items)
    by_key = {
        (o.item.item_id, o.condition.condition_id, o.seed): o for o in whole
    }
    replicas = max(config.execution.replicas, 2)
    for r in range(replicas):
        shard = [i for i in items if assign_replica(i.item_id, replicas) == r]
        sharded, _ = _build_observations(config, shard)
        for o in sharded:
            reference = by_key[(o.item.item_id, o.condition.condition_id, o.seed)]
            assert o.observation_id == reference.observation_id
            assert o.record_id == reference.record_id
            assert o.observation_id == _answer_observation_id(
                config, o.item.item_id, o.condition.condition_id, o.seed
            )
            assert derive_sampling_seed(
                o.seed, o.item.item_id, "answer"
            ) == derive_sampling_seed(reference.seed, reference.item.item_id, "answer")


def test_artifact_identity_ignores_placement_but_pins_batch(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CPIS_STORAGE_ROOT", str(tmp_path / "storage"))
    config = load_matrix_config(SMOKE)
    items = _items(config)
    identity = answer_identity(
        config,
        item_id=items[0].item_id,
        condition=config.inference.answer_conditions[0],
        seed=config.inference.seeds[0],
    )
    assert "replica" not in str(identity).lower()
    assert identity["engine"]["batch_size"] == config.execution.batch_size

    # Changing the replica count re-shards and so is design, not scheduling.
    # The smoke fixture holds too few items to show that, so the sharding claim
    # is made over enough synthetic ids to be meaningful.
    ids = [f"item-{n:05d}" for n in range(2000)]
    two = {i for i in ids if assign_replica(i, 2) == 0}
    four = {i for i in ids if assign_replica(i, 4) == 0}
    assert four < two, "a 4-way shard 0 should be a strict subset of the 2-way shard 0"
    assert len(two) != len(four)

    rescaled = config.model_copy(
        update={"execution": config.execution.model_copy(update={"replicas": 4})}
    )
    assert rescaled.run_id != config.run_id
