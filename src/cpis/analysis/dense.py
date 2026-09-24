"""Model-specific dense decoder-response surfaces from saved matrix outputs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cpis.analysis.confirmatory import _expected_cells
from cpis.analysis.dense_surface import DenseSurfaceRow, bootstrap_dense_surface
from cpis.analysis.matrix import (
    _load_replica_manifests,
    _resampling_design,
    _resolve_official_scores,
)
from cpis.analysis.metrics import confidence_accepts, has_valid_confidence
from cpis.analysis_plan import load_analysis_plan
from cpis.datasets import load_dataset, partition_items
from cpis.dense_plan import DenseGridPlan, load_dense_grid_plan
from cpis.dense_qualification import (
    check_dense_qualification_covers,
    validate_dense_qualification_reference,
)
from cpis.manifest import git_metadata
from cpis.matrix_config import load_matrix_config
from cpis.matrix_records import MatrixParsedRecord
from cpis.policy import (
    validate_certification_decision_reference,
    validate_policy_reference,
)
from cpis.phase_gate import (
    PhaseGate,
    authorize_phase_analysis_plan,
    authorize_phase_config,
    record_dataset_access,
    require_phase_gate,
)
from cpis.storage import StorageLayout, read_json, write_immutable_json
from cpis.study_plan import StudyPlan, load_study_plan


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DenseCoordinateDesign:
    model_condition_id: str
    dataset_panel_id: str
    coordinates: dict[str, tuple[float, float]]
    planned_coordinates: tuple[tuple[float, float], ...]
    unsupported_coordinates: tuple[tuple[float, float], ...]
    reference_temperature: float
    reference_top_p: float
    qualification: dict | None


def _coordinate_design(
    config,
    study: StudyPlan,
    dense: DenseGridPlan,
    repository_root: Path,
    gate: PhaseGate | None,
) -> DenseCoordinateDesign:
    model_matches = [
        entry
        for entry in study.model_conditions
        if entry.integration == config.model_integration
    ]
    if len(model_matches) != 1:
        raise RuntimeError("dense matrix does not identify one frozen model condition")
    model_condition = model_matches[0]
    if model_condition.condition_id not in study.dense_model_conditions:
        raise RuntimeError("matrix model is not declared in the dense study")
    dense_model_matches = [
        entry
        for entry in dense.model_conditions
        if entry.model_condition_id == model_condition.condition_id
    ]
    if len(dense_model_matches) != 1:
        raise RuntimeError("dense-grid plan does not identify the matrix model")
    dense_model = dense_model_matches[0]

    dataset_matches = [
        entry
        for entry in study.core_datasets
        if entry.integration == config.dataset_integration
    ]
    if len(dataset_matches) != 1:
        raise RuntimeError("dense matrix does not identify one frozen dataset panel")
    dataset_panel = dataset_matches[0]
    if (
        dataset_panel.panel_id not in study.dense_dataset_panels
        or dataset_panel.panel_id not in dense.dataset_panels
    ):
        raise RuntimeError("matrix dataset is not declared in the dense study")

    if tuple(study.dense_model_conditions) != tuple(
        entry.model_condition_id for entry in dense.model_conditions
    ):
        raise RuntimeError("study and dense-grid model declarations differ")
    if set(study.dense_dataset_panels) != set(dense.dataset_panels):
        raise RuntimeError("study and dense-grid dataset declarations differ")

    design = config.inference.design
    readouts = {
        readout.readout_id: readout for readout in config.inference.confidence_readouts
    }
    reference_condition = next(
        condition
        for condition in config.inference.answer_conditions
        if condition.condition_id == design.reference_answer_condition_id
    )
    reference_readout = readouts[design.reference_confidence_readout_id]
    ignored_coordinates = {"temperature", "top_p"}
    answer_controls = {
        key: value
        for key, value in reference_condition.sampling.model_dump(mode="json").items()
        if key not in ignored_coordinates
    }
    readout_controls = {
        key: value
        for key, value in reference_readout.sampling.model_dump(mode="json").items()
        if key not in ignored_coordinates
    }
    coordinates: dict[str, tuple[float, float]] = {}
    for condition in config.inference.answer_conditions:
        temperature = condition.sampling.temperature
        top_p = condition.sampling.top_p
        coordinate = (temperature, top_p)
        if coordinate in coordinates.values():
            raise RuntimeError("dense matrix has duplicate temperature/top-p coordinates")
        if condition.sampling.top_k != dense_model.held_top_k:
            raise RuntimeError("dense matrix changed the model-native top-k setting")
        controls = {
            key: value
            for key, value in condition.sampling.model_dump(mode="json").items()
            if key not in ignored_coordinates
        }
        if controls != answer_controls:
            raise RuntimeError(
                "dense answer conditions may change only temperature and top-p"
            )
        coupled_id = design.coupled_readout_by_answer[condition.condition_id]
        coupled = readouts[coupled_id]
        if (
            coupled.sampling.temperature != temperature
            or coupled.sampling.top_p != top_p
        ):
            raise RuntimeError("dense coupled readout does not match its answer coordinate")
        coupled_controls = {
            key: value
            for key, value in coupled.sampling.model_dump(mode="json").items()
            if key not in ignored_coordinates
        }
        if (
            coupled_controls != readout_controls
            or coupled.chat_template_kwargs != reference_readout.chat_template_kwargs
            or coupled.model_mode != reference_readout.model_mode
        ):
            raise RuntimeError(
                "dense coupled readouts may change only temperature and top-p"
            )
        coordinates[condition.condition_id] = coordinate
    planned = {
        (temperature, top_p)
        for temperature in dense.temperatures
        for top_p in dense.top_ps
    }
    observed = set(coordinates.values())
    if not observed <= planned:
        raise RuntimeError("dense matrix contains a coordinate outside the frozen grid")
    if config.dense_qualification is None:
        if observed != planned:
            raise RuntimeError(
                "a dense matrix that omits planned coordinates must cite the frozen "
                "outcome-blind qualification that declared them unsupported"
            )
        qualification = None
    else:
        qualification = check_dense_qualification_covers(
            validate_dense_qualification_reference(
                config.dense_qualification, repository_root, gate
            ),
            dense=dense,
            study=study,
            model_condition_id=model_condition.condition_id,
            planned=planned,
            observed=observed,
        )
    temperatures = {coordinate[0] for coordinate in coordinates.values()}
    top_ps = {coordinate[1] for coordinate in coordinates.values()}
    if len(observed) < 6 or len(temperatures) < 3 or len(top_ps) < 3:
        raise RuntimeError(
            "dense matrix requires at least six supported cells spanning three "
            "temperature and three top-p levels"
        )
    reference_id = design.reference_answer_condition_id
    reference_temperature, reference_top_p = coordinates[reference_id]
    if (
        reference_temperature != dense_model.reference_temperature
        or reference_top_p != dense_model.reference_top_p
    ):
        raise RuntimeError("dense matrix reference differs from the model-native plan")
    return DenseCoordinateDesign(
        model_condition_id=model_condition.condition_id,
        dataset_panel_id=dataset_panel.panel_id,
        coordinates=coordinates,
        planned_coordinates=tuple(sorted(planned)),
        unsupported_coordinates=tuple(sorted(planned - observed)),
        reference_temperature=reference_temperature,
        reference_top_p=reference_top_p,
        qualification=qualification,
    )


def _rows(
    records: list[MatrixParsedRecord],
    coordinates: dict[str, tuple[float, float]],
    cluster_by_item: dict[str, str],
    outcome,
) -> list[DenseSurfaceRow]:
    result = []
    for record in records:
        value = outcome(record)
        if value is None:
            continue
        temperature, top_p = coordinates[record.answer_condition_id]
        result.append(
            DenseSurfaceRow(
                outcome=float(value),
                temperature=temperature,
                top_p=top_p,
                cluster_id=cluster_by_item[record.dataset_item_id],
                categories={},
            )
        )
    return result


def _surface(
    records: list[MatrixParsedRecord],
    *,
    coordinates: dict[str, tuple[float, float]],
    cluster_by_item: dict[str, str],
    outcome,
    reference_temperature: float,
    reference_top_p: float,
    replicates: int,
    seed: int,
    top_p_logit_epsilon: float,
) -> dict:
    rows = _rows(records, coordinates, cluster_by_item, outcome)
    if len(rows) <= 6 or len({row.cluster_id for row in rows}) < 2:
        return {
            "status": "insufficient_observations",
            "observations": len(rows),
            "clusters": len({row.cluster_id for row in rows}),
        }
    return bootstrap_dense_surface(
        rows,
        reference_temperature=reference_temperature,
        reference_top_p=reference_top_p,
        replicates=replicates,
        seed=seed,
        top_p_logit_epsilon=top_p_logit_epsilon,
    )


def _operational_threshold(
    config,
    role: str,
    repository_root: Path,
    analysis_plan_sha256: str,
) -> dict | None:
    if config.dataset.partition != "test":
        return None
    assert config.frozen_policy is not None
    assert config.certification_decision is not None
    policy = validate_policy_reference(config.frozen_policy, repository_root)
    decision = validate_certification_decision_reference(
        config.certification_decision, repository_root
    )
    if (
        policy.analysis_plan_sha256 != analysis_plan_sha256
        or decision.analysis_plan_sha256 != analysis_plan_sha256
        or decision.frozen_policy_sha256 != policy.canonical_sha256
    ):
        raise RuntimeError("dense test artifacts do not share one frozen analysis plan")
    policy_readout = getattr(policy, role)
    decision_readout = getattr(decision, role)
    if decision_readout.status == "certified":
        return {
            "endpoint": "certified_risk_contract",
            "threshold": decision_readout.threshold,
        }
    fixed = next(
        entry
        for entry in policy_readout.fixed_coverage
        if entry.target_coverage == 0.5
    )
    if fixed.threshold is None:
        return None
    return {"endpoint": "fixed_coverage_0.50", "threshold": fixed.threshold}


def analyze_dense_matrix(
    config_path: Path, analysis_plan_path: Path, repository_root: Path
) -> tuple[Path, dict]:
    """Fit prespecified model-specific surfaces without invoking a model."""

    repository_root = repository_root.resolve()
    config_path = config_path.resolve()
    config = load_matrix_config(config_path)
    if config.study not in {"development_matrix", "factorial"}:
        raise ValueError("dense analysis requires a development or factorial matrix")
    if config.study == "factorial" and config.dataset.partition != "test":
        raise ValueError("confirmatory factorial surfaces require the test partition")
    gate = require_phase_gate(config.dataset.partition, repository_root)
    authorize_phase_config(gate, config_path, repository_root)
    plan = load_analysis_plan(analysis_plan_path.resolve())
    authorize_phase_analysis_plan(gate, plan.sha256)
    if plan.schema_version not in {"3.0", "4.0"} or plan.dense_surface is None:
        raise ValueError("dense analysis requires analysis plan v3+")
    study = load_study_plan(
        repository_root / "configs/registry/primary-conditions-v1.yaml"
    )
    dense = load_dense_grid_plan(
        repository_root / "configs/registry/dense-grid-v1.yaml"
    )
    coordinate_design = _coordinate_design(config, study, dense, repository_root, gate)
    coordinates = coordinate_design.coordinates
    reference_temperature = coordinate_design.reference_temperature
    reference_top_p = coordinate_design.reference_top_p
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
    manifests = _load_replica_manifests(run, config, repository_root)
    paths = sorted(run.parsed.glob("*.json"))
    records = [MatrixParsedRecord.model_validate(read_json(path)) for path in paths]
    records = _resolve_official_scores(run, records, paths)
    if any(record.correct is None for record in records):
        raise RuntimeError("dense analysis requires completed deterministic scoring")
    all_items = load_dataset(config.dataset, config_path.parent, storage.dataset_cache)
    items = partition_items(all_items, config.dataset)
    expected = _expected_cells(config, len(items))
    if len(records) != expected:
        raise RuntimeError(f"dense matrix is incomplete: expected {expected}, got {len(records)}")
    cluster_by_item, _, _ = _resampling_design(items, config.dataset.dataset_id)
    design = config.inference.design
    coupled = [record for record in records if "coupled" in record.design_roles]
    standardized = [
        record for record in records if "standardized" in record.design_roles
    ]
    if len(coupled) != len(config.inference.answer_conditions) * len(items) * len(
        config.inference.seeds
    ):
        raise RuntimeError("dense matrix does not have exactly one coupled row per answer")

    common = {
        "coordinates": coordinates,
        "cluster_by_item": cluster_by_item,
        "reference_temperature": reference_temperature,
        "reference_top_p": reference_top_p,
        "replicates": plan.bootstrap.replicates,
        "top_p_logit_epsilon": plan.dense_surface.top_p_logit_epsilon,
    }
    surfaces: dict[str, dict] = {
        "answer_correctness": _surface(
            coupled,
            outcome=lambda record: float(bool(record.correct)),
            seed=plan.bootstrap.seed + 100_000,
            **common,
        )
    }
    for role_index, (role, role_records) in enumerate(
        (("coupled", coupled), ("standardized_deterministic", standardized))
    ):
        offset = 110_000 + 10_000 * role_index
        role_surfaces = {
            "confidence_valid_only": _surface(
                role_records,
                outcome=lambda record: (
                    record.confidence if has_valid_confidence(record) else None
                ),
                seed=plan.bootstrap.seed + offset,
                **common,
            ),
            "invalid_confidence": _surface(
                role_records,
                outcome=lambda record: float(not has_valid_confidence(record)),
                seed=plan.bootstrap.seed + offset + 1_000,
                **common,
            ),
        }
        threshold = _operational_threshold(
            config, role, repository_root, plan.sha256
        )
        if threshold is not None:
            numeric = float(threshold["threshold"])
            role_surfaces["operational_policy"] = {
                **threshold,
                "coverage": _surface(
                    role_records,
                    outcome=lambda record, value=numeric: float(
                        confidence_accepts(record, value)
                    ),
                    seed=plan.bootstrap.seed + offset + 2_000,
                    **common,
                ),
                "selective_error_valid_accepted_only": _surface(
                    role_records,
                    outcome=lambda record, value=numeric: (
                        float(not bool(record.correct))
                        if confidence_accepts(record, value)
                        else None
                    ),
                    seed=plan.bootstrap.seed + offset + 3_000,
                    **common,
                ),
            }
        surfaces[role] = role_surfaces

    result = {
        "schema_version": "1.0",
        "analysis_id": f"{config.experiment_id}-dense-surface-analysis-v1",
        "confirmatory": config.confirmatory,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": config.run_id,
        "generation_git_commit": manifests[0]["git"]["commit"],
        "analysis_git": git_metadata(repository_root),
        "config_sha256": config.config_sha256,
        "analysis_plan_id": plan.plan_id,
        "analysis_plan_sha256": plan.sha256,
        "study_plan_id": study.plan_id,
        "study_plan_sha256": study.canonical_sha256,
        "dense_grid_id": dense.grid_id,
        "dense_grid_sha256": dense.canonical_sha256,
        "model_condition_id": coordinate_design.model_condition_id,
        "dataset_panel_id": coordinate_design.dataset_panel_id,
        "model_id": config.model.model_id,
        "model_mode": config.model.model_mode,
        "dataset_id": config.dataset.dataset_id,
        "dataset_partition": config.dataset.partition,
        "experimental_units": len(set(cluster_by_item.values())),
        "items": len(items),
        "replicate_seeds": list(config.inference.seeds),
        "planned_coordinates": [
            {"temperature": temperature, "top_p": top_p}
            for temperature, top_p in coordinate_design.planned_coordinates
        ],
        "supported_coordinates": [
            {
                "condition_id": condition_id,
                "temperature": temperature,
                "top_p": top_p,
            }
            for condition_id, (temperature, top_p) in sorted(coordinates.items())
        ],
        "unsupported_coordinates": [
            {"temperature": temperature, "top_p": top_p}
            for temperature, top_p in coordinate_design.unsupported_coordinates
        ],
        "unsupported_cell_rule": dense.unsupported_cell_rule,
        "qualification_rule": dense.qualification.unsupported_cell_rule,
        "dense_qualification": coordinate_design.qualification,
        "temperature_coordinates": sorted({value[0] for value in coordinates.values()}),
        "top_p_coordinates": sorted({value[1] for value in coordinates.values()}),
        "reference_coordinate": {
            "condition_id": design.reference_answer_condition_id,
            "temperature": reference_temperature,
            "top_p": reference_top_p,
        },
        "surfaces": surfaces,
    }
    output = run.statistics / "dense-surface-analysis-v1.json"
    if output.exists():
        existing = read_json(output)
        comparable = dict(result)
        comparable["created_at_utc"] = existing.get("created_at_utc")
        comparable["analysis_git"] = existing.get("analysis_git")
        if existing != comparable:
            raise RuntimeError(f"dense analysis collision at {output}")
        return output, existing
    write_immutable_json(output, result)
    return output, result
