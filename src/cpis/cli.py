"""Command-line entry points for validation, schema export, and execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cpis.analysis_plan import AnalysisPlan, load_analysis_plan
from cpis.agent_config import AgentExperimentConfig, load_agent_config
from cpis.agent_audit import audit_agent
from cpis.agent_pipeline import (
    derive_agent_records,
    run_agent_experiment,
    validate_agent_integrations,
)
from cpis.agent_records import AgentEpisodeRecord, AgentParsedRecord
from cpis.analysis.matrix import analyze_matrix
from cpis.analysis.certification import analyze_certification
from cpis.analysis.agent import (
    analyze_agent_certification,
    analyze_agent_development,
    analyze_agent_test,
)
from cpis.analysis.confirmatory import analyze_confirmatory_test
from cpis.analysis.dense import analyze_dense_matrix
from cpis.analysis.panel import analyze_confirmatory_panel
from cpis.config import ExperimentConfig, load_config
from cpis.dense_plan import DenseGridPlan, load_dense_grid_plan
from cpis.dense_qualification import (
    DenseQualificationRecord,
    qualify_dense_preflight,
)
from cpis.integration import (
    DatasetIntegrationRecord,
    EvaluatorIntegrationRecord,
    IntegrationReference,
    ModelIntegrationRecord,
    load_integration_record,
)
from cpis.manifest import Manifest
from cpis.matrix_audit import audit_matrix
from cpis.matrix_config import MatrixExperimentConfig, load_matrix_config
from cpis.matrix_pipeline import run_matrix_experiment, validate_matrix_integrations
from cpis.matrix_records import MatrixParsedRecord
from cpis.official_scoring import OfficialScoreRecord, score_saved_matrix
from cpis.panel_config import PanelAnalysisConfig
from cpis.phase_configs import (
    derive_agent_certification_config,
    derive_agent_test_config,
    derive_certification_config,
    derive_test_config,
)
from cpis.pipeline import run_experiment
from cpis.phase_gate import PhaseGate
from cpis.policy import (
    CertificationDecision,
    FrozenPolicy,
    freeze_development_policy,
    freeze_agent_development_policy,
    resolve_agent_certification_decision,
    resolve_certification_decision,
)
from cpis.records import GenerationRecord, ParsedRecord
from cpis.reporting.matrix import write_matrix_evidence
from cpis.reporting.agent import write_agent_evidence
from cpis.reporting.final import write_panel_evidence
from cpis.study_plan import StudyPlan, load_study_plan


def export_schemas(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    models = {
        "experiment-config": ExperimentConfig,
        "experiment-manifest": Manifest,
        "generation-record": GenerationRecord,
        "parsed-record": ParsedRecord,
        "analysis-plan": AnalysisPlan,
        "matrix-experiment-config": MatrixExperimentConfig,
        "matrix-parsed-record": MatrixParsedRecord,
        "agent-experiment-config": AgentExperimentConfig,
        "agent-episode-record": AgentEpisodeRecord,
        "agent-parsed-record": AgentParsedRecord,
        "model-integration-record": ModelIntegrationRecord,
        "dataset-integration-record": DatasetIntegrationRecord,
        "evaluator-integration-record": EvaluatorIntegrationRecord,
        "official-score-record": OfficialScoreRecord,
        "phase-gate": PhaseGate,
        "frozen-policy": FrozenPolicy,
        "certification-decision": CertificationDecision,
        "study-plan": StudyPlan,
        "panel-analysis-config": PanelAnalysisConfig,
        "dense-grid-plan": DenseGridPlan,
        "dense-qualification-record": DenseQualificationRecord,
    }
    for name, model in models.items():
        path = destination / f"{name}.schema.json"
        path.write_text(
            json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cpis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate an experiment YAML")
    validate.add_argument("--config", type=Path, required=True)

    validate_analysis = subparsers.add_parser(
        "validate-analysis", help="validate a prespecified analysis-plan YAML"
    )
    validate_analysis.add_argument("--config", type=Path, required=True)

    validate_matrix = subparsers.add_parser(
        "validate-matrix", help="validate a development matrix YAML"
    )
    validate_matrix.add_argument("--config", type=Path, required=True)
    validate_matrix.add_argument(
        "--repository-root", type=Path, default=Path.cwd()
    )

    validate_agent = subparsers.add_parser(
        "validate-agent", help="validate a BFCL agent experiment YAML"
    )
    validate_agent.add_argument("--config", type=Path, required=True)
    validate_agent.add_argument(
        "--repository-root", type=Path, default=Path.cwd()
    )
    validate_agent.add_argument("--preflight", action="store_true")

    validate_integration = subparsers.add_parser(
        "validate-integration", help="validate a model or dataset integration record"
    )
    validate_integration.add_argument("--record", type=Path, required=True)

    validate_study = subparsers.add_parser(
        "validate-study-plan", help="validate the frozen model/dataset panel"
    )
    validate_study.add_argument("--config", type=Path, required=True)

    validate_dense_plan = subparsers.add_parser(
        "validate-dense-plan", help="validate the common-coordinate dense grid"
    )
    validate_dense_plan.add_argument("--config", type=Path, required=True)

    qualify_dense = subparsers.add_parser(
        "qualify-dense", help="classify dense-grid cells from a completed preflight"
    )
    qualify_dense.add_argument("--config", type=Path, required=True)
    qualify_dense.add_argument("--dense-plan", type=Path, required=True)
    qualify_dense.add_argument("--study-plan", type=Path, required=True)
    qualify_dense.add_argument(
        "--repository-root", type=Path, default=Path.cwd()
    )
    qualify_dense.add_argument("--destination", type=Path, required=True)

    analyze = subparsers.add_parser(
        "analyze-matrix", help="analyze saved matrix outputs without model access"
    )
    analyze.add_argument("--config", type=Path, required=True)
    analyze.add_argument("--analysis-plan", type=Path, required=True)
    analyze.add_argument("--repository-root", type=Path, default=Path.cwd())

    certify = subparsers.add_parser(
        "analyze-certification", help="certify frozen policies on sealed certification data"
    )
    certify.add_argument("--config", type=Path, required=True)
    certify.add_argument("--analysis-plan", type=Path, required=True)
    certify.add_argument("--repository-root", type=Path, default=Path.cwd())

    analyze_test = subparsers.add_parser(
        "analyze-test", help="analyze untouched test outputs under frozen policies"
    )
    analyze_test.add_argument("--config", type=Path, required=True)
    analyze_test.add_argument("--analysis-plan", type=Path, required=True)
    analyze_test.add_argument("--repository-root", type=Path, default=Path.cwd())

    analyze_panel = subparsers.add_parser(
        "analyze-panel", help="aggregate the complete confirmatory test panel"
    )
    analyze_panel.add_argument("--config", type=Path, required=True)
    analyze_panel.add_argument("--repository-root", type=Path, default=Path.cwd())
    analyze_panel.add_argument("--destination", type=Path, required=True)

    analyze_dense = subparsers.add_parser(
        "analyze-dense", help="fit prespecified dense decoder-response surfaces"
    )
    analyze_dense.add_argument("--config", type=Path, required=True)
    analyze_dense.add_argument("--analysis-plan", type=Path, required=True)
    analyze_dense.add_argument("--repository-root", type=Path, default=Path.cwd())

    analyze_agent = subparsers.add_parser(
        "analyze-agent", help="analyze saved BFCL development outputs"
    )
    analyze_agent.add_argument("--config", type=Path, required=True)
    analyze_agent.add_argument("--analysis-plan", type=Path, required=True)
    analyze_agent.add_argument("--repository-root", type=Path, default=Path.cwd())

    analyze_agent_cert = subparsers.add_parser(
        "analyze-agent-certification", help="certify a frozen BFCL policy"
    )
    analyze_agent_cert.add_argument("--config", type=Path, required=True)
    analyze_agent_cert.add_argument("--analysis-plan", type=Path, required=True)
    analyze_agent_cert.add_argument(
        "--repository-root", type=Path, default=Path.cwd()
    )

    analyze_agent_test_parser = subparsers.add_parser(
        "analyze-agent-test", help="analyze untouched BFCL test outputs"
    )
    analyze_agent_test_parser.add_argument("--config", type=Path, required=True)
    analyze_agent_test_parser.add_argument(
        "--analysis-plan", type=Path, required=True
    )
    analyze_agent_test_parser.add_argument(
        "--repository-root", type=Path, default=Path.cwd()
    )

    audit = subparsers.add_parser(
        "audit-matrix", help="verify immutable matrix records without model access"
    )
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--repository-root", type=Path, default=Path.cwd())

    audit_agent_parser = subparsers.add_parser(
        "audit-agent", help="verify immutable BFCL records without model access"
    )
    audit_agent_parser.add_argument("--config", type=Path, required=True)
    audit_agent_parser.add_argument(
        "--repository-root", type=Path, default=Path.cwd()
    )

    report = subparsers.add_parser(
        "report-matrix", help="regenerate lightweight evidence from matrix statistics"
    )
    report.add_argument("--config", type=Path, required=True)
    report.add_argument("--repository-root", type=Path, default=Path.cwd())
    report.add_argument("--destination", type=Path, required=True)

    report_agent = subparsers.add_parser(
        "report-agent", help="regenerate compact evidence from saved BFCL outputs"
    )
    report_agent.add_argument("--config", type=Path, required=True)
    report_agent.add_argument(
        "--repository-root", type=Path, default=Path.cwd()
    )
    report_agent.add_argument("--destination", type=Path, required=True)

    report_panel = subparsers.add_parser(
        "report-panel", help="write manuscript-ready evidence from panel statistics"
    )
    report_panel.add_argument("--panel-result", type=Path, required=True)
    report_panel.add_argument(
        "--repository-root", type=Path, default=Path.cwd()
    )
    report_panel.add_argument("--destination", type=Path, required=True)

    freeze_policy = subparsers.add_parser(
        "freeze-policy", help="freeze development-selected thresholds before certification"
    )
    freeze_policy.add_argument("--config", type=Path, required=True)
    freeze_policy.add_argument("--analysis-plan", type=Path, required=True)
    freeze_policy.add_argument("--repository-root", type=Path, default=Path.cwd())
    freeze_policy.add_argument("--destination", type=Path, required=True)

    freeze_agent_policy = subparsers.add_parser(
        "freeze-agent-policy", help="freeze BFCL development thresholds"
    )
    freeze_agent_policy.add_argument("--config", type=Path, required=True)
    freeze_agent_policy.add_argument("--analysis-plan", type=Path, required=True)
    freeze_agent_policy.add_argument(
        "--repository-root", type=Path, default=Path.cwd()
    )
    freeze_agent_policy.add_argument("--destination", type=Path, required=True)

    resolve_policy = subparsers.add_parser(
        "resolve-certification",
        help="freeze the final certification decision before test access",
    )
    resolve_policy.add_argument("--policy-id", required=True)
    resolve_policy.add_argument("--policy-path", type=Path, required=True)
    resolve_policy.add_argument("--policy-sha256", required=True)
    resolve_policy.add_argument("--analysis-plan", type=Path, required=True)
    resolve_policy.add_argument("--repository-root", type=Path, default=Path.cwd())
    resolve_policy.add_argument("--certification-config", type=Path)
    resolve_policy.add_argument("--destination", type=Path, required=True)

    resolve_agent_policy = subparsers.add_parser(
        "resolve-agent-certification",
        help="freeze the BFCL certification decision before test access",
    )
    resolve_agent_policy.add_argument("--policy-id", required=True)
    resolve_agent_policy.add_argument("--policy-path", type=Path, required=True)
    resolve_agent_policy.add_argument("--policy-sha256", required=True)
    resolve_agent_policy.add_argument("--analysis-plan", type=Path, required=True)
    resolve_agent_policy.add_argument(
        "--repository-root", type=Path, default=Path.cwd()
    )
    resolve_agent_policy.add_argument("--certification-config", type=Path)
    resolve_agent_policy.add_argument("--destination", type=Path, required=True)

    derive_cert = subparsers.add_parser(
        "derive-certification-config",
        help="derive a certification config from a frozen development policy",
    )
    derive_cert.add_argument("--development-config", type=Path, required=True)
    derive_cert.add_argument("--policy", type=Path, required=True)
    derive_cert.add_argument("--repository-root", type=Path, default=Path.cwd())
    derive_cert.add_argument("--destination", type=Path, required=True)

    derive_test = subparsers.add_parser(
        "derive-test-config",
        help="derive an untouched-test config from frozen gate artifacts",
    )
    derive_test.add_argument("--development-config", type=Path, required=True)
    derive_test.add_argument("--policy", type=Path, required=True)
    derive_test.add_argument("--decision", type=Path, required=True)
    derive_test.add_argument("--repository-root", type=Path, default=Path.cwd())
    derive_test.add_argument("--destination", type=Path, required=True)

    derive_agent_cert = subparsers.add_parser(
        "derive-agent-certification-config",
        help="derive a BFCL certification config from a frozen policy",
    )
    derive_agent_cert.add_argument("--development-config", type=Path, required=True)
    derive_agent_cert.add_argument("--policy", type=Path, required=True)
    derive_agent_cert.add_argument(
        "--repository-root", type=Path, default=Path.cwd()
    )
    derive_agent_cert.add_argument("--destination", type=Path, required=True)

    derive_agent_test = subparsers.add_parser(
        "derive-agent-test-config",
        help="derive an untouched BFCL test config from frozen gate artifacts",
    )
    derive_agent_test.add_argument("--development-config", type=Path, required=True)
    derive_agent_test.add_argument("--policy", type=Path, required=True)
    derive_agent_test.add_argument("--decision", type=Path, required=True)
    derive_agent_test.add_argument(
        "--repository-root", type=Path, default=Path.cwd()
    )
    derive_agent_test.add_argument("--destination", type=Path, required=True)

    score_matrix = subparsers.add_parser(
        "score-matrix", help="apply a pinned official evaluator to saved outputs"
    )
    score_matrix.add_argument("--config", type=Path, required=True)
    score_matrix.add_argument("--repository-root", type=Path, default=Path.cwd())

    score_agent = subparsers.add_parser(
        "score-agent", help="score and parse saved BFCL outputs without model access"
    )
    score_agent.add_argument("--config", type=Path, required=True)
    score_agent.add_argument("--repository-root", type=Path, default=Path.cwd())
    score_agent.add_argument("--replica-index", type=int, default=0)
    score_agent.add_argument("--preflight", action="store_true")

    run = subparsers.add_parser("run", help="run or safely resume an experiment")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--repository-root", type=Path, default=Path.cwd())
    run.add_argument("--replica-index", type=int, default=0)

    run_matrix = subparsers.add_parser(
        "run-matrix", help="run or safely resume a development readout matrix"
    )
    run_matrix.add_argument("--config", type=Path, required=True)
    run_matrix.add_argument("--repository-root", type=Path, default=Path.cwd())
    run_matrix.add_argument("--replica-index", type=int, default=0)
    run_matrix.add_argument(
        "--generate-only",
        action="store_true",
        help="publish immutable raw answer/confidence records without parsing",
    )
    run_matrix.add_argument(
        "--answers-only",
        action="store_true",
        help="stop after the answer stage; confidence readouts depend only on the "
        "saved answers and can be generated later under a revised protocol",
    )

    run_agent = subparsers.add_parser(
        "run-agent", help="run or safely resume a BFCL agent experiment"
    )
    run_agent.add_argument("--config", type=Path, required=True)
    run_agent.add_argument("--repository-root", type=Path, default=Path.cwd())
    run_agent.add_argument("--replica-index", type=int, default=0)
    run_agent.add_argument("--preflight", action="store_true")

    smoke = subparsers.add_parser("smoke-run", help="run a test-only smoke experiment")
    smoke.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    smoke.add_argument("--repository-root", type=Path, default=Path.cwd())
    smoke.add_argument("--replica-index", type=int, default=0)

    schemas = subparsers.add_parser("export-schemas", help="regenerate JSON schemas")
    schemas.add_argument("--destination", type=Path, default=Path("schemas"))
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        if args.command == "validate":
            config = load_config(args.config)
            print(json.dumps({"run_id": config.run_id, "valid": True}))
            return
        if args.command == "validate-analysis":
            plan = load_analysis_plan(args.config)
            print(
                json.dumps(
                    {"plan_id": plan.plan_id, "sha256": plan.sha256, "valid": True},
                    sort_keys=True,
                )
            )
            return
        if args.command == "validate-matrix":
            config = load_matrix_config(args.config)
            validate_matrix_integrations(config, args.repository_root.resolve())
            print(json.dumps({"run_id": config.run_id, "valid": True}, sort_keys=True))
            return
        if args.command == "validate-agent":
            config = load_agent_config(args.config)
            validate_agent_integrations(
                config, args.repository_root.resolve(), preflight=args.preflight
            )
            print(json.dumps({"run_id": config.run_id, "valid": True}, sort_keys=True))
            return
        if args.command == "validate-integration":
            record = load_integration_record(args.record)
            print(
                json.dumps(
                    {
                        "record_id": record.record_id,
                        "record_type": record.record_type,
                        "sha256": record.canonical_sha256,
                        "valid": True,
                    },
                    sort_keys=True,
                )
            )
            return
        if args.command == "validate-study-plan":
            plan = load_study_plan(args.config)
            print(
                json.dumps(
                    {
                        "plan_id": plan.plan_id,
                        "sha256": plan.canonical_sha256,
                        "valid": True,
                    },
                    sort_keys=True,
                )
            )
            return
        if args.command == "validate-dense-plan":
            plan = load_dense_grid_plan(args.config)
            print(
                json.dumps(
                    {
                        "grid_id": plan.grid_id,
                        "sha256": plan.canonical_sha256,
                        "valid": True,
                    },
                    sort_keys=True,
                )
            )
            return
        if args.command == "qualify-dense":
            record = qualify_dense_preflight(
                args.config,
                args.dense_plan,
                args.study_plan,
                args.repository_root,
                args.destination,
            )
            print(
                json.dumps(
                    {
                        "qualification_id": record.qualification_id,
                        "sha256": record.canonical_sha256,
                        "supported_cells": sum(
                            cell.status == "supported" for cell in record.cells
                        ),
                        "unsupported_cells": sum(
                            cell.status == "unsupported" for cell in record.cells
                        ),
                        "destination": str(args.destination.resolve()),
                    },
                    sort_keys=True,
                )
            )
            return
        if args.command == "analyze-matrix":
            output_path, result = analyze_matrix(
                args.config, args.analysis_plan, args.repository_root
            )
            print(
                json.dumps(
                    {
                        "analysis_id": result["analysis_id"],
                        "output": str(output_path),
                        "run_id": result["run_id"],
                    },
                    sort_keys=True,
                )
            )
            return
        if args.command == "analyze-certification":
            output_path, result = analyze_certification(
                args.config, args.analysis_plan, args.repository_root
            )
            print(
                json.dumps(
                    {
                        "analysis_id": result["analysis_id"],
                        "output": str(output_path),
                        "run_id": result["run_id"],
                    },
                    sort_keys=True,
                )
            )
            return
        if args.command == "analyze-test":
            output_path, result = analyze_confirmatory_test(
                args.config, args.analysis_plan, args.repository_root
            )
            print(
                json.dumps(
                    {
                        "analysis_id": result["analysis_id"],
                        "output": str(output_path),
                        "run_id": result["run_id"],
                    },
                    sort_keys=True,
                )
            )
            return
        if args.command == "analyze-panel":
            output_path, result = analyze_confirmatory_panel(
                args.config, args.repository_root, args.destination
            )
            print(
                json.dumps(
                    {
                        "analysis_id": result["analysis_id"],
                        "effects": len(result["effect_inventory"]),
                        "output": str(output_path),
                    },
                    sort_keys=True,
                )
            )
            return
        if args.command == "analyze-dense":
            output_path, result = analyze_dense_matrix(
                args.config, args.analysis_plan, args.repository_root
            )
            print(
                json.dumps(
                    {
                        "analysis_id": result["analysis_id"],
                        "output": str(output_path),
                        "run_id": result["run_id"],
                    },
                    sort_keys=True,
                )
            )
            return
        if args.command == "analyze-agent":
            output_path, result = analyze_agent_development(
                args.config, args.analysis_plan, args.repository_root
            )
            print(
                json.dumps(
                    {
                        "analysis_id": result["analysis_id"],
                        "output": str(output_path),
                        "run_id": result["run_id"],
                    },
                    sort_keys=True,
                )
            )
            return
        if args.command == "analyze-agent-certification":
            output_path, result = analyze_agent_certification(
                args.config, args.analysis_plan, args.repository_root
            )
            print(
                json.dumps(
                    {
                        "analysis_id": result["analysis_id"],
                        "output": str(output_path),
                        "run_id": result["run_id"],
                    },
                    sort_keys=True,
                )
            )
            return
        if args.command == "analyze-agent-test":
            output_path, result = analyze_agent_test(
                args.config, args.analysis_plan, args.repository_root
            )
            print(
                json.dumps(
                    {
                        "analysis_id": result["analysis_id"],
                        "output": str(output_path),
                        "run_id": result["run_id"],
                    },
                    sort_keys=True,
                )
            )
            return
        if args.command == "audit-matrix":
            output_path, result = audit_matrix(args.config, args.repository_root)
            print(
                json.dumps(
                    {
                        "audit_id": result["audit_id"],
                        "output": str(output_path),
                        "run_id": result["run_id"],
                    },
                    sort_keys=True,
                )
            )
            return
        if args.command == "audit-agent":
            output_path, result = audit_agent(args.config, args.repository_root)
            print(
                json.dumps(
                    {
                        "audit_id": result["audit_id"],
                        "output": str(output_path),
                        "run_id": result["run_id"],
                    },
                    sort_keys=True,
                )
            )
            return
        if args.command == "report-matrix":
            outputs = write_matrix_evidence(
                args.config, args.repository_root, args.destination
            )
            print(json.dumps(outputs, sort_keys=True))
            return
        if args.command == "report-agent":
            outputs = write_agent_evidence(
                args.config, args.repository_root, args.destination
            )
            print(json.dumps(outputs, sort_keys=True))
            return
        if args.command == "report-panel":
            outputs = write_panel_evidence(
                args.panel_result, args.repository_root, args.destination
            )
            print(json.dumps(outputs, sort_keys=True))
            return
        if args.command == "freeze-policy":
            policy = freeze_development_policy(
                args.config,
                args.analysis_plan,
                args.repository_root,
                args.destination,
            )
            print(
                json.dumps(
                    {
                        "policy_id": policy.policy_id,
                        "sha256": policy.canonical_sha256,
                        "destination": str(args.destination.resolve()),
                    },
                    sort_keys=True,
                )
            )
            return
        if args.command == "freeze-agent-policy":
            policy = freeze_agent_development_policy(
                args.config,
                args.analysis_plan,
                args.repository_root,
                args.destination,
            )
            print(
                json.dumps(
                    {
                        "policy_id": policy.policy_id,
                        "sha256": policy.canonical_sha256,
                        "destination": str(args.destination.resolve()),
                    },
                    sort_keys=True,
                )
            )
            return
        if args.command == "resolve-certification":
            decision = resolve_certification_decision(
                IntegrationReference(
                    record_id=args.policy_id,
                    record_path=str(args.policy_path),
                    canonical_sha256=args.policy_sha256,
                ),
                args.analysis_plan,
                args.repository_root,
                args.destination,
                args.certification_config,
            )
            print(
                json.dumps(
                    {
                        "decision_id": decision.decision_id,
                        "sha256": decision.canonical_sha256,
                        "destination": str(args.destination.resolve()),
                    },
                    sort_keys=True,
                )
            )
            return
        if args.command == "resolve-agent-certification":
            decision = resolve_agent_certification_decision(
                IntegrationReference(
                    record_id=args.policy_id,
                    record_path=str(args.policy_path),
                    canonical_sha256=args.policy_sha256,
                ),
                args.analysis_plan,
                args.repository_root,
                args.destination,
                args.certification_config,
            )
            print(
                json.dumps(
                    {
                        "decision_id": decision.decision_id,
                        "sha256": decision.canonical_sha256,
                        "destination": str(args.destination.resolve()),
                    },
                    sort_keys=True,
                )
            )
            return
        if args.command == "derive-certification-config":
            config = derive_certification_config(
                args.development_config,
                args.policy,
                args.repository_root,
                args.destination,
            )
            print(
                json.dumps(
                    {"run_id": config.run_id, "destination": str(args.destination)},
                    sort_keys=True,
                )
            )
            return
        if args.command == "derive-test-config":
            config = derive_test_config(
                args.development_config,
                args.policy,
                args.decision,
                args.repository_root,
                args.destination,
            )
            print(
                json.dumps(
                    {"run_id": config.run_id, "destination": str(args.destination)},
                    sort_keys=True,
                )
            )
            return
        if args.command == "derive-agent-certification-config":
            config = derive_agent_certification_config(
                args.development_config,
                args.policy,
                args.repository_root,
                args.destination,
            )
            print(
                json.dumps(
                    {"run_id": config.run_id, "destination": str(args.destination)},
                    sort_keys=True,
                )
            )
            return
        if args.command == "derive-agent-test-config":
            config = derive_agent_test_config(
                args.development_config,
                args.policy,
                args.decision,
                args.repository_root,
                args.destination,
            )
            print(
                json.dumps(
                    {"run_id": config.run_id, "destination": str(args.destination)},
                    sort_keys=True,
                )
            )
            return
        if args.command == "score-matrix":
            result = score_saved_matrix(args.config, args.repository_root)
            print(
                json.dumps(
                    {**result.__dict__, "output_directory": str(result.output_directory)},
                    sort_keys=True,
                )
            )
            return
        if args.command == "score-agent":
            result = derive_agent_records(
                args.config,
                args.repository_root,
                args.replica_index,
                preflight=args.preflight,
            )
            print(
                json.dumps(
                    {**result.__dict__, "run_directory": str(result.run_directory)},
                    sort_keys=True,
                )
            )
            return
        if args.command == "export-schemas":
            export_schemas(args.destination)
            print(json.dumps({"destination": str(args.destination.resolve())}))
            return
        if args.command == "smoke-run":
            config = load_config(args.config)
            if config.study != "smoke":
                raise ValueError("smoke-run requires study=smoke")
        if args.command == "run-matrix":
            result = run_matrix_experiment(
                args.config,
                args.repository_root,
                args.replica_index,
                generate_only=args.generate_only,
                answers_only=args.answers_only,
            )
        elif args.command == "run-agent":
            result = run_agent_experiment(
                args.config,
                args.repository_root,
                args.replica_index,
                preflight=args.preflight,
            )
        else:
            result = run_experiment(args.config, args.repository_root, args.replica_index)
        payload = {
            **result.__dict__,
            "run_directory": str(result.run_directory),
        }
        print(json.dumps(payload, sort_keys=True))
    except Exception as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
