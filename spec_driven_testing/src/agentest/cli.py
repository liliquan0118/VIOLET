"""Command-line entry point for AgentSpecTesting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adaptive_search import (
    AdaptiveSearchError,
    close_adaptive_round,
    generate_adaptive_round_files,
)
from .candidate_generator import CandidateGenerationError, generate_candidate_files
from .compiler import (
    ArtifactBundleError,
    CompilerRequestError,
    ConfigurationError,
    EvaluationContractError,
    ExperimentPlanningError,
    SourceAnchorLoadError,
    TargetCellSelectionError,
    assess_request_file,
    assess_spec_set_file,
    compile_request_file,
)
from .compiler.selected_cell import SelectedCellError
from .execution_runner import (
    ExecutionConfigurationError,
    configure_openai_compatible_environment,
    execute_manifest,
    load_provider_config,
)
from .fixture_materializer import FixtureMaterializationError, materialize_fixture_file
from .driver import (
    DriverBindingError,
    RuntimeDriverError,
    TauOnlineAdapterError,
    bind_registered_driver_file,
    bind_tau_airline_driver_file,
    replay_runtime_trace_file,
    run_tau_online_driver_file,
)
from .input_compiler import InputContractError, compile_input_file
from .guidance import (
    FinalFindingError,
    GuidanceError,
    GuidanceLoopError,
    build_final_finding_bundle_files,
    decide_next_experiment,
    new_guidance_history,
    record_guidance_observation,
    run_approved_guidance_batch_files,
)
from .oracle import (
    BoundOracleError,
    evaluate_bound_online_execution_file,
)
from .result_analyzer import ResultAnalysisError, analyze_execution_batch


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-spec-testing")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser(
        "compile-input",
        help="validate a test-generation input and emit a deterministic generation plan",
    )
    compile_parser.add_argument("--input", required=True, type=Path)
    compile_parser.add_argument("--output", type=Path)
    compile_parser.add_argument(
        "--summary",
        action="store_true",
        help="print a compact compilation summary instead of the full plan",
    )

    accepted_compile_parser = subparsers.add_parser(
        "compile-test",
        help="compile an accepted policy cell and Test Intent into an unbound CompiledTestPlan",
    )
    accepted_compile_parser.add_argument("--request", required=True, type=Path)
    accepted_compile_parser.add_argument("--artifact-bundle", required=True, type=Path)
    accepted_compile_parser.add_argument(
        "--source-base-dir",
        type=Path,
        default=Path.cwd(),
        help="base directory for relative workbook paths in source-resolved requests",
    )
    accepted_compile_parser.add_argument(
        "--compiler-profile",
        type=Path,
        help="versioned method profile required by compiler-request/v0.3",
    )
    accepted_compile_parser.add_argument(
        "--sut-adapter-profile",
        type=Path,
        help="versioned SUT capability profile required by compiler-request/v0.3",
    )
    accepted_compile_parser.add_argument("--output", type=Path)
    accepted_compile_parser.add_argument("--summary", action="store_true")

    assess_compile_parser = subparsers.add_parser(
        "assess-test",
        help=(
            "resolve one selected spec and report readiness at every pipeline "
            "layer without claiming unsupported work compiled"
        ),
    )
    assess_compile_parser.add_argument("--request", required=True, type=Path)
    assess_compile_parser.add_argument("--artifact-bundle", required=True, type=Path)
    assess_compile_parser.add_argument(
        "--source-base-dir", type=Path, default=Path.cwd()
    )
    assess_compile_parser.add_argument("--compiler-profile", required=True, type=Path)
    assess_compile_parser.add_argument(
        "--sut-adapter-profile", required=True, type=Path
    )
    assess_compile_parser.add_argument("--output", type=Path)
    assess_compile_parser.add_argument("--summary", action="store_true")

    assess_set_parser = subparsers.add_parser(
        "assess-spec-set",
        help=(
            "assess every selected workbook branch and separate source, "
            "semantic and end-to-end readiness"
        ),
    )
    assess_set_parser.add_argument("--workbook", required=True, type=Path)
    assess_set_parser.add_argument("--sheet", required=True)
    assess_set_parser.add_argument(
        "--review-ok",
        type=int,
        default=1,
        help="include only rows with this review_ok value; default: 1",
    )
    assess_set_parser.add_argument("--artifact-bundle", required=True, type=Path)
    assess_set_parser.add_argument(
        "--source-base-dir", type=Path, default=Path.cwd()
    )
    assess_set_parser.add_argument("--compiler-profile", required=True, type=Path)
    assess_set_parser.add_argument(
        "--sut-adapter-profile", required=True, type=Path
    )
    assess_set_parser.add_argument("--output", type=Path)
    assess_set_parser.add_argument("--summary", action="store_true")

    bind_driver_parser = subparsers.add_parser(
        "bind-driver",
        help="bind a CompiledTestPlan v0.3 to auditable tau-bench airline fixtures",
    )
    bind_driver_parser.add_argument("--plan", required=True, type=Path)
    bind_driver_parser.add_argument("--run-configuration", required=True, type=Path)
    bind_driver_parser.add_argument("--output", required=True, type=Path)
    bind_driver_parser.add_argument(
        "--database",
        type=Path,
        help="optional airline db.json; otherwise load a fresh tau2 environment",
    )
    bind_driver_parser.add_argument(
        "--reference-time",
        help="observed policy clock; required for a JSON database without policy_clock metadata",
    )
    bind_driver_parser.add_argument("--summary", action="store_true")

    replay_driver_parser = subparsers.add_parser(
        "replay-driver",
        help="replay a deterministic runtime-driver trace without model calls",
    )
    replay_driver_parser.add_argument("--bound-plan", required=True, type=Path)
    replay_driver_parser.add_argument(
        "--run-configuration", required=True, type=Path
    )
    replay_driver_parser.add_argument("--trace", required=True, type=Path)
    replay_driver_parser.add_argument("--output", required=True, type=Path)
    replay_driver_parser.add_argument("--summary", action="store_true")

    execute_bound_driver_parser = subparsers.add_parser(
        "execute-bound-driver",
        help="run one bound fixture with a deterministic driver user and target agent",
    )
    execute_bound_driver_parser.add_argument("--bound-plan", required=True, type=Path)
    execute_bound_driver_parser.add_argument(
        "--run-configuration", required=True, type=Path
    )
    execute_bound_driver_parser.add_argument("--fixture-instance-id", required=True)
    execute_bound_driver_parser.add_argument(
        "--variation",
        action="append",
        default=[],
        metavar="DIMENSION=VALUE",
        help="select at most the compiled number of changed mutation dimensions",
    )
    execute_bound_driver_parser.add_argument("--output", required=True, type=Path)
    execute_bound_driver_parser.add_argument("--api-file", type=Path)
    execute_bound_driver_parser.add_argument("--api-key-env")
    execute_bound_driver_parser.add_argument("--base-url")
    execute_bound_driver_parser.add_argument("--model", default="deepseek-v4-flash")
    execute_bound_driver_parser.add_argument(
        "--approved-agent-call-budget", required=True, type=int
    )
    execute_bound_driver_parser.add_argument("--seed", type=int, default=0)
    execute_bound_driver_parser.add_argument(
        "--timeout-seconds", type=float, default=300.0
    )
    execute_bound_driver_parser.add_argument("--summary", action="store_true")

    evaluate_bound_parser = subparsers.add_parser(
        "evaluate-bound-execution",
        help="apply the bound correctness oracle to one saved online execution",
    )
    evaluate_bound_parser.add_argument("--bound-plan", required=True, type=Path)
    evaluate_bound_parser.add_argument("--execution", required=True, type=Path)
    evaluate_bound_parser.add_argument("--output", required=True, type=Path)
    evaluate_bound_parser.add_argument("--summary", action="store_true")

    fixture_parser = subparsers.add_parser(
        "materialize-fixture",
        help="select a tau-bench entity and bind all compiled boundary variants",
    )
    fixture_parser.add_argument("--plan", required=True, type=Path)
    fixture_parser.add_argument("--output", required=True, type=Path)
    fixture_parser.add_argument(
        "--database",
        type=Path,
        help="optional airline db.json; if omitted, load a fresh database through tau2",
    )
    fixture_parser.add_argument("--summary", action="store_true")

    candidate_parser = subparsers.add_parser(
        "generate-candidates",
        help="generate baseline and bounded single-mutation tau2 task candidates",
    )
    candidate_parser.add_argument("--plan", required=True, type=Path)
    candidate_parser.add_argument("--fixtures", required=True, type=Path)
    candidate_parser.add_argument("--output-dir", required=True, type=Path)
    candidate_parser.add_argument("--adversarial-count", type=int, default=5)
    candidate_parser.add_argument("--summary", action="store_true")

    execute_parser = subparsers.add_parser(
        "execute",
        help="execute an approved candidate manifest and apply the policy oracle",
    )
    execute_parser.add_argument("--manifest", required=True, type=Path)
    execute_parser.add_argument("--workspace", type=Path, default=Path.cwd())
    execute_parser.add_argument("--output-dir", required=True, type=Path)
    execute_parser.add_argument("--api-file", type=Path)
    execute_parser.add_argument("--api-key-env")
    execute_parser.add_argument("--base-url")
    execute_parser.add_argument("--model", default="deepseek-v4-flash")
    execute_parser.add_argument("--max-steps", type=int, default=30)
    execute_parser.add_argument("--approved-task-count", type=int, required=True)
    execute_parser.add_argument("--max-workers", type=int, default=2)
    execute_parser.add_argument("--seed", type=int, default=0)
    execute_parser.add_argument("--timeout-seconds", type=float, default=300.0)

    analyze_parser = subparsers.add_parser(
        "analyze-results",
        help="separate decision correctness, protocol gaps, and harness errors offline",
    )
    analyze_parser.add_argument("--summary", required=True, type=Path)
    analyze_parser.add_argument("--output", required=True, type=Path)

    adaptive_parser = subparsers.add_parser(
        "generate-adaptive-round",
        help="use prior execution feedback to prune search and generate the next candidates",
    )
    adaptive_parser.add_argument("--plan", required=True, type=Path)
    adaptive_parser.add_argument("--fixtures", required=True, type=Path)
    adaptive_parser.add_argument("--source-manifest", required=True, type=Path)
    adaptive_parser.add_argument("--analysis", required=True, type=Path)
    adaptive_parser.add_argument("--execution-summary", required=True, type=Path)
    adaptive_parser.add_argument("--workspace", type=Path, default=Path.cwd())
    adaptive_parser.add_argument("--output-dir", required=True, type=Path)

    close_parser = subparsers.add_parser(
        "close-adaptive-round",
        help="consume adaptive feedback and emit a stop or continue guidance decision",
    )
    close_parser.add_argument("--guidance", required=True, type=Path)
    close_parser.add_argument("--analysis", required=True, type=Path)
    close_parser.add_argument("--execution-summary", required=True, type=Path)
    close_parser.add_argument("--workspace", type=Path, default=Path.cwd())
    close_parser.add_argument("--output", required=True, type=Path)

    init_guidance_parser = subparsers.add_parser(
        "init-guidance",
        help="initialize an append-only deterministic Guidance search history",
    )
    init_guidance_parser.add_argument("--bound-plan", required=True, type=Path)
    init_guidance_parser.add_argument("--max-experiments", required=True, type=int)
    init_guidance_parser.add_argument(
        "--required-confirmations", type=int, default=2
    )
    init_guidance_parser.add_argument(
        "--max-attempts-per-candidate", type=int, default=3
    )
    init_guidance_parser.add_argument(
        "--max-invalid-attempts-per-coordinate", type=int, default=2
    )
    init_guidance_parser.add_argument("--output", required=True, type=Path)

    record_guidance_parser = subparsers.add_parser(
        "record-guidance",
        help="append one Bound Oracle result to a Guidance search history",
    )
    record_guidance_parser.add_argument("--bound-plan", required=True, type=Path)
    record_guidance_parser.add_argument("--history", required=True, type=Path)
    record_guidance_parser.add_argument("--oracle", required=True, type=Path)
    record_guidance_parser.add_argument("--output", required=True, type=Path)

    decide_guidance_parser = subparsers.add_parser(
        "decide-guidance",
        help="select one next bound experiment from structured execution feedback",
    )
    decide_guidance_parser.add_argument("--bound-plan", required=True, type=Path)
    decide_guidance_parser.add_argument("--history", required=True, type=Path)
    decide_guidance_parser.add_argument("--output", required=True, type=Path)
    decide_guidance_parser.add_argument("--summary", action="store_true")

    guidance_batch_parser = subparsers.add_parser(
        "run-guidance-batch",
        help=(
            "execute only an explicitly approved number of Guidance experiments "
            "and checkpoint every Oracle feedback step"
        ),
    )
    guidance_batch_parser.add_argument("--bound-plan", required=True, type=Path)
    guidance_batch_parser.add_argument(
        "--run-configuration", required=True, type=Path
    )
    guidance_batch_parser.add_argument("--history", required=True, type=Path)
    guidance_batch_parser.add_argument("--output-dir", required=True, type=Path)
    guidance_batch_parser.add_argument(
        "--approved-experiment-count", required=True, type=int
    )
    guidance_batch_parser.add_argument(
        "--approved-agent-call-budget-per-experiment", required=True, type=int
    )
    guidance_batch_parser.add_argument("--api-file", type=Path)
    guidance_batch_parser.add_argument("--api-key-env")
    guidance_batch_parser.add_argument("--base-url")
    guidance_batch_parser.add_argument("--model", default="deepseek-v4-flash")
    guidance_batch_parser.add_argument("--seed", type=int, default=0)
    guidance_batch_parser.add_argument(
        "--timeout-seconds", type=float, default=300.0
    )

    finding_parser = subparsers.add_parser(
        "export-finding",
        help="export one confirmed terminal Guidance search as a finding bundle",
    )
    finding_parser.add_argument("--compiled-plan", required=True, type=Path)
    finding_parser.add_argument("--bound-plan", required=True, type=Path)
    finding_parser.add_argument("--history", required=True, type=Path)
    finding_parser.add_argument("--terminal-decision", type=Path)
    finding_parser.add_argument("--output", required=True, type=Path)
    finding_parser.add_argument("--summary", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command in {"init-guidance", "record-guidance", "decide-guidance"}:
        try:
            bound = json.loads(args.bound_plan.read_text(encoding="utf-8"))
            if args.command == "init-guidance":
                artifact = new_guidance_history(
                    bound,
                    max_experiments=args.max_experiments,
                    required_confirmations=args.required_confirmations,
                    max_attempts_per_candidate=args.max_attempts_per_candidate,
                    max_invalid_attempts_per_coordinate=(
                        args.max_invalid_attempts_per_coordinate
                    ),
                )
            else:
                history = json.loads(args.history.read_text(encoding="utf-8"))
                if args.command == "record-guidance":
                    oracle_document = json.loads(
                        args.oracle.read_text(encoding="utf-8")
                    )
                    oracle = oracle_document.get("bound_oracle", oracle_document)
                    artifact = record_guidance_observation(
                        bound, history, oracle
                    )
                else:
                    artifact = decide_next_experiment(bound, history)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except (OSError, json.JSONDecodeError, GuidanceError) as exc:
            print(f"guidance failed: {exc}", file=sys.stderr)
            return 2
        if args.command == "decide-guidance" and args.summary:
            print(
                json.dumps(
                    {
                        "decision_id": artifact["decision_id"],
                        "status": artifact["status"],
                        "phase": artifact["phase"],
                        "reason_code": artifact["reason_code"],
                        "experiment": artifact["experiment"],
                        "budget": artifact["budget"],
                        "llm_calls": artifact["generator_contract"]["llm_calls"],
                        "output": str(args.output),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return 0

    if args.command == "run-guidance-batch":
        try:
            preview_bound = json.loads(
                args.bound_plan.read_text(encoding="utf-8")
            )
            preview_history = json.loads(
                args.history.read_text(encoding="utf-8")
            )
            if decide_next_experiment(preview_bound, preview_history)[
                "status"
            ] == "continue":
                provider = load_provider_config(
                    api_file=args.api_file,
                    api_key_env=args.api_key_env,
                    base_url=args.base_url,
                )
                configure_openai_compatible_environment(provider)
            batch = run_approved_guidance_batch_files(
                bound_plan_path=args.bound_plan,
                run_configuration_path=args.run_configuration,
                history_path=args.history,
                output_dir=args.output_dir,
                approved_experiment_count=args.approved_experiment_count,
                approved_agent_call_budget_per_experiment=(
                    args.approved_agent_call_budget_per_experiment
                ),
                model=args.model,
                seed=args.seed,
                timeout_seconds=args.timeout_seconds,
            )
        except (
            OSError,
            json.JSONDecodeError,
            ExecutionConfigurationError,
            GuidanceError,
            GuidanceLoopError,
            RuntimeDriverError,
            TauOnlineAdapterError,
            ConfigurationError,
            ImportError,
        ) as exc:
            print(f"guidance batch failed: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "batch_status": batch["batch_status"],
                    "stop_reason": batch["stop_reason"],
                    "approval": batch["approval"],
                    "consumption": batch["consumption"],
                    "next_guidance": {
                        key: batch["next_guidance_decision"][key]
                        for key in ("status", "phase", "reason_code", "experiment")
                    },
                    "output_dir": str(args.output_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "export-finding":
        try:
            finding = build_final_finding_bundle_files(
                compiled_plan_path=args.compiled_plan,
                bound_plan_path=args.bound_plan,
                history_path=args.history,
                terminal_decision_path=args.terminal_decision,
                output_path=args.output,
            )
        except (OSError, json.JSONDecodeError, FinalFindingError, GuidanceError) as exc:
            print(f"finding export failed: {exc}", file=sys.stderr)
            return 2
        if args.summary:
            print(
                json.dumps(
                    {
                        "finding_id": finding["finding_id"],
                        "finding_status": finding["finding_status"],
                        "classification": finding["failure_signature"][
                            "classification"
                        ],
                        "trigger_observation_ids": finding["confirmation"][
                            "trigger_observation_ids"
                        ],
                        "passing_surface_contrast_count": len(
                            finding["passing_surface_contrasts"]
                        ),
                        "quarantined_experiment_count": finding[
                            "search_summary"
                        ]["counts"]["quarantined_experiments"],
                        "finding_fingerprint": finding["finding_fingerprint"],
                        "output": str(args.output),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return 0

    if args.command == "compile-input":
        try:
            plan = compile_input_file(args.input, args.output)
        except (OSError, json.JSONDecodeError, InputContractError) as exc:
            print(f"input compilation failed: {exc}", file=sys.stderr)
            return 2

        if args.summary:
            print(
                json.dumps(
                    {
                        "input_id": plan["input_id"],
                        "source_branch": plan["source"]["branch_id"],
                        "action_under_test": plan["decision"]["action_under_test"],
                        "variant_count": len(plan["fixture"]["variants"]),
                        "mutation_dimension_count": len(plan["dialogue_search"]["mutation_catalog"]),
                        "compiler_checks": plan["compiler_checks"],
                        "output": str(args.output) if args.output else None,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.output is None:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    if args.command == "assess-test":
        try:
            assessment = assess_request_file(
                request_path=args.request,
                artifact_bundle_path=args.artifact_bundle,
                source_base_dir=args.source_base_dir,
                compiler_profile_path=args.compiler_profile,
                sut_adapter_profile_path=args.sut_adapter_profile,
                output_path=args.output,
            )
        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
            ArtifactBundleError,
            ConfigurationError,
            EvaluationContractError,
            SourceAnchorLoadError,
        ) as exc:
            print(f"compiler assessment failed: {exc}", file=sys.stderr)
            return 2
        if args.summary:
            print(
                json.dumps(
                    {
                        "source_branch_id": assessment["source_branch_id"],
                        "target_archetype": assessment["target_archetype"],
                        "resolution": assessment["stages"]["semantic_resolution"],
                        "end_to_end_ready": assessment["end_to_end_ready"],
                        "blocked_stages": [
                            name
                            for name, stage in assessment["stages"].items()
                            if stage["status"] in {"blocked", "not_assessed"}
                        ],
                        "output": str(args.output) if args.output else None,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.output is None:
            print(json.dumps(assessment, ensure_ascii=False, indent=2))
        return 0

    if args.command == "assess-spec-set":
        try:
            report = assess_spec_set_file(
                workbook=args.workbook,
                sheet=args.sheet,
                review_ok=args.review_ok,
                artifact_bundle_path=args.artifact_bundle,
                source_base_dir=args.source_base_dir,
                compiler_profile_path=args.compiler_profile,
                sut_adapter_profile_path=args.sut_adapter_profile,
                output_path=args.output,
            )
        except (
            OSError,
            ValueError,
            ArtifactBundleError,
            ConfigurationError,
            EvaluationContractError,
            SourceAnchorLoadError,
        ) as exc:
            print(f"compiler set assessment failed: {exc}", file=sys.stderr)
            return 2
        if args.summary:
            summary = dict(report["summary"])
            summary["output"] = str(args.output) if args.output else None
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        elif args.output is None:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "compile-test":
        try:
            plan = compile_request_file(
                request_path=args.request,
                artifact_bundle_path=args.artifact_bundle,
                output_path=args.output,
                source_base_dir=args.source_base_dir,
                compiler_profile_path=args.compiler_profile,
                sut_adapter_profile_path=args.sut_adapter_profile,
            )
        except (
            OSError,
            json.JSONDecodeError,
            ArtifactBundleError,
            CompilerRequestError,
            ConfigurationError,
            ExperimentPlanningError,
            SourceAnchorLoadError,
            TargetCellSelectionError,
            SelectedCellError,
        ) as exc:
            print(f"accepted-model compilation failed: {exc}", file=sys.stderr)
            return 2
        if args.summary:
            print(
                json.dumps(
                    {
                        "compiled_plan_id": plan["compiled_plan_id"],
                        "target_coverage_cell_id": plan["target"][
                            "focal_configuration"
                        ]["coverage_cell_id"],
                        "expected_operation_decision": plan["target"][
                            "focal_configuration"
                        ]["expected_operation_decision"],
                        "contrast_target_count": len(plan["contrast_targets"]),
                        "boundary_refinement_count": len(plan["boundary_refinements"]),
                        "probe_family": (plan.get("probe_contract") or {}).get(
                            "probe_family"
                        ),
                        "probe_instance_count": len(
                            (plan.get("probe_contract") or {}).get(
                                "probe_instances"
                            )
                            or []
                        ),
                        "generation_readiness": plan["generation_readiness"],
                        "compiled_plan_fingerprint": plan[
                            "compiled_plan_fingerprint"
                        ],
                        "llm_calls": plan["compiler_checks"]["llm_calls"],
                        "output": str(args.output) if args.output else None,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.output is None:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    if args.command == "materialize-fixture":
        try:
            bundle = materialize_fixture_file(args.plan, args.output, args.database)
        except (
            OSError,
            json.JSONDecodeError,
            InputContractError,
            FixtureMaterializationError,
        ) as exc:
            print(f"fixture materialization failed: {exc}", file=sys.stderr)
            return 2
        if args.summary:
            print(
                json.dumps(
                    {
                        "input_id": bundle["input_id"],
                        "candidate_count": bundle["selection"]["candidate_count"],
                        "selected_reservation_id": bundle["selection"]["selected_reservation_id"],
                        "selected_user_id": bundle["selection"]["selected_user_id"],
                        "variant_count": len(bundle["variants"]),
                        "checks": bundle["materialization_checks"],
                        "output": str(args.output),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return 0

    if args.command == "bind-driver":
        try:
            bound = bind_registered_driver_file(
                compiled_plan_path=args.plan,
                run_configuration_path=args.run_configuration,
                output_path=args.output,
                database_path=args.database,
                reference_time=args.reference_time,
            )
        except (
            OSError,
            json.JSONDecodeError,
            DriverBindingError,
            ConfigurationError,
            FixtureMaterializationError,
        ) as exc:
            print(f"driver binding failed: {exc}", file=sys.stderr)
            return 2
        if args.summary:
            print(
                json.dumps(
                    {
                        "bound_driver_plan_id": bound["bound_driver_plan_id"],
                        "selected_reservation_id": bound["selection"][
                            "selected_reservation_id"
                        ],
                        "fixture_instance_count": len(bound["fixture_instances"]),
                        "state_patch_count": sum(
                            fixture["initial_state_patch"]["agent_data"] is not None
                            for fixture in bound["fixture_instances"]
                        ),
                        "binding_checks": bound["binding_checks"],
                        "bound_driver_plan_fingerprint": bound[
                            "bound_driver_plan_fingerprint"
                        ],
                        "output": str(args.output),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return 0

    if args.command == "replay-driver":
        try:
            replay = replay_runtime_trace_file(
                bound_plan_path=args.bound_plan,
                run_configuration_path=args.run_configuration,
                trace_path=args.trace,
                output_path=args.output,
            )
        except (
            OSError,
            json.JSONDecodeError,
            RuntimeDriverError,
            ConfigurationError,
        ) as exc:
            print(f"runtime driver replay failed: {exc}", file=sys.stderr)
            return 2
        if args.summary:
            result = replay["reachability_result"]
            print(
                json.dumps(
                    {
                        "trace_id": replay["trace_id"],
                        "fixture_instance_id": result["fixture_instance_id"],
                        "reached": result["reached"],
                        "decision_opportunity_reached": result[
                            "decision_opportunity_reached"
                        ],
                        "target_action_observed": result[
                            "target_action_observed"
                        ],
                        "stop_reason": result["stop_reason"],
                        "turn_count": result["turn_count"],
                        "llm_calls": replay["replay_checks"]["llm_calls"],
                        "output": str(args.output),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return 0

    if args.command == "execute-bound-driver":
        try:
            variation: dict[str, str] = {}
            for raw in args.variation:
                if "=" not in raw:
                    raise TauOnlineAdapterError(
                        f"variation must use DIMENSION=VALUE: {raw!r}"
                    )
                dimension, value = raw.split("=", 1)
                if not dimension or not value:
                    raise TauOnlineAdapterError(
                        f"variation must use DIMENSION=VALUE: {raw!r}"
                    )
                if dimension in variation:
                    raise TauOnlineAdapterError(
                        f"duplicate variation dimension: {dimension!r}"
                    )
                variation[dimension] = value
            provider = load_provider_config(
                api_file=args.api_file,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
            )
            configure_openai_compatible_environment(provider)
            execution = run_tau_online_driver_file(
                bound_plan_path=args.bound_plan,
                run_configuration_path=args.run_configuration,
                fixture_instance_id=args.fixture_instance_id,
                variation_selection=variation,
                model=args.model,
                approved_agent_call_budget=args.approved_agent_call_budget,
                seed=args.seed,
                timeout_seconds=args.timeout_seconds,
                output_path=args.output,
            )
        except (
            OSError,
            json.JSONDecodeError,
            ExecutionConfigurationError,
            RuntimeDriverError,
            TauOnlineAdapterError,
            ConfigurationError,
            ImportError,
        ) as exc:
            print(f"bound driver execution failed: {exc}", file=sys.stderr)
            return 2
        if args.summary:
            result = execution["runtime_driver"]["reachability_result"]
            print(
                json.dumps(
                    {
                        "fixture_instance_id": execution["fixture_instance_id"],
                        "reached": result["reached"],
                        "target_action_observed": result[
                            "target_action_observed"
                        ],
                        "runtime_stop_reason": execution["runtime_stop_reason"],
                        "tau_termination_reason": execution[
                            "tau_termination_reason"
                        ],
                        "observed_agent_calls": execution["budget"][
                            "observed_agent_calls"
                        ],
                        "user_model_calls": execution["budget"][
                            "user_model_calls"
                        ],
                        "oracle_classification": execution["bound_oracle"][
                            "classification"
                        ],
                        "output": str(args.output),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return 0

    if args.command == "evaluate-bound-execution":
        try:
            oracle = evaluate_bound_online_execution_file(
                bound_plan_path=args.bound_plan,
                execution_path=args.execution,
                output_path=args.output,
            )
        except (
            OSError,
            json.JSONDecodeError,
            BoundOracleError,
        ) as exc:
            print(f"bound execution evaluation failed: {exc}", file=sys.stderr)
            return 2
        if args.summary:
            print(
                json.dumps(
                    {
                        "fixture_instance_id": oracle["fixture_instance_id"],
                        "coverage_cell_id": oracle["coverage_cell_id"],
                        "expected_operation_decision": oracle[
                            "oracle_binding"
                        ]["expected_operation_decision"],
                        "actual_decision": oracle["actual_outcome"]["decision"],
                        "reachability_passed": oracle["reachability_gate"][
                            "passed"
                        ],
                        "correctness_assessed": oracle[
                            "correctness_assessed"
                        ],
                        "policy_decision_correct": oracle[
                            "policy_decision_correct"
                        ],
                        "strict_execution_passed": oracle[
                            "strict_execution_passed"
                        ],
                        "classification": oracle["classification"],
                        "output": str(args.output),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return 0

    if args.command == "generate-candidates":
        try:
            manifest = generate_candidate_files(
                args.plan,
                args.fixtures,
                args.output_dir,
                adversarial_count=args.adversarial_count,
            )
        except (
            OSError,
            json.JSONDecodeError,
            InputContractError,
            FixtureMaterializationError,
            CandidateGenerationError,
        ) as exc:
            print(f"candidate generation failed: {exc}", file=sys.stderr)
            return 2
        if args.summary:
            summary = manifest["generation_summary"]
            print(
                json.dumps(
                    {
                        "input_id": manifest["input_id"],
                        "generation_mode": summary["generation_mode"],
                        "llm_calls": summary["llm_calls"],
                        "fixture_count": summary["fixture_count"],
                        "candidate_count": summary["candidate_count"],
                        "all_semantic_invariants_passed": summary[
                            "all_semantic_invariants_passed"
                        ],
                        "manifest": str(args.output_dir / "manifest.json"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return 0

    if args.command == "execute":
        try:
            provider = load_provider_config(
                api_file=args.api_file,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
            )
            execute_manifest(
                manifest_path=args.manifest,
                workspace=args.workspace,
                output_dir=args.output_dir,
                provider=provider,
                model=args.model,
                max_steps=args.max_steps,
                approved_task_count=args.approved_task_count,
                max_workers=args.max_workers,
                seed=args.seed,
                timeout_seconds=args.timeout_seconds,
            )
        except (
            OSError,
            json.JSONDecodeError,
            ExecutionConfigurationError,
        ) as exc:
            print(f"execution setup failed: {exc}", file=sys.stderr)
            return 2
        return 0

    if args.command == "analyze-results":
        try:
            report = analyze_execution_batch(args.summary, args.output)
        except (OSError, json.JSONDecodeError, ResultAnalysisError) as exc:
            print(f"result analysis failed: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "scheduled_count": report["scheduled_count"],
                    "completed_count": report["completed_count"],
                    "harness_error_count": report["harness_error_count"],
                    "decision_outcome_correct_count": report[
                        "decision_outcome_correct_count"
                    ],
                    "decision_outcome_incorrect_count": report[
                        "decision_outcome_incorrect_count"
                    ],
                    "strict_oracle_pass_count": report["strict_oracle_pass_count"],
                    "analysis_axes": report["analysis_axes"],
                    "output": str(args.output),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "generate-adaptive-round":
        try:
            manifest = generate_adaptive_round_files(
                plan_path=args.plan,
                fixture_path=args.fixtures,
                source_manifest_path=args.source_manifest,
                analysis_path=args.analysis,
                execution_summary_path=args.execution_summary,
                workspace=args.workspace,
                output_dir=args.output_dir,
            )
        except (
            OSError,
            json.JSONDecodeError,
            AdaptiveSearchError,
            CandidateGenerationError,
        ) as exc:
            print(f"adaptive generation failed: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    **manifest["generation_summary"],
                    "guidance": manifest["guidance_path"],
                    "manifest": str(args.output_dir / "manifest.json"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "close-adaptive-round":
        try:
            closure = close_adaptive_round(
                guidance_path=args.guidance,
                analysis_path=args.analysis,
                execution_summary_path=args.execution_summary,
                workspace=args.workspace,
                output_path=args.output,
            )
        except (OSError, json.JSONDecodeError, AdaptiveSearchError) as exc:
            print(f"adaptive round closure failed: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "iteration": closure["iteration"],
                    "decision": closure["decision"],
                    "finding": closure["finding"],
                    "output": str(args.output),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
