"""Step 8 environment/task preflight, without Agent or business-tool execution."""

from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
import inspect
import os
from pathlib import Path
from unittest.mock import patch

from agentest.compiler.artifacts import content_sha256
from agentest.compiler.v5_given_when_preparation_v1 import verify_fingerprint
from .v5_data_sources_v1 import read_json
from .v5_object_references_v1 import seal
from .v5_step7_package_v1 import user_view


class EnvironmentPreparationError(ValueError):
    """A package or local runtime does not meet its declared identity contract."""


@contextmanager
def offline_environment_access():
    """Block networking even during optional tau2/LiteLLM import side effects."""
    with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}), \
         patch("socket.socket.connect", side_effect=EnvironmentPreparationError("network_disabled")), \
         patch("socket.socket.connect_ex", side_effect=EnvironmentPreparationError("network_disabled")), \
         patch("socket.create_connection", side_effect=EnvironmentPreparationError("network_disabled")):
        yield


def load_prepared_package(root, package_path):
    root = Path(root).resolve()
    package = read_json(package_path)
    verify_fingerprint(package, "package_fingerprint")
    if package.get("schema_version") != "agentspectesting.v5-step7-package/v0.1":
        raise EnvironmentPreparationError("unsupported_package_schema")
    cases = package["cases"]
    if len({c["branch_id"] for c in cases}) != len(cases):
        raise EnvironmentPreparationError("duplicate_branch")
    for case in cases:
        verify_fingerprint(case, "case_fingerprint")
    if package["environment_contract"]["database_reset_policy"] != "fresh_full_snapshot_before_each_case":
        raise EnvironmentPreparationError("unsupported_reset_policy")
    dependencies = {}
    for name in ("database_snapshot", "tool_schema_snapshot", "clock", "policy"):
        source = package["artifact_sources"][name]
        path = (root / source["workspace_relative_path"]).resolve()
        if not path.is_relative_to(root):
            raise EnvironmentPreparationError("dependency_outside_workspace")
        if sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise EnvironmentPreparationError("dependency_file_changed:" + name)
        dependencies[name] = read_json(path)
    expected = package["environment_contract"]
    if (content_sha256(dependencies["database_snapshot"]) != expected["database_fingerprint"]
            or content_sha256(dependencies["tool_schema_snapshot"]) != expected["tool_schema_fingerprint"]
            or dependencies["clock"] != expected["reference_clock"]
            or dependencies["policy"] != package["policy"]):
        raise EnvironmentPreparationError("dependency_content_identity_mismatch")
    return package, dependencies


def materialize_task(case):
    """Only the initial request enters Task transport; no hidden Oracle or facts."""
    view = user_view(case)
    return {
        "id": "v5-baseline::" + case["branch_id"],
        "description": {"purpose": "Ordinary functional baseline execution.",
                        "relevant_policies": None,
                        "notes": "A factual Driver supplies user messages; no feedback search."},
        "user_scenario": {"persona": None, "instructions": {
            "domain": "airline", "reason_for_call": view["initial_user_message"],
            "known_info": None, "unknown_info": None,
            "task_instructions": "Transport metadata only; use factual Driver messages."}},
        # Reset by replacing the complete environment DB, not a partial patch.
        "initial_state": {"initialization_data": None, "initialization_actions": None,
                          "message_history": None},
        "evaluation_criteria": {"actions": [], "communicate_info": [],
                                "nl_assertions": [], "reward_basis": []},
        "annotations": None,
    }


def create_environment(database):
    """Fresh validated copy per case. No default database or update_db merge."""
    from tau2.domains.airline.data_model import FlightDB
    from tau2.domains.airline.environment import get_environment
    return get_environment(db=FlightDB.model_validate(deepcopy(database)))


def observe_environment(environment, required_names):
    """Read values from the created environment, never substitute expectations."""
    tools = environment.tools
    records = {t.name: {"description": "\n\n".join(x for x in (t.short_desc, t.long_desc) if x),
                        "parameters": t.params.model_json_schema()}
               for t in environment.get_tools()}
    from tau2.environment.tool import Tool
    from tau2.environment.toolkit import ToolKitBase
    classes = (type(tools.db), type(tools), Tool, ToolKitBase)
    source_files = {str(Path(inspect.getfile(cls)).resolve()):
                    sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest() for cls in classes}
    return {
        "database_fingerprint": content_sha256(tools.db.model_dump(mode="json")),
        "database_counts": {k: len(v) for k, v in tools.db.model_dump(mode="json").items()},
        "reference_clock": tools._get_datetime(),
        "required_tool_records": {name: records[name] for name in required_names if name in records},
        "all_tool_names": sorted(records), "all_tool_records_fingerprint": content_sha256(records),
        "implementation_source_files": source_files,
        "policy_sha256": sha256(environment.get_policy().encode("utf-8")).hexdigest(),
    }


def check_environment(package, dependencies, observed):
    expected = package["environment_contract"]
    snapshot = dependencies["tool_schema_snapshot"]
    checks = {
        "full_database_round_trip": observed["database_fingerprint"] == expected["database_fingerprint"],
        "clock_value": observed["reference_clock"] == expected["reference_clock"]["value"],
        "required_tool_records": observed["required_tool_records"] == snapshot["tools"],
        "schema_source_files": all(observed["implementation_source_files"].get(r["path"]) == r["sha256"]
                                   for r in snapshot["provenance"]["source_files"]),
        "clock_provider_source": observed["implementation_source_files"].get(
            expected["reference_clock"]["provider"]["path"]) == expected["reference_clock"]["provider"]["sha256"],
    }
    return {"status": "matched_with_limits" if all(checks.values()) else "mismatch",
            "checks": checks, "observed": observed,
            "remaining_limits": [
                "Historical snapshot producer version remains unknown; current model lossless round-trip is checked.",
                "Clock is the same naive literal/provider; no timezone meaning is inferred.",
                "Current policy and full tool catalog are recorded, not certified as the original spec extraction environment.",
            ], "online_execution_authorized": False}


def preflight(package, dependencies, *, environment_factory=create_environment,
              observer=observe_environment, task_validator=None):
    verify_fingerprint(package, "package_fingerprint")
    if task_validator is None:
        from tau2.data_model.tasks import Task
        task_validator = Task.model_validate
    rows = []
    for case in package["cases"]:
        verify_fingerprint(case, "case_fingerprint")
        row = {"branch_id": case["branch_id"], "case_fingerprint": case["case_fingerprint"]}
        if not case["step7_prepared"]:
            rows.append({**row, "status": "blocked_upstream", "blockers": deepcopy(case["blockers"]),
                         "environment_created": False, "task": None})
            continue
        try:
            task = materialize_task(case)
            task_validator(task)
            # A brand new environment is constructed even when cases share a user.
            environment = environment_factory(deepcopy(dependencies["database_snapshot"]))
            observed = observer(environment, dependencies["tool_schema_snapshot"]["requested_tools"])
            identity = check_environment(package, dependencies, observed)
            rows.append({**row, "status": "environment_preflight_passed_with_limits" if identity["status"] == "matched_with_limits" else "environment_mismatch",
                         "environment_created": True, "task_schema_valid": True, "task": task,
                         "environment_identity": identity})
        except Exception as exc:
            # Validation exceptions can contain private records; never save repr/str.
            rows.append({**row, "status": "environment_preflight_error", "error_type": type(exc).__name__,
                         "task": None})
    return seal({"schema_version": "agentspectesting.v5-step8-environment-preflight/v0.1",
        "source_package_fingerprint": package["package_fingerprint"], "rows": rows,
        "summary": {"branches": len(rows),
                    "environment_preflight_passed_with_limits": sum(r["status"] == "environment_preflight_passed_with_limits" for r in rows),
                    "blocked_upstream": sum(r["status"] == "blocked_upstream" for r in rows),
                    "environment_failures": sum(r["status"] in ("environment_mismatch", "environment_preflight_error") for r in rows)},
        "target_agent_calls": 0, "business_tool_calls": 0, "endpoint_configuration_read": False,
        "online_execution_performed": False, "dialogue_adapter_implemented": False,
        "oracle_evaluation_performed": False, "feedback_search_performed": False,
        "scope": "environment and task preflight, not completed Step 8 or observed When"}, "preflight_fingerprint")


def render_preflight(result):
    rows = ["# Step 8A environment and task entry acceptance", "", "Only creates isolated local environments; does not run the Agent, business actions or the Oracle.", "",
            "| Branch | Status |", "| --- | --- |"]
    rows.extend(f"| {r['branch_id']} | {r['status']} |" for r in result["rows"])
    return "\n".join(rows + ["", "Real dialogue adaptation, call budgeting and trajectory recording are still pending; this result cannot authorize online runs.", ""])
