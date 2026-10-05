"""Step 8 environment/task preflight loader that also accepts the Step 7 v3
package (see docs/oracle_requirement_pipeline_v0_7.md section 53).

v5_step8_environment_v1.py is hash-pinned by real historical acceptance
records (outputs/v5_step8_environment_v0_1/acceptance.json and others that
include its own file's sha256 in their file_sha256 map) -- editing it
directly broke 3 real regression tests that exist specifically to catch
unreviewed drift in already-audited files. This module does NOT touch v1;
it only reimplements the one function that needs a wider check
(load_prepared_package's schema_version literal), reusing everything else
from v1 unchanged (create_environment, observe_environment,
check_environment, preflight, materialize_task, offline_environment_access,
render_preflight all still come from v1 and are re-exported here for
convenience)."""
from hashlib import sha256
from pathlib import Path

from agentest.compiler.v5_given_when_preparation_v1 import verify_fingerprint
from .v5_data_sources_v1 import read_json
from .v5_step8_environment_v1 import (  # noqa: F401 -- re-exported, unchanged
    EnvironmentPreparationError, check_environment, create_environment, materialize_task,
    observe_environment, offline_environment_access, preflight, render_preflight,
)

# v0.1 is both the original 6-branch pilot package and the 155-branch v0_2
# package (compile_package plus hand-written adapters); v0.3 is the
# 155-branch package built entirely by the generic tool-call synthesis
# pipeline. Both share the exact same environment_contract/artifact_sources
# shape v1's load_prepared_package actually reads -- only the schema_version
# string itself differs.
SUPPORTED_PACKAGE_SCHEMA_VERSIONS = {
    "agentspectesting.v5-step7-package/v0.1",
    "agentspectesting.v5-step7-package/v0.3-generic-tool-call-synthesis",
}


def load_prepared_package(root, package_path):
    """Identical to v5_step8_environment_v1.load_prepared_package, except
    the schema_version check accepts SUPPORTED_PACKAGE_SCHEMA_VERSIONS
    instead of only the single original literal."""
    root = Path(root).resolve()
    package = read_json(package_path)
    verify_fingerprint(package, "package_fingerprint")
    if package.get("schema_version") not in SUPPORTED_PACKAGE_SCHEMA_VERSIONS:
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
    from agentest.compiler.artifacts import content_sha256
    if (content_sha256(dependencies["database_snapshot"]) != expected["database_fingerprint"]
            or content_sha256(dependencies["tool_schema_snapshot"]) != expected["tool_schema_fingerprint"]
            or dependencies["clock"] != expected["reference_clock"]
            or dependencies["policy"] != package["policy"]):
        raise EnvironmentPreparationError("dependency_content_identity_mismatch")
    return package, dependencies
