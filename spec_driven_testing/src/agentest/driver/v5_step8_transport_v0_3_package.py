"""prepare_local_runtime variant that accepts the Step 7 v3 package (see
docs/oracle_requirement_pipeline_v0_7.md section 53/54).

v5_step8_transport_v11.py is hash-pinned by a real historical acceptance
record (outputs/v5_step8_transport_v0_11/acceptance.json includes its own
file's sha256 in its file_sha256 map) -- editing it directly would repeat
the exact mistake already made and corrected once for
v5_step8_environment_v1.py (see section 54.3). This module does NOT touch
v11; it only reimplements prepare_local_runtime, which is the one function
tied to environment_v1.load_prepared_package's narrower schema check, and
reuses everything else from v11 unchanged (CombinedBudget, make_session,
run_transport, TauToolBridge, provider_history, CompletionBridges all still
come from v11 and are re-exported here for convenience).
"""
from copy import deepcopy
from hashlib import sha256
import inspect
from pathlib import Path

from agentest.compiler.artifacts import content_sha256
from .v5_step8_environment_v1 import EnvironmentPreparationError
from .v5_step8_environment_v2 import load_prepared_package
from .v5_step8_transport_v11 import (  # noqa: F401 -- re-exported, unchanged
    CombinedBudget, CompletionBridges, TauToolBridge, make_session, provider_history, run_transport,
)


def create_environment_for_domain(database, *, domain="airline"):
    """Domain-parametrized sibling of v5_step8_environment_v1.create_environment
    (docs/agentcoveragetesting_reuse_log.md section 41). That function is
    hash-pinned by a real historical acceptance record
    (outputs/v5_step8_environment_v0_1/acceptance.json) -- editing it
    directly broke 3 real, unrelated tests that assert its bytes are
    unchanged, so this is a NEW function in this (unpinned) module instead,
    same discipline as this module's own docstring already establishes for
    not touching v11.

    telecom's real get_environment needs TWO databases (TelecomDB + a
    separate TelecomUserDB, unlike airline/retail's single-DB shape), so its
    `database` payload is a dict with `db`/`user_db` keys rather than the
    raw DB fields directly.
    """
    payload = deepcopy(database)
    if domain == "airline":
        from tau2.domains.airline.data_model import FlightDB
        from tau2.domains.airline.environment import get_environment
        return get_environment(db=FlightDB.model_validate(payload))
    if domain == "retail":
        from tau2.domains.retail.data_model import RetailDB
        from tau2.domains.retail.environment import get_environment
        return get_environment(db=RetailDB.model_validate(payload))
    if domain == "telecom":
        from tau2.domains.telecom.data_model import TelecomDB
        from tau2.domains.telecom.user_data_model import TelecomUserDB
        from tau2.domains.telecom.environment import get_environment
        return get_environment(
            db=TelecomDB.model_validate(payload["db"]),
            user_db=TelecomUserDB.model_validate(payload["user_db"]),
        )
    raise ValueError(f"unsupported domain: {domain!r}")


def prepare_local_runtime(root, package_path, branch_id, *, domain="airline"):
    """Identical to v5_step8_transport_v11.prepare_local_runtime, except it
    loads the package via v5_step8_environment_v2.load_prepared_package
    (accepts both the v0.1 and the v0.3-generic-tool-call-synthesis package
    schema_version) instead of v1's loader.

    `domain` (docs/agentcoveragetesting_reuse_log.md section 41) selects the
    real tau2 domain via create_environment_for_domain (this module's own
    function, not v1's hash-pinned create_environment) -- default "airline"
    keeps every existing airline call site unchanged."""
    package, dependencies = load_prepared_package(root, package_path)
    matches = [c for c in package['cases'] if c['branch_id'] == branch_id]
    if len(matches) != 1 or not matches[0]['step7_prepared']:
        raise EnvironmentPreparationError('unique_prepared_case_required')
    from .v5_step8_environment_v1 import observe_environment, check_environment
    env = create_environment_for_domain(dependencies['database_snapshot'], domain=domain)
    identity = check_environment(package, dependencies,
        observe_environment(env, dependencies['tool_schema_snapshot']['requested_tools']))
    if identity['status'] != 'matched_with_limits':
        raise EnvironmentPreparationError('runtime_identity_mismatch')
    from tau2.agent import llm_agent
    system = llm_agent.SYSTEM_PROMPT.format(agent_instruction=llm_agent.AGENT_INSTRUCTION,
                                          domain_policy=env.get_policy())
    schemas = [deepcopy(t.openai_schema) for t in env.get_tools()]
    source = Path(inspect.getfile(llm_agent)).resolve()
    metadata = {'source_package_fingerprint': package['package_fingerprint'],
        'source_case_fingerprint': matches[0]['case_fingerprint'], 'environment_identity': identity,
        'target_system_prompt_sha256': sha256(system.encode()).hexdigest(),
        'target_tool_schemas_fingerprint': content_sha256(schemas),
        'target_instruction_source': {'path': str(source), 'sha256': sha256(source.read_bytes()).hexdigest()},
        'transport_implementation': 'direct_openai_compatible_not_tau2_orchestrator',
        'online_execution_authorized': False}
    return deepcopy(matches[0]), TauToolBridge(env), system, schemas, metadata
