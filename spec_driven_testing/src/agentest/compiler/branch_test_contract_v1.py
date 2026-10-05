"""Join Given, When, fixture support, and Oracle artifacts at branch level."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .artifacts import content_sha256
from .given_evidence_binding_v2 import validate_given_evidence_binding_set_v2
from .given_logical_form_v1 import validate_given_logical_form_set
from .given_when_contract_v2 import validate_given_when_contract_set_v2
from .oracle_evaluator_contract_v1 import validate_oracle_evaluator_contract_set
from .runtime_observation_binding_v1 import validate_runtime_observation_binding_set
from .semantic_judge_contract_v1 import validate_semantic_judge_contract_set
from .then_atomization import ThenAtomizationError


CONTRACT_SET_VERSION = "agentspectesting.branch-test-contract-set/v0.1"
CONTRACT_VERSION = "agentspectesting.branch-test-contract/v0.1"

FIXTURE_LOOKUP_STATUSES = frozenset(
    {"no_conditions", "matched", "makeable", "unmakeable"}
)
FIXTURE_GATE_STATUSES = frozenset(
    {"ready", "needs_fixture_construction", "blocked"}
)
GIVEN_GATE_STATUSES = frozenset({"ready", "blocked"})
WHEN_GATE_STATUSES = frozenset({"ready", "needs_runtime_binding"})
ORACLE_GATE_STATUSES = frozenset({"ready", "needs_runtime_binding", "missing"})
ADMISSION_STATUSES = frozenset({"ready_for_driver", "work_required"})
WORK_KINDS = (
    "blocked_by_given",
    "blocked_by_fixture",
    "needs_fixture_construction",
    "blocked_by_oracle",
    "needs_runtime_binding",
)
PRIMARY_STATUSES = frozenset({"ready_for_driver", *WORK_KINDS})


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThenAtomizationError(f"{path} must be an object")
    return value


def _index_unique(
    values: Sequence[Mapping[str, Any]], key: str, label: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for raw in values:
        item = _mapping(raw, f"${label}[]")
        identity = item.get(key)
        if not isinstance(identity, str) or not identity or identity in result:
            raise ThenAtomizationError(f"{label} identities are missing or overlap")
        result[identity] = item
    return result


def _same_gwt(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(left.get(key) == right.get(key) for key in ("given", "when", "then"))


def _validate_fixture_support(
    value: Any,
) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(value, list):
        raise ThenAtomizationError("fixture support must be an array")
    rows: list[dict[str, Any]] = []
    branch_ids: list[str] = []
    for index, raw in enumerate(value):
        row = deepcopy(dict(_mapping(raw, f"$fixture_support[{index}]")))
        gwt = _mapping(row.get("gwt"), f"$fixture_support[{index}].gwt")
        branch_id = gwt.get("branch_id")
        if not isinstance(branch_id, str) or not branch_id:
            raise ThenAtomizationError("fixture support branch ID is missing")
        if row.get("lookup_status") not in FIXTURE_LOOKUP_STATUSES:
            raise ThenAtomizationError("invalid fixture support lookup status")
        matches = row.get("matches")
        if not isinstance(matches, list) or row.get("n_matches") != len(matches):
            raise ThenAtomizationError("fixture support match count is inconsistent")
        if not isinstance(row.get("conditions"), list):
            raise ThenAtomizationError("fixture support conditions must be an array")
        if not isinstance(row.get("dropped"), list) or not isinstance(
            row.get("errors"), list
        ):
            raise ThenAtomizationError("fixture support diagnostics must be arrays")
        branch_ids.append(branch_id)
        rows.append(row)
    if len(branch_ids) != len(set(branch_ids)):
        raise ThenAtomizationError("fixture support branch IDs overlap")
    return rows, content_sha256(rows)


def _fixture_contract(row: Mapping[str, Any], row_index: int) -> dict[str, Any]:
    status = row["lookup_status"]
    if status in {"no_conditions", "matched"}:
        gate_status = "ready"
        mode = "existing_candidate_pool"
    elif status == "makeable":
        gate_status = "needs_fixture_construction"
        mode = "construction_required"
    else:
        gate_status = "blocked"
        mode = "unsupported_by_fixture_resolver"
    candidate_fields = sorted(
        {
            key
            for match in row.get("matches") or []
            if isinstance(match, Mapping)
            for key in match
        }
    )
    return {
        "gate_status": gate_status,
        "mode": mode,
        "lookup_status": status,
        "fixture_root": row.get("root"),
        "candidate_count": row.get("n_matches"),
        "candidate_fields": candidate_fields,
        "construction_or_lookup_conditions": deepcopy(row.get("conditions") or []),
        "dropped_conditions": deepcopy(row.get("dropped") or []),
        "diagnostics": deepcopy(row.get("errors") or []),
        "candidate_source": {
            "source_kind": "branch_fixture_support_row",
            "source_row_index": row_index,
            "branch_id": row["gwt"]["branch_id"],
            "source_row_fingerprint": content_sha256(row),
            "selection_policy": "defer_concrete_candidate_selection_to_driver_binder",
        },
    }


def _oracle_contract(
    branch_id: str,
    runtime_branch: Mapping[str, Any] | None,
    evaluator_branch: Mapping[str, Any] | None,
    semantic_contracts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if runtime_branch is None and evaluator_branch is None:
        return {
            "gate_status": "missing",
            "runtime_observation_bindings": [],
            "evaluator_contracts": [],
            "semantic_judge_contracts": [],
            "checks": [],
        }
    if runtime_branch is None or evaluator_branch is None:
        raise ThenAtomizationError(
            f"Oracle runtime/evaluator branch coverage differs for {branch_id}"
        )
    runtime_bindings = _index_unique(
        runtime_branch.get("bindings") or [], "binding_id", "runtime bindings"
    )
    evaluators = _index_unique(
        evaluator_branch.get("evaluator_contracts") or [],
        "binding_id",
        "evaluator contracts",
    )
    if set(runtime_bindings) != set(evaluators):
        raise ThenAtomizationError(f"Oracle binding/evaluator membership differs for {branch_id}")
    semantic_by_binding = _index_unique(
        semantic_contracts, "binding_id", "semantic judge contracts"
    )
    if not set(semantic_by_binding) <= set(runtime_bindings):
        raise ThenAtomizationError(
            f"semantic judge contract does not belong to Oracle branch {branch_id}"
        )
    checks = []
    fully_ready = True
    for binding_id in sorted(runtime_bindings):
        binding = runtime_bindings[binding_id]
        evaluator = evaluators[binding_id]
        if evaluator.get("source_binding_fingerprint") != binding.get(
            "binding_fingerprint"
        ):
            raise ThenAtomizationError(f"Oracle check lineage mismatch for {binding_id}")
        mechanically_ready = (
            binding.get("binding_status") == "bound"
            and evaluator.get("evaluator_status") == "executable"
        )
        semantic = semantic_by_binding.get(binding_id)
        semantically_ready = (
            semantic is not None
            and semantic.get("contract_status") == "ready"
            and binding.get("binding_status") == "semantic_deferred"
            and evaluator.get("evaluator_status") == "deferred"
            and semantic.get("source_binding_fingerprint")
            == binding.get("binding_fingerprint")
            and semantic.get("source_evaluator_contract_fingerprint")
            == evaluator.get("evaluator_contract_fingerprint")
        )
        ready = mechanically_ready or semantically_ready
        fully_ready = fully_ready and ready
        checks.append(
            {
                "binding_id": binding_id,
                "requirement_id": binding.get("requirement_id"),
                "check_status": "ready" if ready else "needs_runtime_binding",
                "evaluation_mode": (
                    "mechanical"
                    if mechanically_ready
                    else "semantic_judge"
                    if semantically_ready
                    else "unavailable"
                ),
                "runtime_observation_binding": deepcopy(binding),
                "evaluator_contract": deepcopy(evaluator),
                "semantic_judge_contract": deepcopy(semantic),
            }
        )
    return {
        "gate_status": "ready" if fully_ready else "needs_runtime_binding",
        "runtime_observation_bindings": [
            deepcopy(runtime_bindings[key]) for key in sorted(runtime_bindings)
        ],
        "evaluator_contracts": [deepcopy(evaluators[key]) for key in sorted(evaluators)],
        "semantic_judge_contracts": [
            deepcopy(semantic_by_binding[key]) for key in sorted(semantic_by_binding)
        ],
        "checks": checks,
    }


def _required_work(
    given_status: str, when_status: str, fixture_status: str, oracle_status: str
) -> list[str]:
    work = []
    if given_status != "ready":
        work.append("blocked_by_given")
    if fixture_status == "blocked":
        work.append("blocked_by_fixture")
    elif fixture_status == "needs_fixture_construction":
        work.append("needs_fixture_construction")
    if oracle_status == "missing":
        work.append("blocked_by_oracle")
    elif oracle_status == "needs_runtime_binding":
        work.append("needs_runtime_binding")
    if when_status != "ready" and "needs_runtime_binding" not in work:
        work.append("needs_runtime_binding")
    return work


def _summary(contracts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    admission = Counter(item["admission_status"] for item in contracts)
    primary = Counter(item["primary_status"] for item in contracts)
    work = Counter(kind for item in contracts for kind in item["required_work"])
    gates = {
        "given": Counter(item["given_contract"]["gate_status"] for item in contracts),
        "when": Counter(item["when_contract"]["gate_status"] for item in contracts),
        "fixture": Counter(item["fixture_contract"]["gate_status"] for item in contracts),
        "oracle": Counter(item["oracle_contract"]["gate_status"] for item in contracts),
    }
    return {
        "branch_count": len(contracts),
        "admission_status_counts": dict(sorted(admission.items())),
        "primary_status_counts": dict(sorted(primary.items())),
        "required_work_counts": dict(sorted(work.items())),
        "gate_status_counts": {
            name: dict(sorted(counts.items())) for name, counts in gates.items()
        },
    }


def compile_branch_test_contracts(
    given_when_contract_set: Mapping[str, Any],
    given_logical_form_set: Mapping[str, Any],
    given_binding_set: Mapping[str, Any],
    fixture_support: Any,
    runtime_observation_binding_set: Mapping[str, Any],
    oracle_evaluator_contract_set: Mapping[str, Any],
    semantic_judge_contract_set: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile a lossless branch-level readiness and handoff contract set."""

    given_when = validate_given_when_contract_set_v2(given_when_contract_set)
    logical_forms = validate_given_logical_form_set(given_logical_form_set)
    given_bindings = validate_given_evidence_binding_set_v2(given_binding_set)
    runtime_bindings = validate_runtime_observation_binding_set(
        runtime_observation_binding_set
    )
    evaluators = validate_oracle_evaluator_contract_set(
        oracle_evaluator_contract_set
    )
    semantic_judges = validate_semantic_judge_contract_set(
        semantic_judge_contract_set
    )
    fixture_rows, fixture_fingerprint = _validate_fixture_support(fixture_support)

    if evaluators.get("source_binding_set_fingerprint") != runtime_bindings.get(
        "binding_set_fingerprint"
    ):
        raise ThenAtomizationError("Oracle evaluator/runtime binding lineage mismatch")
    if (
        semantic_judges.get("source_binding_set_fingerprint")
        != runtime_bindings.get("binding_set_fingerprint")
        or semantic_judges.get("source_evaluator_contract_set_fingerprint")
        != evaluators.get("evaluator_contract_set_fingerprint")
    ):
        raise ThenAtomizationError("semantic judge/Oracle lineage mismatch")

    gw_index = _index_unique(given_when["contracts"], "branch_id", "Given/When")
    logical_index = _index_unique(
        logical_forms["forms"], "branch_id", "Given logical forms"
    )
    given_branch_index = _index_unique(
        given_bindings["branches"], "branch_id", "Given binding branches"
    )
    given_requirement_index = _index_unique(
        given_bindings["bindings"], "requirement_id", "Given bindings"
    )
    fixture_index = {
        row["gwt"]["branch_id"]: (index, row)
        for index, row in enumerate(fixture_rows)
    }
    runtime_index = _index_unique(
        runtime_bindings.get("branches") or [], "branch_id", "Oracle runtime branches"
    )
    evaluator_index = _index_unique(
        evaluators.get("branches") or [], "branch_id", "Oracle evaluator branches"
    )
    semantic_index: dict[str, list[Mapping[str, Any]]] = {}
    for semantic in semantic_judges.get("contracts") or []:
        semantic_index.setdefault(semantic["branch_id"], []).append(semantic)
    branch_ids = set(gw_index)
    if (
        set(logical_index) != branch_ids
        or set(given_branch_index) != branch_ids
        or set(fixture_index) != branch_ids
    ):
        raise ThenAtomizationError(
            "branch coverage differs across Given/When, logical form, Given binding, and fixture support"
        )
    if not set(runtime_index) <= branch_ids or set(evaluator_index) != set(runtime_index):
        raise ThenAtomizationError("Oracle branch coverage is not a consistent branch subset")
    if not set(semantic_index) <= set(runtime_index):
        raise ThenAtomizationError("semantic judge branch coverage exceeds Oracle coverage")

    contracts = []
    for branch_id in [item["branch_id"] for item in given_when["contracts"]]:
        gw = gw_index[branch_id]
        logical_form = logical_index[branch_id]
        given_branch = given_branch_index[branch_id]
        fixture_row_index, fixture_row = fixture_index[branch_id]
        if not _same_gwt(gw["gwt"], fixture_row["gwt"]):
            raise ThenAtomizationError(f"fixture/GWT branch content mismatch for {branch_id}")
        if not _same_gwt(gw["gwt"], logical_form["gwt"]):
            raise ThenAtomizationError(f"logical-form/GWT branch content mismatch for {branch_id}")
        requirement_ids = given_branch.get("requirement_ids") or []
        requirements = []
        for requirement_id in requirement_ids:
            requirement = given_requirement_index.get(requirement_id)
            if requirement is None or requirement.get("branch_id") != branch_id:
                raise ThenAtomizationError(f"Given binding membership mismatch for {branch_id}")
            if requirement.get("when") != gw["gwt"]["when"]:
                raise ThenAtomizationError(f"Given/When lineage mismatch for {branch_id}")
            requirements.append(deepcopy(requirement))
        active_condition_ids = logical_form["branch_alignment"][
            "active_condition_ids"
        ]
        if [item["condition_id"] for item in requirements] != active_condition_ids:
            raise ThenAtomizationError(
                f"logical-form/Given binding membership mismatch for {branch_id}"
            )

        trigger = _mapping(gw.get("when_trigger"), f"$.contracts[{branch_id}].when_trigger")
        when_ready = (
            isinstance(trigger.get("source_text"), str)
            and bool(trigger["source_text"].strip())
            and trigger.get("binding_status") == "requires_driver_binding"
        )
        when_contract = {
            "gate_status": "ready" if when_ready else "needs_runtime_binding",
            "user_goal": gw["gwt"]["when"],
            "source_trigger": deepcopy(trigger),
            "realization_requirement": "driver_must_realize_trigger_without_changing_business_semantics",
        }
        fixture_contract = _fixture_contract(fixture_row, fixture_row_index)
        oracle_contract = _oracle_contract(
            branch_id,
            runtime_index.get(branch_id),
            evaluator_index.get(branch_id),
            semantic_index.get(branch_id, []),
        )
        given_contract = {
            "gate_status": given_branch["binding_status"],
            "logical_form": deepcopy(logical_form),
            "requirement_ids": deepcopy(requirement_ids),
            "evidence_bindings": requirements,
            "source_branch_binding_fingerprint": given_branch[
                "branch_binding_fingerprint"
            ],
        }
        work = _required_work(
            given_contract["gate_status"],
            when_contract["gate_status"],
            fixture_contract["gate_status"],
            oracle_contract["gate_status"],
        )
        primary = next((kind for kind in WORK_KINDS if kind in work), "ready_for_driver")
        contract = {
            "schema_version": CONTRACT_VERSION,
            "branch_test_contract_id": f"{branch_id}::BTC01",
            "branch_id": branch_id,
            "spec_id": gw["spec_id"],
            "kind": gw["kind"],
            "origin": gw["origin"],
            "gwt_index": gw["gwt_index"],
            "gwt": deepcopy(gw["gwt"]),
            "source_rule_context": deepcopy(gw["source_rule_context"]),
            "given_contract": given_contract,
            "when_contract": when_contract,
            "fixture_contract": fixture_contract,
            "oracle_contract": oracle_contract,
            "required_work": work,
            "primary_status": primary,
            "admission_status": "ready_for_driver" if not work else "work_required",
            "source_given_when_contract_fingerprint": gw[
                "given_when_contract_fingerprint"
            ],
        }
        contract["branch_test_contract_fingerprint"] = content_sha256(contract)
        contracts.append(contract)

    result = {
        "schema_version": CONTRACT_SET_VERSION,
        "contracts": contracts,
        "summary": _summary(contracts),
        "source_given_when_contract_set_fingerprint": given_when[
            "given_when_contract_set_fingerprint"
        ],
        "source_given_logical_form_set_fingerprint": logical_forms[
            "logical_form_set_fingerprint"
        ],
        "source_given_binding_set_fingerprint": given_bindings[
            "binding_set_fingerprint"
        ],
        "source_fixture_support_fingerprint": fixture_fingerprint,
        "source_runtime_observation_binding_set_fingerprint": runtime_bindings[
            "binding_set_fingerprint"
        ],
        "source_oracle_evaluator_contract_set_fingerprint": evaluators[
            "evaluator_contract_set_fingerprint"
        ],
        "source_semantic_judge_contract_set_fingerprint": semantic_judges[
            "semantic_judge_contract_set_fingerprint"
        ],
    }
    result["branch_test_contract_set_fingerprint"] = content_sha256(result)
    return validate_branch_test_contract_set(result)


def validate_branch_test_contract_set(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(_mapping(value, "$branch_test_contract_set")))
    supplied = result.pop("branch_test_contract_set_fingerprint", None)
    if (
        result.get("schema_version") != CONTRACT_SET_VERSION
        or supplied != content_sha256(result)
    ):
        raise ThenAtomizationError("invalid branch test contract set")
    contracts = result.get("contracts")
    if not isinstance(contracts, list):
        raise ThenAtomizationError("branch test contracts must be an array")
    identities = []
    for raw in contracts:
        item = deepcopy(dict(_mapping(raw, "$.contracts[]")))
        fingerprint = item.pop("branch_test_contract_fingerprint", None)
        if (
            item.get("schema_version") != CONTRACT_VERSION
            or fingerprint != content_sha256(item)
        ):
            raise ThenAtomizationError("invalid branch test contract")
        if item.get("admission_status") not in ADMISSION_STATUSES:
            raise ThenAtomizationError("invalid branch test admission status")
        if item.get("primary_status") not in PRIMARY_STATUSES:
            raise ThenAtomizationError("invalid branch test primary status")
        work = item.get("required_work")
        if not isinstance(work, list) or any(kind not in WORK_KINDS for kind in work):
            raise ThenAtomizationError("invalid branch test required work")
        if len(work) != len(set(work)):
            raise ThenAtomizationError("branch test required work overlaps")
        branch_id = item.get("branch_id")
        if (
            not isinstance(branch_id, str)
            or item.get("branch_test_contract_id") != f"{branch_id}::BTC01"
        ):
            raise ThenAtomizationError("invalid branch test contract identity")
        given_gate = item["given_contract"].get("gate_status")
        when_gate = item["when_contract"].get("gate_status")
        fixture_gate = item["fixture_contract"].get("gate_status")
        oracle_gate = item["oracle_contract"].get("gate_status")
        if given_gate not in GIVEN_GATE_STATUSES:
            raise ThenAtomizationError("invalid branch Given gate status")
        if when_gate not in WHEN_GATE_STATUSES:
            raise ThenAtomizationError("invalid branch When gate status")
        expected_primary = next(
            (kind for kind in WORK_KINDS if kind in work), "ready_for_driver"
        )
        expected_admission = "ready_for_driver" if not work else "work_required"
        if item["primary_status"] != expected_primary or item[
            "admission_status"
        ] != expected_admission:
            raise ThenAtomizationError("branch test status derivation is inconsistent")
        if fixture_gate not in FIXTURE_GATE_STATUSES:
            raise ThenAtomizationError("invalid branch fixture gate status")
        if oracle_gate not in ORACLE_GATE_STATUSES:
            raise ThenAtomizationError("invalid branch Oracle gate status")
        if work != _required_work(given_gate, when_gate, fixture_gate, oracle_gate):
            raise ThenAtomizationError("branch test required work/gate derivation differs")
        logical_form = _mapping(
            item["given_contract"].get("logical_form"), "$.given_contract.logical_form"
        )
        if (
            logical_form.get("branch_id") != branch_id
            or logical_form.get("logic_status") != "resolved"
        ):
            raise ThenAtomizationError("branch test logical form is not closed")
        identities.append(branch_id)
    if len(identities) != len(set(identities)):
        raise ThenAtomizationError("branch test contract identities overlap")
    if result.get("summary") != _summary(contracts):
        raise ThenAtomizationError("branch test contract summary mismatch")
    result["branch_test_contract_set_fingerprint"] = supplied
    return result
