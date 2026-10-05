"""Consume reviewed lifecycle labels as binding constraints, never as bound facts."""
from collections import Counter
from copy import deepcopy

from ..compiler.artifacts import content_sha256
from ..compiler.v5_given_when_preparation_v1 import verify_fingerprint
from .v5_role_questions_v1 import prepare_questions, validate_answer, seal


SCHEMA = "agentspectesting.v5-role-handoff/v0.1"
TARGET_ORIGINS = {"create_new": "future_result", "use_existing": "snapshot_object",
                  "not_determined": None}


def _index(rows, key):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate source row")
    return result


def compile_handoff(contracts, references, questions, manifest, review, answers):
    """All source associations must agree; review is a trusted local audit input.

    A matching excerpt or hash does not independently prove semantic correctness.
    This consumes the recorded review, not a second semantic classification.
    """
    if questions != prepare_questions(contracts, references):
        raise ValueError("questions do not describe current contracts/references")
    fp = questions["bundle_fingerprint"]
    if manifest["source_bundle_fingerprint"] != fp or review["source_bundle_fingerprint"] != fp:
        raise ValueError("review/run source mismatch")
    if review["schema_version"] != "agentspectesting.step7c-semantic-review/v0.1":
        raise ValueError("unsupported review schema")
    jobs = _index(questions["jobs"], "job_id")
    records = _index(manifest["records"], "job_id")
    audited = _index(review["rows"], "job_id")
    if not (jobs.keys() == records.keys() == audited.keys() == answers.keys()):
        raise ValueError("answer/review membership mismatch")
    if manifest["status"] != "completed_semantic_review_required" or manifest["role_bindings_applied"] is not False:
        raise ValueError("run is incomplete or already applied")
    if review["role_bindings_applied"] is not False:
        raise ValueError("review cannot authorize binding")
    ref_rows = _index(references["rows"], "branch_id")
    rows = []
    for job in questions["jobs"]:
        name, bid = job["job_id"], job["branch_id"]
        record, audit, answer = records[name], audited[name], answers[name]
        validate_answer(job["packet"], answer)
        if record["branch_id"] != bid or audit["branch_id"] != bid:
            raise ValueError("branch association mismatch")
        if record["status"] != "interface_valid_review_required" or record["finish_reason"] != "stop":
            raise ValueError("answer did not complete validation")
        if record["decision"] != answer["decision"] or audit["answer"] != answer:
            raise ValueError("review/answer disagreement")
        if audit["packet_fingerprint"] != job["packet"]["packet_fingerprint"] or audit["source_texts"] != job["packet"]["texts"]:
            raise ValueError("review text evidence source mismatch")
        if audit["semantic_review"] != "accepted_for_this_question" or audit["role_binding_authorized"] is not False:
            raise ValueError("semantic review missing or exceeds its authority")
        if not isinstance(audit.get("review_reason"), str) or not audit["review_reason"].strip():
            raise ValueError("missing review rationale")
        reference = ref_rows[bid]
        verify_fingerprint(reference, "reference_row_fingerprint")
        decision = answer["decision"]
        row = {
            "branch_id": bid,
            "source_contract_fingerprint": job["source_contract_fingerprint"],
            "source_reference_row_fingerprint": reference["reference_row_fingerprint"],
            "source_review_row_fingerprint": content_sha256(audit),
            "lifecycle": {"decision": decision, "evidence": deepcopy(answer["evidence"])},
            "target_constraint": {
                "required_origin": TARGET_ORIGINS[decision],
                "object_identity": None, "object_type": None,
                "state": "unresolved" if decision == "not_determined" else "origin_only_known",
                "existing_candidate_is_not_automatically_the_target": True},
            "candidate_pool_reference": {"reference_row_fingerprint": reference["reference_row_fingerprint"],
                "candidate_count": reference["summary"]["supplied_pairs"],
                "role_assignment": "none", "preserve_supplied_pairs": True},
            "role_bindings": {"requester": None, "operation_target": None, "context_anchor": None},
            "source_user_requirements": deepcopy(job["packet"]["texts"]),
            "user_view_policy": {"allowed_fields": ["request", "when"],
                "bound_fact_grants": [], "complete_user_input": False},
            "pending_driver_binding_names": deepcopy(reference["pending_driver_binding_names"]),
            "remaining_work": ["requester_and_object_type_scope", "candidate_selection_and_given_verification",
                "conversation_requirements_and_when_realization", "verified_user_fact_projection"],
            "runtime_ready": False,
        }
        rows.append(seal(row, "role_row_fingerprint"))
    return seal({"schema_version": SCHEMA,
        "source_question_bundle_fingerprint": fp,
        "source_review_fingerprint": content_sha256(review),
        "source_manifest_fingerprint": content_sha256(manifest),
        "source_reference_set_fingerprint": references["reference_set_fingerprint"],
        "source_compatibility_limits": deepcopy(references["source_compatibility_limits"]),
        "rows": rows,
        "summary": {"branches": len(rows), "decision_counts": dict(Counter(r["lifecycle"]["decision"] for r in rows)),
            "bound_roles": 0, "runtime_ready_branches": 0, "external_calls": 0},
        "scope": "reviewed semantic constraints and source-only field projection; no selection, concrete test input or feedback search"},
        "handoff_fingerprint")


def check_target_origin(row, origin):
    """Necessary lifecycle guard only; a compatible origin never proves binding."""
    verify_fingerprint(row, "role_row_fingerprint")
    if origin not in ("future_result", "snapshot_object"):
        raise ValueError("unsupported target origin")
    required = TARGET_ORIGINS[row["lifecycle"]["decision"]]
    if required is None:
        return {"status": "blocked", "reason": "lifecycle_not_determined", "binding_authorized": False}
    if required != origin:
        return {"status": "rejected", "reason": "target_origin_conflict", "binding_authorized": False}
    return {"status": "compatible", "reason": "other_binding_checks_still_required", "binding_authorized": False}


def project_source_requirements(row):
    """Return two unchanged source fields, not a generated/runtime-ready user turn.

    This is field separation, not redaction of sensitive text inside source fields.
    No database grants are supported by this source-only projection.
    """
    verify_fingerprint(row, "role_row_fingerprint")
    source = row["source_user_requirements"]
    return {key: deepcopy(source[key]) for key in ("request", "when")}


def render_handoff(result):
    lines = ["# Step 7D role constraint handoff", "", "These are constraints only; objects are not yet bound. The user-side projection contains only the original request and When.", "",
             "| Branch | Semantics | Required target origin | Candidate count |", "| --- | --- | --- | ---: |"]
    for row in result["rows"]:
        lines.append(f"| {row['branch_id']} | {row['lifecycle']['decision']} | {row['target_constraint']['required_origin'] or 'undetermined'} | {row['candidate_pool_reference']['candidate_count']} |")
    return "\n".join(lines + ["", "requester, operation_target and context_anchor are all unbound; candidates remain in the private reference layer.", ""])
