"""Build a real Step3 `assessments` document for any v5 domain, without the
v1.1.0-era XLSX SourceAnchor loader.

`compile_assessment.py::assess_selected_spec`/`assess_spec_set` already
implement the entire deterministic resolver chain (target resolution, policy
lineage, runtime projection, compile assessment) generically over any
`SourceSpecRecord` + artifact bundle -- the ONLY domain-specific,
XLSX-dependent step in that chain is `assess_selected_spec`'s own call to
`load_source_spec_record`, which reads one row out of an Excel workbook that
does not exist for telecom/retail (their true root input is already JSON,
see docs/end_to_end_implementation_walkthrough.md). Everything downstream of
that one call only ever consumes the resulting `SourceSpecRecord` value, per
`make_source_spec_record_v2`'s own docstring ("later resolvers do not need to
know about XLSX").

This module does not modify or duplicate that resolver chain -- it builds a
`SourceSpecRecord` directly from a v5 Step2/Step3 branch's own real
given/when/then/rule_text/kind/origin fields (which are the same underlying
content the Excel row would have carried) plus one externally-supplied
`deontic` label (the one field the Excel row had that v5's JSON does not),
then re-invokes the exact same generic resolver chain that
`assess_selected_spec` uses, in the same order, with the same call shapes.
See docs/agentcoveragetesting_reuse_log.md section 12 for the real-data
validation of this design (verified byte-for-byte capability parity with the
airline artifact bundle, and a real airline branch reproducing the same
`resolved_unique`/`accepted-operation-factor-resolver/v0.1` resolution the
existing frozen assessment-set already has for that branch).

`deontic` is not present anywhere in v5's own JSON source and is not
mechanically derivable from `then`'s surface wording alone (a naive
keyword classifier scored only 88% against 166 real human-reviewed airline
labels; a rubric grounded primarily in `rule_text`'s own polarity, applied by
an LLM judge, scored 98% on the same real held-out data -- see
docs/agentcoveragetesting_reuse_log.md section 13). Callers supply
`deontic` per branch_id from whatever classification process they trust;
this module does not perform that classification itself, so a real external
LLM call (the production path) can be substituted for the calibration
mechanism used during this project's own development without touching this
module.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Mapping, Sequence

from .artifacts import (
    CAP_ACCEPTED_POLICY_MODEL,
    CAP_POLICY_SOURCE_EVIDENCE,
    CAP_TOOL_OBSERVABLE_ENDPOINTS,
    artifact_document_for_capability,
    content_sha256,
)
from .capabilities import DEFAULT_PIPELINE_CAPABILITIES, CapabilityRegistry
from .compile_assessment import COMPILE_ASSESSMENT_SET_SCHEMA_VERSION, _probe_family, _stage
from .evaluation_contracts import make_compile_assessment
from .evaluation_resolver import resolve_evaluation_target
from .policy_lineage import resolve_policy_lineage
from .profiles import validate_compiler_profile, validate_sut_adapter_profile
from .runtime_projection import (
    materialize_runtime_projection_target,
    resolve_runtime_projection,
)
from .source_contracts import make_source_spec_record_v2
from .target_resolution import make_target_resolution_report


V5_SOURCE_SPEC_ADAPTER_ID = "v5_native_source_spec_adapter/v0.1"


def _tokens(text: str) -> set[str]:
    raw = re.findall(r"[a-z0-9]+", text.lower())
    out = set()
    for token in raw:
        # Cheap, language-general plural normalization (not an airline-specific
        # rule): the real policy.md text mixes singular/plural noun forms
        # across otherwise-identical sibling lines (e.g. "1 free checked bag"
        # vs "2 free checked bags"), which would otherwise spuriously favor
        # whichever sibling happens to share the same plural form as
        # rule_text, drowning out the real discriminating word.
        if len(token) > 3 and token.endswith("s") and not token.isdigit():
            token = token[:-1]
        out.add(token)
    return out


def _reconstruct_tool_argument_evidence_quote(
    rule_text: str, then: str, resolved_artifacts: Mapping[str, Any],
) -> str | None:
    """Real endpoint descriptions from the domain's own tool catalog, for
    whichever argument name(s) `rule_text`/`then` actually mention.

    `accepted-tool-argument-resolver/v0.1` only matches an endpoint whose
    `description` is contained in (or equal to) one of `evidence_quote`'s
    pipe-separated parts -- v5's JSON never captured this quote, but the real
    text it would have quoted is the SAME tool catalog already in the
    artifact bundle, so it is recovered exactly, not invented. Downstream
    tool-name disambiguation in the resolver itself narrows a broad candidate
    set back down to the correct single tool, so precision here is not
    required -- completeness (not missing the right endpoint) is what matters.
    """

    try:
        catalog = artifact_document_for_capability(resolved_artifacts, CAP_TOOL_OBSERVABLE_ENDPOINTS)
    except Exception:
        return None
    # Argument names are identifiers (e.g. "user_id"), not English words --
    # `_tokens()` drops underscores to split natural-language text into real
    # words, which would silently split "user_id" into "user"/"id" and never
    # match either against the literal identifier. Keep underscores here.
    identifier_tokens = set(re.findall(r"[a-z0-9_]+", (rule_text + " " + then).lower()))
    seen: list[str] = []
    seen_normalized: set[str] = set()
    for tool in catalog.get("tools") or []:
        for endpoint in tool.get("observable_endpoints") or []:
            argument = str(endpoint.get("source_path") or "").rsplit(".", 1)[-1].replace("[]", "")
            if not argument or argument.lower() not in identifier_tokens:
                continue
            description = str(endpoint.get("description") or "")
            normalized = re.sub(r"\s+", " ", description.strip().lower())
            if normalized and normalized not in seen_normalized:
                seen_normalized.add(normalized)
                seen.append(description)
    return " | ".join(seen) if seen else None


def _reconstruct_policy_lineage_evidence_quote(
    rule_text: str, resolved_artifacts: Mapping[str, Any], *, source_origin: str,
) -> str | None:
    """The single real policy.md evidence line that best matches `rule_text`,
    recovered from the domain's own accepted policy model + source evidence
    index (both already in the artifact bundle) -- see the module-level
    `_reconstruct_tool_argument_evidence_quote` docstring for why this is
    recovery, not invention.

    Two-stage search: (1) find the accepted policy *statement* (a full
    sentence, not a short bullet) with the highest cross-statement-IDF-
    weighted token overlap with `rule_text` (deliberately NOT normalized by
    statement length/Jaccard -- empirically, on the real 110-branch
    validation set, that normalization fixed one known confounder but
    regressed 3 other previously-correct branches net, so the unnormalized
    sum is kept; see docs/agentcoveragetesting_reuse_log.md section 17) --
    this is what actually identifies the right policy, since
    short bullet-list evidence lines within one policy are often near-
    duplicates of each other (e.g. "0/1/2/3/4 free checked bags", one line
    per membership tier and cabin class); (2) within *that* policy's own
    linked evidence lines only, score each by matching-token weight (a
    numeric-token match is required whenever `rule_text` has one -- numbers
    are the actual discriminator between otherwise near-identical sibling
    lines; a token's weight is inversely proportional to how many of this
    policy's OWN evidence lines already contain it, so boilerplate shared by
    every sibling line contributes nothing).
    """

    # resolve_policy_lineage itself refuses any source whose origin is not
    # "prompt" (policy.md), before ever looking at evidence_quote -- e.g. a
    # domain_knowledge-origin rule has no real policy.md evidence at all, so
    # any reconstructed quote for one would necessarily be a false match
    # (this was caught by a real regression: airline_049_arg#b0's rule has no
    # policy.md basis, and a naive best-effort quote spuriously matched an
    # unrelated cabin-class policy statement, producing a wrong
    # `needs_adjudication` result where the real pipeline correctly falls
    # back to `source_grounded`).
    if source_origin != "prompt":
        return None
    try:
        policy_document = artifact_document_for_capability(resolved_artifacts, CAP_ACCEPTED_POLICY_MODEL)
        evidence_document = artifact_document_for_capability(resolved_artifacts, CAP_POLICY_SOURCE_EVIDENCE)
    except Exception:
        return None
    rule_tokens = _tokens(rule_text)
    if not rule_tokens:
        return None

    evidence_by_passage: dict[str, dict[str, str]] = {}
    for proposal in evidence_document.get("policy_proposal_inputs") or []:
        passage_id = str(proposal.get("target_source_passage_id") or "")
        for item in proposal.get("target_evidence") or []:
            evidence_id = item.get("evidence_id")
            if evidence_id:
                evidence_by_passage.setdefault(passage_id, {})[str(evidence_id)] = str(item.get("text") or "")

    all_statements: list[tuple[str, Mapping[str, Any], set[str]]] = []
    for proposal in policy_document.get("proposal_results") or []:
        passage_id = str(proposal.get("target_source_passage_id") or "")
        for policy in proposal.get("policies") or []:
            statement_tokens = _tokens(str(policy.get("statement") or ""))
            if statement_tokens:
                all_statements.append((passage_id, policy, statement_tokens))
    if not all_statements:
        return None

    # Plain token-overlap (or even Jaccard) picks the wrong policy whenever a
    # short, topically-adjacent statement happens to share more of its own
    # words with rule_text than the real match does (real regression: for
    # "...obtain the user id...before booking a flight", "...obtain the user
    # id...before MODIFYING a flight" outscored the correct booking-flow
    # statement on raw Jaccard, since "modifying" is one word in a shorter
    # statement while "booking"'s real match sits in a longer one). Weight by
    # cross-statement IDF instead, same principle as the evidence-line
    # disambiguation above but computed over the whole policy corpus, so a
    # rare, meaning-carrying word like "booking"/"modifying" dominates the
    # score instead of being diluted by boilerplate ("obtain", "the", "user",
    # "id", "before", "must") that recurs across many unrelated statements.
    statement_document_frequency: Counter[str] = Counter()
    for _, _, statement_tokens in all_statements:
        statement_document_frequency.update(statement_tokens)
    statement_count = len(all_statements)

    def statement_weight(token: str) -> float:
        frequency = statement_document_frequency.get(token, 0)
        if frequency == 0 or frequency == statement_count:
            return 0.0
        return 1.0 / frequency

    best_passage_id = best_policy = None
    best_statement_score = -1.0
    for passage_id, policy, statement_tokens in all_statements:
        score = sum(statement_weight(token) for token in (rule_tokens & statement_tokens))
        if score > best_statement_score:
            best_statement_score = score
            best_passage_id, best_policy = passage_id, policy
    if best_policy is None or best_statement_score <= 0:
        return None

    evidence_ids = best_policy.get("semantic_support_evidence_ids") or best_policy.get("target_evidence_ids") or []
    candidates = {
        eid: text for eid in evidence_ids
        if (text := evidence_by_passage.get(best_passage_id, {}).get(str(eid)))
    }
    if not candidates:
        return None
    candidate_tokens = {eid: _tokens(text) for eid, text in candidates.items()}
    document_frequency: Counter[str] = Counter()
    for tokens in candidate_tokens.values():
        document_frequency.update(tokens)
    candidate_count = len(candidate_tokens)

    def weight(token: str) -> float:
        frequency = document_frequency.get(token, 0)
        if frequency == 0 or frequency == candidate_count:
            return 0.0
        return 1.0 / frequency

    rule_numbers = {token for token in rule_tokens if token.isdigit()}
    pool = candidate_tokens.items()
    if rule_numbers:
        numeric_matches = {eid: tokens for eid, tokens in pool if tokens & rule_numbers}
        if numeric_matches:
            pool = numeric_matches.items()

    best_eid, best_score = None, -1.0
    for eid, tokens in pool:
        score = sum(weight(token) for token in (rule_tokens & tokens))
        if score > best_score:
            best_eid, best_score = eid, score
    if best_eid is None:
        return None
    return _narrow_to_best_sentence(candidates[best_eid], rule_tokens)


def _narrow_to_best_sentence(evidence_text: str, rule_tokens: set[str]) -> str:
    """A winning evidence span sometimes bundles more than one real sentence
    (a source extraction artifact, not this module's own construction) --
    e.g. a policy condition sentence immediately followed by an unrelated
    operational how-to sentence in the same span. Real regression:
    airline_076_norm#b0's evidence span is "You should transfer the user...
    if and only if the request cannot be handled... To transfer, first make
    a tool call to transfer_to_human_agents...", but the real evidence_quote
    only ever needed the first (condition) sentence -- passing the whole
    span made `accepted-path-contract-resolver` match against both the
    condition AND the unrelated tool-call instruction text, producing a
    spurious `ambiguous` result the real pipeline never had. If the span is
    a single sentence already, this is a no-op."""

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", evidence_text) if s.strip()]
    if len(sentences) <= 1:
        return evidence_text
    best_sentence, best_sentence_score = evidence_text, -1.0
    for sentence in sentences:
        score = len(rule_tokens & _tokens(sentence))
        if score > best_sentence_score:
            best_sentence_score = score
            best_sentence = sentence
    return best_sentence


def reconstruct_evidence_quote(
    rule_text: str, then: str, resolved_artifacts: Mapping[str, Any], *, source_origin: str,
) -> str | None:
    """Recover the real source text `evidence_quote` would have quoted, from
    the domain's own artifact bundle -- see the two helper docstrings above.
    Both resolvers that consume `evidence_quote` only check substring
    containment/equality, so combining both reconstructions into one string
    serves both without either interfering with the other."""

    parts = [
        part for part in (
            _reconstruct_tool_argument_evidence_quote(rule_text, then, resolved_artifacts),
            _reconstruct_policy_lineage_evidence_quote(rule_text, resolved_artifacts, source_origin=source_origin),
        ) if part
    ]
    return " | ".join(parts) if parts else None


def build_v5_source_spec_record(
    *, spec_id: str, branch_id: str, source_kind: str, source_origin: str,
    rule_text: str, given: str, when: str, then: str, deontic: str,
    evidence_quote: str | None = None,
) -> dict[str, Any]:
    """Construct a `SourceSpecRecord` v2 straight from v5's own branch text.

    `selected_spec_ref` normally locates one Excel row (workbook/sheet/row);
    v5 has no such row, so a synthetic-but-honest locator is used instead --
    it is never read by the resolver chain (see module docstring), only
    carried through as an opaque identity field.

    `evidence_quote` defaults to `None` (an honest gap: two resolvers in the
    chain then structurally cannot fire -- see this module's evidence-quote
    reconstruction functions). Callers that already resolved an artifact
    bundle should pass `reconstruct_evidence_quote(rule_text, then,
    resolved_artifacts)` here instead of leaving it `None`.
    """

    return make_source_spec_record_v2(
        selected_spec_ref={
            "schema_version": "agentspectesting.selected-spec-ref/v0.1",
            "workbook": f"v5:{source_kind}", "sheet": "v5_native", "row": 1,
            "branch_id": branch_id,
        },
        spec_id=spec_id, branch_id=branch_id,
        source_kind=source_kind, source_origin=source_origin,
        rule_text=rule_text, given=given, when=when, then=then, deontic=deontic,
        evidence_quote=evidence_quote, review_status=None, review_note=None,
        source_columns={}, loader_provenance={"adapter_id": V5_SOURCE_SPEC_ADAPTER_ID},
    )


def assess_v5_source_spec_record(
    source: Mapping[str, Any], resolved_artifacts: Mapping[str, Any], *,
    compiler_profile: Mapping[str, Any], sut_adapter_profile: Mapping[str, Any],
    capability_registry: CapabilityRegistry = DEFAULT_PIPELINE_CAPABILITIES,
) -> dict[str, Any]:
    """Same chain as `compile_assessment.assess_selected_spec`, minus XLSX."""

    validate_compiler_profile(compiler_profile)
    sut = validate_sut_adapter_profile(sut_adapter_profile)
    target = resolve_evaluation_target(source, resolved_artifacts)
    policy_lineage = resolve_policy_lineage(source, resolved_artifacts)
    runtime_projection = resolve_runtime_projection(source, policy_lineage, resolved_artifacts)
    target = materialize_runtime_projection_target(source, target, runtime_projection, resolved_artifacts)
    resolution_report = make_target_resolution_report(
        source_spec_record=source, resolved_evaluation_target=target,
        policy_lineage_report=policy_lineage, runtime_projection_report=runtime_projection,
    )
    binding = target.get("evaluation_binding") or {}
    archetype = binding.get("target_archetype")
    subject = binding.get("evaluation_subject") or {}
    tools = subject.get("verified_tool_names") or []
    probe_family = _probe_family(binding) if binding else None
    context = {
        "target_archetype": archetype, "sut": sut["sut"], "domain": sut["domain"],
        "sut_profile_id": sut["profile_id"], "verified_tool_names": tools,
        "probe_family": probe_family,
    }

    stages: dict[str, dict[str, Any]] = {"source_loading": _stage("ready", reason_code="v5_native_source_loaded")}
    if resolution_report["readiness"]["runtime_target_ready"]:
        stages["semantic_resolution"] = _stage("ready", reason_code="runtime_evaluation_target_ready")
    elif resolution_report["readiness"]["target_ready"]:
        stages["semantic_resolution"] = _stage("partial", reason_code="abstract_target_ready_runtime_binding_incomplete")
    else:
        stages["semantic_resolution"] = _stage("blocked", reason_code=target["resolution_status"])

    downstream = resolution_report["readiness"]["runtime_target_ready"] and isinstance(archetype, str)
    for output_name, registry_stage in (
        ("experiment_planning", "experiment_planner"), ("probe_synthesis", "probe_synthesizer"),
        ("fixture_binding", "fixture_binder"), ("runtime_protocol", "runtime_protocol"),
        ("surface_realization", "surface_realizer"), ("outcome_oracle", "outcome_oracle"),
        ("mutation_provider", "mutation_provider"), ("guidance", "guidance_controller"),
    ):
        if not downstream:
            stages[output_name] = _stage("not_assessed", reason_code="semantic_resolution_not_ready")
            continue
        capability = capability_registry.select(registry_stage, context)
        if capability is None:
            stages[output_name] = _stage("blocked", reason_code=f"missing_{registry_stage}_capability")
        else:
            stages[output_name] = _stage("ready", reason_code="registered_capability_selected", capability_id=capability.capability_id)

    snapshot = capability_registry.snapshot()
    registry_identity = {
        "schema_version": snapshot["schema_version"], "fingerprint": content_sha256(snapshot),
        "capability_count": len(snapshot["capabilities"]),
    }
    return make_compile_assessment(
        source_spec_record=source, resolved_evaluation_target=target,
        target_resolution_report=resolution_report, stages=stages,
        capability_registry=registry_identity, diagnostics=target["diagnostics"],
    )


def assess_v5_spec_set(
    source_records: Sequence[Mapping[str, Any]], resolved_artifacts: Mapping[str, Any], *,
    compiler_profile: Mapping[str, Any], sut_adapter_profile: Mapping[str, Any],
    capability_registry: CapabilityRegistry = DEFAULT_PIPELINE_CAPABILITIES,
) -> dict[str, Any]:
    """Same aggregation as `compile_assessment.assess_spec_set`, over a real
    list of v5-native SourceSpecRecords instead of an Excel-enumerated set."""

    items = []
    for source in source_records:
        assessment = assess_v5_source_spec_record(
            source, resolved_artifacts, compiler_profile=compiler_profile,
            sut_adapter_profile=sut_adapter_profile, capability_registry=capability_registry,
        )
        items.append({"selected_spec_ref": source["selected_spec_ref"], "assessment": assessment})

    resolution_counts = Counter(item["assessment"]["resolved_evaluation_target_identity"]["resolution_status"] for item in items)
    resolver_counts = Counter(item["assessment"]["resolved_evaluation_target_identity"]["resolver_id"] for item in items)
    archetype_counts = Counter(item["assessment"]["target_archetype"] or "unresolved" for item in items)
    source_kind_counts = Counter(item["assessment"]["source_kind"] or "unclassified" for item in items)
    resolution_by_source_kind: dict[str, Counter] = {}
    for item in items:
        assessment = item["assessment"]
        kind = assessment["source_kind"] or "unclassified"
        resolution_by_source_kind.setdefault(kind, Counter()).update([assessment["resolved_evaluation_target_identity"]["resolution_status"]])
    dimension_counts = {
        dimension: Counter(item["assessment"]["target_resolution_report"]["dimensions"][dimension]["status"] for item in items)
        for dimension in ("target_compilation", "accepted_binding", "lineage", "assertion_formalization", "observation_binding")
    }
    policy_lineage_counts = Counter((item["assessment"]["target_resolution_report"].get("policy_lineage_report") or {"status": "not_available"})["status"] for item in items)
    runtime_projection_counts = Counter((item["assessment"]["target_resolution_report"].get("runtime_projection_report") or {"status": "not_available"})["status"] for item in items)
    readiness_counts = {
        readiness: sum(bool(item["assessment"]["target_resolution_report"]["readiness"][readiness]) for item in items)
        for readiness in ("target_ready", "accepted_binding_ready", "assertion_ready", "runtime_target_ready")
    }
    payload = {
        "schema_version": COMPILE_ASSESSMENT_SET_SCHEMA_VERSION,
        "selection": {"kind": "v5_native_branch_set", "branch_count": len(items)},
        "summary": {
            "spec_count": len(items),
            "resolution_status_counts": dict(sorted(resolution_counts.items())),
            "resolver_counts": dict(sorted(resolver_counts.items())),
            "target_archetype_counts": dict(sorted(archetype_counts.items())),
            "source_kind_counts": dict(sorted(source_kind_counts.items())),
            "resolution_by_source_kind": {kind: dict(sorted(counts.items())) for kind, counts in sorted(resolution_by_source_kind.items())},
            "target_resolution_dimension_counts": {name: dict(sorted(counts.items())) for name, counts in sorted(dimension_counts.items())},
            "policy_lineage_status_counts": dict(sorted(policy_lineage_counts.items())),
            "runtime_projection_status_counts": dict(sorted(runtime_projection_counts.items())),
            "target_resolution_readiness_counts": readiness_counts,
            "end_to_end_ready_count": sum(bool(item["assessment"]["end_to_end_ready"]) for item in items),
        },
        "items": items,
        "llm_calls": 0,
    }
    payload["compile_assessment_set_fingerprint"] = content_sha256(payload)
    return payload
