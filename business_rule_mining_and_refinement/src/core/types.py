from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class SemanticTag(Enum):
    ENTITY_ID = "entity_id"
    QUANTITY = "quantity"
    STATUS = "status"
    OTHER = "other"


@dataclass
class ToolParameter:
    name: str
    type: str
    description: str
    enum_values: list[str] | None = None
    required: bool = False
    semantic_tag: SemanticTag | None = None
    belongs_to_tool: str = ""


@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    return_type: dict | None = None
    raw_schema: dict[str, Any] = field(default_factory=dict)
    source_code: str | None = None


@dataclass
class AgentUnderTest:
    tools: list[ToolSchema]
    system_prompt: str
    rules: list[str]
    domain: str = ""
    agent_id: str = ""
    orchestration: str | None = None

    def to_dict(self, include_source: bool = False) -> dict:
        d = asdict(self)
        if not include_source:
            for t in d.get("tools", []):
                t.pop("source_code", None)
        return d

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @staticmethod
    def load(path: str) -> "AgentUnderTest":
        with open(path) as f:
            return AgentUnderTest.from_dict(json.load(f))

    @staticmethod
    def from_dict(d: dict) -> "AgentUnderTest":
        def _parse_tag(val):
            if not val:
                return None
            s = str(val)
            if s.startswith("SemanticTag."):
                s = s.split(".", 1)[1].lower()
            try:
                return SemanticTag(s)
            except ValueError:
                return None

        tools = []
        for t in d.get("tools", []):
            params = [
                ToolParameter(
                    name=p["name"],
                    type=p.get("type", "string"),
                    description=p.get("description", ""),
                    enum_values=p.get("enum_values"),
                    required=p.get("required", False),
                    semantic_tag=_parse_tag(p.get("semantic_tag")),
                    belongs_to_tool=p.get("belongs_to_tool", ""),
                )
                for p in t.get("parameters", [])
            ]
            tools.append(ToolSchema(
                name=t["name"],
                description=t.get("description", ""),
                parameters=params,
                return_type=t.get("return_type"),
                raw_schema=t.get("raw_schema", {}),
                source_code=t.get("source_code"),
            ))
        return AgentUnderTest(
            tools=tools,
            system_prompt=d.get("system_prompt", ""),
            rules=d.get("rules", []),
            domain=d.get("domain", ""),
            agent_id=d.get("agent_id", ""),
            orchestration=d.get("orchestration"),
        )


# ── Contract IR ──────────────────────────────────────────────────────────────


@dataclass
class OwnershipPredicate:
    entity_param: str
    tool_name: str
    ownership_source: str
    description: str
    grounding_evidence: list[str] = field(default_factory=list)


@dataclass
class NumericBoundExpr:
    param: str
    operator: str  # "<=", ">=", "<", ">", "==", "!="
    bound_type: str  # "absolute" | "relative"
    absolute_value: float | None = None
    reference_tool: str | None = None
    reference_field: str | None = None
    entity_scope_param: str | None = None


@dataclass
class NumericPredicate:
    bound_expr: NumericBoundExpr
    tool_name: str
    accumulator_mode: str = "per_call"  # "per_call" | "cumulative"
    description: str = ""
    grounding_evidence: list[str] = field(default_factory=list)


@dataclass
class SequencingPredicate:
    predecessor_tool: str
    tool_name: str
    predecessor_args_constraints: dict[str, str] | None = None
    ordering_type: str = "must_precede"
    confidence: str = "prompt_explicit"  # "structural" | "prompt_explicit" | "prompt_inferred"
    description: str = ""
    grounding_evidence: list[str] = field(default_factory=list)


@dataclass
class FormatPredicate:
    param: str
    tool_name: str
    constraint_type: str  # "length" | "pattern" | "case" | "enum" | "date_format" | "value_range" | "reference_set"
    min_length: int | None = None
    max_length: int | None = None
    exact_length: int | None = None
    pattern: str | None = None
    case_requirement: str | None = None  # "upper" | "lower"
    valid_values: list[str] | None = None
    date_format: str | None = None
    min_value: str | None = None
    max_value: str | None = None
    reference_tool: str | None = None
    description: str = ""
    grounding_evidence: list[str] = field(default_factory=list)


@dataclass
class CrossParamPredicate:
    param_a: str
    param_b: str
    tool_name: str
    relation: str  # "not_equal" | "less_than" | "less_than_or_equal" | "greater_than" | "greater_than_or_equal"
    description: str = ""
    grounding_evidence: list[str] = field(default_factory=list)


@dataclass
class PolicyPredicate:
    rule_id: str  # unique ID like "cancel_24h_window"
    tool_name: str  # which tool this constrains
    category: str  # "conditional" | "cardinality" | "immutability" | "consistency" | "confirmation" | "info_requirement" | "compensation"
    rule_text: str  # the policy rule in natural language
    violation_scenario: str  # how to violate this rule
    expected_behavior: str  # "deny" | "allow_with_conditions" | "must_collect_info" | "must_confirm"
    condition_params: dict[str, Any] = field(default_factory=dict)  # structured condition details
    description: str = ""
    grounding_evidence: list[str] = field(default_factory=list)


@dataclass
class OwnershipAcquisitionRule:
    tool_name: str
    entity_field_in_result: str
    acquisition_type: str  # "creates" | "session_bound_retrieval"
    entity_field_in_args: str | None = None


@dataclass
class ToolContract:
    tool_name: str
    ownership_predicates: list[OwnershipPredicate] = field(default_factory=list)
    numeric_predicates: list[NumericPredicate] = field(default_factory=list)
    sequencing_predicates: list[SequencingPredicate] = field(default_factory=list)
    format_predicates: list[FormatPredicate] = field(default_factory=list)
    cross_param_predicates: list[CrossParamPredicate] = field(default_factory=list)
    policy_predicates: list[PolicyPredicate] = field(default_factory=list)


@dataclass
class ContractSet:
    agent_id: str
    contracts: dict[str, ToolContract] = field(default_factory=dict)
    acquisition_rules: list[OwnershipAcquisitionRule] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @staticmethod
    def load(path: str) -> "ContractSet":
        with open(path) as f:
            return ContractSet.from_dict(json.load(f))

    @staticmethod
    def from_dict(d: dict) -> "ContractSet":
        cs = ContractSet(agent_id=d["agent_id"], metadata=d.get("metadata", {}))
        for rule_d in d.get("acquisition_rules", []):
            cs.acquisition_rules.append(OwnershipAcquisitionRule(**rule_d))
        for tool_name, tc_d in d.get("contracts", {}).items():
            tc = ToolContract(tool_name=tool_name)
            for op in tc_d.get("ownership_predicates", []):
                tc.ownership_predicates.append(OwnershipPredicate(**op))
            for np_d in tc_d.get("numeric_predicates", []):
                expr_d = np_d.pop("bound_expr")
                expr = NumericBoundExpr(**expr_d)
                tc.numeric_predicates.append(NumericPredicate(bound_expr=expr, **np_d))
                np_d["bound_expr"] = expr_d
            for sp in tc_d.get("sequencing_predicates", []):
                tc.sequencing_predicates.append(SequencingPredicate(**sp))
            for fp in tc_d.get("format_predicates", []):
                tc.format_predicates.append(FormatPredicate(**fp))
            for cp in tc_d.get("cross_param_predicates", []):
                tc.cross_param_predicates.append(CrossParamPredicate(**cp))
            for pp in tc_d.get("policy_predicates", []):
                tc.policy_predicates.append(PolicyPredicate(**pp))
            cs.contracts[tool_name] = tc
        return cs


# ── Declared Vocabulary ──────────────────────────────────────────────────────


@dataclass
class DeclaredVocabulary:
    entity_ids: list[tuple[str, str]] = field(default_factory=list)
    quantities: list[tuple[str, str]] = field(default_factory=list)
    statuses: list[tuple[str, str]] = field(default_factory=list)
    all_fields: set[str] = field(default_factory=set)
    tool_names: set[str] = field(default_factory=set)
    dataflow_edges: list[tuple[str, str, str, str]] = field(default_factory=list)

    def formatted(self) -> str:
        lines = [
            "ENTITY-ID PARAMETERS (use ONLY the param_name in predicates, "
            "NOT the tool prefix):"
        ]
        for param, tool in self.entity_ids:
            lines.append(f"  - param_name=\"{param}\"  (on tool: {tool})")
        lines.append(
            "\nQUANTITY PARAMETERS (use ONLY the param_name in predicates):"
        )
        for param, tool in self.quantities:
            lines.append(f"  - param_name=\"{param}\"  (on tool: {tool})")
        lines.append(
            "\nSTATUS PARAMETERS (use ONLY the param_name in predicates):"
        )
        for param, tool in self.statuses:
            lines.append(f"  - param_name=\"{param}\"  (on tool: {tool})")
        lines.append("\nALL TOOL NAMES:")
        for t in sorted(self.tool_names):
            lines.append(f"  - {t}")
        return "\n".join(lines)


# ── Trace & Verdict ──────────────────────────────────────────────────────────


@dataclass
class ToolCall:
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any] | None = None
    timestamp: int = 0
    turn_index: int = 0


@dataclass
class ConversationTurn:
    role: str
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class MultiTurnTrace:
    trace_id: str
    session_identity: str
    conversation: list[ConversationTurn]
    tool_call_sequence: list[ToolCall]
    target_contract: ToolContract | None = None
    database_snapshot_before: str | None = None


@dataclass
class ViolationEvidence:
    violating_call: ToolCall
    violated_predicate_type: str
    violated_predicate: OwnershipPredicate | NumericPredicate | SequencingPredicate | FormatPredicate | CrossParamPredicate | PolicyPredicate
    explanation: str


@dataclass
class Verdict:
    trace_id: str
    result: str  # "pass" | "fail"
    evidence: list[ViolationEvidence] = field(default_factory=list)


# ── Planner Types ───────────────────────────────────────────────────────────


@dataclass
class SetupStep:
    user_message: str
    expected_tool_call: str
    expected_arguments: dict[str, Any] | None = None
    purpose: str = ""  # "authenticate" | "lookup_entity" | "satisfy_predecessor"


@dataclass
class BreachStep:
    user_message: str
    expected_tool_call: str
    violating_arguments: dict[str, Any] = field(default_factory=dict)
    negation_explanation: str = ""


@dataclass
class BreachPlan:
    plan_id: str
    target_tool: str
    target_predicate_type: str  # "ownership" | "numeric" | "sequencing" | "format" | "cross_param" | "policy"
    target_predicate: OwnershipPredicate | NumericPredicate | SequencingPredicate | FormatPredicate | CrossParamPredicate | PolicyPredicate
    negation_strategy: str  # "use_foreign_entity_id" | "exceed_bound" | "skip_predecessor" | "violate_format" | "violate_cross_param" | "violate_policy"
    setup_steps: list[SetupStep]
    breach_step: BreachStep
    session_identity: str
    foreign_entity_ids: list[str] | None = None
    business_scenario: str = ""
    scenario_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ── BehaviorSpec IR ─────────────────────────────────────────────────────────
# Captures complete decision logic: for a given request type, what should the
# agent do under each combination of conditions? Both "allow" and "deny"
# branches are first-class, enabling positive AND negative test generation
# from the same specification.


@dataclass
class StateCondition:
    """A single condition on DB/session state."""
    field: str              # e.g. "reservation.cabin", "user.membership", "flight.date"
    operator: str           # "==", "!=", "in", "not_in", ">=", "<=", ">", "<", "exists", "not_exists"
    value: Any              # e.g. "basic_economy", ["silver", "gold"], 24
    source: str = "db"      # "db" | "user_input" | "computed" | "temporal"
    description: str = ""   # human-readable, e.g. "cabin class is basic economy"


@dataclass
class ExpectedAction:
    """What the agent SHOULD do when conditions are met."""
    action_type: str                    # "allow", "deny", "compute", "transfer_to_human", "inform"
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # Each tool_call: {"tool": "cancel_reservation", "args": {}, "check": "called"|"not_called"|"arg_match"|"response_contains"|"called_n_times"}
    response_should_contain: list[str] = field(default_factory=list)
    response_should_not_contain: list[str] = field(default_factory=list)
    description: str = ""               # e.g. "agent should cancel and issue refund"


@dataclass
class DecisionBranch:
    """One branch of the decision tree — conditions → expected action."""
    branch_id: str
    conditions: list[StateCondition]    # AND of all conditions
    expected_action: ExpectedAction
    description: str = ""               # e.g. "business class cancellation within 24h"
    policy_reference: str = ""          # quote from policy document
    priority: int = 0                   # lower = higher priority (first match wins)
    is_default: bool = False            # true for the fallback/else branch


@dataclass
class BehaviorSpec:
    """Complete decision logic for one request type."""
    spec_id: str                        # e.g. "cancel_reservation_logic"
    request_type: str                   # e.g. "cancel", "modify_cabin", "add_baggage", "compensate"
    description: str                    # human-readable summary
    decision_branches: list[DecisionBranch]  # ordered by priority
    applicable_tools: list[str] = field(default_factory=list)
    policy_section: str = ""            # which section of policy this comes from
    grounding_evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @staticmethod
    def from_dict(d: dict) -> "BehaviorSpec":
        branches = []
        for b in d.get("decision_branches", []):
            conditions = [StateCondition(**c) for c in b.get("conditions", [])]
            ea = b.get("expected_action", {})
            expected_action = ExpectedAction(
                action_type=ea.get("action_type", "deny"),
                tool_calls=ea.get("tool_calls", []),
                response_should_contain=ea.get("response_should_contain", []),
                response_should_not_contain=ea.get("response_should_not_contain", []),
                description=ea.get("description", ""),
            )
            branches.append(DecisionBranch(
                branch_id=b.get("branch_id", ""),
                conditions=conditions,
                expected_action=expected_action,
                description=b.get("description", ""),
                policy_reference=b.get("policy_reference", ""),
                priority=b.get("priority", 0),
                is_default=b.get("is_default", False),
            ))
        return BehaviorSpec(
            spec_id=d.get("spec_id", ""),
            request_type=d.get("request_type", ""),
            description=d.get("description", ""),
            decision_branches=branches,
            applicable_tools=d.get("applicable_tools", []),
            policy_section=d.get("policy_section", ""),
            grounding_evidence=d.get("grounding_evidence", []),
        )


@dataclass
class BehaviorSpecSet:
    """Collection of all BehaviorSpecs for a domain."""
    domain: str
    specs: list[BehaviorSpec] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "specs": [s.to_dict() for s in self.specs],
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @staticmethod
    def load(path: str) -> "BehaviorSpecSet":
        with open(path) as f:
            d = json.load(f)
        specs = [BehaviorSpec.from_dict(s) for s in d.get("specs", [])]
        return BehaviorSpecSet(
            domain=d.get("domain", ""),
            specs=specs,
            metadata=d.get("metadata", {}),
        )
