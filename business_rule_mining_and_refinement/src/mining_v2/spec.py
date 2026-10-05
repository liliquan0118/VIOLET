"""Spec IR for mining_v2.

Deliberately minimal: a spec is a natural-language business rule plus the
metadata needed to (a) know where it came from, (b) know what kind of
constraint it is, and (c) drive coverage checking. No test-case material,
no guard/DNF representation — that is a downstream concern.

kind taxonomy (the four "shapes" of a per-tool-call constraint):
  ARG   — constrains the call's own arguments (format / range / relation)
  STATE — refers to DB / session state at call time (ownership, eligibility)
  ORDER — refers to what must happen before/after the call (sequencing,
          confirmation, info collection)
  NORM  — a behavioral norm not reducible to the above (no proactive X, tone)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


KINDS = ("ARG", "STATE", "ORDER", "NORM")


@dataclass
class Evidence:
    source: str   # "prompt" | "schema" | "source_code"
    quote: str    # the policy sentence / code line / schema fact it is grounded in
    anchor: str = ""  # locator for the quote's occurrence in the source (the
                      # immediately preceding sentence + "occ k/n" when the same
                      # text appears more than once); "" when unambiguous or unknown


# ── Intermediate unit: a rule fragment (one statement-level piece) ────────────
# A fragment is what Pass 1 emits from a single policy statement. Multiple
# fragments about the SAME target action are aggregated into one Spec.

@dataclass
class Fragment:
    text: str                            # one clause, plain English
    kind: str                            # ARG | STATE | ORDER | NORM
    target_action: str                   # short label of the action it governs
    relevant_tools: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)


# ── Final unit: a Spec ────────────────────────────────────────────────────────
# A Spec is one testable business rule. Its kind is by WHAT IT CONSTRAINS:
#   ARG   — a tool call's own argument(s)
#   STATE — DB / account state
#   ORDER — the call sequence / history
#   NORM  — the agent's language output / behavior
# Rules that are independently testable per tool are split into one spec per
# tool (each binds that tool's params), all sharing a `source_rule` id so their
# common origin is not lost. Rules that are inherently cross-tool / cumulative
# (e.g. "one certificate per incident across all tools") stay a single spec
# binding several tools.

@dataclass
class Spec:
    spec_id: str
    target_action: str                   # the action/decision this spec governs
    rule_text: str                       # the rule in natural language
    kind: str                            # ARG | STATE | ORDER | NORM (what it constrains)
    clauses: list[str] = field(default_factory=list)   # composing clauses (>1 = multi-branch)
    relevant_tools: list[str] = field(default_factory=list)
    bindings: list[dict] = field(default_factory=list)  # [{"tool": str, "params": [str]}]
    cross_tool: bool = False             # True = inherently cross-tool/cumulative, do not split
    source_rule: str = ""                # shared id for specs split from one policy rule
    confidence: str = "stated"           # "stated" | "hypothesized" | "structural"
    origin: str = "prompt"               # which pass produced it
    evidence: list[Evidence] = field(default_factory=list)
    gwt: list[dict] | None = None        # natural-language GWT sub-scenarios:
                                         # {"test_direction","given","when","then"}

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SpecSet:
    domain: str
    specs: list[Spec] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "metadata": self.metadata,
            "specs": [s.to_dict() for s in self.specs],
        }

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @staticmethod
    def load(path: str) -> "SpecSet":
        with open(path) as f:
            d = json.load(f)
        specs = []
        for s in d.get("specs", []):
            ev = [Evidence(**e) for e in s.get("evidence", [])]
            specs.append(Spec(
                spec_id=s["spec_id"],
                target_action=s.get("target_action", ""),
                rule_text=s.get("rule_text", ""),
                kind=s.get("kind", "NORM"),
                clauses=s.get("clauses", []),
                relevant_tools=s.get("relevant_tools", []),
                bindings=s.get("bindings", []),
                cross_tool=s.get("cross_tool", False),
                source_rule=s.get("source_rule", ""),
                confidence=s.get("confidence", "stated"),
                origin=s.get("origin", "prompt"),
                evidence=ev,
                gwt=(
                    [s["gwt"]] if isinstance(s.get("gwt"), dict)
                    else s.get("gwt")
                ),
            ))
        return SpecSet(domain=d.get("domain", ""), specs=specs, metadata=d.get("metadata", {}))
