# GWT Then Predicate Partition

## Goal

Partition the supplied `then` into the smallest independently assessable
correctness claims. Two claims are independent when one can be satisfied or
violated while the other is not.

`given_context` and `when_context` may resolve the operation or object already
named by the branch. They are context only: do not turn them into new outcome
claims.

## Boundary Rules

- Separate coordinated actions, checks, communications, or resulting state
  properties when each can independently be correct or wrong.
- Separate an event from a separately stated result when the event can occur
  while the result is wrong.
- Keep a property and its allowed or prohibited value set together.
- Keep comparison operands, comparison direction, quantity, unit, quantifier,
  object identity, scope, polarity, and negation with the claim they qualify.
- Keep a condition that restricts a prohibited or permitted event with that
  event. For example, "must not book more than five passengers" is one
  prohibited guarded event, not a prohibition on all booking plus a separate
  passenger-count claim.
- Preserve `must`/`should` as `required`, `must not`/`should not` as
  `prohibited`, and `may`/`can` as `permitted`.
- Represent an explicit ordering, same-turn, mutual-exclusion, or dependency
  constraint as a relation between claims. Do not merge ordered actions into
  one claim.
- Do not classify observation channels, bind tools, consult a Coverage Model,
  or write executable Oracle logic.

Every atom and relation must cite one or more exact substrings of `then`.
Return `ambiguous` rather than choosing between materially different
partitions.

## Output

```json
{
  "resolution_status": "resolved",
  "atoms": [
    {
      "atom_id": "T01",
      "modality": "required",
      "claim": "A self-contained independently assessable claim.",
      "evidence_spans": ["exact substring of then"]
    }
  ],
  "relations": [
    {
      "relation_id": "R01",
      "relation": "before",
      "left_atom_id": "T01",
      "right_atom_id": "T02",
      "evidence_spans": ["exact substring of then"]
    }
  ],
  "reason": "Why these are the independently assessable boundaries."
}
```

Allowed relations are `before`, `after`, `same_turn`, `mutually_exclusive`,
and `requires`. Use consecutive IDs in source order. With `ambiguous`, return
empty `atoms` and `relations`.

## Input

```json
{predicate_partition_input}
```
