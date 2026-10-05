# GWT Then Independent Claim Inventory

## Goal

Independently inventory every correctness claim in the supplied `then`. You
are not shown another decomposition. The inventory will be mechanically
compared with a separately generated partition.

Two claims are independent when one can be satisfied or violated while the
other is not. Preserve modality, polarity, operation, object, values,
quantities, units, quantifiers, scopes, and comparison direction.

Keep one property's allowed/prohibited alternatives together. Keep a guard
that qualifies an event with that event. Split coordinated actions and split
an action from a separately assessable result. Record explicit ordering,
same-turn, mutual-exclusion, and dependency constraints as relations between
claims.

`given_context` and `when_context` are for reference resolution only. Do not
invent outcome requirements from them. Do not classify Oracle types, bind
tools, or use a Coverage Model.

Every claim and relation must cite exact substrings of `then`. Return
`ambiguous` when materially different inventories remain supported.

## Output

```json
{
  "resolution_status": "resolved",
  "claims": [
    {
      "claim_id": "C01",
      "claim": "A self-contained independently assessable claim.",
      "evidence_spans": ["exact substring of then"]
    }
  ],
  "relations": [
    {
      "relation_id": "Q01",
      "relation": "before",
      "left_claim_id": "C01",
      "right_claim_id": "C02",
      "evidence_spans": ["exact substring of then"]
    }
  ],
  "reason": "Why this inventory is complete."
}
```

Allowed relations are `before`, `after`, `same_turn`, `mutually_exclusive`,
and `requires`. Use consecutive IDs in source order. With `ambiguous`, return
empty `claims` and `relations`.

## Input

```json
{claim_inventory_input}
```
