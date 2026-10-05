"""Adapt an explicitly selected coverage cell to the upstream contract materializer."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Mapping

from agentest.coverage_parity.constants import GAP_SCHEMA_VERSION
from agentest.coverage_parity.generation_contract import materialize_contracts


class SelectedCellError(ValueError):
    """Raised when a selected coverage cell cannot be materialized."""


_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def selected_cell(manifest: Mapping[str, Any], coverage_cell_id: str) -> dict[str, Any]:
    matches = [
        value
        for value in manifest.get("coverage_cells") or []
        if value.get("coverage_cell_id") == coverage_cell_id
    ]
    if len(matches) != 1:
        raise SelectedCellError(
            f"expected exactly one coverage cell {coverage_cell_id!r}, found {len(matches)}"
        )
    return deepcopy(matches[0])


def materialize_selected_cell_contract(
    *,
    domain: str,
    manifest: dict[str, Any],
    coverage_cell_id: str,
    selection_id: str,
) -> dict[str, Any]:
    """Use the parity implementation without requiring a coverage gap."""

    selected_cell(manifest, coverage_cell_id)
    safe_id = _SAFE_ID_RE.sub("_", selection_id).strip("_")
    if not safe_id:
        raise SelectedCellError("selection_id must contain at least one safe character")
    synthetic_gap_id = f"SELECTED::{domain}::{safe_id}"
    synthetic_inventory = {
        "schema_version": GAP_SCHEMA_VERSION,
        "domain": domain,
        "coverage_gap_records": [
            {
                "coverage_gap_id": synthetic_gap_id,
                "coverage_cell_id": coverage_cell_id,
                "gap_status": "explicitly_selected",
            }
        ],
        "failures": [],
    }
    materialized = materialize_contracts(
        domain=domain,
        manifest=manifest,
        gaps=synthetic_inventory,
    )
    contracts = materialized["test_generation_contracts"]
    if len(contracts) != 1:
        raise SelectedCellError(
            f"selected-cell materialization produced {len(contracts)} contracts"
        )
    contract = deepcopy(contracts[0])
    contract["selection_origin"] = {
        "kind": "explicit_coverage_cell_selection",
        "selection_id": selection_id,
        "synthetic_inventory_record_id": synthetic_gap_id,
        "was_coverage_gap_required": False,
    }
    return contract
