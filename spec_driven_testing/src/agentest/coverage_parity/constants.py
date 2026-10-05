"""Upstream constants required by the generation-contract parity port.

The values are copied from the pinned AgentCoverageTesting source lineage.  They
are kept in a small local module so importing the parity implementation does not
pull in the upstream execution-coverage dependency graph.
"""

DEFAULT_DOMAINS = ("airline", "retail", "telecom")
GAP_SCHEMA_VERSION = "adequacy.policy_coverage_cell_gap_inventory.v1"
MANIFEST_SCHEMA_VERSION = "adequacy.policy_coverage_match_manifest.v1"
