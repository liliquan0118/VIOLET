"""Resolve and validate pinned, read-only policy artifact bundles."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ARTIFACT_BUNDLE_SCHEMA_VERSION = "agentspectesting.artifact-bundle/v0.1"
ACCEPTED_ARTIFACT_CATALOG_SCHEMA_VERSION = (
    "agentspectesting.accepted-artifact-catalog/v0.1"
)

CAP_OPERATION_POLICY_CLOSURES = "policy.operation-closures"
CAP_COVERAGE_MATCH_MANIFEST = "coverage.match-manifest"
CAP_DECISION_CONFIGURATION_MODEL = "decision.configuration-model"
CAP_EXECUTION_MATCHING_CONTRACTS = "execution.matching-contracts"
CAP_AGENT_TOOL_SCHEMAS = "source.agent-tool-schemas"
CAP_TOOL_OBSERVABLE_ENDPOINTS = "execution.tool-observable-endpoints"
CAP_ACCEPTED_POLICY_MODEL = "policy.accepted-model"
CAP_POLICY_SEMANTIC_UNITS = "policy.semantic-units"
CAP_POLICY_PATH_TEST_SPECIFICATIONS = "policy.path-test-specifications"
CAP_POLICY_SOURCE_EVIDENCE = "source.policy-evidence-index"


@dataclass(frozen=True)
class ArtifactSchemaRegistration:
    """One accepted artifact schema and the capabilities it provides."""

    schema_version: str
    capabilities: tuple[str, ...]
    required_array_fields: tuple[str, ...]

    def validate(self, document: Mapping[str, Any], *, artifact_id: str) -> None:
        for field in self.required_array_fields:
            if not isinstance(document.get(field), list):
                raise ArtifactBundleError(
                    f"artifact {artifact_id} field {field!r} must be an array"
                )


DEFAULT_ARTIFACT_SCHEMA_REGISTRY = (
    ArtifactSchemaRegistration(
        schema_version="adequacy.agent_spec.v3",
        capabilities=(
            "source.agent-spec",
            "source.system-prompt",
            CAP_AGENT_TOOL_SCHEMAS,
        ),
        required_array_fields=("tools",),
    ),
    ArtifactSchemaRegistration(
        schema_version="adequacy.policy_tool_implementation_catalog.v1",
        capabilities=(
            "execution.tool-implementation-catalog",
            CAP_TOOL_OBSERVABLE_ENDPOINTS,
        ),
        required_array_fields=("tools",),
    ),
    ArtifactSchemaRegistration(
        schema_version="adequacy.policy_extraction_proposal.v1",
        capabilities=(
            CAP_ACCEPTED_POLICY_MODEL,
            "source.accepted-policy-lineage",
        ),
        required_array_fields=(
            "selected_input_ids",
            "proposal_results",
        ),
    ),
    ArtifactSchemaRegistration(
        schema_version="adequacy.policy_extraction_proposal_input.v1",
        capabilities=(
            CAP_POLICY_SOURCE_EVIDENCE,
            "source.policy-proposal-inputs",
        ),
        required_array_fields=(
            "policy_proposal_inputs",
            "blocked_policy_extraction_inputs",
        ),
    ),
    ArtifactSchemaRegistration(
        schema_version="adequacy.policy_extraction_semantic_unit_model.v1",
        capabilities=(
            CAP_POLICY_SEMANTIC_UNITS,
            "policy.policy-index",
            "source.semantic-unit-lineage",
        ),
        required_array_fields=(
            "semantic_units",
            "policy_index",
        ),
    ),
    ArtifactSchemaRegistration(
        schema_version=(
            "adequacy.policy_path_test_specifications_with_eligibility_witnesses.v1"
        ),
        capabilities=(
            CAP_POLICY_PATH_TEST_SPECIFICATIONS,
            "evaluation.dataset-independent-path-specifications",
            "source.policy-path-lineage",
        ),
        required_array_fields=("path_test_specifications",),
    ),
    ArtifactSchemaRegistration(
        schema_version="adequacy.policy_operation_closures.v2",
        capabilities=(
            CAP_OPERATION_POLICY_CLOSURES,
            "policy.operation-families",
            "source.operation-requirements",
        ),
        required_array_fields=(
            "operation_policy_closures",
            "operation_policy_families",
            "human_reviewed_operation_requirements",
        ),
    ),
    ArtifactSchemaRegistration(
        schema_version="adequacy.policy_coverage_match_manifest.v1",
        capabilities=(
            CAP_COVERAGE_MATCH_MANIFEST,
            "coverage.cells",
            "evaluation.factor-contracts",
        ),
        required_array_fields=(
            "coverage_cells",
            "operation_family_match_manifests",
        ),
    ),
    ArtifactSchemaRegistration(
        schema_version="adequacy.policy_decision_configuration_model.v1",
        capabilities=(
            CAP_DECISION_CONFIGURATION_MODEL,
            "experiment.operation-decision-regions",
            "experiment.local-behavior-regions",
            "experiment.unconditional-requirements",
            "evaluation.predicate-roles",
        ),
        required_array_fields=(
            "operation_outcome_decision_regions",
            "local_behavior_decision_regions",
            "unconditional_requirement_cells",
            "predicate_role_inventory",
        ),
    ),
    ArtifactSchemaRegistration(
        schema_version="adequacy.policy_execution_matching_contracts.v1",
        capabilities=(
            CAP_EXECUTION_MATCHING_CONTRACTS,
            "evaluation.operation-contracts",
            "evaluation.path-contracts",
            "evaluation.coverage-contract-index",
        ),
        required_array_fields=(
            "operation_family_contracts",
            "path_match_contracts",
            "coverage_contract_index",
        ),
    ),
)


class ArtifactBundleError(ValueError):
    """Raised when an artifact bundle is incomplete, changed, or incompatible."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic JSON bytes used for compiler fingerprints."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _load_json_bytes(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ArtifactBundleError(f"cannot read artifact {path}: {exc}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ArtifactBundleError(f"artifact {path} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactBundleError(f"artifact {path} must contain one JSON object")
    return value, digest


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactBundleError(f"{path} must be an object")
    return value


def _schema_registry(
    registrations: tuple[ArtifactSchemaRegistration, ...],
) -> dict[str, ArtifactSchemaRegistration]:
    result: dict[str, ArtifactSchemaRegistration] = {}
    all_capabilities = set()
    for registration in registrations:
        if (
            not isinstance(registration.schema_version, str)
            or not registration.schema_version
        ):
            raise ArtifactBundleError(
                "artifact schema registration must have a non-empty schema_version"
            )
        if registration.schema_version in result:
            raise ArtifactBundleError(
                "duplicate artifact schema registration: "
                f"{registration.schema_version!r}"
            )
        if not registration.capabilities or any(
            not isinstance(value, str) or not value
            for value in registration.capabilities
        ):
            raise ArtifactBundleError(
                f"artifact schema {registration.schema_version!r} must provide capabilities"
            )
        duplicate_capabilities = sorted(
            value
            for value in set(registration.capabilities)
            if registration.capabilities.count(value) > 1
        )
        if duplicate_capabilities:
            raise ArtifactBundleError(
                f"artifact schema {registration.schema_version!r} repeats capabilities: "
                f"{duplicate_capabilities!r}"
            )
        result[registration.schema_version] = registration
        all_capabilities.update(registration.capabilities)
    if not result or not all_capabilities:
        raise ArtifactBundleError("artifact schema registry must not be empty")
    return result


def _sha256(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
    ):
        raise ArtifactBundleError(f"{path} must be a lowercase SHA-256 hex digest")
    return value


def _catalog_fingerprint_payload(catalog: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": catalog["schema_version"],
        "catalog_id": catalog["catalog_id"],
        "domain": catalog["domain"],
        "bundle_identity": deepcopy(catalog["bundle_identity"]),
        "artifacts": {
            artifact_id: {
                "sha256": entry["metadata"]["sha256"],
                "schema_version": entry["metadata"]["schema_version"],
                "capabilities": deepcopy(entry["metadata"]["capabilities"]),
                "document_fingerprint": entry["metadata"]["document_fingerprint"],
            }
            for artifact_id, entry in sorted(catalog["artifacts"].items())
        },
        "capability_index": deepcopy(catalog["capability_index"]),
    }


def resolve_artifact_bundle(
    bundle: Mapping[str, Any],
    *,
    base_dir: Path,
    schema_registry: tuple[ArtifactSchemaRegistration, ...] = (
        DEFAULT_ARTIFACT_SCHEMA_REGISTRY
    ),
) -> dict[str, Any]:
    """Load accepted artifacts and index them by declared semantic capability."""

    if bundle.get("schema_version") != ARTIFACT_BUNDLE_SCHEMA_VERSION:
        raise ArtifactBundleError(
            f"unsupported artifact bundle schema: {bundle.get('schema_version')!r}"
        )
    domain = bundle.get("domain")
    if not isinstance(domain, str) or not domain:
        raise ArtifactBundleError("artifact bundle domain must be a non-empty string")
    bundle_id = bundle.get("bundle_id")
    if not isinstance(bundle_id, str) or not bundle_id:
        raise ArtifactBundleError("artifact bundle ID must be a non-empty string")
    compile_run_id = bundle.get("compile_run_id")
    if not isinstance(compile_run_id, str) or not compile_run_id:
        raise ArtifactBundleError(
            "artifact bundle compile_run_id must be a non-empty string"
        )
    if bundle.get("read_only") is not True:
        raise ArtifactBundleError("accepted artifact bundle must declare read_only=true")
    artifact_specs = _require_mapping(bundle.get("artifacts"), "$.artifacts")
    if not artifact_specs:
        raise ArtifactBundleError("artifact bundle must contain at least one artifact")
    registered = _schema_registry(schema_registry)

    resolved: dict[str, Any] = {}
    capability_index: dict[str, list[str]] = {}
    for name in sorted(artifact_specs):
        if not isinstance(name, str) or not name:
            raise ArtifactBundleError("artifact IDs must be non-empty strings")
        spec = _require_mapping(artifact_specs[name], f"$.artifacts.{name}")
        raw_path = spec.get("path")
        expected_hash = _sha256(spec.get("sha256"), f"$.artifacts.{name}.sha256")
        expected_schema = spec.get("schema_version")
        if not isinstance(raw_path, str) or not raw_path:
            raise ArtifactBundleError(f"$.artifacts.{name}.path must be a string")
        if not isinstance(expected_schema, str) or not expected_schema:
            raise ArtifactBundleError(f"$.artifacts.{name}.schema_version must be a string")
        registration = registered.get(expected_schema)
        if registration is None:
            raise ArtifactBundleError(
                f"artifact {name} uses unregistered schema {expected_schema!r}"
            )
        path = Path(raw_path)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        document, actual_hash = _load_json_bytes(path)
        if actual_hash != expected_hash:
            raise ArtifactBundleError(
                f"artifact hash mismatch for {name}: expected {expected_hash}, got {actual_hash}"
            )
        if document.get("schema_version") != expected_schema:
            raise ArtifactBundleError(
                f"artifact schema mismatch for {name}: expected {expected_schema!r}, "
                f"got {document.get('schema_version')!r}"
            )
        if document.get("domain") != domain:
            raise ArtifactBundleError(
                f"artifact domain mismatch for {name}: expected {domain!r}, "
                f"got {document.get('domain')!r}"
            )
        if document.get("failures"):
            raise ArtifactBundleError(f"artifact {name} contains recorded failures")
        registration.validate(document, artifact_id=name)
        capabilities = sorted(registration.capabilities)
        resolved[name] = {
            "metadata": {
                "artifact_id": name,
                "path": str(path),
                "sha256": actual_hash,
                "schema_version": expected_schema,
                "capabilities": capabilities,
                "document_fingerprint": content_sha256(document),
            },
            "document": document,
        }
        for capability in capabilities:
            capability_index.setdefault(capability, []).append(name)

    capability_index = {
        capability: sorted(providers)
        for capability, providers in sorted(capability_index.items())
    }
    catalog = {
        "schema_version": ACCEPTED_ARTIFACT_CATALOG_SCHEMA_VERSION,
        "catalog_id": f"accepted-artifacts::{bundle_id}",
        "domain": domain,
        "bundle_identity": {
            "artifact_bundle_id": bundle_id,
            "compile_run_id": compile_run_id,
            "bundle_schema_version": ARTIFACT_BUNDLE_SCHEMA_VERSION,
        },
        "artifacts": resolved,
        "capability_index": capability_index,
    }
    catalog["catalog_fingerprint"] = content_sha256(
        _catalog_fingerprint_payload(catalog)
    )
    validate_accepted_artifact_catalog(catalog)

    return {
        "bundle": deepcopy(dict(bundle)),
        "bundle_path": None,
        "domain": domain,
        "catalog": catalog,
        # Compatibility projection: old consumers see the same entry objects.
        "artifacts": catalog["artifacts"],
    }


def validate_accepted_artifact_catalog(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate catalog structure, capability index, and content fingerprints."""

    catalog = _require_mapping(value, "$catalog")
    if catalog.get("schema_version") != ACCEPTED_ARTIFACT_CATALOG_SCHEMA_VERSION:
        raise ArtifactBundleError(
            f"unsupported accepted artifact catalog schema: {catalog.get('schema_version')!r}"
        )
    for field in ("catalog_id", "domain"):
        if not isinstance(catalog.get(field), str) or not catalog.get(field):
            raise ArtifactBundleError(f"$.catalog.{field} must be a non-empty string")
    identity = _require_mapping(catalog.get("bundle_identity"), "$.catalog.bundle_identity")
    for field in ("artifact_bundle_id", "compile_run_id", "bundle_schema_version"):
        if not isinstance(identity.get(field), str) or not identity.get(field):
            raise ArtifactBundleError(
                f"$.catalog.bundle_identity.{field} must be a non-empty string"
            )
    artifacts = _require_mapping(catalog.get("artifacts"), "$.catalog.artifacts")
    if not artifacts:
        raise ArtifactBundleError("accepted artifact catalog must not be empty")
    rebuilt_index: dict[str, list[str]] = {}
    for artifact_id, raw_entry in sorted(artifacts.items()):
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ArtifactBundleError("catalog artifact IDs must be non-empty strings")
        entry = _require_mapping(raw_entry, f"$.catalog.artifacts.{artifact_id}")
        metadata = _require_mapping(
            entry.get("metadata"),
            f"$.catalog.artifacts.{artifact_id}.metadata",
        )
        if metadata.get("artifact_id") != artifact_id:
            raise ArtifactBundleError(
                f"catalog artifact key {artifact_id!r} disagrees with metadata artifact_id"
            )
        _sha256(metadata.get("sha256"), f"$.catalog.artifacts.{artifact_id}.metadata.sha256")
        _sha256(
            metadata.get("document_fingerprint"),
            f"$.catalog.artifacts.{artifact_id}.metadata.document_fingerprint",
        )
        if not isinstance(metadata.get("schema_version"), str) or not metadata.get(
            "schema_version"
        ):
            raise ArtifactBundleError(
                f"$.catalog.artifacts.{artifact_id}.metadata.schema_version must be non-empty"
            )
        capabilities = metadata.get("capabilities")
        if (
            not isinstance(capabilities, list)
            or not capabilities
            or capabilities != sorted(set(capabilities))
            or any(not isinstance(item, str) or not item for item in capabilities)
        ):
            raise ArtifactBundleError(
                f"$.catalog.artifacts.{artifact_id}.metadata.capabilities must be a sorted unique string array"
            )
        document = _require_mapping(
            entry.get("document"),
            f"$.catalog.artifacts.{artifact_id}.document",
        )
        if content_sha256(document) != metadata["document_fingerprint"]:
            raise ArtifactBundleError(
                f"catalog document fingerprint mismatch for {artifact_id}"
            )
        for capability in capabilities:
            rebuilt_index.setdefault(capability, []).append(artifact_id)
    rebuilt_index = {
        capability: sorted(providers)
        for capability, providers in sorted(rebuilt_index.items())
    }
    actual_index = _require_mapping(
        catalog.get("capability_index"),
        "$.catalog.capability_index",
    )
    if dict(actual_index) != rebuilt_index:
        raise ArtifactBundleError(
            "catalog capability index does not match artifact capability declarations"
        )
    expected = _sha256(
        catalog.get("catalog_fingerprint"),
        "$.catalog.catalog_fingerprint",
    )
    actual = content_sha256(_catalog_fingerprint_payload(catalog))
    if actual != expected:
        raise ArtifactBundleError(
            f"accepted artifact catalog fingerprint mismatch: expected {expected}, computed {actual}"
        )
    return deepcopy(dict(catalog))


def resolve_artifact_bundle_file(
    path: str | Path,
    *,
    schema_registry: tuple[ArtifactSchemaRegistration, ...] = (
        DEFAULT_ARTIFACT_SCHEMA_REGISTRY
    ),
) -> dict[str, Any]:
    bundle_path = Path(path).resolve()
    try:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactBundleError(f"cannot load artifact bundle {bundle_path}: {exc}") from exc
    if not isinstance(bundle, dict):
        raise ArtifactBundleError("artifact bundle must contain one JSON object")
    resolved = resolve_artifact_bundle(
        bundle,
        base_dir=bundle_path.parent,
        schema_registry=schema_registry,
    )
    resolved["bundle_path"] = str(bundle_path)
    return resolved


def artifact_capability_status(
    resolved: Mapping[str, Any],
    capability: str,
) -> dict[str, Any]:
    """Describe capability availability without guessing a provider."""

    if not isinstance(capability, str) or not capability:
        raise ArtifactBundleError("artifact capability must be a non-empty string")
    catalog = _require_mapping(resolved.get("catalog"), "$.catalog")
    index = _require_mapping(catalog.get("capability_index"), "$.catalog.capability_index")
    providers = index.get(capability) or []
    if not isinstance(providers, list) or any(
        not isinstance(value, str) or not value for value in providers
    ):
        raise ArtifactBundleError(
            f"$.catalog.capability_index.{capability} must be an artifact ID array"
        )
    if not providers:
        status = "missing"
    elif len(providers) == 1:
        status = "available_unique"
    else:
        status = "available_multiple"
    return {
        "capability": capability,
        "status": status,
        "provider_artifact_ids": deepcopy(providers),
    }


def artifacts_for_capability(
    resolved: Mapping[str, Any],
    capability: str,
) -> list[dict[str, Any]]:
    status = artifact_capability_status(resolved, capability)
    catalog = _require_mapping(resolved.get("catalog"), "$.catalog")
    artifacts = _require_mapping(catalog.get("artifacts"), "$.catalog.artifacts")
    return [
        artifacts[artifact_id]
        for artifact_id in status["provider_artifact_ids"]
    ]


def artifact_for_capability(
    resolved: Mapping[str, Any],
    capability: str,
) -> Mapping[str, Any]:
    """Return the unique provider or report missing/ambiguous capability."""

    status = artifact_capability_status(resolved, capability)
    if status["status"] == "missing":
        raise ArtifactBundleError(
            f"accepted artifact capability {capability!r} is missing"
        )
    if status["status"] == "available_multiple":
        raise ArtifactBundleError(
            f"accepted artifact capability {capability!r} has multiple providers: "
            f"{status['provider_artifact_ids']!r}"
        )
    return artifacts_for_capability(resolved, capability)[0]


def artifact_document_for_capability(
    resolved: Mapping[str, Any],
    capability: str,
) -> Mapping[str, Any]:
    entry = artifact_for_capability(resolved, capability)
    return _require_mapping(entry.get("document"), f"artifact[{capability}].document")


def artifact_identity_view(resolved: Mapping[str, Any]) -> dict[str, Any]:
    catalog = _require_mapping(resolved.get("catalog"), "$.catalog")
    identity = _require_mapping(
        catalog.get("bundle_identity"),
        "$.catalog.bundle_identity",
    )
    artifacts = _require_mapping(catalog.get("artifacts"), "$.catalog.artifacts")
    return {
        "artifact_bundle_id": identity.get("artifact_bundle_id"),
        "compile_run_id": identity.get("compile_run_id"),
        "artifact_catalog_fingerprint": catalog.get("catalog_fingerprint"),
        "artifact_hashes": {
            artifact_id: _require_mapping(
                entry.get("metadata"),
                f"$.catalog.artifacts.{artifact_id}.metadata",
            ).get("sha256")
            for artifact_id, entry in sorted(artifacts.items())
        },
    }


def provenance_view(resolved: Mapping[str, Any]) -> dict[str, Any]:
    catalog = _require_mapping(resolved.get("catalog"), "$.catalog")
    artifacts = _require_mapping(catalog.get("artifacts"), "$.catalog.artifacts")
    identity = _require_mapping(catalog.get("bundle_identity"), "$.catalog.bundle_identity")
    return {
        "artifact_bundle_id": identity.get("artifact_bundle_id"),
        "compile_run_id": identity.get("compile_run_id"),
        "artifact_bundle_path": resolved.get("bundle_path"),
        "artifacts": {
            name: {
                "path": value["metadata"]["path"],
                "sha256": value["metadata"]["sha256"],
                "schema_version": value["metadata"]["schema_version"],
            }
            for name, value in sorted(artifacts.items())
        },
    }
