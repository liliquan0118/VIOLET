"""Deterministic feedback-guided experiment selection."""

from .controller import (
    GUIDANCE_DECISION_SCHEMA_VERSION,
    GUIDANCE_HISTORY_SCHEMA_VERSION,
    GuidanceError,
    decide_next_experiment,
    new_guidance_history,
    record_guidance_observation,
)
from .loop import (
    GUIDANCE_BATCH_SCHEMA_VERSION,
    GuidanceLoopError,
    run_approved_guidance_batch,
    run_approved_guidance_batch_files,
)
from .finding import (
    FINAL_FINDING_SCHEMA_VERSION,
    FinalFindingError,
    build_final_finding_bundle,
    build_final_finding_bundle_files,
)

__all__ = [
    "GUIDANCE_DECISION_SCHEMA_VERSION",
    "GUIDANCE_HISTORY_SCHEMA_VERSION",
    "GuidanceError",
    "new_guidance_history",
    "record_guidance_observation",
    "decide_next_experiment",
    "GUIDANCE_BATCH_SCHEMA_VERSION",
    "GuidanceLoopError",
    "run_approved_guidance_batch",
    "run_approved_guidance_batch_files",
    "FINAL_FINDING_SCHEMA_VERSION",
    "FinalFindingError",
    "build_final_finding_bundle",
    "build_final_finding_bundle_files",
]
