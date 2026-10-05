"""Standardised output path resolver for experiment results.

Directory layout:
    ExperimentResult/
      {model}/            deepseek-v4-flash | gpt-4o | sonnet-4.6
        {method}/         benchmark | benchmark+ourspec | ablation-db | ablation-tactics | ourmethod
          {domain}/       airline | retail | telecom
            {spec_id}/    one subfolder per spec (for per-spec parallelism)
              db.json     isolated DB snapshot for this spec's runs
              1.json      run 1 result
              2.json      run 2 result
              3.json      run 3 result
              1_metrics.json
              2_metrics.json
              3_metrics.json

    ExperimentResult/spec/
      {domain}/
        ownership.json    ownership-class specs
        numeric.json      numeric-class specs
        sequencing.json   sequencing-class specs
        all.json          all specs combined

Usage::

    from src.core.experiment_paths import resolve_spec_output, resolve_spec_db_path
    out, metrics_out = resolve_spec_output("deepseek-chat", "ourmethod", "airline", "neg_0001", 1)
    db_path = resolve_spec_db_path("deepseek-chat", "ourmethod", "airline", "neg_0001")
"""

from __future__ import annotations

import os

_EXPERIMENT_DIR = "ExperimentResult"

# Maps substrings of model name (lowercase) → directory label
_MODEL_MAP: list[tuple[str, str]] = [
    ("deepseek",        "deepseek-v4-flash"),
    ("flash",           "deepseek-v4-flash"),
    ("gpt-4o",          "gpt-4o"),
    ("gpt4o",           "gpt-4o"),
    ("sonnet",          "sonnet-4.6"),
    ("claude-sonnet",   "sonnet-4.6"),
]

_VALID_METHODS = {
    "benchmark",
    "benchmark+ourspec",
    "ablation-db",
    "ablation-tactics",
    "ourmethod",
}

# spec_id substring → constraint class (for ExperimentResult/spec/ splits)
_SPEC_CLASS_MAP: list[tuple[str, str]] = [
    ("_own_",  "ownership"),
    ("_num_",  "numeric"),
    ("_seq_",  "sequencing"),
    ("_fmt_",  "format"),
]


def model_to_dir(model: str) -> str:
    """Map a model name to the experiment directory label."""
    lower = model.lower()
    # strip a litellm provider prefix such as "openai/qwen3.8-max"
    if "/" in lower:
        lower = lower.rsplit("/", 1)[1]
    if "qwen" in lower:
        # keep the exact model name; the substring map would mislabel e.g. "qwen3.8-flash"
        return lower.replace(":", "-").replace(" ", "-")
    for key, label in _MODEL_MAP:
        if key in lower:
            return label
    return model.replace("/", "_").replace(":", "-").replace(" ", "-")


def spec_class(spec_id: str) -> str:
    """Return the constraint class for a spec_id (ownership/numeric/sequencing/format/policy)."""
    for key, cls in _SPEC_CLASS_MAP:
        if key in spec_id:
            return cls
    return "policy"


def _spec_dir(
    agent_model: str,
    method: str,
    domain: str,
    spec_id: str,
    experiment_dir: str = _EXPERIMENT_DIR,
) -> str:
    if method not in _VALID_METHODS:
        raise ValueError(f"Unknown method '{method}'. Valid: {sorted(_VALID_METHODS)}")
    model_dir = model_to_dir(agent_model)
    # Sanitise spec_id to be a safe directory name
    safe_spec = spec_id.replace("/", "_").replace(":", "-")
    d = os.path.join(experiment_dir, model_dir, method, domain, safe_spec)
    os.makedirs(d, exist_ok=True)
    return d


def resolve_spec_output(
    agent_model: str,
    method: str,
    domain: str,
    spec_id: str,
    run_id: int,
    experiment_dir: str = _EXPERIMENT_DIR,
) -> tuple[str, str]:
    """Return (result_path, metrics_path) for one (spec, run_id) pair.

    Directory: ExperimentResult/{model}/{method}/{domain}/{spec_id}/
    Files:     {run_id}.json, {run_id}_metrics.json
    """
    d = _spec_dir(agent_model, method, domain, spec_id, experiment_dir)
    return os.path.join(d, f"{run_id}.json"), os.path.join(d, f"{run_id}_metrics.json")


def resolve_spec_db_path(
    agent_model: str,
    method: str,
    domain: str,
    spec_id: str,
    experiment_dir: str = _EXPERIMENT_DIR,
) -> str:
    """Return the path for this spec's isolated DB snapshot (db.json)."""
    d = _spec_dir(agent_model, method, domain, spec_id, experiment_dir)
    return os.path.join(d, "db.json")


# ── Legacy helper (kept for backward compatibility with baseline.py etc.) ──────

def resolve_output(
    agent_model: str,
    method: str,
    domain: str,
    run_id: int,
    experiment_dir: str = _EXPERIMENT_DIR,
) -> tuple[str, str]:
    """Return (result_path, metrics_path) at the domain level (no spec subdirectory).

    Used by benchmark and benchmark+ourspec which don't split by spec.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"Unknown method '{method}'. Valid: {sorted(_VALID_METHODS)}")
    model_dir = model_to_dir(agent_model)
    out_dir = os.path.join(experiment_dir, model_dir, method, domain)
    os.makedirs(out_dir, exist_ok=True)
    return (
        os.path.join(out_dir, f"{run_id}.json"),
        os.path.join(out_dir, f"{run_id}_metrics.json"),
    )
