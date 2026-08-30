from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any


_REQUIRED_SUPPORTCOVER_COEFFICIENTS = {
    "alpha_relevance",
    "beta_coverage",
    "gamma_redundancy",
    "delta_token_cost",
    "title_bonus",
}


def canonicalize(value: Any) -> Any:
    """Convert resolved configuration data to a deterministic JSON-compatible form."""
    if is_dataclass(value) and not isinstance(value, type):
        return canonicalize(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): canonicalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [canonicalize(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported value in frozen configuration: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def build_frozen_manifest(
    *,
    supportcover_coefficients: Mapping[str, float],
    mmr_lambda_relevance: float,
    development_split_sha256: str,
    final_split_sha256: str,
    dataset: Any,
    model: Any,
    prompt_settings: Mapping[str, Any],
    decoding_settings: Mapping[str, Any],
    token_budget: int,
    retrieval_depth: int,
    selection_evidence: Mapping[str, Any] | None = None,
    supportcover_method: str = "supportcover_final",
    supportcover_variant: str = "development_selected",
) -> dict[str, Any]:
    missing = sorted(_REQUIRED_SUPPORTCOVER_COEFFICIENTS - set(supportcover_coefficients))
    if missing:
        raise ValueError(f"Missing final SupportCover coefficients: {', '.join(missing)}")
    if not development_split_sha256:
        raise ValueError("A non-empty development split SHA256 is required.")
    if not final_split_sha256:
        raise ValueError("A non-empty final split SHA256 is required.")

    configuration = canonicalize(
        {
            "supportcover_method": supportcover_method,
            "supportcover_variant": supportcover_variant,
            "supportcover_coefficients": dict(supportcover_coefficients),
            "mmr_lambda_relevance": mmr_lambda_relevance,
            "development_split_sha256": development_split_sha256,
            "final_split_sha256": final_split_sha256,
            "dataset": dataset,
            "model": model,
            "prompt_settings": dict(prompt_settings),
            "decoding_settings": dict(decoding_settings),
            "token_budget": token_budget,
            "retrieval_depth": retrieval_depth,
            "selection_evidence": dict(selection_evidence or {}),
        }
    )
    return {
        "schema_version": 2,
        "configuration": configuration,
        "config_sha256": canonical_sha256(configuration),
    }
