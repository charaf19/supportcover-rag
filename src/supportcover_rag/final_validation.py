from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from supportcover_rag.freeze import canonical_json


FAIR_COMPARISON_FIELDS = (
    "split_sha256",
    "model",
    "prompt_settings",
    "decoding_settings",
    "token_budget",
    "retrieval_depth",
    "dataset",
)


@dataclass(frozen=True, slots=True)
class FairComparisonValidation:
    methods: tuple[str, ...]
    checked_fields: tuple[str, ...]
    passed: bool = True


def validate_fair_comparison(
    resolved_configs: Mapping[str, Mapping[str, Any]],
) -> FairComparisonValidation:
    if len(resolved_configs) < 2:
        raise ValueError("Fair-comparison validation requires at least two resolved method configurations.")

    methods = tuple(resolved_configs)
    reference_method = methods[0]
    reference = resolved_configs[reference_method]
    mismatches: list[str] = []
    for field in FAIR_COMPARISON_FIELDS:
        missing = [method for method, config in resolved_configs.items() if field not in config]
        if missing:
            raise ValueError(f"Resolved comparison field '{field}' is missing for: {', '.join(missing)}")
        reference_value = canonical_json(reference[field])
        different = [
            method
            for method in methods[1:]
            if canonical_json(resolved_configs[method][field]) != reference_value
        ]
        if different:
            mismatches.append(
                f"{field} differs from {reference_method} for: {', '.join(different)}"
            )

    if mismatches:
        raise ValueError("Unfair resolved experiment comparison: " + "; ".join(mismatches))
    return FairComparisonValidation(
        methods=methods,
        checked_fields=FAIR_COMPARISON_FIELDS,
    )
