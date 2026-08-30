from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from supportcover_rag.config import AppConfig
from supportcover_rag.freeze import canonical_json


FAIR_COMPARISON_FIELDS = (
    "dataset",
    "split_role",
    "split_sha256",
    "model",
    "model_revision",
    "model_precision",
    "prompt_settings",
    "decoding_settings",
    "token_budget",
    "retrieval_depth",
    "retrieval_parameters",
    "retrieval_mode",
    "retrieval_corpus",
    "seed",
)


@dataclass(frozen=True, slots=True)
class FairComparisonValidation:
    methods: tuple[str, ...]
    checked_fields: tuple[str, ...]
    passed: bool = True


def build_final_protocol_descriptor(
    config: AppConfig,
    *,
    split_sha256: str,
    method_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build shared and method-specific provenance from one resolved configuration."""
    return {
        "dataset": {
            "path": config.raw_data.dataset_path,
            "config": config.raw_data.dataset_config,
        },
        "split_role": config.split.role,
        "split_sha256": split_sha256,
        "model": config.generation.model_name_or_path,
        "model_revision": config.generation.model_revision,
        "model_precision": config.generation.dtype,
        "prompt_settings": {
            "include_titles": config.prompting.include_titles,
            "allow_abstain": config.prompting.allow_abstain,
            "system_instruction": config.prompting.system_instruction,
            "user_instruction": config.prompting.user_instruction,
        },
        "decoding_settings": {
            "backend": config.generation.backend,
            "temperature": config.generation.temperature,
            "max_new_tokens": config.generation.max_new_tokens,
            "do_sample": config.generation.do_sample,
            "think": config.generation.think,
            "stream": config.generation.stream,
        },
        "token_budget": config.supportcover.token_budget,
        "retrieval_depth": config.retrieval.top_k_paragraphs,
        "retrieval_parameters": {
            "method": config.retrieval.method,
            "bm25_k1": config.retrieval.bm25_k1,
            "bm25_b": config.retrieval.bm25_b,
        },
        "retrieval_mode": config.retrieval.evaluation_mode,
        "retrieval_corpus": {
            "manifest": config.retrieval.corpus_manifest or None,
            "index": config.retrieval.index_path or None,
            "tokenizer_identity": config.retrieval.tokenizer_identity,
        },
        "seed": config.seed,
        "method_parameters": dict(method_parameters or {}),
        "execution": {
            "batch_size": config.generation.batch_size,
            "device": config.generation.device,
        },
    }


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
