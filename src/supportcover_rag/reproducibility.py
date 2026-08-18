from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from supportcover_rag.splits import validate_unique_ids


@dataclass(frozen=True, slots=True)
class VerificationCheck:
    name: str
    passed: bool
    details: str


def _manifest_ids(manifest: Mapping[str, Any]) -> list[str]:
    ids = manifest.get("ids")
    if not isinstance(ids, list) or not all(isinstance(item_id, str) for item_id in ids):
        raise ValueError("Each split manifest must include an already-loaded 'ids' array of strings.")
    validate_unique_ids(ids)
    return ids


def check_development_final_overlap(manifests: Sequence[Mapping[str, Any]]) -> VerificationCheck:
    development_ids: set[str] = set()
    final_ids: set[str] = set()
    for manifest in manifests:
        role = manifest.get("role")
        if not isinstance(role, str):
            raise ValueError("Each split manifest must include a string role.")
        normalized_role = role.strip().lower()
        if normalized_role in {"development", "dev"}:
            development_ids.update(_manifest_ids(manifest))
        elif normalized_role == "final":
            final_ids.update(_manifest_ids(manifest))

    if not development_ids or not final_ids:
        return VerificationCheck(
            name="development_final_overlap",
            passed=False,
            details="Both development and final IDs are required.",
        )
    overlap = sorted(development_ids & final_ids)
    return VerificationCheck(
        name="development_final_overlap",
        passed=not overlap,
        details="overlap=0" if not overlap else f"overlapping IDs: {', '.join(overlap)}",
    )


def check_identical_final_ids(manifests: Sequence[Mapping[str, Any]]) -> VerificationCheck:
    final_ids_by_method: dict[str, list[str]] = {}
    for manifest in manifests:
        if str(manifest.get("role", "")).strip().lower() != "final":
            continue
        method = manifest.get("method")
        if not isinstance(method, str) or not method:
            raise ValueError("Each final split manifest must include a non-empty method.")
        if method in final_ids_by_method:
            raise ValueError(f"Multiple final split manifests were supplied for method '{method}'.")
        final_ids_by_method[method] = _manifest_ids(manifest)

    if len(final_ids_by_method) < 2:
        return VerificationCheck(
            name="identical_final_ids",
            passed=False,
            details="Final IDs for at least two compared methods are required.",
        )
    reference_method, reference_ids = next(iter(final_ids_by_method.items()))
    mismatches = [method for method, ids in final_ids_by_method.items() if ids != reference_ids]
    return VerificationCheck(
        name="identical_final_ids",
        passed=not mismatches,
        details=(
            f"all methods match {reference_method}"
            if not mismatches
            else f"ordered final IDs differ from {reference_method}: {', '.join(mismatches)}"
        ),
    )


def check_expected_hashes(
    records: Mapping[str, Mapping[str, Any]],
    expected: Mapping[str, str],
    *,
    field: str,
) -> VerificationCheck:
    failures: list[str] = []
    for name, expected_hash in expected.items():
        record = records.get(name)
        if record is None:
            failures.append(f"{name}: missing record")
            continue
        actual_hash = record.get(field)
        if actual_hash != expected_hash:
            failures.append(f"{name}: expected {expected_hash}, got {actual_hash!r}")
    return VerificationCheck(
        name=f"expected_{field}",
        passed=not failures,
        details="all expected hashes match" if not failures else "; ".join(failures),
    )


def check_required_metrics(
    prediction_metadata: Mapping[str, Sequence[Mapping[str, Any]]],
    required_metrics: Sequence[str],
) -> VerificationCheck:
    failures: list[str] = []
    if not prediction_metadata:
        failures.append("no prediction metadata supplied")
    for method, records in prediction_metadata.items():
        if not records:
            failures.append(f"{method}: no prediction records")
            continue
        for index, record in enumerate(records):
            missing = [metric for metric in required_metrics if metric not in record]
            if missing:
                failures.append(f"{method}[{index}]: missing {', '.join(missing)}")
    return VerificationCheck(
        name="required_metrics_present",
        passed=not failures,
        details="all required metrics are present" if not failures else "; ".join(failures),
    )


def check_no_duplicate_example_ids(
    prediction_metadata: Mapping[str, Sequence[Mapping[str, Any]]],
) -> VerificationCheck:
    failures: list[str] = []
    for method, records in prediction_metadata.items():
        ids: list[str] = []
        for index, record in enumerate(records):
            example_id = record.get("example_id")
            if not isinstance(example_id, str):
                failures.append(f"{method}[{index}]: missing string example_id")
            else:
                ids.append(example_id)
        seen: set[str] = set()
        duplicates: list[str] = []
        for example_id in ids:
            if example_id in seen and example_id not in duplicates:
                duplicates.append(example_id)
            seen.add(example_id)
        if duplicates:
            failures.append(f"{method}: duplicate IDs {', '.join(duplicates)}")
    return VerificationCheck(
        name="no_duplicate_example_ids",
        passed=not failures,
        details="no duplicate example IDs" if not failures else "; ".join(failures),
    )


def check_fixed_setting(
    config_summaries: Mapping[str, Mapping[str, Any]],
    *,
    field: str,
    required: bool = True,
) -> VerificationCheck:
    if not required:
        return VerificationCheck(name=f"fixed_{field}", passed=True, details="not required")
    if not config_summaries:
        return VerificationCheck(name=f"fixed_{field}", passed=False, details="no config summaries supplied")

    missing = [name for name, summary in config_summaries.items() if field not in summary]
    if missing:
        return VerificationCheck(
            name=f"fixed_{field}",
            passed=False,
            details=f"missing from: {', '.join(missing)}",
        )
    encoded_values = {
        json.dumps(summary[field], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        for summary in config_summaries.values()
    }
    return VerificationCheck(
        name=f"fixed_{field}",
        passed=len(encoded_values) == 1,
        details="fixed across methods" if len(encoded_values) == 1 else "values differ across methods",
    )


def verify_reproducibility(
    manifests: Sequence[Mapping[str, Any]],
    config_summaries: Mapping[str, Mapping[str, Any]],
    prediction_metadata: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    expected_config_hashes: Mapping[str, str],
    expected_split_hashes: Mapping[str, str],
    required_metrics: Sequence[str],
    require_fixed_token_budget: bool = False,
    require_fixed_retrieval_settings: bool = False,
) -> list[VerificationCheck]:
    """Verify already-loaded protocol metadata without performing file discovery."""
    final_manifests: dict[str, Mapping[str, Any]] = {}
    for manifest in manifests:
        if str(manifest.get("role", "")).strip().lower() != "final":
            continue
        method = manifest.get("method")
        if isinstance(method, str):
            if method in final_manifests:
                raise ValueError(f"Multiple final manifests were supplied for method '{method}'.")
            final_manifests[method] = manifest

    return [
        check_development_final_overlap(manifests),
        check_identical_final_ids(manifests),
        check_expected_hashes(config_summaries, expected_config_hashes, field="config_sha256"),
        check_expected_hashes(final_manifests, expected_split_hashes, field="split_sha256"),
        check_required_metrics(prediction_metadata, required_metrics),
        check_no_duplicate_example_ids(prediction_metadata),
        check_fixed_setting(config_summaries, field="model"),
        check_fixed_setting(config_summaries, field="prompt_settings"),
        check_fixed_setting(config_summaries, field="decoding_settings"),
        check_fixed_setting(
            config_summaries,
            field="token_budget",
            required=require_fixed_token_budget,
        ),
        check_fixed_setting(
            config_summaries,
            field="retrieval_settings",
            required=require_fixed_retrieval_settings,
        ),
    ]
