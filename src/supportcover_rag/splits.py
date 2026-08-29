from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def select_seeded_stratified_ids(
    ids: Sequence[str],
    *,
    sample_size: int,
    seed: int,
    strata: Mapping[str, str] | None = None,
) -> list[str]:
    """Select IDs deterministically, proportionally allocating across strata."""
    ordered_ids = list(ids)
    validate_unique_ids(ordered_ids)
    if sample_size < 0 or sample_size > len(ordered_ids):
        raise ValueError("sample_size must be between zero and the number of IDs.")
    if sample_size == 0:
        return []

    buckets: dict[str, list[str]] = {}
    if strata is None:
        buckets[""] = ordered_ids
    else:
        missing = [item_id for item_id in ordered_ids if item_id not in strata]
        if missing:
            raise ValueError(f"Missing strata for IDs: {', '.join(missing)}")
        for item_id in ordered_ids:
            buckets.setdefault(strata[item_id], []).append(item_id)

    total = len(ordered_ids)
    bucket_items = list(buckets.items())
    allocations: list[int] = []
    remainders: list[int] = []
    for _, bucket_ids in bucket_items:
        allocation, remainder = divmod(sample_size * len(bucket_ids), total)
        allocations.append(allocation)
        remainders.append(remainder)

    remaining = sample_size - sum(allocations)
    remainder_order = sorted(range(len(bucket_items)), key=lambda index: (-remainders[index], index))
    for index in remainder_order[:remaining]:
        allocations[index] += 1

    rng = random.Random(seed)
    selected: set[str] = set()
    for (_, bucket_ids), allocation in zip(bucket_items, allocations, strict=True):
        selected.update(rng.sample(bucket_ids, allocation))
    return [item_id for item_id in ordered_ids if item_id in selected]


def build_record_strata(
    records: Sequence[Mapping[str, Any]],
    dimensions: Sequence[str],
    *,
    id_field: str = "id",
) -> dict[str, str]:
    """Build deterministic combined stratum labels from processed records."""
    normalized_dimensions = [dimension.strip() for dimension in dimensions if dimension.strip()]
    if len(normalized_dimensions) != len(set(normalized_dimensions)):
        raise ValueError("Stratification dimensions must be unique.")
    if not normalized_dimensions:
        return {}

    strata: dict[str, str] = {}
    for index, record in enumerate(records):
        item_id = record.get(id_field)
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"Record {index} is missing a non-empty string '{id_field}'.")
        if item_id in strata:
            raise ValueError(f"Duplicate IDs: {item_id}")
        missing = [dimension for dimension in normalized_dimensions if dimension not in record]
        if missing:
            raise ValueError(
                f"Record {item_id} is missing stratification fields: {', '.join(missing)}"
            )
        values = [record[dimension] for dimension in normalized_dimensions]
        if any(isinstance(value, (dict, list)) for value in values):
            raise ValueError(f"Record {item_id} has a non-scalar stratification value.")
        strata[item_id] = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return strata


def load_json_ids(path: str | Path) -> list[str]:
    """Load an ID list from a JSON array or an object containing an ``ids`` array."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload: Any = json.load(handle)
    expected_sha256: str | None = None
    if isinstance(payload, dict):
        raw_sha256 = payload.get("split_sha256")
        if raw_sha256 is not None and not isinstance(raw_sha256, str):
            raise ValueError("JSON split_sha256 must be a string when provided.")
        expected_sha256 = raw_sha256
        payload = payload.get("ids")
    if not isinstance(payload, list) or not all(isinstance(item_id, str) for item_id in payload):
        raise ValueError("JSON IDs must be an array of strings or an object containing an 'ids' array.")
    validate_unique_ids(payload)
    if expected_sha256 is not None:
        actual_sha256 = ordered_ids_sha256(payload)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Split SHA256 mismatch: expected {expected_sha256}, calculated {actual_sha256}."
            )
    return payload


def validate_unique_ids(ids: Sequence[str]) -> None:
    """Raise when an ID occurs more than once."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for item_id in ids:
        if item_id in seen and item_id not in duplicates:
            duplicates.append(item_id)
        seen.add(item_id)
    if duplicates:
        raise ValueError(f"Duplicate IDs: {', '.join(duplicates)}")


def validate_disjoint_splits(splits: Mapping[str, Sequence[str]]) -> None:
    """Raise when an ID belongs to more than one named split."""
    owners: dict[str, str] = {}
    overlaps: list[str] = []
    for split_name, ids in splits.items():
        validate_unique_ids(ids)
        for item_id in ids:
            owner = owners.setdefault(item_id, split_name)
            if owner != split_name:
                overlaps.append(f"{item_id} ({owner}, {split_name})")
    if overlaps:
        raise ValueError(f"Split IDs are not disjoint: {', '.join(overlaps)}")


def ordered_ids_sha256(ids: Sequence[str]) -> str:
    """Return a stable SHA256 for the ordered ID sequence."""
    validate_unique_ids(ids)
    encoded = json.dumps(list(ids), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_split_manifest(
    ids: Sequence[str],
    *,
    ids_file: str | Path,
    role: str,
    seed: int,
    stratify_by: Sequence[str] | str | None = None,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a self-contained, JSON-serializable manifest for a selected split."""
    ordered_ids = list(ids)
    validate_unique_ids(ordered_ids)
    normalized_role = role.strip().lower()
    if not normalized_role:
        raise ValueError("Split role cannot be empty.")
    if isinstance(stratify_by, str):
        dimensions = [stratify_by] if stratify_by.strip() else []
    else:
        dimensions = [dimension for dimension in (stratify_by or []) if dimension]

    manifest: dict[str, Any] = {
        "ids_file": str(ids_file),
        "role": normalized_role,
        "seed": seed,
        "count": len(ordered_ids),
        "split_sha256": ordered_ids_sha256(ordered_ids),
        "ids": ordered_ids,
    }
    if dimensions:
        manifest["stratify_by"] = dimensions
    if source_path is not None:
        manifest["source_path"] = str(source_path)
    return manifest
