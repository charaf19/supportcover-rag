from __future__ import annotations

import json
from pathlib import Path

import pytest

from supportcover_rag.data import load_examples_by_ids
from supportcover_rag.splits import (
    ordered_ids_sha256,
    select_seeded_stratified_ids,
    validate_disjoint_splits,
    validate_unique_ids,
)


def _processed_row(example_id: str) -> dict[str, object]:
    return {
        "id": example_id,
        "question": f"Question {example_id}?",
        "answer": example_id,
        "type": "bridge",
        "level": "easy",
        "context": [{"title": "Title", "sentences": ["Evidence."]}],
        "supporting_facts": [{"title": "Title", "sent_id": 0}],
    }


def _write_processed(path: Path, ids: list[str]) -> None:
    path.write_text(
        "".join(json.dumps(_processed_row(example_id)) + "\n" for example_id in ids),
        encoding="utf-8",
    )


def test_seeded_stratified_selection_is_deterministic() -> None:
    ids = [f"id-{index}" for index in range(12)]
    strata = {item_id: ("bridge" if index < 8 else "comparison") for index, item_id in enumerate(ids)}

    first = select_seeded_stratified_ids(ids, sample_size=6, seed=17, strata=strata)
    second = select_seeded_stratified_ids(ids, sample_size=6, seed=17, strata=strata)

    assert first == second
    assert len(first) == 6
    assert len(set(first)) == 6


def test_split_disjointness_accepts_zero_overlap_and_identifies_overlap() -> None:
    validate_disjoint_splits({"development": ["a", "b"], "final": ["c", "d"]})

    with pytest.raises(ValueError, match="shared"):
        validate_disjoint_splits({"development": ["a", "shared"], "final": ["shared", "d"]})


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        validate_unique_ids(["a", "b", "a"])


def test_ordered_id_sha256_is_stable_and_order_sensitive() -> None:
    ids = ["a", "b", "c"]

    assert ordered_ids_sha256(ids) == ordered_ids_sha256(list(ids))
    assert ordered_ids_sha256(ids) != ordered_ids_sha256(list(reversed(ids)))


def test_load_examples_by_ids_preserves_requested_order(tmp_path: Path) -> None:
    processed_path = tmp_path / "processed.jsonl"
    _write_processed(processed_path, ["a", "b", "c"])

    examples = load_examples_by_ids(processed_path, ["c", "a"])

    assert [example.example_id for example in examples] == ["c", "a"]


def test_load_examples_by_ids_reports_missing_ids(tmp_path: Path) -> None:
    processed_path = tmp_path / "processed.jsonl"
    _write_processed(processed_path, ["a", "b"])

    with pytest.raises(ValueError, match="missing"):
        load_examples_by_ids(processed_path, ["a", "missing"])
