from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from supportcover_rag.freeze import build_frozen_manifest
from supportcover_rag.robustness import (
    CROSS_DATASET_ROLE,
    FrozenPackedEvidence,
    aggregate_robustness_runs,
    build_robustness_manifest,
    deterministic_run_id,
    frozen_protocol_descriptor,
    validate_budget_isolation,
    validate_cross_dataset_protocol,
    validate_freeze_boundary,
    validate_main_study_completion,
    validate_model_isolation,
    validate_packed_evidence_reuse,
    load_frozen_packed_evidence,
    write_frozen_packed_evidence,
    verify_robustness_readiness,
)


def _freeze() -> dict[str, object]:
    manifest = build_frozen_manifest(
        supportcover_coefficients={
            "alpha_relevance": 1.0,
            "beta_coverage": 1.2,
            "gamma_redundancy": 0.6,
            "delta_token_cost": 0.15,
            "title_bonus": 0.3,
        },
        mmr_lambda_relevance=0.9,
        development_split_sha256="a" * 64,
        final_split_sha256="b" * 64,
        dataset={"path": "synthetic", "config": "fixture", "final_count": 3},
        model={"model_name_or_path": "fixture/model", "revision": "r1"},
        prompt_settings={"identity": "prompt-v1"},
        decoding_settings={"temperature": 0.0, "do_sample": False},
        token_budget=160,
        retrieval_depth=5,
    )
    manifest.update({"selection_role": "development_only", "final_predictions_inspected": False})
    return manifest


def _reference() -> dict[str, object]:
    return frozen_protocol_descriptor(validate_freeze_boundary(_freeze()))


def _model_descriptor(model_id: str) -> dict[str, object]:
    return {
        "implementation": "transformers",
        "model_id": model_id,
        "revision": "r1",
        "backend": "transformers",
        "precision": "float16",
        "prompt_format_id": "chat-v1",
        "batch_size": 2,
        "generation_tokenizer": f"{model_id}@r1",
    }


def test_budget_isolation_allows_only_budget() -> None:
    reference = _reference()
    candidate = deepcopy(reference)
    candidate["token_budget"] = 96
    validate_budget_isolation(reference, candidate)

    for field, changed in (
        ("coefficients", {**reference["coefficients"], "beta_coverage": 2.4}),
        ("retrieval_depth", 10),
        ("generator", {"model_name_or_path": "other"}),
        ("split_sha256", "c" * 64),
    ):
        invalid = deepcopy(reference)
        invalid[field] = changed
        with pytest.raises(ValueError, match="forbidden"):
            validate_budget_isolation(reference, invalid)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("variant", "changed"),
        ("coefficients", {"alpha_relevance": 1.0, "beta_coverage": 0.3}),
    ],
)
def test_frozen_method_changes_fail_budget_validation(field: str, value: object) -> None:
    reference = _reference()
    candidate = deepcopy(reference)
    candidate[field] = value
    with pytest.raises(ValueError, match="forbidden"):
        validate_budget_isolation(reference, candidate)


@pytest.mark.parametrize("coefficient", ["beta_coverage", "title_bonus", "gamma_redundancy", "delta_token_cost"])
def test_each_frozen_coefficient_is_immutable(coefficient: str) -> None:
    reference = _reference()
    candidate = deepcopy(reference)
    candidate["coefficients"][coefficient] += 0.1
    with pytest.raises(ValueError, match="coefficients"):
        validate_budget_isolation(reference, candidate)


def test_model_isolation_allows_only_explicit_generator_identity() -> None:
    reference = _reference()
    reference["generator"] = _model_descriptor("primary")
    candidate = deepcopy(reference)
    candidate["generator"] = _model_descriptor("alternative")
    validate_model_isolation(reference, candidate)

    for field, changed in (
        ("coefficients", {**reference["coefficients"], "title_bonus": 0.0}),
        ("dataset", {"path": "different"}),
        ("split_sha256", "c" * 64),
        ("token_budget", 128),
    ):
        invalid = deepcopy(candidate)
        invalid[field] = changed
        with pytest.raises(ValueError, match="forbidden"):
            validate_model_isolation(reference, invalid)


def _cross_protocol() -> dict[str, object]:
    return {
        "role": CROSS_DATASET_ROLE,
        "tuning_permitted": False,
        "primary_final_split_sha256": "b" * 64,
        "dataset": {
            "name": "synthetic-multihop",
            "version_or_revision": "v1",
            "split": "test",
            "id_file": "fixture_ids.json",
            "count": 2,
            "id_sha256": "c" * 64,
        },
    }


def test_cross_dataset_requires_explicit_role_and_provenance() -> None:
    validate_cross_dataset_protocol(_cross_protocol())
    for role in ("development", "primary_final"):
        invalid = _cross_protocol()
        invalid["role"] = role
        with pytest.raises(ValueError, match="cross_dataset_robustness"):
            validate_cross_dataset_protocol(invalid)
    missing = _cross_protocol()
    del missing["dataset"]["version_or_revision"]
    with pytest.raises(ValueError, match="provenance"):
        validate_cross_dataset_protocol(missing)
    masquerading = _cross_protocol()
    masquerading["dataset"]["id_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="masquerade"):
        validate_cross_dataset_protocol(masquerading)


def test_main_study_dependency_fails_closed(tmp_path: Path) -> None:
    main_results = tmp_path / "main_results.csv"
    main_results.write_text("method,answer_f1\nsupportcover_final,0.5\n", encoding="utf-8")
    import hashlib

    results_sha = hashlib.sha256(main_results.read_bytes()).hexdigest()
    valid = {
        "schema_version": 1,
        "status": "completed",
        "scientific_role": "primary_final",
        "freeze_sha256": "a" * 64,
        "split_sha256": "b" * 64,
        "num_examples": 3,
        "run_ids": ["EXP001"],
        "main_results": {"path": str(main_results), "sha256": results_sha},
    }
    validate_main_study_completion(valid, freeze_sha256="a" * 64, split_sha256="b" * 64)
    for mutation, message in (
        ({"status": "pending"}, "incomplete"),
        ({"freeze_sha256": "c" * 64}, "freeze"),
        ({"split_sha256": "d" * 64}, "split"),
    ):
        invalid = {**valid, **mutation}
        with pytest.raises(ValueError, match=message):
            validate_main_study_completion(invalid, freeze_sha256="a" * 64, split_sha256="b" * 64)


def test_packed_evidence_reuse_is_exact() -> None:
    cached = FrozenPackedEvidence(
        example_id="id-1",
        retrieved_source_ids=("p1", "p2"),
        selected_support_keys=(("p1", 0),),
        rendered_evidence="evidence",
        used_evidence_tokens=12,
        token_budget=160,
        method="supportcover_final",
        config_id="frozen",
        freeze_sha256="a" * 64,
        split_sha256="b" * 64,
        retrieval_depth=5,
        retrieval_settings={"method": "bm25"},
        packing_tokenizer="fixture-tokenizer@r1",
    )
    request = {
        "freeze_sha256": "a" * 64,
        "split_sha256": "b" * 64,
        "method": "supportcover_final",
        "config_id": "frozen",
        "token_budget": 160,
        "retrieval_depth": 5,
        "retrieval_settings": {"method": "bm25"},
        "packing_tokenizer": "fixture-tokenizer@r1",
    }
    validate_packed_evidence_reuse(cached, request)
    for field, value in (
        ("token_budget", 128),
        ("retrieval_depth", 10),
        ("method", "relevance_only"),
        ("split_sha256", "c" * 64),
    ):
        invalid = {**request, field: value}
        with pytest.raises(ValueError, match=field):
            validate_packed_evidence_reuse(cached, invalid)


def test_packed_evidence_round_trip_is_deterministic(tmp_path: Path) -> None:
    record = FrozenPackedEvidence(
        example_id="synthetic-1",
        retrieved_source_ids=("p1",),
        selected_support_keys=(("p1", 0),),
        rendered_evidence="synthetic evidence",
        used_evidence_tokens=4,
        token_budget=96,
        method="supportcover_final",
        config_id="frozen",
        freeze_sha256="a" * 64,
        split_sha256="b" * 64,
        retrieval_depth=5,
        retrieval_settings={"method": "bm25"},
        packing_tokenizer="fixture@r1",
    )
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_frozen_packed_evidence(first, [record])
    loaded = load_frozen_packed_evidence(first)
    write_frozen_packed_evidence(second, loaded)
    assert loaded == [record]
    assert first.read_bytes() == second.read_bytes()


def _write_run(root: Path, family: str, index: int) -> Path:
    run_dir = root / f"run-{index}"
    run_dir.mkdir(parents=True)
    metrics = {
        "status": "completed",
        "num_examples": 2,
        "answer_em": 0.5,
        "answer_f1": 0.6,
        "support_precision": 0.4,
        "support_recall": 0.7,
        "support_f1": 0.5,
        "coverage_at_budget": 0.7,
        "evidence_tokens": 100.0,
        "retrieval_latency_ms": 1.0,
        "packing_latency_ms": 2.0,
        "generation_latency_ms": 3.0,
        "total_latency_ms": 6.0,
    }
    provenance: dict[str, object] = {
        "family": family,
        "scientific_role": CROSS_DATASET_ROLE if family == "cross_dataset" else "primary_final",
        "run_id": f"run-{index}",
        "num_examples": 2,
        "token_budget": 96 + index,
        "split_sha256": "b" * 64,
        "freeze_sha256": "a" * 64,
        "config_sha256": "c" * 64,
        "code_revision": "fixture",
    }
    if family == "models":
        provenance.update({"generator": _model_descriptor(f"model-{index}"), "packing_tokenizer": "pack@r1"})
    if family == "cross_dataset":
        provenance["dataset"] = _cross_protocol()["dataset"]
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "predictions.jsonl").write_text(
        "".join(json.dumps({"example_id": f"id-{item}"}) + "\n" for item in range(2)),
        encoding="utf-8",
    )
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump({"robustness_run": provenance}), encoding="utf-8"
    )
    return run_dir


@pytest.mark.parametrize(
    ("family", "filename"),
    [
        ("budget", "budget_results.csv"),
        ("models", "model_results.csv"),
        ("cross_dataset", "cross_dataset_results.csv"),
    ],
)
def test_robustness_aggregation_is_deterministic(tmp_path: Path, family: str, filename: str) -> None:
    runs = [_write_run(tmp_path / family, family, index) for index in (2, 1)]
    first_path = tmp_path / "first" / filename
    second_path = tmp_path / "second" / filename
    first = aggregate_robustness_runs(family, runs, output_path=first_path)
    second = aggregate_robustness_runs(family, list(reversed(runs)), output_path=second_path)
    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    with first_path.open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2


def test_run_ids_are_deterministic_and_family_specific() -> None:
    variation = {"token_budget": 96}
    assert deterministic_run_id("budget", variation, "a" * 64) == deterministic_run_id(
        "budget", variation, "a" * 64
    )
    assert deterministic_run_id("budget", variation, "a" * 64) != deterministic_run_id(
        "models", variation, "a" * 64
    )


def test_completed_robustness_manifest_hashes_source_artifacts(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "budget", "budget", 1)
    aggregate = tmp_path / "budget_results.csv"
    aggregate_robustness_runs("budget", [run], output_path=aggregate)
    manifest = build_robustness_manifest(
        family="budget",
        freeze_sha256="a" * 64,
        main_study_completion={"path": "synthetic-main-completion.json", "sha256": "d" * 64},
        run_dirs=[run],
        aggregate_path=aggregate,
        population={"role": "primary_final", "split_sha256": "b" * 64, "N": 2},
        frozen_method={"method": "supportcover_final"},
        allowed_changed_fields=["token_budget"],
        forbidden_changed_fields=["coefficients", "retrieval_depth"],
        code_revision="fixture",
        environment_reference="synthetic-environment.json",
        created_at="2026-01-01T00:00:00Z",
    )
    assert manifest["status"] == "completed"
    assert manifest["source_artifacts"][0]["predictions_sha256"]
    assert manifest["aggregate"]["sha256"]


def test_readiness_is_metadata_only_and_blocked_without_prerequisites(tmp_path: Path) -> None:
    plan = {
        "schema_version": 1,
        "robustness_family": "budget",
        "freeze_manifest": "configs/frozen/final_manifest.json",
        "freeze_sha256": "UNRESOLVED",
        "main_study_completion_manifest": "outputs/final/main_study_completion.json",
        "supported_budgets": [96, 128, 160, 192],
        "allowed_changed_fields": ["token_budget"],
    }
    paths = []
    for family in ("budget", "models", "cross_dataset"):
        path = tmp_path / f"{family}.yaml"
        payload = {**plan, "robustness_family": family}
        if family == "models":
            payload.update({"allowed_changed_fields": ["generator"], "models": []})
        if family == "cross_dataset":
            payload.update(_cross_protocol())
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        paths.append(path)
    report = verify_robustness_readiness(
        freeze_manifest_path=tmp_path / "missing-freeze.json",
        main_completion_path=tmp_path / "missing-main.json",
        budget_plan_path=paths[0],
        model_plan_path=paths[1],
        cross_dataset_plan_path=paths[2],
    )
    assert not report.ready
    assert report.to_dict()["evaluation_scope"] == "metadata_only_no_final_outcomes"
