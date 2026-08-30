from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import yaml

from supportcover_rag.config import (
    AppConfig,
    ExperimentsConfig,
    ExternalCompressorConfig,
    FinalStudyConfig,
    FreezeConfig,
    PathsConfig,
    RawDataConfig,
    RuntimeConfig,
    SplitConfig,
)
from supportcover_rag.experiment_outputs import ExperimentFamily, ExperimentOutputManager
from supportcover_rag.evaluation import aggregate_records
from supportcover_rag.external_baselines import ExternalCompressorMetadata, validate_external_compressor_metadata
from supportcover_rag.final_execution import (
    OPTIONAL_EXTERNAL_METHOD,
    PRIMARY_MAIN_METHODS,
    aggregate_final_main_runs,
    validate_final_execution_config,
    verify_final_readiness,
)
from supportcover_rag.final_validation import build_final_protocol_descriptor, validate_fair_comparison
from supportcover_rag.freeze import build_frozen_manifest, canonical_sha256
from supportcover_rag.main_analysis import PAIRED_COMPARATORS
from supportcover_rag.splits import ordered_ids_sha256
from supportcover_rag.types import PredictionRecord


def _write_ids(path: Path, ids: list[str]) -> str:
    path.write_text(json.dumps({"ids": ids}), encoding="utf-8")
    return ordered_ids_sha256(ids)


def _final_fixture(tmp_path: Path) -> tuple[AppConfig, Path, Path, Path, dict[str, object]]:
    development_ids = ["dev-1", "dev-2"]
    final_ids = ["final-1", "final-2", "final-3"]
    development_path = tmp_path / "development_ids.json"
    final_path = tmp_path / "final_ids.json"
    development_sha = _write_ids(development_path, development_ids)
    final_sha = _write_ids(final_path, final_ids)
    config = AppConfig(
        paths=PathsConfig(data_root=str(tmp_path / "data"), output_root=str(tmp_path / "outputs")),
        runtime=RuntimeConfig(limit=None, overwrite=False, resume=True),
        raw_data=RawDataConfig(splits=["validation"]),
        split=SplitConfig(ids_file=str(final_path), role="final"),
        experiments=ExperimentsConfig(split="validation", methods=list(PRIMARY_MAIN_METHODS)),
        final_study=FinalStudyConfig(
            expected_development_count=2,
            expected_development_sha256=development_sha,
            expected_final_count=3,
            expected_final_sha256=final_sha,
            development_ids_file=str(development_path),
            final_ids_file=str(final_path),
        ),
    )
    manifest = build_frozen_manifest(
        supportcover_coefficients={
            field: getattr(config.supportcover, field)
            for field in (
                "alpha_relevance",
                "beta_coverage",
                "gamma_redundancy",
                "delta_token_cost",
                "title_bonus",
            )
        },
        mmr_lambda_relevance=config.retrieval.mmr_lambda_relevance,
        development_split_sha256=development_sha,
        final_split_sha256=final_sha,
        dataset={
            "path": config.raw_data.dataset_path,
            "config": config.raw_data.dataset_config,
            "final_count": len(final_ids),
        },
        model=asdict(config.generation),
        prompt_settings=asdict(config.prompting),
        decoding_settings={
            "temperature": config.generation.temperature,
            "max_new_tokens": config.generation.max_new_tokens,
            "do_sample": config.generation.do_sample,
        },
        token_budget=config.supportcover.token_budget,
        retrieval_depth=config.retrieval.top_k_paragraphs,
    )
    manifest.update({"selection_role": "development_only", "final_predictions_inspected": False})
    manifest_path = tmp_path / "final_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = replace(
        config,
        freeze=FreezeConfig(
            manifest_file=str(manifest_path),
            sha256=str(manifest["config_sha256"]),
            require_sha256=True,
        ),
    )
    return config, manifest_path, development_path, final_path, manifest


def test_final_gate_accepts_complete_metadata_only_fixture(tmp_path: Path) -> None:
    config, manifest_path, _, _, _ = _final_fixture(tmp_path)

    result = validate_final_execution_config(config, manifest_path=manifest_path)

    assert result["final_count"] == 3
    assert result["final_split_sha256"] == config.final_study.expected_final_sha256


def test_readiness_check_is_metadata_only_and_safely_blocked(tmp_path: Path) -> None:
    config, _, _, _, _ = _final_fixture(tmp_path)
    template_path = tmp_path / "template.yaml"
    template_path.write_text(yaml.safe_dump(asdict(config)), encoding="utf-8")

    report = verify_final_readiness(
        template_path=template_path,
        frozen_config_path=tmp_path / "missing_frozen.yaml",
        manifest_path=tmp_path / "missing_manifest.json",
    )

    assert not report.ready
    assert report.to_dict()["evaluation_scope"] == "metadata_only_no_final_outcomes"


def test_final_gate_rejects_missing_manifest(tmp_path: Path) -> None:
    config, _, _, _, _ = _final_fixture(tmp_path)

    with pytest.raises(FileNotFoundError, match="freeze manifest"):
        validate_final_execution_config(config, manifest_path=tmp_path / "missing.json")


def test_final_gate_rejects_invalid_freeze_sha(tmp_path: Path) -> None:
    config, manifest_path, _, _, _ = _final_fixture(tmp_path)
    config = replace(config, freeze=replace(config.freeze, sha256="f" * 64))

    with pytest.raises(ValueError, match="freeze SHA"):
        validate_final_execution_config(config, manifest_path=manifest_path)


def test_final_gate_rejects_null_frozen_parameter(tmp_path: Path) -> None:
    config, manifest_path, _, _, _ = _final_fixture(tmp_path)
    config = replace(config, retrieval=replace(config.retrieval, top_k_paragraphs=None))

    with pytest.raises(ValueError, match="retrieval depth"):
        validate_final_execution_config(config, manifest_path=manifest_path)


def test_final_gate_rejects_wrong_final_sha_and_count(tmp_path: Path) -> None:
    config, manifest_path, _, _, _ = _final_fixture(tmp_path)
    wrong_sha = replace(config.final_study, expected_final_sha256="f" * 64)
    with pytest.raises(ValueError, match="final split SHA256"):
        validate_final_execution_config(replace(config, final_study=wrong_sha), manifest_path=manifest_path)

    wrong_count = replace(config.final_study, expected_final_count=4)
    with pytest.raises(ValueError, match="population count"):
        validate_final_execution_config(replace(config, final_study=wrong_count), manifest_path=manifest_path)


def test_final_gate_rejects_development_role(tmp_path: Path) -> None:
    config, manifest_path, _, _, _ = _final_fixture(tmp_path)
    config = replace(config, split=replace(config.split, role="development"))

    with pytest.raises(ValueError, match="role='final'"):
        validate_final_execution_config(config, manifest_path=manifest_path)


def test_final_gate_requires_evidence_for_changed_batch_size(tmp_path: Path) -> None:
    config, manifest_path, _, _, _ = _final_fixture(tmp_path)
    config = replace(config, generation=replace(config.generation, batch_size=config.generation.batch_size + 1))

    with pytest.raises(ValueError, match="batch-equivalence"):
        validate_final_execution_config(config, manifest_path=manifest_path)


def test_final_gate_rejects_overlapping_populations(tmp_path: Path) -> None:
    config, manifest_path, development_path, _, _ = _final_fixture(tmp_path)
    final_path = Path(config.final_study.final_ids_file)
    final_sha = _write_ids(final_path, ["dev-1", "final-2", "final-3"])
    config = replace(
        config,
        final_study=replace(config.final_study, expected_final_sha256=final_sha),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["configuration"]["final_split_sha256"] = final_sha
    manifest["config_sha256"] = canonical_sha256(manifest["configuration"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = replace(config, freeze=replace(config.freeze, sha256=manifest["config_sha256"]))

    with pytest.raises(ValueError, match="overlap"):
        validate_final_execution_config(
            config,
            manifest_path=manifest_path,
            development_ids_path=development_path,
            final_ids_path=final_path,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda config: replace(config, prompting=replace(config.prompting, user_instruction="changed")), "prompt_settings"),
        (lambda config: replace(config, generation=replace(config.generation, model_name_or_path="other/model")), "model"),
        (lambda config: replace(config, supportcover=replace(config.supportcover, token_budget=96)), "token_budget"),
        (lambda config: replace(config, retrieval=replace(config.retrieval, top_k_paragraphs=10)), "retrieval_depth"),
    ],
)
def test_fair_comparison_rejects_shared_protocol_changes(mutation, message: str) -> None:
    config = AppConfig(split=SplitConfig(role="final"))
    reference = build_final_protocol_descriptor(config, split_sha256="a" * 64)
    comparison = build_final_protocol_descriptor(mutation(config), split_sha256="a" * 64)

    with pytest.raises(ValueError, match=message):
        validate_fair_comparison({"reference": reference, "comparison": comparison})


def test_fair_comparison_rejects_split_sha_but_allows_method_parameters() -> None:
    config = AppConfig(split=SplitConfig(role="final"))
    reference = build_final_protocol_descriptor(config, split_sha256="a" * 64, method_parameters={"gamma": 0.6})
    allowed = build_final_protocol_descriptor(config, split_sha256="a" * 64, method_parameters={"lambda": 0.9})
    assert validate_fair_comparison({"reference": reference, "comparison": allowed}).passed

    changed_split = build_final_protocol_descriptor(config, split_sha256="b" * 64)
    with pytest.raises(ValueError, match="split_sha256"):
        validate_fair_comparison({"reference": reference, "comparison": changed_split})


def test_main_method_set_includes_paragraph_and_external_is_conditional(tmp_path: Path) -> None:
    assert "paragraph_topk" in PAIRED_COMPARATORS
    assert PRIMARY_MAIN_METHODS == (
        "paragraph_topk",
        "relevance_only",
        "mmr_sentence",
        "greedy_query_cover",
        "supportcover_final",
    )
    config, manifest_path, _, _, _ = _final_fixture(tmp_path)
    validate_final_execution_config(config, manifest_path=manifest_path)
    with_external = replace(
        config,
        experiments=replace(config.experiments, methods=[*PRIMARY_MAIN_METHODS, OPTIONAL_EXTERNAL_METHOD]),
    )
    with pytest.raises(ValueError, match="No external compressor"):
        validate_final_execution_config(with_external, manifest_path=manifest_path)

    unavailable = replace(
        with_external,
        external_compressor=ExternalCompressorConfig(
            enabled=True,
            adapter="definitely_unavailable_supportcover_adapter:build",
            implementation_id="fixture",
            version="1.0",
            revision="fixture-revision",
        ),
    )
    with pytest.raises(ValueError, match="unavailable"):
        validate_final_execution_config(unavailable, manifest_path=manifest_path)


def test_unmapped_external_support_metrics_remain_explicitly_unavailable() -> None:
    record = PredictionRecord(
        example_id="id-1",
        method="external_compressor",
        token_budget=160,
        question="q",
        gold_answer="a",
        predicted_answer="a",
        gold_supporting_facts=[],
        predicted_supporting_facts=[],
        answer_em=1.0,
        answer_f1=1.0,
        support_em=None,
        support_precision=None,
        support_recall=None,
        support_f1=None,
        coverage_at_budget=None,
        evidence_tokens=10,
        retrieval_latency_ms=1.0,
        packing_latency_ms=1.0,
        generation_latency_ms=1.0,
        total_latency_ms=3.0,
        peak_rss_mb=1.0,
        metadata={"support_metrics_supported": False},
    )

    aggregates = aggregate_records([record])

    assert aggregates["support_f1"] is None
    assert aggregates["support_recall"] is None
    assert aggregates["coverage_at_budget"] is None


def test_external_compressor_identity_must_match_configuration() -> None:
    class FixtureCompressor:
        metadata = ExternalCompressorMetadata(
            implementation_id="fixture",
            version="1.0",
            revision="abc123",
            preserves_support_keys=True,
        )

        def compress(self, *, question, retrieved_paragraphs, token_budget):
            raise AssertionError("Metadata validation must not execute compression.")

    configured = ExternalCompressorConfig(
        enabled=True,
        adapter="fixture:build",
        implementation_id="fixture",
        version="1.0",
        revision="abc123",
        preserves_support_keys=True,
    )
    assert validate_external_compressor_metadata(FixtureCompressor(), configured).version == "1.0"
    with pytest.raises(ValueError, match="provenance"):
        validate_external_compressor_metadata(
            FixtureCompressor(),
            replace(configured, revision="different"),
        )


def test_resume_rejects_changed_config_split_and_freeze(tmp_path: Path) -> None:
    config = AppConfig(
        paths=PathsConfig(output_root=str(tmp_path / "outputs")),
        runtime=RuntimeConfig(resume=True),
    )
    manager = ExperimentOutputManager(config.paths.output_root)
    kwargs = {
        "config": config,
        "family": ExperimentFamily.MAIN,
        "method": "relevance_only",
        "split_name": "validation",
        "token_budget": 160,
        "retrieval_depth": 5,
        "variant": "full",
        "experiment_id": "EXP001",
        "config_sha256": "a" * 64,
        "freeze_sha256": "b" * 64,
        "split_sha256": "c" * 64,
    }
    context = manager.prepare_run(**kwargs)
    context.output_dir.mkdir(parents=True)
    manager.write_config_snapshot(context.output_dir / "config.resolved.yaml", config, context)
    assert manager.prepare_run(**kwargs).experiment_id == "EXP001"

    with pytest.raises(ValueError, match="configuration has changed"):
        manager.prepare_run(**{**kwargs, "config": replace(config, seed=99)})
    with pytest.raises(ValueError, match="split_sha256"):
        manager.prepare_run(**{**kwargs, "split_sha256": "d" * 64})
    with pytest.raises(ValueError, match="freeze_sha256"):
        manager.prepare_run(**{**kwargs, "freeze_sha256": "e" * 64})


def _write_complete_run(root: Path, method: str, split_sha: str, freeze_sha: str) -> Path:
    run_dir = root / method
    run_dir.mkdir(parents=True)
    config_payload: dict[str, object] = {
        "split": {"role": "final"},
        "experiments": {"split": "validation"},
    }
    config_sha = canonical_sha256(config_payload)
    experiment_id = f"EXP-{method}"
    metrics = {
        "experiment_id": experiment_id,
        "status": "completed",
        "method": method,
        "num_examples": 2,
        "split_sha256": split_sha,
        "freeze_sha256": freeze_sha,
        "config_sha256": config_sha,
        **{field: 0.5 for field in (
            "answer_em", "answer_f1", "support_em", "support_precision", "support_recall", "support_f1",
            "coverage_at_budget", "evidence_tokens", "retrieval_latency_ms", "packing_latency_ms",
            "generation_latency_ms", "total_latency_ms", "peak_rss_mb",
        )},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    predictions = [
        {
            "example_id": f"id-{index}",
            "method": method,
            "answer_em": 0.0,
            "answer_f1": 0.5,
            "support_f1": 0.5,
            "support_recall": 0.5,
            "coverage_at_budget": 0.5,
        }
        for index in range(2)
    ]
    (run_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in predictions),
        encoding="utf-8",
    )
    config_payload["run"] = {
        "experiment_id": experiment_id,
        "method": method,
        "split_sha256": split_sha,
        "freeze_sha256": freeze_sha,
        "config_sha256": config_sha,
    }
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(config_payload), encoding="utf-8")
    return run_dir


def test_complete_synthetic_runs_produce_deterministic_main_results(tmp_path: Path) -> None:
    split_sha = "a" * 64
    freeze_sha = "b" * 64
    run_dirs = [_write_complete_run(tmp_path / "runs", method, split_sha, freeze_sha) for method in PRIMARY_MAIN_METHODS]

    first = aggregate_final_main_runs(
        run_dirs,
        output_path=tmp_path / "first.csv",
        expected_final_count=2,
        expected_final_sha256=split_sha,
        expected_freeze_sha256=freeze_sha,
        require_external=False,
    )
    second = aggregate_final_main_runs(
        list(reversed(run_dirs)),
        output_path=tmp_path / "second.csv",
        expected_final_count=2,
        expected_final_sha256=split_sha,
        expected_freeze_sha256=freeze_sha,
        require_external=False,
    )

    assert [row["method"] for row in first] == list(PRIMARY_MAIN_METHODS)
    assert first == second
    assert (tmp_path / "first.csv").read_bytes() == (tmp_path / "second.csv").read_bytes()
