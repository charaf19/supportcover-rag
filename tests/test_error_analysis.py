from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from supportcover_rag.config import AppConfig, ErrorAnalysisConfig, PathsConfig
from supportcover_rag.error_analysis import assign_error_category, run_error_analysis
from supportcover_rag.io_utils import write_jsonl


def _write_run_dir(
    run_dir: Path,
    *,
    experiment_id: str,
    model_alias: str,
    rows: list[dict[str, object]],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump({"run": {"experiment_id": experiment_id, "model_alias": model_alias}}, sort_keys=False),
        encoding="utf-8",
    )
    write_jsonl(run_dir / "predictions.jsonl", rows)


def _record(
    *,
    example_id: str,
    method: str,
    question: str,
    gold_answer: str,
    predicted_answer: str,
    answer_em: float,
    answer_f1: float,
    support_f1: float,
    coverage: float,
    evidence_text: str,
    question_type: str = "bridge",
) -> dict[str, object]:
    return {
        "example_id": example_id,
        "method": method,
        "token_budget": 160,
        "question": question,
        "gold_answer": gold_answer,
        "predicted_answer": predicted_answer,
        "gold_supporting_facts": [],
        "predicted_supporting_facts": [],
        "answer_em": answer_em,
        "answer_f1": answer_f1,
        "support_em": 0.0,
        "support_precision": 0.0,
        "support_recall": coverage,
        "support_f1": support_f1,
        "coverage_at_budget": coverage,
        "evidence_tokens": 120,
        "retrieval_latency_ms": 1.0,
        "packing_latency_ms": 2.0,
        "generation_latency_ms": 3.0,
        "total_latency_ms": 6.0,
        "peak_rss_mb": 100.0,
        "metadata": {
            "evidence_text": evidence_text,
            "question_type": question_type,
        },
    }


def test_assign_error_category_handles_main_taxonomy_cases() -> None:
    support_missing = _record(
        example_id="ex-a",
        method="supportcover_final",
        question="Were A and B from the same country?",
        gold_answer="yes",
        predicted_answer="insufficient evidence",
        answer_em=0.0,
        answer_f1=0.0,
        support_f1=0.0,
        coverage=0.0,
        evidence_text="Title: A - A was a writer.",
        question_type="comparison",
    )
    formatting = _record(
        example_id="ex-b",
        method="supportcover_final",
        question="Where is the company headquartered?",
        gold_answer="Mumbai",
        predicted_answer="Mumbai, Maharashtra",
        answer_em=0.0,
        answer_f1=2.0 / 3.0,
        support_f1=0.4,
        coverage=1.0,
        evidence_text="Title: Company - The company is headquartered in Mumbai.",
    )

    assert assign_error_category(support_missing) == "support_missing"
    assert assign_error_category(formatting) == "formatting_mismatch"


def test_run_error_analysis_writes_annotations_and_summary(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    relevance_run = output_root / "robustness" / "EXP030_relevance_only_qwen_val_b160_d5_full"
    supportcover_run = output_root / "robustness" / "EXP031_supportcover_final_qwen_val_b160_d5_final"

    relevance_rows = [
        _record(
            example_id="ex-a",
            method="relevance_only",
            question="Were A and B from the same country?",
            gold_answer="yes",
            predicted_answer="insufficient evidence",
            answer_em=0.0,
            answer_f1=0.0,
            support_f1=0.0,
            coverage=0.0,
            evidence_text="Title: A - A was a writer.",
            question_type="comparison",
        ),
        _record(
            example_id="ex-b",
            method="relevance_only",
            question="Which race track hosts the race?",
            gold_answer="Indianapolis Motor Speedway",
            predicted_answer="insufficient evidence",
            answer_em=0.0,
            answer_f1=0.0,
            support_f1=0.0,
            coverage=0.0,
            evidence_text="Title: Allison Transmission - The first Indianapolis 500 was held there.",
        ),
        _record(
            example_id="ex-c",
            method="relevance_only",
            question="Where is the company headquartered?",
            gold_answer="Mumbai",
            predicted_answer="Mumbai",
            answer_em=1.0,
            answer_f1=1.0,
            support_f1=0.4,
            coverage=1.0,
            evidence_text="Title: Company - The company is headquartered in Mumbai.",
        ),
        _record(
            example_id="ex-d",
            method="relevance_only",
            question="What population does the country have?",
            gold_answer="9,984",
            predicted_answer="insufficient evidence",
            answer_em=0.0,
            answer_f1=0.0,
            support_f1=0.2,
            coverage=0.5,
            evidence_text="Title: Lake - Brown Lake is in Botswana. Title: Botswana - Population is 9,984.",
            question_type="bridge",
        ),
    ]
    supportcover_rows = [
        _record(
            example_id="ex-a",
            method="supportcover_final",
            question="Were A and B from the same country?",
            gold_answer="yes",
            predicted_answer="insufficient evidence",
            answer_em=0.0,
            answer_f1=0.0,
            support_f1=0.0,
            coverage=0.0,
            evidence_text="Title: A - A was a writer.",
            question_type="comparison",
        ),
        _record(
            example_id="ex-b",
            method="supportcover_final",
            question="Which race track hosts the race?",
            gold_answer="Indianapolis Motor Speedway",
            predicted_answer="Indianapolis Motor Speedway",
            answer_em=1.0,
            answer_f1=1.0,
            support_f1=0.3,
            coverage=0.5,
            evidence_text="Title: Indianapolis Motor Speedway - It hosts the Indianapolis 500.",
        ),
        _record(
            example_id="ex-c",
            method="supportcover_final",
            question="Where is the company headquartered?",
            gold_answer="Mumbai",
            predicted_answer="Mumbai, Maharashtra",
            answer_em=0.0,
            answer_f1=2.0 / 3.0,
            support_f1=0.4,
            coverage=1.0,
            evidence_text="Title: Company - The company is headquartered in Mumbai.",
        ),
        _record(
            example_id="ex-d",
            method="supportcover_final",
            question="What population does the country have?",
            gold_answer="9,984",
            predicted_answer="insufficient evidence",
            answer_em=0.0,
            answer_f1=0.0,
            support_f1=0.2,
            coverage=0.5,
            evidence_text="Title: Lake - Brown Lake is in Botswana. Title: Botswana - Population is 9,984.",
            question_type="bridge",
        ),
    ]

    _write_run_dir(relevance_run, experiment_id="EXP030", model_alias="qwen", rows=relevance_rows)
    _write_run_dir(supportcover_run, experiment_id="EXP031", model_alias="qwen", rows=supportcover_rows)

    config = AppConfig(
        paths=PathsConfig(output_root=str(output_root)),
        error_analysis=ErrorAnalysisConfig(
            output_dir=str(output_root / "error_analysis"),
            source_runs={
                "relevance_only": str(relevance_run),
                "supportcover_final": str(supportcover_run),
            },
            canonical_sample_size=2,
            comparison_sample_size=2,
            representative_examples=3,
        ),
    )

    artifacts = run_error_analysis(config)

    assert artifacts["annotation_path"].exists()
    assert artifacts["summary_path"].exists()
    assert artifacts["analysis_path"].exists()

    with artifacts["annotation_path"].open("r", encoding="utf-8", newline="") as handle:
        annotation_rows = list(csv.DictReader(handle))
    assert len(annotation_rows) == 4
    assert {row["comparison_outcome"] for row in annotation_rows} == {
        "both_fail",
        "relevance_only_fail_only",
        "supportcover_final_fail_only",
    }

    with artifacts["summary_path"].open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    counts = {
        (row["method"], row["error_category"]): int(row["count"])
        for row in summary_rows
    }
    assert counts[("relevance_only", "support_missing")] == 2
    assert counts[("supportcover_final", "formatting_mismatch")] == 1

    analysis_text = artifacts["analysis_path"].read_text(encoding="utf-8")
    assert "Phase 7 Error Analysis" in analysis_text
    assert "supportcover_final = no_redundancy" in analysis_text
