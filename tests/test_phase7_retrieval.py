from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from supportcover_rag.retrieval import (
    CONTROLLED_CONTEXT,
    GLOBAL_CORPUS,
    BM25ParagraphRetriever,
    GlobalBM25Index,
    GlobalCorpusRetriever,
    build_paragraph_retriever,
    validate_retrieval_mode,
)
from supportcover_rag.retrieval_realism import (
    GLOBAL_RETRIEVAL_ROLE,
    FrozenRetrievedCandidates,
    aggregate_retrieval_metrics,
    aggregate_retrieval_diagnostics,
    build_retrieval_manifest,
    deterministic_corpus_object_id,
    evaluate_global_retrieval,
    load_retrieval_cache,
    map_support_facts_to_paragraph_ids,
    paragraph_retrieval_metrics,
    validate_corpus_manifest,
    validate_query_role,
    validate_retrieval_cache_reuse,
    validate_retrieval_fairness,
    verify_retrieval_readiness,
    write_retrieval_cache,
)
from supportcover_rag.types import CorpusParagraph, HotpotExample, Paragraph


def _corpus() -> list[CorpusParagraph]:
    return [
        CorpusParagraph(
            corpus_id="synthetic-corpus",
            document_id="doc-b",
            paragraph_id="p-2",
            title="Duplicate title",
            sentences=("orange orange fruit",),
            source_dataset="synthetic",
            source_split="test",
            source_record_id="source-2",
        ),
        CorpusParagraph(
            corpus_id="synthetic-corpus",
            document_id="doc-a",
            paragraph_id="p-1",
            title="Duplicate title",
            sentences=("apple fruit",),
            source_dataset="synthetic",
            source_split="test",
            source_record_id="source-1",
        ),
        CorpusParagraph(
            corpus_id="synthetic-corpus",
            document_id="doc-c",
            paragraph_id="p-3",
            title="Other",
            sentences=("unrelated words",),
            source_dataset="synthetic",
            source_split="test",
            source_record_id="source-3",
        ),
    ]


def _index(corpus_sha: str = "a" * 64) -> GlobalBM25Index:
    return GlobalBM25Index.build(_corpus(), corpus_sha256=corpus_sha, k1=1.5, b=0.75)


def test_controlled_context_ranking_and_mode_remain_reproducible() -> None:
    example = HotpotExample(
        example_id="fixture-1",
        question="orange fruit",
        answer="unused",
        qtype="bridge",
        level="easy",
        context=[
            Paragraph(title="A", sentences=["apple"]),
            Paragraph(title="B", sentences=["orange fruit"]),
        ],
        supporting_facts=[("B", 0)],
    )
    retriever = BM25ParagraphRetriever()
    first = retriever.retrieve(example=example, top_k=2)
    second = retriever.retrieve(example=example, top_k=2)
    assert [paragraph.title for paragraph in first] == ["B", "A"]
    assert [paragraph.paragraph_id for paragraph in first] == [
        "controlled:fixture-1:0001",
        "controlled:fixture-1:0000",
    ]
    assert [paragraph.score for paragraph in first] == [paragraph.score for paragraph in second]
    assert retriever.evaluation_mode == CONTROLLED_CONTEXT
    assert all(paragraph.metadata["retrieval_mode"] == CONTROLLED_CONTEXT for paragraph in first)


def test_global_bm25_ranking_top_k_and_duplicate_title_identity() -> None:
    retriever = GlobalCorpusRetriever(_index())
    first = retriever.retrieve(question="orange", top_k=2, query_id="q1")
    second = retriever.retrieve(question="orange", top_k=2, query_id="q1")
    assert first[0].paragraph_id == "p-2"
    assert len(first) == 2
    assert [item.paragraph_id for item in first] == [item.paragraph_id for item in second]
    assert len({item.paragraph_id for item in retriever.retrieve(question="fruit", top_k=3)}) == 3
    assert retriever.evaluation_mode == GLOBAL_CORPUS
    with pytest.raises(ValueError, match="query-specific context"):
        retriever.retrieve(
            question="orange",
            top_k=2,
            context=[Paragraph(title="leak", sentences=["gold-only paragraph"])],
        )


def test_global_bm25_ties_use_document_and_paragraph_identity() -> None:
    retriever = GlobalCorpusRetriever(_index())
    tied = retriever.retrieve(question="term-not-present", top_k=3)
    assert [paragraph.paragraph_id for paragraph in tied] == ["p-1", "p-2", "p-3"]


def _write_manifest_fixture(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        "".join(json.dumps({"paragraph_id": item.paragraph_id}) + "\n" for item in _corpus()),
        encoding="utf-8",
    )
    corpus_sha = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    index = _index(corpus_sha)
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index.to_dict(), sort_keys=True), encoding="utf-8")
    index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
    manifest: dict[str, object] = {
        "schema_version": 1,
        "corpus_id": "synthetic-corpus",
        "construction_protocol": "benchmark_defined_corpus",
        "dataset": {
            "identity": "synthetic",
            "version_or_config": "v1",
            "source_splits": ["test"],
            "source_revision": "r1",
        },
        "number_of_unique_documents": 3,
        "number_of_unique_paragraphs": 3,
        "deduplication_rule": "paragraph_id exact",
        "normalization_rule": "none",
        "ordering_rule": "document_id,paragraph_id",
        "tokenizer_identity": "supportcover_rag.text.tokenize:v1",
        "construction_inputs": ["source_text", "source_identity", "source_provenance"],
        "corpus_artifact": {"path": str(corpus_path), "sha256": corpus_sha},
        "index": {
            "type": "supportcover_rag.global_bm25",
            "path": str(index_path),
            "sha256": index_sha,
            "corpus_sha256": corpus_sha,
        },
        "created_at": "2026-01-01T00:00:00Z",
        "code_revision": "fixture",
    }
    return manifest, corpus_path, index_path


def test_corpus_manifest_and_index_binding(tmp_path: Path) -> None:
    manifest, _, _ = _write_manifest_fixture(tmp_path)
    validate_corpus_manifest(manifest)

    wrong_corpus = json.loads(json.dumps(manifest))
    wrong_corpus["corpus_artifact"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="Index corpus SHA256"):
        validate_corpus_manifest(wrong_corpus)

    wrong_index_binding = json.loads(json.dumps(manifest))
    wrong_index_binding["index"]["corpus_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="Index corpus SHA256"):
        validate_corpus_manifest(wrong_index_binding)

    missing_identity = json.loads(json.dumps(manifest))
    missing_identity["corpus_id"] = ""
    with pytest.raises(ValueError, match="corpus identity"):
        validate_corpus_manifest(missing_identity)

    unresolved_protocol = json.loads(json.dumps(manifest))
    unresolved_protocol["construction_protocol"] = "__PREREGISTER_LATER__"
    with pytest.raises(ValueError, match="construction protocol"):
        validate_corpus_manifest(unresolved_protocol)

    leaked = json.loads(json.dumps(manifest))
    leaked["construction_inputs"].append("support_annotations")
    with pytest.raises(ValueError, match="labels or outcomes"):
        validate_corpus_manifest(leaked)


def test_deterministic_corpus_ids_ignore_mapping_insertion_order() -> None:
    first = deterministic_corpus_object_id("paragraph", {"record": "r1", "index": 2})
    second = deterministic_corpus_object_id("paragraph", {"index": 2, "record": "r1"})
    assert first == second


def test_retrieval_modes_are_explicit_not_path_inferred() -> None:
    validate_retrieval_mode(CONTROLLED_CONTEXT)
    validate_retrieval_mode(GLOBAL_CORPUS, corpus_manifest="corpus_manifest.json")
    with pytest.raises(ValueError, match="requires an explicit corpus"):
        validate_retrieval_mode(GLOBAL_CORPUS)
    with pytest.raises(ValueError, match="must not silently bind"):
        validate_retrieval_mode(CONTROLLED_CONTEXT, corpus_manifest="corpus_manifest.json")
    assert build_paragraph_retriever(evaluation_mode=CONTROLLED_CONTEXT).evaluation_mode == CONTROLLED_CONTEXT
    assert build_paragraph_retriever(
        evaluation_mode=GLOBAL_CORPUS,
        global_index=_index(),
    ).evaluation_mode == GLOBAL_CORPUS


@pytest.mark.parametrize(
    ("retrieved_ids", "expected_recall", "expected_all", "expected_any"),
    [
        (["p-1", "p-2"], 1.0, 1.0, 1.0),
        (["p-1"], 0.5, 0.0, 1.0),
        (["p-3"], 0.0, 0.0, 0.0),
    ],
)
def test_paragraph_support_metrics(
    retrieved_ids: list[str], expected_recall: float, expected_all: float, expected_any: float
) -> None:
    retrieved = [
        next(item for item in GlobalCorpusRetriever(_index()).retrieve(question="", top_k=3) if item.paragraph_id == identifier)
        for identifier in retrieved_ids
    ]
    for rank, item in enumerate(retrieved, start=1):
        item.rank = rank
    metrics = paragraph_retrieval_metrics(retrieved, ["p-1", "p-2"])
    assert metrics["support_paragraph_recall"] == expected_recall
    assert metrics["all_support_paragraphs_retrieved_rate"] == expected_all
    assert metrics["at_least_one_support_retrieved_rate"] == expected_any


def test_multiple_support_sentences_map_to_one_paragraph() -> None:
    facts = [("Title", 0), ("Title", 1), ("Other", 0)]
    mapping = {
        ("Title", 0): "paragraph-a",
        ("Title", 1): "paragraph-a",
        ("Other", 0): "paragraph-b",
    }
    assert map_support_facts_to_paragraph_ids(facts, mapping) == ("paragraph-a", "paragraph-b")


def test_packing_fairness_rejects_different_candidate_sets() -> None:
    shared = [
        {"paragraph_id": "p-1", "rank": 1, "score": 1.0},
        {"paragraph_id": "p-2", "rank": 2, "score": 0.5},
    ]
    validate_retrieval_fairness({"supportcover": shared, "relevance_only": list(shared)})
    with pytest.raises(ValueError, match="different retrieved candidates"):
        validate_retrieval_fairness(
            {
                "supportcover": shared,
                "relevance_only": [{"paragraph_id": "p-3", "rank": 1, "score": 1.0}],
            }
        )


def test_global_retrieval_role_and_population_are_isolated() -> None:
    validate_query_role(
        GLOBAL_RETRIEVAL_ROLE,
        query_population_sha256="a" * 64,
        expected_population_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="global_retrieval_evaluation"):
        validate_query_role(
            "development",
            query_population_sha256="a" * 64,
            expected_population_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="population SHA256"):
        validate_query_role(
            GLOBAL_RETRIEVAL_ROLE,
            query_population_sha256="b" * 64,
            expected_population_sha256="a" * 64,
        )


def test_frozen_retrieval_cache_requires_exact_protocol() -> None:
    paragraphs = tuple(GlobalCorpusRetriever(_index()).retrieve(question="fruit", top_k=2))
    cached = FrozenRetrievedCandidates(
        example_id="q1",
        query_role=GLOBAL_RETRIEVAL_ROLE,
        query_population_sha256="a" * 64,
        corpus_id="synthetic-corpus",
        corpus_sha256="b" * 64,
        retrieval_mode=GLOBAL_CORPUS,
        retrieval_method="bm25",
        retrieval_parameters={"k1": 1.5, "b": 0.75},
        tokenizer_identity="supportcover_rag.text.tokenize:v1",
        top_k=2,
        paragraphs=paragraphs,
    )
    valid = {
        "query_population_sha256": "a" * 64,
        "corpus_sha256": "b" * 64,
        "retrieval_method": "bm25",
        "retrieval_parameters": {"k1": 1.5, "b": 0.75},
        "tokenizer_identity": "supportcover_rag.text.tokenize:v1",
        "top_k": 2,
    }
    validate_retrieval_cache_reuse(cached, **valid)
    for field, value in (
        ("corpus_sha256", "c" * 64),
        ("top_k", 3),
        ("retrieval_parameters", {"k1": 0.5, "b": 0.75}),
    ):
        with pytest.raises(ValueError, match=field):
            validate_retrieval_cache_reuse(cached, **{**valid, field: value})


def test_retrieval_cache_round_trip_is_deterministic(tmp_path: Path) -> None:
    cached = FrozenRetrievedCandidates(
        example_id="q1",
        query_role=GLOBAL_RETRIEVAL_ROLE,
        query_population_sha256="a" * 64,
        corpus_id="synthetic-corpus",
        corpus_sha256="b" * 64,
        retrieval_mode=GLOBAL_CORPUS,
        retrieval_method="bm25",
        retrieval_parameters={"k1": 1.5, "b": 0.75},
        tokenizer_identity="supportcover_rag.text.tokenize:v1",
        top_k=2,
        paragraphs=tuple(GlobalCorpusRetriever(_index()).retrieve(question="fruit", top_k=2)),
    )
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_retrieval_cache(first, [cached])
    loaded = load_retrieval_cache(first)
    write_retrieval_cache(second, loaded)
    assert loaded == [cached]
    assert first.read_bytes() == second.read_bytes()


def test_generator_free_retrieval_diagnostics_write_no_answers(tmp_path: Path) -> None:
    records, aggregate = evaluate_global_retrieval(
        retriever=GlobalCorpusRetriever(_index()),
        queries=[
            {
                "example_id": "q1",
                "question": "orange",
                "gold_support_paragraph_ids": ["p-2"],
            }
        ],
        query_role=GLOBAL_RETRIEVAL_ROLE,
        query_population_sha256="a" * 64,
        expected_population_sha256="a" * 64,
        top_k=2,
        per_example_output=tmp_path / "per_example.jsonl",
        aggregate_output=tmp_path / "metrics.csv",
    )
    assert records[0]["support_paragraph_recall"] == 1.0
    assert aggregate["N"] == 1
    assert "answer" not in records[0]
    assert aggregate_retrieval_metrics(records)["support_paragraph_recall"] == 1.0
    regenerated = aggregate_retrieval_diagnostics(
        tmp_path / "per_example.jsonl",
        output_path=tmp_path / "regenerated.csv",
    )
    assert regenerated["support_paragraph_recall"] == aggregate["support_paragraph_recall"]


def test_readiness_is_generator_independent_and_correctly_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("Generator construction must not occur.")

    monkeypatch.setattr("supportcover_rag.generation.build_generator", forbidden)
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "evaluation_role": GLOBAL_RETRIEVAL_ROLE,
                "tuning_permitted": False,
                "retrieval": {
                    "evaluation_mode": GLOBAL_CORPUS,
                    "corpus_manifest": "__PREREGISTER_LATER__",
                },
                "query_population": {"sha256": "a" * 64},
            }
        ),
        encoding="utf-8",
    )
    report = verify_retrieval_readiness(
        protocol_path=protocol,
        freeze_manifest_path=tmp_path / "missing-freeze.json",
    )
    assert not report.ready
    assert report.to_dict()["evaluation_scope"] == "metadata_only_no_final_examples"


def test_retrieval_manifest_hashes_completed_synthetic_artifacts(tmp_path: Path) -> None:
    corpus_manifest, _, _ = _write_manifest_fixture(tmp_path)
    source = tmp_path / "per_example.jsonl"
    aggregate = tmp_path / "metrics.csv"
    source.write_text('{"example_id":"q1"}\n', encoding="utf-8")
    aggregate.write_text("N\n1\n", encoding="utf-8")
    manifest = build_retrieval_manifest(
        evaluation_role=GLOBAL_RETRIEVAL_ROLE,
        retrieval_mode=GLOBAL_CORPUS,
        corpus_manifest=corpus_manifest,
        query_population={"sha256": "a" * 64, "N": 1},
        retrieval_parameters={"k1": 1.5, "b": 0.75},
        top_k=5,
        tokenizer_identity="supportcover_rag.text.tokenize:v1",
        freeze_sha256="b" * 64,
        code_revision="fixture",
        environment_reference="synthetic-environment.json",
        source_artifacts=[source],
        aggregate_artifacts=[aggregate],
        created_at="2026-01-01T00:00:00Z",
    )
    assert manifest["status"] == "completed"
    assert manifest["source_artifacts"][0]["sha256"]
    assert manifest["aggregate_artifacts"][0]["sha256"]
