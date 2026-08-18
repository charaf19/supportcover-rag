from __future__ import annotations

from supportcover_rag.corpus import build_corpus_manifest, build_paragraph_corpus
from supportcover_rag.retrieval_evaluation import evaluate_supporting_paragraph_recall
from supportcover_rag.retrieval_global import CorpusBM25Retriever, RetrievedCorpusParagraph
from supportcover_rag.types import HotpotExample, Paragraph


def _example(example_id: str, paragraphs: list[Paragraph]) -> HotpotExample:
    return HotpotExample(
        example_id=example_id,
        question="question",
        answer="answer",
        qtype="bridge",
        level="easy",
        context=paragraphs,
        supporting_facts=[],
    )


def _retrieved(corpus_id: str, title: str, rank: int) -> RetrievedCorpusParagraph:
    return RetrievedCorpusParagraph(
        corpus_id=corpus_id,
        title=title,
        sentences=("text",),
        text="text",
        source_example_ids=(),
        rank=rank,
        score=1.0 / rank,
    )


def test_global_bm25_ranking_is_deterministic_and_matches_known_query() -> None:
    examples = [
        _example("e1", [Paragraph("Mars", ["Mars is the red planet."]), Paragraph("Shared", ["Common text."])]),
        _example("e2", [Paragraph("Ocean", ["The ocean is blue."]), Paragraph("Shared", ["Common text."])]),
    ]
    corpus = build_paragraph_corpus(examples)
    retriever = CorpusBM25Retriever(corpus)

    first = retriever.retrieve("red planet Mars", top_k=2)
    second = retriever.retrieve("red planet Mars", top_k=2)

    assert [item.corpus_id for item in first] == [item.corpus_id for item in second]
    assert first[0].title == "Mars"
    assert len(first) == 2
    assert [item.rank for item in first] == [1, 2]


def test_global_bm25_applies_exact_top_k_truncation() -> None:
    corpus = build_paragraph_corpus(
        [_example("e1", [Paragraph("A", ["alpha"]), Paragraph("B", ["beta"]), Paragraph("C", ["gamma"])])]
    )
    retriever = CorpusBM25Retriever(corpus)

    assert retriever.retrieve("alpha", top_k=0) == []
    assert len(retriever.retrieve("alpha", top_k=1)) == 1
    assert len(retriever.retrieve("alpha", top_k=20)) == 3


def test_duplicate_source_paragraphs_map_to_one_corpus_identity() -> None:
    shared = Paragraph("Shared", ["The same paragraph."])
    corpus = build_paragraph_corpus([_example("e2", [shared]), _example("e1", [shared])])
    manifest = build_corpus_manifest(corpus)

    assert len(corpus) == 1
    assert corpus[0].source_example_ids == ("e1", "e2")
    assert manifest["paragraph_count"] == 1
    assert build_corpus_manifest(list(reversed(corpus))) == manifest


def test_supporting_paragraph_recall_handles_multiple_and_missing_supports() -> None:
    retrieved = [_retrieved("p-a", "A", 1), _retrieved("p-x", "X", 2), _retrieved("p-b", "B", 3)]

    records = evaluate_supporting_paragraph_recall(
        example_id="e1",
        retrieved=retrieved,
        gold_support_keys=[("A", 0), ("B", 1), ("B", 2)],
        ks=(1, 2, 3),
    )

    assert [record.support_recall for record in records] == [0.5, 0.5, 1.0]
    assert records[1].matched_titles == ("A",)


def test_supporting_paragraph_recall_handles_no_retrieved_or_no_gold_support() -> None:
    no_retrieval = evaluate_supporting_paragraph_recall(
        example_id="e1",
        retrieved=[],
        gold_support_keys=[("A", 0)],
        ks=(5,),
    )
    no_gold = evaluate_supporting_paragraph_recall(
        example_id="e2",
        retrieved=[],
        gold_support_keys=[],
        ks=(5,),
    )

    assert no_retrieval[0].support_recall == 0.0
    assert no_gold[0].support_recall == 1.0
