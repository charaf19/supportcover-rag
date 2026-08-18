from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean

from supportcover_rag.corpus import paragraph_corpus_id
from supportcover_rag.retrieval_global import RetrievedCorpusParagraph
from supportcover_rag.types import HotpotExample, SupportKey


DEFAULT_RECALL_KS = (5, 10, 20)


@dataclass(frozen=True, slots=True)
class SupportingParagraphRecall:
    example_id: str
    k: int
    gold_titles: tuple[str, ...]
    gold_corpus_ids: tuple[str, ...]
    retrieved_corpus_ids: tuple[str, ...]
    retrieved_titles: tuple[str, ...]
    matched_titles: tuple[str, ...]
    matched_corpus_ids: tuple[str, ...]
    match_basis: str
    support_recall: float


def gold_support_corpus_ids(example: HotpotExample) -> tuple[str, ...]:
    paragraphs_by_title: dict[str, list[str]] = defaultdict(list)
    for paragraph in example.context:
        paragraphs_by_title[paragraph.title].append(paragraph.text)
    corpus_ids: set[str] = set()
    for title in sorted({support_title for support_title, _ in example.supporting_facts}):
        texts = paragraphs_by_title.get(title, [])
        if not texts:
            raise ValueError(f"Gold support title is absent from example {example.example_id}: {title}")
        if len(texts) != 1:
            raise ValueError(f"Gold support title is ambiguous in example {example.example_id}: {title}")
        corpus_ids.add(paragraph_corpus_id(title, texts[0]))
    return tuple(sorted(corpus_ids))


def evaluate_supporting_paragraph_recall(
    *,
    example_id: str,
    retrieved: Sequence[RetrievedCorpusParagraph],
    gold_support_keys: Sequence[SupportKey],
    gold_corpus_ids: Sequence[str] | None = None,
    ks: Sequence[int] = DEFAULT_RECALL_KS,
) -> list[SupportingParagraphRecall]:
    if len(ks) != len(set(ks)) or any(k < 0 for k in ks):
        raise ValueError("Retrieval recall cutoffs must be unique non-negative integers.")
    gold_titles = tuple(sorted({title for title, _ in gold_support_keys}))
    gold_title_set = set(gold_titles)
    ordered_gold_corpus_ids = tuple(sorted(set(gold_corpus_ids or ())))
    if gold_corpus_ids is not None and gold_titles and not ordered_gold_corpus_ids:
        raise ValueError("Gold corpus IDs cannot be empty when canonical gold support is present.")
    gold_corpus_id_set = set(ordered_gold_corpus_ids)
    match_basis = "corpus_id" if gold_corpus_ids is not None else "title"
    diagnostics: list[SupportingParagraphRecall] = []
    for k in ks:
        top_k = list(retrieved[:k])
        retrieved_titles = tuple(item.title for item in top_k)
        retrieved_corpus_ids = tuple(item.corpus_id for item in top_k)
        matched_corpus_ids = tuple(sorted(gold_corpus_id_set & set(retrieved_corpus_ids)))
        if gold_corpus_ids is None:
            matched_titles = tuple(sorted(gold_title_set & set(retrieved_titles)))
            recall = 1.0 if not gold_titles else len(matched_titles) / len(gold_titles)
        else:
            matched_titles = tuple(sorted({item.title for item in top_k if item.corpus_id in matched_corpus_ids}))
            recall = 1.0 if not ordered_gold_corpus_ids else len(matched_corpus_ids) / len(ordered_gold_corpus_ids)
        diagnostics.append(
            SupportingParagraphRecall(
                example_id=example_id,
                k=k,
                gold_titles=gold_titles,
                gold_corpus_ids=ordered_gold_corpus_ids,
                retrieved_corpus_ids=retrieved_corpus_ids,
                retrieved_titles=retrieved_titles,
                matched_titles=matched_titles,
                matched_corpus_ids=matched_corpus_ids,
                match_basis=match_basis,
                support_recall=recall,
            )
        )
    return diagnostics


def aggregate_supporting_paragraph_recall(
    records: Sequence[SupportingParagraphRecall],
) -> list[dict[str, float | int]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for record in records:
        grouped[record.k].append(record.support_recall)
    return [
        {"k": k, "num_examples": len(grouped[k]), "support_recall": mean(grouped[k])}
        for k in sorted(grouped)
    ]
