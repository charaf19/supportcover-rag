from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from supportcover_rag.corpus import CorpusParagraph, paragraph_corpus_id
from supportcover_rag.text import tokenize


@dataclass(frozen=True, slots=True)
class RetrievedCorpusParagraph:
    corpus_id: str
    title: str
    sentences: tuple[str, ...]
    text: str
    source_example_ids: tuple[str, ...]
    rank: int
    score: float


class CorpusBM25Retriever:
    def __init__(
        self,
        corpus: Sequence[CorpusParagraph],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = tuple(sorted(corpus, key=lambda paragraph: paragraph.corpus_id))
        corpus_ids = [paragraph.corpus_id for paragraph in self.corpus]
        if len(corpus_ids) != len(set(corpus_ids)):
            raise ValueError("CorpusBM25Retriever requires unique corpus IDs.")
        for paragraph in self.corpus:
            if paragraph.corpus_id != paragraph_corpus_id(paragraph.title, paragraph.text):
                raise ValueError(
                    f"Corpus paragraph ID does not match title/text identity: {paragraph.corpus_id}"
                )

        self._tokenized_documents = tuple(
            tuple(tokenize(f"{paragraph.title} {paragraph.text}")) for paragraph in self.corpus
        )
        self._average_document_length = sum(map(len, self._tokenized_documents)) / max(len(self.corpus), 1)
        self._document_frequencies: Counter[str] = Counter()
        for document in self._tokenized_documents:
            self._document_frequencies.update(set(document))

    def retrieve(self, question: str, top_k: int) -> list[RetrievedCorpusParagraph]:
        if top_k < 0:
            raise ValueError("top_k must be non-negative.")
        query_tokens = tokenize(question)
        num_documents = len(self.corpus)
        scored: list[tuple[CorpusParagraph, float]] = []
        for paragraph, document in zip(self.corpus, self._tokenized_documents, strict=True):
            term_frequencies = Counter(document)
            score = 0.0
            for term in query_tokens:
                if term not in term_frequencies:
                    continue
                document_frequency = self._document_frequencies.get(term, 0)
                inverse_document_frequency = math.log(
                    ((num_documents - document_frequency + 0.5) / (document_frequency + 0.5)) + 1.0
                )
                numerator = term_frequencies[term] * (self.k1 + 1.0)
                denominator = term_frequencies[term] + self.k1 * (
                    1.0
                    - self.b
                    + self.b * (len(document) / max(self._average_document_length, 1e-8))
                )
                score += inverse_document_frequency * (numerator / max(denominator, 1e-8))
            scored.append((paragraph, score))

        scored.sort(key=lambda item: (-item[1], item[0].corpus_id))
        return [
            RetrievedCorpusParagraph(
                corpus_id=paragraph.corpus_id,
                title=paragraph.title,
                sentences=paragraph.sentences,
                text=paragraph.text,
                source_example_ids=paragraph.source_example_ids,
                rank=rank,
                score=score,
            )
            for rank, (paragraph, score) in enumerate(scored[:top_k], start=1)
        ]
