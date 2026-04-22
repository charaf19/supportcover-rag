from __future__ import annotations

import math
from collections import Counter

from supportcover_rag.text import tokenize
from supportcover_rag.types import HotpotExample, RetrievedParagraph


class BM25ParagraphRetriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def retrieve(self, example: HotpotExample, top_k: int) -> list[RetrievedParagraph]:
        documents = [f"{paragraph.title} {paragraph.text}" for paragraph in example.context]
        tokenized_docs = [tokenize(document) for document in documents]
        query_tokens = tokenize(example.question)
        avgdl = sum(len(doc) for doc in tokenized_docs) / max(len(tokenized_docs), 1)
        doc_freqs = Counter()
        for doc in tokenized_docs:
            doc_freqs.update(set(doc))

        scores: list[tuple[int, float]] = []
        num_docs = len(tokenized_docs)
        for index, doc in enumerate(tokenized_docs):
            tf = Counter(doc)
            score = 0.0
            doc_len = len(doc)
            for term in query_tokens:
                if term not in tf:
                    continue
                n_q = doc_freqs.get(term, 0)
                idf = math.log(((num_docs - n_q + 0.5) / (n_q + 0.5)) + 1.0)
                numerator = tf[term] * (self.k1 + 1.0)
                denominator = tf[term] + self.k1 * (1.0 - self.b + self.b * (doc_len / max(avgdl, 1e-8)))
                score += idf * (numerator / max(denominator, 1e-8))
            scores.append((index, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        retrieved: list[RetrievedParagraph] = []
        for rank, (index, score) in enumerate(scores[:top_k], start=1):
            paragraph = example.context[index]
            retrieved.append(
                RetrievedParagraph(
                    title=paragraph.title,
                    sentences=paragraph.sentences,
                    text=paragraph.text,
                    rank=rank,
                    score=score,
                )
            )
        return retrieved
