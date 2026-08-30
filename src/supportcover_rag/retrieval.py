from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from supportcover_rag.freeze import canonical_sha256
from supportcover_rag.text import tokenize
from supportcover_rag.types import CorpusParagraph, HotpotExample, Paragraph, RetrievedParagraph


CONTROLLED_CONTEXT = "controlled_context"
GLOBAL_CORPUS = "global_corpus"
RETRIEVAL_MODES = (CONTROLLED_CONTEXT, GLOBAL_CORPUS)
DEFAULT_TOKENIZER_IDENTITY = "supportcover_rag.text.tokenize:v1"
GLOBAL_INDEX_SCHEMA_VERSION = 1


class ParagraphRetriever(Protocol):
    evaluation_mode: str

    def retrieve(
        self,
        *,
        question: str,
        top_k: int,
        context: Sequence[Paragraph] | None = None,
        query_id: str | None = None,
    ) -> list[RetrievedParagraph]: ...


def _validate_bm25_parameters(k1: float, b: float) -> None:
    if not math.isfinite(k1) or k1 < 0:
        raise ValueError("BM25 k1 must be finite and non-negative.")
    if not math.isfinite(b) or not 0 <= b <= 1:
        raise ValueError("BM25 b must be finite and between 0 and 1.")


def _bm25_score(
    query_tokens: Sequence[str],
    document_tokens: Sequence[str],
    *,
    document_frequencies: Mapping[str, int],
    number_of_documents: int,
    average_document_length: float,
    k1: float,
    b: float,
    term_frequencies: Mapping[str, int] | None = None,
) -> float:
    frequencies = Counter(document_tokens) if term_frequencies is None else term_frequencies
    document_length = len(document_tokens)
    score = 0.0
    for term in query_tokens:
        if term not in frequencies:
            continue
        containing_documents = document_frequencies.get(term, 0)
        inverse_document_frequency = math.log(
            ((number_of_documents - containing_documents + 0.5) / (containing_documents + 0.5)) + 1.0
        )
        numerator = frequencies[term] * (k1 + 1.0)
        denominator = frequencies[term] + k1 * (
            1.0 - b + b * (document_length / max(average_document_length, 1e-8))
        )
        score += inverse_document_frequency * (numerator / max(denominator, 1e-8))
    return score


class BM25ParagraphRetriever:
    """Backwards-compatible BM25 over one example's supplied distractor context."""

    evaluation_mode = CONTROLLED_CONTEXT

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        _validate_bm25_parameters(k1, b)
        self.k1 = k1
        self.b = b

    def retrieve(
        self,
        example: HotpotExample | None = None,
        top_k: int = 5,
        *,
        question: str | None = None,
        context: Sequence[Paragraph] | None = None,
        query_id: str | None = None,
    ) -> list[RetrievedParagraph]:
        if example is not None:
            question = example.question
            context = example.context
            query_id = example.example_id
        if question is None or context is None:
            raise ValueError("controlled_context retrieval requires question and supplied context.")
        if top_k < 0:
            raise ValueError("top_k must be non-negative.")
        documents = [f"{paragraph.title} {paragraph.text}" for paragraph in context]
        tokenized_documents = [tokenize(document) for document in documents]
        average_length = sum(len(document) for document in tokenized_documents) / max(len(tokenized_documents), 1)
        document_frequencies: Counter[str] = Counter()
        for document in tokenized_documents:
            document_frequencies.update(set(document))
        query_tokens = tokenize(question)
        scores = [
            (
                index,
                _bm25_score(
                    query_tokens,
                    document,
                    document_frequencies=document_frequencies,
                    number_of_documents=len(tokenized_documents),
                    average_document_length=average_length,
                    k1=self.k1,
                    b=self.b,
                ),
            )
            for index, document in enumerate(tokenized_documents)
        ]
        scores.sort(key=lambda item: (-item[1], item[0]))
        retrieved: list[RetrievedParagraph] = []
        for rank, (index, score) in enumerate(scores[:top_k], start=1):
            paragraph = context[index]
            controlled_id = f"controlled:{query_id or 'query'}:{index:04d}"
            retrieved.append(
                RetrievedParagraph(
                    title=paragraph.title,
                    sentences=paragraph.sentences,
                    text=paragraph.text,
                    rank=rank,
                    score=score,
                    corpus_id=CONTROLLED_CONTEXT,
                    document_id=f"controlled:{query_id or 'query'}",
                    paragraph_id=controlled_id,
                    source_record_id=query_id,
                    metadata={"retrieval_mode": CONTROLLED_CONTEXT, "context_index": index},
                )
            )
        return retrieved


ControlledContextRetriever = BM25ParagraphRetriever


@dataclass(frozen=True, slots=True)
class GlobalBM25Index:
    corpus_id: str
    corpus_sha256: str
    tokenizer_identity: str
    k1: float
    b: float
    paragraphs: tuple[CorpusParagraph, ...]
    tokenized_documents: tuple[tuple[str, ...], ...]
    term_frequencies: tuple[Mapping[str, int], ...]
    document_frequencies: Mapping[str, int]
    average_document_length: float

    @classmethod
    def build(
        cls,
        paragraphs: Sequence[CorpusParagraph],
        *,
        corpus_sha256: str,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer_identity: str = DEFAULT_TOKENIZER_IDENTITY,
    ) -> GlobalBM25Index:
        _validate_bm25_parameters(k1, b)
        if (
            len(corpus_sha256) != 64
            or any(character not in "0123456789abcdef" for character in corpus_sha256.lower())
        ):
            raise ValueError("Global corpus SHA256 must be a 64-character hexadecimal digest.")
        if not tokenizer_identity.strip():
            raise ValueError("Global BM25 tokenizer identity must be explicit.")
        if not paragraphs:
            raise ValueError("A global BM25 index requires at least one corpus paragraph.")
        corpus_ids = {paragraph.corpus_id for paragraph in paragraphs}
        if len(corpus_ids) != 1:
            raise ValueError("All indexed paragraphs must share one corpus_id.")
        paragraph_ids = [paragraph.paragraph_id for paragraph in paragraphs]
        if any(not paragraph_id for paragraph_id in paragraph_ids):
            raise ValueError("Global corpus paragraph IDs must be non-empty.")
        if any(not paragraph.document_id for paragraph in paragraphs):
            raise ValueError("Global corpus document IDs must be non-empty.")
        if len(paragraph_ids) != len(set(paragraph_ids)):
            raise ValueError("Global corpus paragraph IDs must be unique.")
        ordered = tuple(sorted(paragraphs, key=lambda item: (item.document_id, item.paragraph_id)))
        tokenized = tuple(tuple(tokenize(f"{paragraph.title} {paragraph.text}")) for paragraph in ordered)
        frequencies: Counter[str] = Counter()
        for document in tokenized:
            frequencies.update(set(document))
        term_frequencies = tuple(dict(Counter(document)) for document in tokenized)
        average_length = sum(len(document) for document in tokenized) / len(tokenized)
        return cls(
            corpus_id=next(iter(corpus_ids)),
            corpus_sha256=corpus_sha256,
            tokenizer_identity=tokenizer_identity,
            k1=k1,
            b=b,
            paragraphs=ordered,
            tokenized_documents=tokenized,
            term_frequencies=term_frequencies,
            document_frequencies=dict(sorted(frequencies.items())),
            average_document_length=average_length,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GLOBAL_INDEX_SCHEMA_VERSION,
            "implementation": "supportcover_rag.global_bm25",
            "corpus_id": self.corpus_id,
            "corpus_sha256": self.corpus_sha256,
            "tokenizer_identity": self.tokenizer_identity,
            "parameters": {"k1": self.k1, "b": self.b},
            "average_document_length": self.average_document_length,
            "document_frequencies": dict(self.document_frequencies),
            "paragraphs": [asdict(paragraph) for paragraph in self.paragraphs],
            "tokenized_documents": [list(document) for document in self.tokenized_documents],
            "term_frequencies": [dict(frequencies) for frequencies in self.term_frequencies],
        }

    @property
    def index_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, expected_corpus_sha256: str) -> GlobalBM25Index:
        if value.get("schema_version") != GLOBAL_INDEX_SCHEMA_VERSION:
            raise ValueError("Unsupported global BM25 index schema.")
        if value.get("corpus_sha256") != expected_corpus_sha256:
            raise ValueError("Global BM25 index corpus SHA256 does not match the requested corpus.")
        parameters = value.get("parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError("Global BM25 index is missing parameters.")
        paragraphs = tuple(
            CorpusParagraph(
                corpus_id=str(item["corpus_id"]),
                document_id=str(item["document_id"]),
                paragraph_id=str(item["paragraph_id"]),
                title=str(item["title"]),
                sentences=tuple(item["sentences"]),
                source_dataset=str(item["source_dataset"]),
                source_split=str(item["source_split"]),
                source_record_id=item.get("source_record_id"),
                source_revision=item.get("source_revision"),
                metadata=dict(item.get("metadata") or {}),
            )
            for item in value["paragraphs"]
        )
        tokenized = tuple(tuple(document) for document in value["tokenized_documents"])
        term_frequencies = tuple(
            {str(term): int(count) for term, count in frequencies.items()}
            for frequencies in value["term_frequencies"]
        )
        if len(paragraphs) != len(tokenized) or len(paragraphs) != len(term_frequencies):
            raise ValueError("Global BM25 index paragraph/token/frequency arrays differ in length.")
        return cls(
            corpus_id=str(value["corpus_id"]),
            corpus_sha256=str(value["corpus_sha256"]),
            tokenizer_identity=str(value["tokenizer_identity"]),
            k1=float(parameters["k1"]),
            b=float(parameters["b"]),
            paragraphs=paragraphs,
            tokenized_documents=tokenized,
            term_frequencies=term_frequencies,
            document_frequencies={str(key): int(count) for key, count in value["document_frequencies"].items()},
            average_document_length=float(value["average_document_length"]),
        )


class GlobalCorpusRetriever:
    evaluation_mode = GLOBAL_CORPUS

    def __init__(self, index: GlobalBM25Index) -> None:
        self.index = index

    def retrieve(
        self,
        *,
        question: str,
        top_k: int,
        context: Sequence[Paragraph] | None = None,
        query_id: str | None = None,
    ) -> list[RetrievedParagraph]:
        if context is not None:
            raise ValueError("global_corpus retrieval cannot accept a query-specific context candidate pool.")
        if top_k < 0:
            raise ValueError("top_k must be non-negative.")
        query_tokens = tokenize(question)
        scored = [
            (
                paragraph,
                _bm25_score(
                    query_tokens,
                    document,
                    document_frequencies=self.index.document_frequencies,
                    number_of_documents=len(self.index.paragraphs),
                    average_document_length=self.index.average_document_length,
                    k1=self.index.k1,
                    b=self.index.b,
                    term_frequencies=frequencies,
                ),
            )
            for paragraph, document, frequencies in zip(
                self.index.paragraphs,
                self.index.tokenized_documents,
                self.index.term_frequencies,
                strict=True,
            )
        ]
        scored.sort(key=lambda item: (-item[1], item[0].document_id, item[0].paragraph_id))
        return [
            RetrievedParagraph(
                title=paragraph.title,
                sentences=list(paragraph.sentences),
                text=paragraph.text,
                rank=rank,
                score=score,
                corpus_id=paragraph.corpus_id,
                document_id=paragraph.document_id,
                paragraph_id=paragraph.paragraph_id,
                source_dataset=paragraph.source_dataset,
                source_split=paragraph.source_split,
                source_record_id=paragraph.source_record_id,
                metadata={**paragraph.metadata, "retrieval_mode": GLOBAL_CORPUS},
            )
            for rank, (paragraph, score) in enumerate(scored[:top_k], start=1)
        ]


def build_paragraph_retriever(
    *,
    evaluation_mode: str,
    k1: float = 1.5,
    b: float = 0.75,
    global_index: GlobalBM25Index | None = None,
) -> ParagraphRetriever:
    if evaluation_mode == CONTROLLED_CONTEXT:
        if global_index is not None:
            raise ValueError("controlled_context retrieval must not receive a global index.")
        return BM25ParagraphRetriever(k1=k1, b=b)
    if evaluation_mode == GLOBAL_CORPUS:
        if global_index is None:
            raise ValueError("global_corpus retrieval requires a precomputed global index.")
        if global_index.k1 != k1 or global_index.b != b:
            raise ValueError("Configured BM25 parameters do not match the frozen global index.")
        return GlobalCorpusRetriever(global_index)
    raise ValueError(f"Unknown retrieval mode: {evaluation_mode}")


def validate_retrieval_mode(mode: str, *, corpus_manifest: str = "") -> None:
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"retrieval.evaluation_mode must be one of: {', '.join(RETRIEVAL_MODES)}")
    if mode == CONTROLLED_CONTEXT and corpus_manifest:
        raise ValueError("controlled_context mode must not silently bind a global corpus manifest.")
    if mode == GLOBAL_CORPUS and not corpus_manifest.strip():
        raise ValueError("global_corpus mode requires an explicit corpus manifest.")
