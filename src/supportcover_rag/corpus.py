from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

from supportcover_rag.types import HotpotExample


@dataclass(frozen=True, slots=True)
class CorpusParagraph:
    corpus_id: str
    title: str
    text: str
    sentences: tuple[str, ...]
    source_example_ids: tuple[str, ...]


def paragraph_corpus_id(title: str, text: str) -> str:
    identity = json.dumps([title, text], ensure_ascii=False, separators=(",", ":"))
    return "paragraph-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def build_paragraph_corpus(examples: Sequence[HotpotExample]) -> list[CorpusParagraph]:
    sentence_variants_by_identity: dict[tuple[str, str], set[tuple[str, ...]]] = {}
    source_ids_by_identity: dict[tuple[str, str], set[str]] = {}
    for example in examples:
        for paragraph in example.context:
            key = (paragraph.title, paragraph.text)
            sentence_variants_by_identity.setdefault(key, set()).add(tuple(paragraph.sentences))
            source_ids_by_identity.setdefault(key, set()).add(example.example_id)

    corpus: list[CorpusParagraph] = []
    for title, text in sorted(sentence_variants_by_identity):
        sentence_variants = sentence_variants_by_identity[(title, text)]
        source_ids = source_ids_by_identity[(title, text)]
        corpus.append(
            CorpusParagraph(
                corpus_id=paragraph_corpus_id(title, text),
                title=title,
                text=text,
                sentences=min(sentence_variants),
                source_example_ids=tuple(sorted(source_ids)),
            )
        )
    return corpus


def corpus_sha256(corpus: Sequence[CorpusParagraph]) -> str:
    ordered = sorted(corpus, key=lambda paragraph: paragraph.corpus_id)
    for paragraph in ordered:
        expected_id = paragraph_corpus_id(paragraph.title, paragraph.text)
        if paragraph.corpus_id != expected_id:
            raise ValueError(
                f"Corpus paragraph ID does not match title/text identity: {paragraph.corpus_id}"
            )
    identities = [
        {"corpus_id": paragraph.corpus_id, "title": paragraph.title, "text": paragraph.text}
        for paragraph in ordered
    ]
    encoded = json.dumps(identities, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_corpus_manifest(corpus: Sequence[CorpusParagraph]) -> dict[str, object]:
    ordered_ids = [paragraph.corpus_id for paragraph in sorted(corpus, key=lambda item: item.corpus_id)]
    if len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError("Global paragraph corpus contains duplicate corpus IDs.")
    return {
        "paragraph_count": len(ordered_ids),
        "corpus_sha256": corpus_sha256(corpus),
        "corpus_ids": ordered_ids,
    }
