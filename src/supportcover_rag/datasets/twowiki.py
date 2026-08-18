from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from supportcover_rag.types import HotpotExample, Paragraph, SupportKey


def _required_text(record: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"2Wiki record is missing a non-empty string field from: {', '.join(keys)}")


def _normalize_sentences(value: Any, *, title: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"2Wiki context sentences for title '{title}' must be an array of strings.")
    sentences: list[str] = []
    for index, sentence in enumerate(value):
        if not isinstance(sentence, str):
            raise ValueError(f"2Wiki sentence {index} for title '{title}' must be a string.")
        sentences.append(sentence.strip())
    return sentences


def _normalize_context(value: Any) -> list[Paragraph]:
    raw_paragraphs: list[tuple[Any, Any]] = []
    if isinstance(value, Mapping):
        titles = value.get("title")
        sentence_groups = value.get("sentences", value.get("content"))
        if not isinstance(titles, Sequence) or isinstance(titles, (str, bytes)):
            raise ValueError("2Wiki context.title must be an array.")
        if not isinstance(sentence_groups, Sequence) or isinstance(sentence_groups, (str, bytes)):
            raise ValueError("2Wiki context sentences/content must be an array.")
        if len(titles) != len(sentence_groups):
            raise ValueError("2Wiki context titles and sentence groups must have equal lengths.")
        raw_paragraphs = list(zip(titles, sentence_groups, strict=True))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, paragraph in enumerate(value):
            if not isinstance(paragraph, Sequence) or isinstance(paragraph, (str, bytes)) or len(paragraph) != 2:
                raise ValueError(f"2Wiki context entry {index} must be [title, sentences].")
            raw_paragraphs.append((paragraph[0], paragraph[1]))
    else:
        raise ValueError("2Wiki context must be an array of paragraph pairs or a title/sentences object.")

    paragraphs: list[Paragraph] = []
    seen_titles: set[str] = set()
    for raw_title, raw_sentences in raw_paragraphs:
        if not isinstance(raw_title, str) or not raw_title.strip():
            raise ValueError("Every 2Wiki context paragraph must have a non-empty string title.")
        title = raw_title.strip()
        if title in seen_titles:
            raise ValueError(f"Duplicate 2Wiki context title cannot be represented unambiguously: {title}")
        seen_titles.add(title)
        paragraphs.append(Paragraph(title=title, sentences=_normalize_sentences(raw_sentences, title=title)))
    if not paragraphs:
        raise ValueError("2Wiki record must contain at least one context paragraph.")
    return paragraphs


def _normalize_supporting_facts(value: Any, paragraphs: Sequence[Paragraph]) -> list[SupportKey]:
    raw_facts: list[tuple[Any, Any]] = []
    if isinstance(value, Mapping):
        titles = value.get("title")
        sentence_ids = value.get("sent_id", value.get("sentence_id"))
        if not isinstance(titles, Sequence) or isinstance(titles, (str, bytes)):
            raise ValueError("2Wiki supporting_facts.title must be an array.")
        if not isinstance(sentence_ids, Sequence) or isinstance(sentence_ids, (str, bytes)):
            raise ValueError("2Wiki supporting_facts sentence IDs must be an array.")
        if len(titles) != len(sentence_ids):
            raise ValueError("2Wiki support titles and sentence IDs must have equal lengths.")
        raw_facts = list(zip(titles, sentence_ids, strict=True))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, fact in enumerate(value):
            if not isinstance(fact, Sequence) or isinstance(fact, (str, bytes)) or len(fact) != 2:
                raise ValueError(f"2Wiki supporting fact {index} must be [title, sentence_id].")
            raw_facts.append((fact[0], fact[1]))
    else:
        raise ValueError("2Wiki supporting_facts must be an array or title/sentence-ID object.")
    if not raw_facts:
        raise ValueError("2Wiki record must contain at least one supporting fact.")

    paragraph_by_title = {paragraph.title: paragraph for paragraph in paragraphs}
    support_keys: list[SupportKey] = []
    seen: set[SupportKey] = set()
    for raw_title, raw_sentence_id in raw_facts:
        if not isinstance(raw_title, str) or not raw_title.strip():
            raise ValueError("Every 2Wiki supporting fact must have a non-empty string title.")
        if not isinstance(raw_sentence_id, int) or isinstance(raw_sentence_id, bool):
            raise ValueError(f"Supporting sentence ID for '{raw_title}' must be an integer.")
        title = raw_title.strip()
        paragraph = paragraph_by_title.get(title)
        if paragraph is None:
            raise ValueError(f"Supporting fact title is absent from 2Wiki context: {title}")
        if raw_sentence_id < 0 or raw_sentence_id >= len(paragraph.sentences):
            raise ValueError(
                f"Supporting sentence ID {raw_sentence_id} is out of range for 2Wiki title '{title}'."
            )
        support_key = (title, raw_sentence_id)
        if support_key in seen:
            raise ValueError(f"Duplicate 2Wiki supporting fact: {title}[{raw_sentence_id}]")
        seen.add(support_key)
        support_keys.append(support_key)
    return support_keys


def normalize_twowiki_record(record: Mapping[str, Any]) -> HotpotExample:
    example_id = _required_text(record, "_id", "id")
    question = _required_text(record, "question")
    answer = _required_text(record, "answer")
    if "context" not in record:
        raise ValueError(f"2Wiki record {example_id} is missing context.")
    if "supporting_facts" not in record:
        raise ValueError(f"2Wiki record {example_id} is missing supporting_facts.")

    paragraphs = _normalize_context(record["context"])
    support_keys = _normalize_supporting_facts(record["supporting_facts"], paragraphs)
    raw_type = record.get("type", "unknown")
    raw_level = record.get("level", "unknown")
    if not isinstance(raw_type, str) or not isinstance(raw_level, str):
        raise ValueError(f"2Wiki record {example_id} type and level must be strings when supplied.")
    return HotpotExample(
        example_id=example_id,
        question=question,
        answer=answer,
        qtype=raw_type,
        level=raw_level,
        context=paragraphs,
        supporting_facts=support_keys,
    )


class TwoWikiMultiHopQAAdapter:
    dataset_name = "2WikiMultiHopQA"

    def normalize_record(self, record: Mapping[str, Any]) -> HotpotExample:
        return normalize_twowiki_record(record)
