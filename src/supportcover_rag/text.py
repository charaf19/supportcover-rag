from __future__ import annotations

import re
import string
from functools import lru_cache

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
    "were", "will", "with", "which", "what", "who", "when", "where", "why", "how",
    "did", "do", "does", "into", "than", "then", "their", "them", "this", "those",
    "these", "or", "if", "during", "after", "before", "over", "under", "about",
}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


@lru_cache(maxsize=20000)
def filtered_terms(text: str) -> tuple[str, ...]:
    return tuple(token for token in tokenize(text) if token not in _STOPWORDS)


def informative_term_set(text: str) -> set[str]:
    return set(filtered_terms(text))


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    without_punc = "".join(ch for ch in lowered if ch not in string.punctuation)
    without_articles = _ARTICLES_RE.sub(" ", without_punc)
    normalized = _WHITESPACE_RE.sub(" ", without_articles).strip()
    return normalized


def whitespace_token_estimate(text: str) -> int:
    tokens = text.split()
    return max(1, len(tokens))
