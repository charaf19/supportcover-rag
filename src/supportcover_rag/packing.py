from __future__ import annotations

import random
from dataclasses import replace
from typing import Iterable

from supportcover_rag.config import SupportCoverConfig
from supportcover_rag.generation import TokenCounter
from supportcover_rag.text import informative_term_set, jaccard_similarity
from supportcover_rag.types import PackedEvidence, RetrievedParagraph, SelectedSentence, SentenceCandidate


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    max_score = max(scores)
    min_score = min(scores)
    if max_score == min_score:
        return [1.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]


def _base_relevance(candidate: SentenceCandidate) -> float:
    return (
        0.55 * candidate.raw_features.get("query_overlap", 0.0)
        + 0.35 * candidate.raw_features.get("paragraph_score_norm", 0.0)
        + 0.10 * candidate.raw_features.get("title_overlap", 0.0)
    )


def build_sentence_candidates(
    question: str,
    retrieved_paragraphs: list[RetrievedParagraph],
    token_counter: TokenCounter,
) -> list[SentenceCandidate]:
    question_terms = informative_term_set(question)
    paragraph_score_values = [paragraph.score for paragraph in retrieved_paragraphs]
    normalized_paragraph_scores = _normalize_scores(paragraph_score_values)

    candidates: list[SentenceCandidate] = []
    for paragraph, normalized_score in zip(retrieved_paragraphs, normalized_paragraph_scores, strict=True):
        title_terms = informative_term_set(paragraph.title)
        for sentence_id, sentence in enumerate(paragraph.sentences):
            sentence_terms = informative_term_set(sentence)
            if not sentence_terms:
                continue
            overlap = len(question_terms & sentence_terms) / max(len(question_terms), 1)
            title_overlap = len(question_terms & title_terms) / max(len(question_terms), 1)
            candidate = SentenceCandidate(
                title=paragraph.title,
                sentence_id=sentence_id,
                text=sentence,
                paragraph_rank=paragraph.rank,
                paragraph_score=paragraph.score,
                token_count=token_counter.count(sentence),
                question_terms=question_terms,
                title_terms=title_terms,
                sentence_terms=sentence_terms,
                raw_features={
                    "query_overlap": overlap,
                    "title_overlap": title_overlap,
                    "paragraph_score_norm": normalized_score,
                },
            )
            candidates.append(candidate)
    return candidates


class SupportCoverSelector:
    def __init__(self, config: SupportCoverConfig) -> None:
        self.config = config

    def _candidate_score(
        self,
        candidate: SentenceCandidate,
        selected: list[SelectedSentence],
        token_budget: int,
    ) -> tuple[float, dict[str, float]]:
        selected_terms = set().union(*(item.candidate.sentence_terms for item in selected)) if selected else set()
        selected_titles = {item.candidate.title for item in selected}

        relevance = _base_relevance(candidate)
        newly_covered_terms = candidate.sentence_terms & candidate.question_terms - selected_terms
        coverage_gain = len(newly_covered_terms) / max(len(candidate.question_terms), 1)
        title_gain = 1.0 if candidate.title not in selected_titles else 0.0
        redundancy = max(
            (jaccard_similarity(candidate.sentence_terms, item.candidate.sentence_terms) for item in selected),
            default=0.0,
        )
        token_cost = candidate.token_count / max(token_budget, 1)

        score = (
            self.config.alpha_relevance * relevance
            + self.config.beta_coverage * coverage_gain
            + self.config.title_bonus * title_gain
            - self.config.gamma_redundancy * redundancy
            - self.config.delta_token_cost * token_cost
        )
        contributions = {
            "relevance": relevance,
            "coverage_gain": coverage_gain,
            "title_gain": title_gain,
            "redundancy": redundancy,
            "token_cost": token_cost,
        }
        return score, contributions

    def select(self, candidates: list[SentenceCandidate], token_budget: int) -> PackedEvidence:
        remaining = list(candidates)
        selected: list[SelectedSentence] = []
        used_tokens = 0

        while remaining:
            feasible = [candidate for candidate in remaining if used_tokens + candidate.token_count <= token_budget]
            if not feasible:
                break

            scored = [(candidate, *self._candidate_score(candidate, selected, token_budget)) for candidate in feasible]
            best_candidate, best_score, contributions = max(scored, key=lambda item: item[1])
            if selected and best_score < self.config.stop_threshold:
                break

            selected.append(SelectedSentence(candidate=best_candidate, score=best_score, contributions=contributions))
            used_tokens += best_candidate.token_count
            remaining = [candidate for candidate in remaining if candidate.support_key != best_candidate.support_key]

        return PackedEvidence(method="supportcover", selected=selected, token_budget=token_budget)


def pack_paragraphs(
    question: str,
    retrieved_paragraphs: list[RetrievedParagraph],
    token_budget: int,
    token_counter: TokenCounter,
) -> PackedEvidence:
    candidates: list[SelectedSentence] = []
    explicit_support_keys: list[tuple[str, int]] = []
    seen_support_keys: set[tuple[str, int]] = set()
    used_tokens = 0
    for paragraph in retrieved_paragraphs:
        paragraph_text = " ".join(paragraph.sentences)
        paragraph_tokens = token_counter.count(paragraph_text)
        if used_tokens + paragraph_tokens > token_budget:
            break
        pseudo_candidate = SentenceCandidate(
            title=paragraph.title,
            sentence_id=-1,
            text=paragraph_text,
            paragraph_rank=paragraph.rank,
            paragraph_score=paragraph.score,
            token_count=paragraph_tokens,
            question_terms=informative_term_set(question),
            title_terms=informative_term_set(paragraph.title),
            sentence_terms=informative_term_set(paragraph_text),
            raw_features={"query_overlap": 0.0, "title_overlap": 0.0, "paragraph_score_norm": 1.0},
        )
        candidates.append(SelectedSentence(candidate=pseudo_candidate, score=paragraph.score, contributions={}))
        for sentence_id in range(len(paragraph.sentences)):
            support_key = (paragraph.title, sentence_id)
            if support_key not in seen_support_keys:
                explicit_support_keys.append(support_key)
                seen_support_keys.add(support_key)
        used_tokens += paragraph_tokens
    return PackedEvidence(
        method="paragraph_topk",
        selected=candidates,
        token_budget=token_budget,
        explicit_support_keys=explicit_support_keys,
    )


def pack_relevance_only(candidates: list[SentenceCandidate], token_budget: int) -> PackedEvidence:
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            candidate.raw_features.get("query_overlap", 0.0),
            candidate.raw_features.get("paragraph_score_norm", 0.0),
            candidate.raw_features.get("title_overlap", 0.0),
        ),
        reverse=True,
    )
    selected: list[SelectedSentence] = []
    used_tokens = 0
    for candidate in ranked:
        if used_tokens + candidate.token_count > token_budget:
            continue
        selected.append(SelectedSentence(candidate=candidate, score=candidate.raw_features.get("query_overlap", 0.0), contributions={}))
        used_tokens += candidate.token_count
    return PackedEvidence(method="relevance_only", selected=selected, token_budget=token_budget)


def pack_mmr(
    candidates: list[SentenceCandidate],
    token_budget: int,
    lambda_relevance: float,
) -> PackedEvidence:
    if not 0.0 <= lambda_relevance <= 1.0:
        raise ValueError("lambda_relevance must be between 0 and 1.")

    remaining = list(candidates)
    selected: list[SelectedSentence] = []
    used_tokens = 0
    while remaining:
        feasible = [candidate for candidate in remaining if used_tokens + candidate.token_count <= token_budget]
        if not feasible:
            break

        scored: list[tuple[SentenceCandidate, float, float, float]] = []
        for candidate in feasible:
            relevance = _base_relevance(candidate)
            max_similarity = max(
                (
                    jaccard_similarity(candidate.sentence_terms, item.candidate.sentence_terms)
                    for item in selected
                ),
                default=0.0,
            )
            score = lambda_relevance * relevance - (1.0 - lambda_relevance) * max_similarity
            scored.append((candidate, score, relevance, max_similarity))

        best_candidate, best_score, relevance, max_similarity = max(scored, key=lambda item: item[1])
        selected.append(
            SelectedSentence(
                candidate=best_candidate,
                score=best_score,
                contributions={
                    "relevance": relevance,
                    "max_similarity_to_selected": max_similarity,
                },
            )
        )
        used_tokens += best_candidate.token_count
        remaining = [candidate for candidate in remaining if candidate.support_key != best_candidate.support_key]

    return PackedEvidence(method="mmr_sentence", selected=selected, token_budget=token_budget)


def pack_greedy_query_cover(
    candidates: list[SentenceCandidate],
    token_budget: int,
) -> PackedEvidence:
    if any(candidate.token_count <= 0 for candidate in candidates):
        raise ValueError("Greedy query-cover candidates must have a positive token cost.")

    remaining = list(candidates)
    selected: list[SelectedSentence] = []
    covered_question_terms: set[str] = set()
    used_tokens = 0
    while remaining:
        feasible = [candidate for candidate in remaining if used_tokens + candidate.token_count <= token_budget]
        if not feasible:
            break

        scored: list[tuple[SentenceCandidate, float, set[str]]] = []
        for candidate in feasible:
            newly_covered_terms = (candidate.sentence_terms & candidate.question_terms) - covered_question_terms
            score = len(newly_covered_terms) / candidate.token_count
            scored.append((candidate, score, newly_covered_terms))

        best_candidate, best_score, newly_covered_terms = max(scored, key=lambda item: item[1])
        if not newly_covered_terms:
            break
        selected.append(
            SelectedSentence(
                candidate=best_candidate,
                score=best_score,
                contributions={
                    "new_query_terms": float(len(newly_covered_terms)),
                    "coverage_per_token": best_score,
                },
            )
        )
        covered_question_terms.update(newly_covered_terms)
        used_tokens += best_candidate.token_count
        remaining = [candidate for candidate in remaining if candidate.support_key != best_candidate.support_key]

    return PackedEvidence(method="greedy_query_cover", selected=selected, token_budget=token_budget)


def pack_random(candidates: list[SentenceCandidate], token_budget: int, seed: int) -> PackedEvidence:
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    selected: list[SelectedSentence] = []
    used_tokens = 0
    for candidate in shuffled:
        if used_tokens + candidate.token_count > token_budget:
            continue
        selected.append(SelectedSentence(candidate=candidate, score=0.0, contributions={}))
        used_tokens += candidate.token_count
    return PackedEvidence(method="random_sentence", selected=selected, token_budget=token_budget)


def apply_variant(selector: SupportCoverSelector, variant: str) -> SupportCoverSelector:
    if variant == "full":
        return selector
    cfg = selector.config
    if variant == "no_query_coverage":
        return SupportCoverSelector(replace(cfg, beta_coverage=0.0))
    if variant == "no_title_gain":
        return SupportCoverSelector(replace(cfg, title_bonus=0.0))
    if variant == "no_coverage":
        return SupportCoverSelector(replace(cfg, beta_coverage=0.0, title_bonus=0.0))
    if variant == "no_redundancy":
        return SupportCoverSelector(replace(cfg, gamma_redundancy=0.0))
    if variant == "no_token_penalty":
        return SupportCoverSelector(replace(cfg, delta_token_cost=0.0))
    if variant == "relevance_only":
        return SupportCoverSelector(replace(cfg, beta_coverage=0.0, title_bonus=0.0, gamma_redundancy=0.0, delta_token_cost=0.0))
    raise ValueError(f"Unknown ablation variant: {variant}")
