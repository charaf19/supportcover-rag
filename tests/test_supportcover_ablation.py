from __future__ import annotations

from dataclasses import asdict

import pytest

from supportcover_rag.config import SupportCoverConfig
from supportcover_rag.packing import SupportCoverSelector, apply_variant


@pytest.mark.parametrize(
    ("variant", "removed_field"),
    [
        ("no_query_coverage", "beta_coverage"),
        ("no_title_gain", "title_bonus"),
        ("no_redundancy", "gamma_redundancy"),
        ("no_token_penalty", "delta_token_cost"),
    ],
)
def test_one_factor_ablation_changes_only_one_component(variant: str, removed_field: str) -> None:
    config = SupportCoverConfig()

    ablated = apply_variant(SupportCoverSelector(config), variant).config

    before = asdict(config)
    after = asdict(ablated)
    changed = {field for field in before if before[field] != after[field]}
    assert changed == {removed_field}
    assert after[removed_field] == 0.0


def test_full_variant_preserves_selector_and_configuration() -> None:
    selector = SupportCoverSelector(SupportCoverConfig())

    assert apply_variant(selector, "full") is selector


def test_relevance_only_remains_available() -> None:
    config = apply_variant(SupportCoverSelector(SupportCoverConfig()), "relevance_only").config

    assert config.beta_coverage == 0.0
    assert config.title_bonus == 0.0
    assert config.gamma_redundancy == 0.0
    assert config.delta_token_cost == 0.0
    assert config.alpha_relevance == SupportCoverConfig().alpha_relevance


def test_legacy_no_coverage_variant_remains_conflated_for_historical_compatibility() -> None:
    original = SupportCoverConfig()

    legacy = apply_variant(SupportCoverSelector(original), "no_coverage").config

    before = asdict(original)
    after = asdict(legacy)
    changed = {field for field in before if before[field] != after[field]}
    assert changed == {"beta_coverage", "title_bonus"}
