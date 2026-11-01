import math

import pytest

from furui.scoring import (
    DEFAULT_HALF_LIFE_DAYS,
    ScoringConfig,
    current_score,
    days_between,
    document_terms,
    query_terms,
    recency,
    reinforcement,
    relevance,
    score_breakdown,
    tokenize,
)

# --------------------------------------------------------------------------
# Decay
# --------------------------------------------------------------------------


def test_recency_is_one_when_fresh():
    assert recency(0) == 1.0


def test_recency_halves_every_half_life():
    assert recency(30, 30) == pytest.approx(0.5)
    assert recency(60, 30) == pytest.approx(0.25)
    assert recency(90, 30) == pytest.approx(0.125)


def test_recency_custom_half_life():
    assert recency(7, 7) == pytest.approx(0.5)
    assert recency(7, 14) == pytest.approx(math.sqrt(0.5))


def test_recency_is_monotonically_decreasing():
    values = [recency(d) for d in range(0, 365, 5)]
    assert all(a > b for a, b in zip(values, values[1:]))
    assert all(0 < v <= 1 for v in values)


def test_recency_clamps_negative_days():
    assert recency(-10) == 1.0


def test_recency_rejects_non_positive_half_life():
    with pytest.raises(ValueError):
        recency(1, 0)


def test_days_between_fractional_and_clamped():
    from datetime import datetime, timedelta, timezone

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert days_between(t0, t0 + timedelta(hours=36)) == pytest.approx(1.5)
    assert days_between(t0 + timedelta(days=1), t0) == 0.0


# --------------------------------------------------------------------------
# Reinforcement
# --------------------------------------------------------------------------


def test_reinforcement_baseline_is_one():
    assert reinforcement(0) == 1.0


def test_reinforcement_grows_with_diminishing_returns():
    values = [reinforcement(n) for n in range(0, 50)]
    gains = [b - a for a, b in zip(values, values[1:])]
    assert all(g > 0 for g in gains)
    assert all(a > b for a, b in zip(gains, gains[1:]))


def test_reinforcement_known_values():
    assert reinforcement(1, 0.5) == pytest.approx(1 + 0.5 * math.log(2))
    assert reinforcement(10, 1.0) == pytest.approx(1 + math.log(11))


def test_reinforcement_weight_zero_disables_it():
    assert reinforcement(1000, 0.0) == 1.0


# --------------------------------------------------------------------------
# Combined score
# --------------------------------------------------------------------------


def test_score_is_product_of_factors():
    cfg = ScoringConfig(half_life_days=30)
    s = current_score(importance=4, idle_days=30, recall_count=3, pinned=False, config=cfg)
    assert s == pytest.approx(4 * 0.5 * reinforcement(3))


def test_score_breakdown_exposes_factors():
    b = score_breakdown(5, 60, 0, False, ScoringConfig(half_life_days=30))
    assert b.importance == 5
    assert b.recency == pytest.approx(0.25)
    assert b.reinforcement == 1.0
    assert b.score == pytest.approx(1.25)


def test_fresh_memory_score_equals_importance():
    for imp in range(1, 6):
        assert current_score(imp, 0, 0, False) == imp


def test_higher_importance_decays_to_same_ratio():
    a = current_score(5, 45, 0, False)
    b = current_score(1, 45, 0, False)
    assert a / b == pytest.approx(5)


def test_pinned_memory_never_drops_below_floor():
    cfg = ScoringConfig(half_life_days=30, pin_floor=0.5)
    ancient_pinned = score_breakdown(3, 10_000, 0, True, cfg)
    ancient_plain = score_breakdown(3, 10_000, 0, False, cfg)
    assert ancient_pinned.recency == 0.5
    assert ancient_pinned.score == pytest.approx(1.5)
    assert ancient_plain.recency < 1e-50


def test_pinned_floor_does_not_cap_fresh_memories():
    cfg = ScoringConfig(pin_floor=0.5)
    assert score_breakdown(3, 0, 0, True, cfg).recency == 1.0
    assert score_breakdown(3, 10, 0, True, cfg).recency == pytest.approx(recency(10))


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def test_config_defaults():
    cfg = ScoringConfig.from_env({})
    assert cfg.half_life_days == DEFAULT_HALF_LIFE_DAYS


def test_config_from_env():
    cfg = ScoringConfig.from_env({"FURUI_HALF_LIFE_DAYS": "7", "FURUI_PIN_FLOOR": "0.2"})
    assert cfg.half_life_days == 7
    assert cfg.pin_floor == 0.2


def test_config_from_process_env(monkeypatch):
    monkeypatch.setenv("FURUI_HALF_LIFE_DAYS", "14.5")
    assert ScoringConfig.from_env().half_life_days == 14.5


@pytest.mark.parametrize(
    "env",
    [
        {"FURUI_HALF_LIFE_DAYS": "abc"},
        {"FURUI_HALF_LIFE_DAYS": "0"},
        {"FURUI_HALF_LIFE_DAYS": "-3"},
        {"FURUI_PIN_FLOOR": "1.5"},
    ],
)
def test_config_rejects_bad_values(env):
    with pytest.raises(ValueError):
        ScoringConfig.from_env(env)


# --------------------------------------------------------------------------
# Relevance
# --------------------------------------------------------------------------


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("User prefers Dark-Mode, in VS_Code!") == [
        "user",
        "prefers",
        "dark",
        "mode",
        "in",
        "vs",
        "code",
    ]


def test_tokenize_empty():
    assert tokenize("  ...  ") == []


def test_query_terms_drop_stopwords_and_fragments():
    assert query_terms("What is the user's editor?") == {"user", "editor"}


def test_query_terms_keep_stopwords_if_nothing_else():
    assert query_terms("the") == {"the"}


def test_relevance_is_fraction_of_query_terms_matched():
    doc = document_terms("The user prefers dark mode", ["ui"])
    assert relevance(frozenset({"dark", "mode"}), doc) == 1.0
    assert relevance(frozenset({"dark", "python"}), doc) == 0.5
    assert relevance(frozenset({"python"}), doc) == 0.0
    assert relevance(frozenset(), doc) == 0.0


def test_relevance_counts_tags():
    doc = document_terms("Deploys go through CI", ["release-process"])
    assert relevance(frozenset({"release"}), doc) == 1.0
