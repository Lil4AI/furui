from datetime import datetime, timedelta, timezone

import pytest

from furui.scoring import ScoringConfig, reinforcement
from furui.store import MemoryStore, normalize_tags

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


class FakeClock:
    """Deterministic, manually advanced clock."""

    def __init__(self, start: datetime = T0) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, days: float) -> None:
        self.now += timedelta(days=days)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def store(clock):
    cfg = ScoringConfig(half_life_days=30, pin_floor=0.5)
    s = MemoryStore(":memory:", config=cfg, clock=clock)
    yield s
    s.close()


# --------------------------------------------------------------------------
# CRUD
# --------------------------------------------------------------------------


def test_add_and_get_roundtrip(store, clock):
    m = store.add("User prefers dark mode", importance=4, tags=["UI", " prefs "], pinned=True)
    assert m.id == 1
    got = store.get(m.id)
    assert got == m
    assert got.text == "User prefers dark mode"
    assert got.tags == ("ui", "prefs")
    assert got.importance == 4
    assert got.pinned is True
    assert got.created_at == clock.now
    assert got.last_recalled_at is None
    assert got.recall_count == 0


def test_add_defaults(store):
    m = store.add("plain fact")
    assert (m.importance, m.tags, m.pinned) == (3, (), False)


def test_ids_increment(store):
    ids = [store.add(f"memory {i}").id for i in range(3)]
    assert ids == [1, 2, 3]
    assert store.count() == 3
    assert [m.id for m in store.all()] == ids


def test_get_missing_returns_none(store):
    assert store.get(999) is None


@pytest.mark.parametrize("text", ["", "   ", None, 42])
def test_add_rejects_bad_text(store, text):
    with pytest.raises(ValueError):
        store.add(text)


@pytest.mark.parametrize("importance", [0, 6, -1, 2.5, "3", True])
def test_add_rejects_bad_importance(store, importance):
    with pytest.raises(ValueError):
        store.add("x", importance=importance)


def test_normalize_tags():
    assert normalize_tags(["A", "a", "  b  c ", "", "B  C"]) == ("a", "b c")
    assert normalize_tags("python, tooling ,python") == ("python", "tooling")
    assert normalize_tags(None) == ()


def test_update_fields(store):
    m = store.add("old text", importance=2, tags=["a"])
    updated = store.update(m.id, text="new text", importance=5, tags=["B"], pinned=True)
    assert updated.text == "new text"
    assert updated.importance == 5
    assert updated.tags == ("b",)
    assert updated.pinned is True
    assert updated.recall_count == 0  # editing is not recalling


def test_update_missing_returns_none(store):
    assert store.update(42, text="nope") is None


def test_delete(store):
    m = store.add("temporary")
    assert store.delete(m.id) is True
    assert store.get(m.id) is None
    assert store.delete(m.id) is False


def test_delete_works_on_pinned(store):
    m = store.add("pinned but wrong", pinned=True)
    assert store.delete(m.id) is True


def test_persists_across_reopen(tmp_path, clock):
    path = tmp_path / "nested" / "furui.db"
    with MemoryStore(path, config=ScoringConfig(), clock=clock) as s:
        s.add("survives restarts", importance=5, tags=["durable"])
    with MemoryStore(path, config=ScoringConfig(), clock=clock) as s:
        [m] = s.all()
        assert m.text == "survives restarts"
        assert m.tags == ("durable",)
        assert m.created_at == clock.now


# --------------------------------------------------------------------------
# Scoring through the store (decay)
# --------------------------------------------------------------------------


def test_score_decays_with_time(store, clock):
    m = store.add("decaying", importance=4)
    assert store.score(m).score == pytest.approx(4.0)
    clock.advance(30)
    assert store.score(m).score == pytest.approx(2.0)
    clock.advance(30)
    assert store.score(m).score == pytest.approx(1.0)


def test_half_life_env_is_respected(monkeypatch, clock):
    monkeypatch.setenv("FURUI_HALF_LIFE_DAYS", "10")
    with MemoryStore(clock=clock) as s:
        m = s.add("short-lived", importance=2)
        clock.advance(10)
        assert s.score(m).score == pytest.approx(1.0)


def test_pinned_memory_holds_floor(store, clock):
    pinned = store.add("never forget", importance=2, pinned=True)
    plain = store.add("ordinary", importance=2)
    clock.advance(3650)
    assert store.score(pinned).score == pytest.approx(1.0)  # 2 * 0.5 floor
    assert store.score(plain).score < 1e-30


# --------------------------------------------------------------------------
# Recall & reinforcement
# --------------------------------------------------------------------------


def test_recall_returns_only_relevant(store):
    store.add("User prefers dark mode in the editor", tags=["ui"])
    store.add("Project uses PostgreSQL 16", tags=["database"])
    hits = store.recall("dark mode")
    assert [h.memory.text for h in hits] == ["User prefers dark mode in the editor"]
    assert hits[0].relevance == 1.0


def test_recall_matches_tags(store):
    m = store.add("Use pnpm, never npm", tags=["javascript", "tooling"])
    assert [h.memory.id for h in store.recall("tooling")] == [m.id]


def test_recall_no_match_returns_empty(store):
    store.add("something")
    assert store.recall("unrelated query") == []


def test_recall_reinforces_returned_memories(store, clock):
    m = store.add("Deploy on Fridays is forbidden", importance=3)
    clock.advance(5)
    [hit] = store.recall("deploy fridays")
    assert hit.memory.recall_count == 1
    assert hit.memory.last_recalled_at == clock.now
    stored = store.get(m.id)
    assert stored.recall_count == 1
    assert stored.last_recalled_at == clock.now
    store.recall("deploy")
    assert store.get(m.id).recall_count == 2


def test_recall_resets_decay_clock_and_boosts(store, clock):
    m = store.add("API rate limit is 100 rps", importance=4)
    clock.advance(60)
    before = store.score(store.get(m.id)).score
    assert before == pytest.approx(1.0)  # 4 * 0.25
    [hit] = store.recall("rate limit")
    assert hit.score == pytest.approx(before)  # ranking uses pre-recall score
    after = store.score(store.get(m.id)).score
    assert after == pytest.approx(4 * 1.0 * reinforcement(1))


def test_recall_does_not_reinforce_unreturned(store):
    a = store.add("python tip one", importance=5)
    b = store.add("python tip two", importance=1)
    hits = store.recall("python", limit=1)
    assert [h.memory.id for h in hits] == [a.id]
    assert store.get(b.id).recall_count == 0


def test_recall_ranks_by_relevance_times_score(store, clock):
    old = store.add("python packaging uses uv", importance=5)
    clock.advance(90)  # old: 5 * 0.125 = 0.625
    new = store.add("python style uses ruff", importance=2)  # 2.0
    full = store.add("python packaging lockfile policy", importance=1)  # 1.0, full match

    hits = store.recall("python packaging")
    ranks = {h.memory.id: h.rank for h in hits}
    assert ranks[full.id] == pytest.approx(1.0 * 1.0)
    assert ranks[new.id] == pytest.approx(0.5 * 2.0)
    assert ranks[old.id] == pytest.approx(1.0 * 0.625)
    assert hits[-1].memory.id == old.id
    assert [h.rank for h in hits] == sorted((h.rank for h in hits), reverse=True)


def test_recall_importance_breaks_equal_relevance(store):
    low = store.add("coffee order: flat white", importance=1)
    high = store.add("coffee allergy: none, but no oat milk", importance=5)
    assert [h.memory.id for h in store.recall("coffee")] == [high.id, low.id]


def test_recall_limit(store):
    for i in range(10):
        store.add(f"note {i} about testing")
    assert len(store.recall("testing", limit=3)) == 3
    with pytest.raises(ValueError):
        store.recall("testing", limit=0)


def test_empty_query_returns_strongest(store):
    store.add("weak", importance=1)
    strong = store.add("strong", importance=5)
    hits = store.recall("", limit=1)
    assert [h.memory.id for h in hits] == [strong.id]


# --------------------------------------------------------------------------
# Prune
# --------------------------------------------------------------------------


@pytest.fixture
def aged_store(store, clock):
    """Memories with known scores 120 days later (half-life 30 => recency 1/16)."""
    ids = {
        "stale_low": store.add("stale low", importance=1).id,  # 0.0625
        "stale_high": store.add("stale high", importance=5).id,  # 0.3125
        "pinned": store.add("pinned low", importance=1, pinned=True).id,  # 0.5 (floor)
    }
    clock.advance(120)
    ids["fresh"] = store.add("fresh", importance=1).id  # 1.0
    return store, ids


def test_prune_dry_run_lists_but_keeps(aged_store):
    store, ids = aged_store
    weak = store.prune(threshold=0.5, dry_run=True)
    assert [s.memory.id for s in weak] == [ids["stale_low"], ids["stale_high"]]
    assert weak[0].score == pytest.approx(0.0625)
    assert store.count() == 4


def test_prune_is_dry_run_by_default(aged_store):
    store, _ = aged_store
    store.prune(threshold=10)
    assert store.count() == 4


def test_prune_deletes_when_not_dry_run(aged_store):
    store, ids = aged_store
    deleted = store.prune(threshold=0.5, dry_run=False)
    assert {s.memory.id for s in deleted} == {ids["stale_low"], ids["stale_high"]}
    assert {m.id for m in store.all()} == {ids["pinned"], ids["fresh"]}


def test_prune_never_selects_pinned(aged_store):
    store, ids = aged_store
    weak = store.prune(threshold=1_000, dry_run=False)
    assert ids["pinned"] not in {s.memory.id for s in weak}
    assert [m.id for m in store.all()] == [ids["pinned"]]


def test_prune_threshold_is_strict(aged_store):
    store, ids = aged_store
    assert ids["fresh"] not in {s.memory.id for s in store.prune(threshold=1.0)}


def test_prune_zero_threshold_selects_nothing(aged_store):
    store, _ = aged_store
    assert store.prune(threshold=0) == []


def test_recall_saves_memory_from_prune(aged_store, clock):
    store, ids = aged_store
    store.recall("stale high")  # reinforce
    weak_ids = {s.memory.id for s in store.prune(threshold=0.5)}
    assert ids["stale_high"] not in weak_ids


@pytest.mark.parametrize("threshold", [-1, float("nan"), float("inf"), "0.5", None])
def test_prune_rejects_bad_threshold(store, threshold):
    with pytest.raises(ValueError):
        store.prune(threshold)


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------


def test_stats_empty(store):
    s = store.stats()
    assert s["total"] == 0
    assert s["score"] is None
    assert s["strongest"] is None


def test_stats_counts(aged_store):
    store, ids = aged_store
    store.recall("fresh")
    s = store.stats()
    assert s["total"] == 4
    assert s["pinned"] == 1
    assert s["total_recalls"] == 1
    assert s["never_recalled"] == 3
    assert s["prunable_below"] == {"0.25": 1, "0.5": 2, "1.0": 2}
    assert s["importance_histogram"]["1"] == 3
    assert s["strongest"]["id"] == ids["fresh"]
    assert s["weakest"]["id"] == ids["stale_low"]
    assert s["config"]["half_life_days"] == 30
