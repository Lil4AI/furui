"""furui demo -- seeds 120 days of simulated memories, then recalls and prunes.

Fully offline: uses an in-memory SQLite database and a simulated clock, and
does not need the MCP SDK installed. Run from the repo root:

    python scripts/demo.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from furui.scoring import ScoringConfig  # noqa: E402
from furui.store import MemoryStore, ScoredMemory  # noqa: E402

START = datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc)
TODAY_OFFSET = 120  # days of simulated history

# (day, text, importance, tags, pinned)
SEED = [
    (0, "User's name is Aiko; address her by first name", 5, ["user", "identity"], True),
    (0, "User prefers dark mode and a monospace font in every editor", 3, ["ui", "preferences"], False),
    (2, "Project database is PostgreSQL 16 listening on port 5433", 4, ["database", "infra"], False),
    (5, "Lunch order on July 6th was a tuna onigiri", 1, ["trivia"], False),
    (10, "Never deploy to production on Fridays (team rule)", 4, ["deploy", "process"], False),
    (20, "Workaround: pin requests<2.32 until the upstream SSL fix lands", 2, ["python", "workaround"], False),
    (40, "User found the weather API docs confusing", 1, ["trivia"], False),
    (60, "CI runs on GitHub Actions; releases go through release.yml", 3, ["ci", "deploy"], False),
    (100, "Use ruff for linting and formatting Python code", 3, ["python", "tooling"], False),
    (115, "User's favourite test runner is pytest with -q", 2, ["python", "testing"], False),
]

# Recalls that happened during the simulated history (day, query).
HISTORY_RECALLS = [
    (50, "database port"),
    (90, "database"),
    (110, "deploy production"),
]


class SimClock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def set_day(self, day: float) -> None:
        self.now = START + timedelta(days=day)


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(f" {title}")
    print("=" * 78)


def short(text: str, width: int = 50) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def print_table(items: list[ScoredMemory]) -> None:
    print(f"{'id':>3}  {'imp':>3}  {'pin':>3}  {'idle':>6}  {'recalls':>7}  {'score':>6}  text")
    print(f"{'-' * 3}  {'-' * 3}  {'-' * 3}  {'-' * 6}  {'-' * 7}  {'-' * 6}  {'-' * 40}")
    for s in items:
        m = s.memory
        print(
            f"{m.id:>3}  {m.importance:>3}  {'yes' if m.pinned else '':>3}  "
            f"{s.idle_days:>5.0f}d  {m.recall_count:>7}  {s.score:>6.3f}  {short(m.text)}"
        )


def recall(store: MemoryStore, query: str, limit: int = 3) -> None:
    print(f'\n> recall("{query}", limit={limit})')
    hits = store.recall(query, limit=limit)
    if not hits:
        print("  (nothing relevant)")
    for h in hits:
        after = store.score(h.memory).score
        print(
            f"  #{h.memory.id:<2} rank={h.rank:.3f} (relevance {h.relevance:.2f} x score {h.score:.3f})"
            f"  -> reinforced to {after:.3f}, recall #{h.memory.recall_count}"
        )
        print(f"      {short(h.memory.text, 70)}")


def main() -> None:
    clock = SimClock(START)
    config = ScoringConfig.from_env()  # honours FURUI_HALF_LIFE_DAYS / FURUI_PIN_FLOOR
    store = MemoryStore(":memory:", config=config, clock=clock)

    # Replay history in chronological order.
    events = [(d, "add", row) for d, *row in SEED] + [(d, "recall", q) for d, q in HISTORY_RECALLS]
    for day, kind, payload in sorted(events, key=lambda e: (e[0], e[1] == "recall")):
        clock.set_day(day)
        if kind == "add":
            text, importance, tags, pinned = payload
            store.add(text, importance=importance, tags=tags, pinned=pinned)
        else:
            store.recall(payload)
    clock.set_day(TODAY_OFFSET)

    print("furui demo -- memory that knows what to forget")
    print(
        f"half-life {config.half_life_days:g} days, pin floor {config.pin_floor:g}; "
        f"simulated {TODAY_OFFSET} days, today is {clock.now:%Y-%m-%d}"
    )

    rule(f"1. All memories on day {TODAY_OFFSET}, strongest first")
    print_table(store.scored())
    print("\nUnused memories have decayed; recalled and pinned ones held up.")

    rule("2. Recall: relevance x score, and every hit gets reinforced")
    recall(store, "dark mode editor")
    recall(store, "which database port?")
    recall(store, "python tooling", limit=2)
    recall(store, "kubernetes helm chart")

    rule("3. prune(threshold=0.5, dry_run=True) -- what would be forgotten?")
    weak = store.prune(0.5, dry_run=True)
    print_table(weak)
    print(f"\n{len(weak)} candidate(s). The dark-mode preference was saved by the recall above;")
    print("the pinned identity memory is never a candidate.")

    rule("4. prune(threshold=0.5, dry_run=False) -- forget them")
    deleted = store.prune(0.5, dry_run=False)
    print(f"deleted ids: {[s.memory.id for s in deleted]}")
    print()
    print_table(store.scored())

    rule("5. stats()")
    st = store.stats()
    print(f"total memories : {st['total']}  (pinned: {st['pinned']})")
    print(f"total recalls  : {st['total_recalls']}  (never recalled: {st['never_recalled']})")
    sc = st["score"]
    print(f"score min/median/max : {sc['min']:.3f} / {sc['median']:.3f} / {sc['max']:.3f}")
    print(f"prunable below : {st['prunable_below']}")
    print(f"top tags       : {', '.join(f'{t} ({n})' for t, n in st['top_tags'][:6])}")

    store.close()


if __name__ == "__main__":
    main()
