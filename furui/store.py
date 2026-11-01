"""SQLite storage for furui.

Uses only the standard library. Scores are never stored -- they are a function
of time, so they are computed on read from the persisted fields.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import statistics
import threading
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from furui.scoring import (
    MAX_IMPORTANCE,
    MIN_IMPORTANCE,
    ScoreBreakdown,
    ScoringConfig,
    days_between,
    document_terms,
    query_terms,
    relevance,
    score_breakdown,
)

ENV_DB_PATH = "FURUI_DB_PATH"
DEFAULT_DB_PATH = Path.home() / ".furui" / "furui.db"
MAX_RECALL_LIMIT = 100
SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    text             TEXT    NOT NULL,
    tags             TEXT    NOT NULL DEFAULT '[]',
    importance       INTEGER NOT NULL CHECK (importance BETWEEN 1 AND 5),
    created_at       TEXT    NOT NULL,
    last_recalled_at TEXT,
    recall_count     INTEGER NOT NULL DEFAULT 0 CHECK (recall_count >= 0),
    pinned           INTEGER NOT NULL DEFAULT 0 CHECK (pinned IN (0, 1))
);
"""

Clock = Callable[[], datetime]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def default_db_path() -> Path:
    raw = os.environ.get(ENV_DB_PATH)
    return Path(raw).expanduser() if raw else DEFAULT_DB_PATH


def _to_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _fmt(dt: datetime | None) -> str | None:
    return None if dt is None else _to_utc(dt).isoformat()


def _parse(raw: str | None) -> datetime | None:
    return None if raw is None else _to_utc(datetime.fromisoformat(raw))


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Memory:
    id: int
    text: str
    tags: tuple[str, ...]
    importance: int
    created_at: datetime
    last_recalled_at: datetime | None
    recall_count: int
    pinned: bool

    @property
    def last_active(self) -> datetime:
        """Decay anchor: last recall, or creation if never recalled."""
        return self.last_recalled_at or self.created_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "tags": list(self.tags),
            "importance": self.importance,
            "created_at": _fmt(self.created_at),
            "last_recalled_at": _fmt(self.last_recalled_at),
            "recall_count": self.recall_count,
            "pinned": self.pinned,
        }


@dataclass(frozen=True)
class ScoredMemory:
    memory: Memory
    breakdown: ScoreBreakdown
    idle_days: float

    @property
    def score(self) -> float:
        return self.breakdown.score

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.memory.to_dict(),
            "score": round(self.score, 4),
            "recency": round(self.breakdown.recency, 4),
            "reinforcement": round(self.breakdown.reinforcement, 4),
            "idle_days": round(self.idle_days, 2),
        }


@dataclass(frozen=True)
class RecallHit:
    """A recall result.

    ``memory`` reflects the state *after* reinforcement (bumped recall_count,
    fresh last_recalled_at). ``score`` and ``rank`` are the values used for
    ranking, i.e. *before* reinforcement.
    """

    memory: Memory
    relevance: float
    score: float

    @property
    def rank(self) -> float:
        return self.relevance * self.score

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.memory.to_dict(),
            "relevance": round(self.relevance, 4),
            "score": round(self.score, 4),
            "rank": round(self.rank, 4),
        }


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _validate_text(text: Any) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    return text.strip()


def _validate_importance(importance: Any) -> int:
    if isinstance(importance, bool) or not isinstance(importance, int):
        raise ValueError(f"importance must be an integer 1-5, got {importance!r}")
    if not MIN_IMPORTANCE <= importance <= MAX_IMPORTANCE:
        raise ValueError(f"importance must be between 1 and 5, got {importance}")
    return importance


def normalize_tags(tags: Iterable[str] | str | None) -> tuple[str, ...]:
    """Lowercase, strip, de-duplicate (order preserved). Accepts 'a, b' too."""
    if tags is None:
        return ()
    if isinstance(tags, str):
        tags = tags.split(",")
    seen: dict[str, None] = {}
    for tag in tags:
        if not isinstance(tag, str):
            raise ValueError(f"tags must be strings, got {tag!r}")
        clean = " ".join(tag.lower().split())
        if clean:
            seen.setdefault(clean, None)
    return tuple(seen)


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------


class MemoryStore:
    """A curated memory store backed by a single SQLite file.

    ``clock`` is injectable so tests (and the demo) can simulate time passing.
    """

    def __init__(
        self,
        path: str | os.PathLike[str] = ":memory:",
        config: ScoringConfig | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.config = config or ScoringConfig.from_env()
        self._clock = clock or utcnow
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            self.path = str(Path(self.path).expanduser())
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def now(self) -> datetime:
        return _to_utc(self._clock())

    # -- CRUD --------------------------------------------------------------

    def add(
        self,
        text: str,
        importance: int = 3,
        tags: Iterable[str] | str | None = None,
        pinned: bool = False,
    ) -> Memory:
        text = _validate_text(text)
        importance = _validate_importance(importance)
        clean_tags = normalize_tags(tags)
        created = self.now()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO memories (text, tags, importance, created_at, pinned)"
                " VALUES (?, ?, ?, ?, ?)",
                (text, json.dumps(clean_tags), importance, _fmt(created), int(bool(pinned))),
            )
            new_id = cur.lastrowid
        assert new_id is not None
        return Memory(
            id=new_id,
            text=text,
            tags=clean_tags,
            importance=importance,
            created_at=created,
            last_recalled_at=None,
            recall_count=0,
            pinned=bool(pinned),
        )

    def get(self, memory_id: int) -> Memory | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return None if row is None else self._row_to_memory(row)

    def all(self) -> list[Memory]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM memories ORDER BY id").fetchall()
        return [self._row_to_memory(r) for r in rows]

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def update(
        self,
        memory_id: int,
        *,
        text: str | None = None,
        importance: int | None = None,
        tags: Iterable[str] | str | None = None,
        pinned: bool | None = None,
    ) -> Memory | None:
        """Edit fields in place. Does not count as a recall."""
        fields: dict[str, Any] = {}
        if text is not None:
            fields["text"] = _validate_text(text)
        if importance is not None:
            fields["importance"] = _validate_importance(importance)
        if tags is not None:
            fields["tags"] = json.dumps(normalize_tags(tags))
        if pinned is not None:
            fields["pinned"] = int(bool(pinned))
        if fields:
            assignments = ", ".join(f"{k} = ?" for k in fields)
            with self._lock, self._conn:
                self._conn.execute(
                    f"UPDATE memories SET {assignments} WHERE id = ?",
                    (*fields.values(), memory_id),
                )
        return self.get(memory_id)

    def delete(self, memory_id: int) -> bool:
        """Remove a memory unconditionally (pinned or not)."""
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return cur.rowcount > 0

    # -- scoring -----------------------------------------------------------

    def score(self, memory: Memory, now: datetime | None = None) -> ScoredMemory:
        now = now or self.now()
        idle = days_between(memory.last_active, now)
        breakdown = score_breakdown(
            memory.importance, idle, memory.recall_count, memory.pinned, self.config
        )
        return ScoredMemory(memory=memory, breakdown=breakdown, idle_days=idle)

    def scored(self) -> list[ScoredMemory]:
        """Every memory with its current score, highest first."""
        now = self.now()
        items = [self.score(m, now) for m in self.all()]
        items.sort(key=lambda s: (-s.score, -s.memory.id))
        return items

    # -- recall ------------------------------------------------------------

    def recall(self, query: str, limit: int = 5) -> list[RecallHit]:
        """Rank by ``relevance x score``, return the top ``limit``, reinforce them.

        An empty / punctuation-only query matches everything (relevance 1), so
        it returns the strongest memories overall.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError(f"limit must be a positive integer, got {limit!r}")
        limit = min(limit, MAX_RECALL_LIMIT)
        terms = query_terms(query or "")
        now = self.now()

        candidates: list[tuple[float, float, Memory]] = []
        for memory in self.all():
            rel = relevance(terms, document_terms(memory.text, memory.tags)) if terms else 1.0
            if rel <= 0:
                continue
            candidates.append((rel, self.score(memory, now).score, memory))

        candidates.sort(key=lambda c: (-(c[0] * c[1]), -c[1], -c[2].id))
        top = candidates[:limit]
        if not top:
            return []

        ids = [m.id for _, _, m in top]
        placeholders = ", ".join("?" for _ in ids)
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE memories SET last_recalled_at = ?, recall_count = recall_count + 1"
                f" WHERE id IN ({placeholders})",
                (_fmt(now), *ids),
            )

        return [
            RecallHit(
                memory=replace(m, last_recalled_at=now, recall_count=m.recall_count + 1),
                relevance=rel,
                score=score,
            )
            for rel, score, m in top
        ]

    # -- prune -------------------------------------------------------------

    def prune_candidates(self, threshold: float) -> list[ScoredMemory]:
        """Unpinned memories whose current score is below ``threshold``, weakest first."""
        threshold = _validate_threshold(threshold)
        weak = [s for s in self.scored() if not s.memory.pinned and s.score < threshold]
        weak.sort(key=lambda s: (s.score, s.memory.id))
        return weak

    def prune(self, threshold: float, dry_run: bool = True) -> list[ScoredMemory]:
        """List (``dry_run``) or delete memories scoring below ``threshold``."""
        weak = self.prune_candidates(threshold)
        if not dry_run and weak:
            ids = [s.memory.id for s in weak]
            placeholders = ", ".join("?" for _ in ids)
            with self._lock, self._conn:
                self._conn.execute(
                    f"DELETE FROM memories WHERE pinned = 0 AND id IN ({placeholders})", ids
                )
        return weak

    # -- stats -------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        items = self.scored()
        scores = [s.score for s in items]
        tag_counts = Counter(tag for s in items for tag in s.memory.tags)
        created = [s.memory.created_at for s in items]
        unpinned = [s.score for s in items if not s.memory.pinned]

        return {
            "total": len(items),
            "pinned": sum(1 for s in items if s.memory.pinned),
            "total_recalls": sum(s.memory.recall_count for s in items),
            "never_recalled": sum(1 for s in items if s.memory.recall_count == 0),
            "score": None
            if not scores
            else {
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
                "mean": round(statistics.fmean(scores), 4),
                "median": round(statistics.median(scores), 4),
            },
            "prunable_below": {
                str(t): sum(1 for sc in unpinned if sc < t) for t in (0.25, 0.5, 1.0)
            },
            "importance_histogram": {
                str(i): sum(1 for s in items if s.memory.importance == i)
                for i in range(MIN_IMPORTANCE, MAX_IMPORTANCE + 1)
            },
            "top_tags": [[tag, n] for tag, n in tag_counts.most_common(10)],
            "oldest_created_at": _fmt(min(created)) if created else None,
            "newest_created_at": _fmt(max(created)) if created else None,
            "strongest": items[0].to_dict() if items else None,
            "weakest": items[-1].to_dict() if items else None,
            "config": self.config.to_dict(),
            "db_path": self.path,
        }

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            text=row["text"],
            tags=tuple(json.loads(row["tags"])),
            importance=row["importance"],
            created_at=_parse(row["created_at"]),  # type: ignore[arg-type]
            last_recalled_at=_parse(row["last_recalled_at"]),
            recall_count=row["recall_count"],
            pinned=bool(row["pinned"]),
        )


def _validate_threshold(threshold: Any) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError(f"threshold must be a number, got {threshold!r}")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError(f"threshold must be a finite number >= 0, got {threshold!r}")
    return float(threshold)


__all__ = [
    "DEFAULT_DB_PATH",
    "ENV_DB_PATH",
    "Memory",
    "MemoryStore",
    "RecallHit",
    "ScoredMemory",
    "default_db_path",
    "normalize_tags",
    "utcnow",
]
