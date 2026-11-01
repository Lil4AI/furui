"""Score computation for furui.

Everything in this module is pure: no I/O, no clocks, no database. The store
feeds in plain values (importance, idle days, recall count, ...) and gets
numbers back, which keeps the math easy to reason about and easy to test.

    score = importance x recency x reinforcement

* ``importance``    -- 1..5, chosen by the agent when the memory is stored.
* ``recency``       -- ``0.5 ** (idle_days / half_life_days)``, where
  ``idle_days`` is the time since the memory was last recalled (or created,
  if it has never been recalled). Pinned memories never drop below
  ``pin_floor``.
* ``reinforcement`` -- ``1 + weight * ln(1 + recall_count)``. Every recall
  helps, with diminishing returns.

Relevance for ``recall`` is keyword overlap: the fraction of distinct query
terms that appear in the memory's text + tags.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime

DEFAULT_HALF_LIFE_DAYS = 30.0
DEFAULT_PIN_FLOOR = 0.5
DEFAULT_REINFORCEMENT_WEIGHT = 0.5

MIN_IMPORTANCE = 1
MAX_IMPORTANCE = 5

ENV_HALF_LIFE = "FURUI_HALF_LIFE_DAYS"
ENV_PIN_FLOOR = "FURUI_PIN_FLOOR"

SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True)
class ScoringConfig:
    """Tunable parameters for the score formula."""

    half_life_days: float = DEFAULT_HALF_LIFE_DAYS
    pin_floor: float = DEFAULT_PIN_FLOOR
    reinforcement_weight: float = DEFAULT_REINFORCEMENT_WEIGHT

    def __post_init__(self) -> None:
        if not (math.isfinite(self.half_life_days) and self.half_life_days > 0):
            raise ValueError(f"half_life_days must be a positive number, got {self.half_life_days!r}")
        if not (math.isfinite(self.pin_floor) and 0.0 <= self.pin_floor <= 1.0):
            raise ValueError(f"pin_floor must be between 0 and 1, got {self.pin_floor!r}")
        if not (math.isfinite(self.reinforcement_weight) and self.reinforcement_weight >= 0):
            raise ValueError(
                f"reinforcement_weight must be >= 0, got {self.reinforcement_weight!r}"
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ScoringConfig":
        """Build a config from ``FURUI_HALF_LIFE_DAYS`` / ``FURUI_PIN_FLOOR``."""
        env = os.environ if env is None else env
        return cls(
            half_life_days=_float_from_env(env, ENV_HALF_LIFE, DEFAULT_HALF_LIFE_DAYS),
            pin_floor=_float_from_env(env, ENV_PIN_FLOOR, DEFAULT_PIN_FLOOR),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "half_life_days": self.half_life_days,
            "pin_floor": self.pin_floor,
            "reinforcement_weight": self.reinforcement_weight,
        }


def _float_from_env(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{key} must be a number, got {raw!r}") from None


@dataclass(frozen=True)
class ScoreBreakdown:
    """The individual factors of a score, useful for explaining a ranking."""

    importance: int
    recency: float
    reinforcement: float

    @property
    def score(self) -> float:
        return self.importance * self.recency * self.reinforcement


# --------------------------------------------------------------------------
# Decay & reinforcement
# --------------------------------------------------------------------------


def days_between(start: datetime, end: datetime) -> float:
    """Fractional days from ``start`` to ``end``, clamped at 0 (clock skew)."""
    return max(0.0, (end - start).total_seconds() / SECONDS_PER_DAY)


def recency(idle_days: float, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> float:
    """Half-life decay: 1.0 when fresh, 0.5 after one half-life, 0.25 after two..."""
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    return 0.5 ** (max(0.0, idle_days) / half_life_days)


def reinforcement(recall_count: int, weight: float = DEFAULT_REINFORCEMENT_WEIGHT) -> float:
    """Logarithmic boost for memories that keep being useful.

    0 recalls -> 1.0, 1 -> ~1.35, 10 -> ~2.20, 100 -> ~3.31 (with weight 0.5).
    """
    return 1.0 + weight * math.log1p(max(0, recall_count))


def score_breakdown(
    importance: int,
    idle_days: float,
    recall_count: int,
    pinned: bool,
    config: ScoringConfig | None = None,
) -> ScoreBreakdown:
    """Compute every factor of the current score."""
    cfg = config or ScoringConfig()
    rec = recency(idle_days, cfg.half_life_days)
    if pinned:
        rec = max(rec, cfg.pin_floor)
    return ScoreBreakdown(
        importance=importance,
        recency=rec,
        reinforcement=reinforcement(recall_count, cfg.reinforcement_weight),
    )


def current_score(
    importance: int,
    idle_days: float,
    recall_count: int,
    pinned: bool,
    config: ScoringConfig | None = None,
) -> float:
    """``importance x recency x reinforcement``."""
    return score_breakdown(importance, idle_days, recall_count, pinned, config).score


# --------------------------------------------------------------------------
# Keyword relevance
# --------------------------------------------------------------------------

_SPLIT_RE = re.compile(r"[\W_]+", re.UNICODE)

# Deliberately small: just enough that "what is the user's editor" doesn't
# match every memory containing "the" or "is".
STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be but by do does for from had has have how i if in is
    it its me my of on or so than that the their them then there these they
    this to was we were what when where which who why will with you your
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase and split on whitespace / punctuation."""
    return [tok for tok in _SPLIT_RE.split(text.lower()) if tok]


def query_terms(query: str) -> frozenset[str]:
    """Distinct query tokens, minus stopwords and 1-char fragments like the
    "s" in "user's" (unless that would leave nothing)."""
    tokens = set(tokenize(query))
    meaningful = {t for t in tokens if t not in STOPWORDS and len(t) > 1}
    return frozenset(meaningful or tokens)


def document_terms(text: str, tags: Iterable[str] = ()) -> frozenset[str]:
    """Distinct tokens over a memory's text and tags."""
    return frozenset(tokenize(" ".join([text, *tags])))


def relevance(terms: frozenset[str] | set[str], doc_terms: frozenset[str] | set[str]) -> float:
    """Fraction of query terms found in the document, in ``[0, 1]``."""
    if not terms:
        return 0.0
    return len(terms & doc_terms) / len(terms)
