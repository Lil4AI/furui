"""furui MCP server.

Run with ``furui`` (console script) or ``python -m furui.server``. Speaks MCP
over stdio. Configuration is via environment variables:

* ``FURUI_DB_PATH``         -- SQLite file (default ``~/.furui/furui.db``)
* ``FURUI_HALF_LIFE_DAYS``  -- recency half-life in days (default 30)
* ``FURUI_PIN_FLOOR``       -- minimum recency factor for pinned memories (default 0.5)
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from furui import __version__
from furui.store import MemoryStore, default_db_path

INSTRUCTIONS = """\
furui is your long-term memory. Memories fade unless they are used.

- remember: store durable facts, preferences, decisions and lessons -- not
  transient chatter. Use importance 1-5 (5 = critical, 3 = normal) and a few
  short tags. Pin only things that must never be forgotten.
- recall: search before answering questions that may depend on past context.
  Every memory returned is reinforced, so recall only what you actually need.
- forget: delete a memory that is wrong or obsolete.
- prune: periodically run with dry_run=true, review the list, then run again
  with dry_run=false to delete stale memories. Pinned memories are never pruned.
- stats: inspect the overall health of memory (sizes, score spread, prunable counts).
"""

mcp = FastMCP("furui", instructions=INSTRUCTIONS, version=__version__)

_store: MemoryStore | None = None
_store_lock = threading.Lock()


def get_store() -> MemoryStore:
    """Lazily open the store so importing this module has no side effects."""
    global _store
    with _store_lock:
        if _store is None:
            _store = MemoryStore(default_db_path())
        return _store


@contextmanager
def _user_errors() -> Iterator[None]:
    """Report bad arguments to the agent as a clean tool error, not a crash."""
    try:
        yield
    except ValueError as exc:
        raise ToolError(str(exc)) from None


@mcp.tool()
def remember(
    text: str,
    importance: int = 3,
    tags: list[str] | None = None,
    pinned: bool = False,
) -> dict[str, Any]:
    """Store a new long-term memory.

    Args:
        text: The fact, preference, decision or lesson to remember.
        importance: 1 (trivia) to 5 (critical). Scales the memory's score.
        tags: Optional short keywords; they also count toward recall matching.
        pinned: If true, the memory never decays below a floor and is never pruned.
    """
    store = get_store()
    with _user_errors():
        memory = store.add(text, importance=importance, tags=tags or [], pinned=pinned)
    return {"stored": store.score(memory).to_dict()}


@mcp.tool()
def recall(query: str, limit: int = 5) -> dict[str, Any]:
    """Search memories by keyword overlap, ranked by relevance x current score.

    Every returned memory is reinforced: its recall_count goes up and its decay
    clock resets. An empty query returns the strongest memories overall.

    Args:
        query: Natural-language or keyword query.
        limit: Maximum number of memories to return (1-100).
    """
    with _user_errors():
        hits = get_store().recall(query, limit=limit)
    return {"query": query, "count": len(hits), "memories": [h.to_dict() for h in hits]}


@mcp.tool()
def forget(id: int) -> dict[str, Any]:  # noqa: A002 - tool parameter name is part of the API
    """Permanently delete one memory by id (works on pinned memories too).

    Args:
        id: The memory id returned by remember / recall.
    """
    deleted = get_store().delete(id)
    return {"id": id, "deleted": deleted}


@mcp.tool()
def prune(threshold: float, dry_run: bool = True) -> dict[str, Any]:
    """List or delete unpinned memories whose current score is below a threshold.

    A brand-new importance-1 memory scores 1.0; an importance-3 memory left
    untouched for one half-life scores 1.5. Thresholds around 0.25-1.0 are typical.

    Args:
        threshold: Memories with score strictly below this are candidates.
        dry_run: If true (default) only list candidates; if false, delete them.
    """
    with _user_errors():
        weak = get_store().prune(threshold, dry_run=dry_run)
    return {
        "threshold": threshold,
        "dry_run": dry_run,
        "count": len(weak),
        "deleted": 0 if dry_run else len(weak),
        "memories": [s.to_dict() for s in weak],
    }


@mcp.tool()
def stats() -> dict[str, Any]:
    """Summary of the memory store: counts, score distribution, prunable counts, config."""
    return get_store().stats()


def main() -> None:
    # stdio transport; skip the startup banner so client logs stay clean.
    mcp.run(show_banner=False)


if __name__ == "__main__":
    main()
