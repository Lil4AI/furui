# furui

**Memory that knows what to forget.**

furui is an [MCP](https://modelcontextprotocol.io) server that gives AI agents a
*curated* long-term memory. Memories fade unless they are used, get stronger
every time they are recalled, and fall through the sieve when they stop
mattering. Pinned memories stay put.

- **Decay.** Every memory loses strength on a half-life curve (default 30 days).
- **Reinforcement.** Recalling a memory resets its decay clock and boosts it.
- **Pruning.** `prune` lists (or deletes) whatever has faded below a threshold.
- **Fully local.** Python + stdlib `sqlite3`. No API keys, no network, no embeddings.

> *furui* (篩) is Japanese for "sieve". It also sounds like 古い, "old", which is
> what this server is for: deciding which old things are still worth keeping.

---

## Why not just a vector DB?

A vector database answers "what is *similar* to this?" It does not answer
"what is still *worth knowing*?" Agent memory goes stale for reasons that have
nothing to do with similarity:

| Problem | Vector DB | furui |
|---|---|---|
| Last month's workaround is still there and outranks today's fix | Similarity doesn't care about age | Unused memories decay; fresh or recently used ones win |
| Memory grows forever and recall gets noisier | You write your own TTL/cleanup jobs | `prune(threshold)` with a dry-run preview is built in |
| Facts the agent keeps needing look the same as one-off trivia | All vectors are equal | Every recall reinforces a memory |
| Some facts must never be lost (the user's name, hard rules) | No such concept | `pinned=true`: floor on decay, never pruned |
| Setup | Embedding model, API key or GPU, index, extra service | One SQLite file |
| Why did this rank first? | Cosine distance between opaque vectors | `relevance × importance × recency × reinforcement`, every factor returned |

Embeddings are good at fuzzy matching, and furui may add them as an optional
relevance signal later. The core idea here is **lifecycle**, not retrieval:
deciding what to keep matters more than finding the nearest neighbour in a pile
that only grows.

---

## Quickstart

Requires Python 3.11+.

```bash
git clone <this repo> furui && cd furui
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"

python scripts/demo.py    # offline demo: 120 simulated days, recall, prune
pytest                    # run the test suite
furui                     # start the MCP server on stdio
```

The demo doesn't need the MCP SDK. It uses an in-memory database and a
simulated clock, so you can watch memories decay, get reinforced and get pruned:

```text
 id  imp  pin    idle  recalls   score  text
  5    4          10d        1   4.275  Never deploy to production on Fridays (team rule)
  3    4          30d        2   3.099  Project database is PostgreSQL 16 listening on ...
  1    5  yes    120d        0   2.500  User's name is Aiko; address her by first name
  ...
  7    1          80d        0   0.157  User found the weather API docs confusing
  4    1         115d        0   0.070  Lunch order on July 6th was a tuna onigiri
```

### MCP client configuration

Add furui to any MCP client that launches stdio servers (Claude Desktop, Cursor,
Antigravity, ...). Point `command` at the `furui` script inside your virtualenv:

```json
{
  "mcpServers": {
    "furui": {
      "command": "/absolute/path/to/furui/.venv/bin/furui",
      "args": [],
      "env": {
        "FURUI_DB_PATH": "/absolute/path/to/memories/furui.db",
        "FURUI_HALF_LIFE_DAYS": "30"
      }
    }
  }
}
```

On Windows the command is `C:\\path\\to\\furui\\.venv\\Scripts\\furui.exe`. You can
also use `"command": "python", "args": ["-m", "furui.server"]` with the right
interpreter.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `FURUI_DB_PATH` | `~/.furui/furui.db` | SQLite file (created on first use) |
| `FURUI_HALF_LIFE_DAYS` | `30` | Days for an untouched memory to lose half its recency |
| `FURUI_PIN_FLOOR` | `0.5` | Lowest recency factor a pinned memory can reach (0–1) |

---

## Tools

| Tool | Signature | What it does |
|---|---|---|
| `remember` | `remember(text, importance=3, tags=[], pinned=false)` | Store a memory. `importance` is 1 (trivia) to 5 (critical). Tags are normalised to lowercase. |
| `recall` | `recall(query, limit=5)` | Keyword search ranked by `relevance × score`. **Every returned memory is reinforced.** An empty query returns the strongest memories. |
| `forget` | `forget(id)` | Delete one memory permanently. This works on pinned memories too, because it's an explicit request. |
| `prune` | `prune(threshold, dry_run=true)` | List unpinned memories scoring below `threshold`, weakest first. Deletes them only when `dry_run=false`. |
| `stats` | `stats()` | Counts, score min/median/mean/max, how many memories each threshold would prune, importance histogram, top tags, config. |

Results include each memory's `score`, its factors (`recency`, `reinforcement`),
`idle_days` and, for recall, its `relevance` and `rank`. The agent can see why
something ranked where it did.

Suggested agent loop: `recall` before answering anything that depends on past
context, `remember` durable facts and decisions, then now and then run
`prune(0.5)` (dry run), review the list, and run it again with `dry_run=false`.

---

## How scoring works

Each memory's **current score** is computed at read time, never stored:

$$
\text{score} = \text{importance} \times \text{recency} \times \text{reinforcement}
$$

**Importance** is the 1–5 value given at `remember` time. A brand-new memory
scores exactly its importance.

**Recency** is half-life decay over the days since the memory was last recalled,
or since it was created if it has never been recalled:

$$
\text{recency} = 0.5^{\,\text{idle\_days} / \text{half\_life}}
$$

| idle days (half-life 30) | 0 | 15 | 30 | 60 | 90 | 180 |
|---|---|---|---|---|---|---|
| recency | 1.00 | 0.71 | 0.50 | 0.25 | 0.125 | 0.016 |
| score, importance 3 | 3.00 | 2.12 | 1.50 | 0.75 | 0.375 | 0.047 |

Pinned memories use `max(recency, FURUI_PIN_FLOOR)`, so a pinned importance-5
memory never drops below 2.5 with the default floor. Pinned memories are never
prune candidates.

**Reinforcement** rewards memories that keep proving useful, with diminishing
returns:

$$
\text{reinforcement} = 1 + 0.5 \cdot \ln(1 + \text{recall\_count})
$$

| recall_count | 0 | 1 | 3 | 10 | 100 |
|---|---|---|---|---|---|
| reinforcement | 1.00 | 1.35 | 1.69 | 2.20 | 3.31 |

Recalling a memory has two effects: `recall_count` goes up, and the decay clock
resets (`last_recalled_at = now`). A memory that's used often stays near the
top. One that's never used fades out.

**Relevance** (for `recall`) is keyword overlap. The query and each memory's
text + tags are lowercased and split on whitespace and punctuation. Common
stopwords and one-letter fragments are dropped from the query. Then:

$$
\text{relevance} = \frac{|\text{query terms} \cap \text{memory terms}|}{|\text{query terms}|}
$$

Memories with zero overlap are never returned. The rest are ranked by
`relevance × score`, so a strong partial match can beat a faded exact match.

**Picking a prune threshold.** A fresh importance-1 memory scores 1.0. An
importance-3 memory untouched for three half-lives scores 0.375. Thresholds
between 0.25 and 1.0 work well. `stats()` reports how many memories each of
0.25 / 0.5 / 1.0 would remove.

---

## furui vs plain memory servers

"Plain" here means the common append-only design: store text (or
entities/observations), search it, and keep everything until someone deletes it
by hand.

| | Plain memory server | Vector-DB memory | **furui** |
|---|---|---|---|
| Storage | JSON / SQLite file | Vector index + often a separate service | Single SQLite file (stdlib) |
| Retrieval | Substring / keyword | Embedding similarity | Keyword overlap × lifecycle score |
| Ranking uses age | ✗ | ✗ (usually) | ✓ half-life decay |
| Usage strengthens memories | ✗ | ✗ | ✓ reinforcement on recall |
| Importance levels | ✗ | ✗ (metadata at most) | ✓ 1–5, scales score |
| Never-forget memories | n/a (nothing is forgotten) | n/a | ✓ pinned, with decay floor |
| Built-in forgetting | ✗ manual delete only | ✗ TTL/cleanup scripts | ✓ `prune` with dry-run |
| Explainable ranking | partly | ✗ opaque distances | ✓ every factor returned |
| Needs API keys / network | no | often | **no** |
| Extra dependencies | few | embedding model + DB | `fastmcp` only |

---

## Development

```text
furui/
  server.py    FastMCP server and the five tools
  store.py     SQLite storage, recall, prune, stats
  scoring.py   pure functions: decay, reinforcement, relevance
tests/
  test_scoring.py
  test_store.py
scripts/demo.py
```

- `scoring.py` is pure and has no I/O. `store.py` takes an injectable clock, so
  tests simulate months of decay without sleeping.
- Scores are derived, not stored, so changing `FURUI_HALF_LIFE_DAYS` takes
  effect at once for every memory.
- v1 scores every memory in Python on each recall. That's fine for thousands
  of memories. An FTS5 pre-filter is the natural next step beyond that.

## License

MIT. See [LICENSE](LICENSE).
