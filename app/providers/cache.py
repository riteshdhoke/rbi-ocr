"""SQLite response cache for model calls — BUILD-BRIEF M2, "not optional".

Its job is economic, not architectural: without it every prompt experiment on
the laptop costs two seconds and a rate-limit slot, and Phase 0 iteration stops
feeling local.  Everything about the design follows from "a stale hit must be
impossible":

* the key is ``sha256(cache_schema_version | model | prompt | canonical params)``.
  ``cache_schema_version`` is *inside the hash* **and** stored as a column, so
  bumping :data:`CACHE_SCHEMA_VERSION` misses every old row rather than
  returning an answer produced under different framing;
* parameters are canonicalised (sorted keys, no incidental whitespace) so
  ``{"temperature": 0, "top_p": 1}`` and ``{"top_p": 1, "temperature": 0}`` are
  one entry;
* the value is stored as an opaque string.  The cache never parses a response,
  so a schema change downstream cannot make it serve something that no longer
  validates — the caller revalidates every hit.

Thread-safe by a single connection behind an ``RLock``.  One connection rather
than one per thread because ``:memory:`` must behave like a file for tests, and
because write contention here is nowhere near the bottleneck.

This module does NOT handle:
  * deciding *what* is cacheable.  A non-deterministic call (``temperature``
    above 0) is still cached if the caller asks; the caller puts the sampling
    parameters in ``params`` and thereby decides;
  * TTL or eviction.  A model call is a pure function of its key, so entries do
    not go stale; the developer clears the file, or calls
    :meth:`ResponseCache.invalidate_model` after redeploying a model id;
  * encryption or PII scrubbing.  Prompts contain complainant text, so the
    cache file lives under the gitignored ``.cache/`` directory and is never
    shipped.  It stores the prompt **hash**, never the prompt;
  * cross-process locking beyond what SQLite itself provides.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Self

from prometheus_client import Counter
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "MEMORY_PATH",
    "CacheEntry",
    "CacheStats",
    "ResponseCache",
    "canonical_params",
    "make_cache_key",
]

#: Bump whenever the *meaning* of a cached response changes — a new prompt
#: template, a different system message, a changed post-processing contract.
#: Old rows then miss instead of lying.  It is part of the hashed key, so a
#: bump cannot be defeated by an existing database file.
CACHE_SCHEMA_VERSION: Final[str] = "1"

#: Field separator inside the hashed payload.  ASCII Unit Separator cannot
#: occur in a model id and is vanishingly unlikely in a prompt, so
#: ``("ab", "c")`` and ``("a", "bc")`` cannot collide.
_KEY_SEPARATOR: Final[str] = "\x1f"

#: Sentinel path for an in-process cache (tests, dry runs).
MEMORY_PATH: Final[str] = ":memory:"

_CACHE_EVENTS = Counter(
    "provider_cache_events_total",
    "Provider response-cache events by outcome.",
    ["event"],
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS provider_cache (
    key            TEXT    PRIMARY KEY,
    schema_version TEXT    NOT NULL,
    model          TEXT    NOT NULL,
    response       TEXT    NOT NULL,
    created_at     REAL    NOT NULL,
    hit_count      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_provider_cache_model
    ON provider_cache (model);
CREATE INDEX IF NOT EXISTS idx_provider_cache_version
    ON provider_cache (schema_version);
"""


def _json_default(value: object) -> str:
    """Render a non-JSON parameter reproducibly rather than failing the key."""
    return f"{type(value).__name__}:{value!r}"


def canonical_params(params: Mapping[str, Any] | None) -> str:
    """Serialise call parameters to a stable string.

    Sorted keys, no incidental whitespace, ``ensure_ascii=False`` so Devanagari
    inside a parameter is not re-encoded differently across Python versions.
    ``None`` and ``{}`` both canonicalise to ``"{}"``.
    """
    return json.dumps(
        dict(params or {}),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    )


def make_cache_key(
    *,
    model: str,
    prompt: str,
    params: Mapping[str, Any] | None = None,
    schema_version: str = CACHE_SCHEMA_VERSION,
) -> str:
    """``sha256(schema_version | model | prompt | canonical params)``, hex.

    ``prompt`` should be the fully rendered request — system message included —
    because two different system messages must not share a key.
    """
    payload = _KEY_SEPARATOR.join(
        (schema_version, model, prompt, canonical_params(params))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CacheEntry(BaseModel):
    """One stored response, as returned by :meth:`ResponseCache.get_entry`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=64, max_length=64)
    schema_version: str
    model: str
    response: str
    created_at: float
    hit_count: int = Field(ge=0)


class CacheStats(BaseModel):
    """In-process counters since the cache object was constructed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hits: int = Field(default=0, ge=0)
    misses: int = Field(default=0, ge=0)
    stores: int = Field(default=0, ge=0)
    invalidations: int = Field(default=0, ge=0)

    @property
    def lookups(self) -> int:
        """Total ``get`` calls."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Hits over lookups; ``0.0`` when nothing has been looked up."""
        return self.hits / self.lookups if self.lookups else 0.0


class ResponseCache:
    """Thread-safe SQLite cache of model responses.

    Args:
        path: file path, or :data:`MEMORY_PATH` for an in-process cache.
            Parent directories are created for a file path.
        schema_version: overridable for tests that assert version isolation.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        schema_version: str = CACHE_SCHEMA_VERSION,
    ) -> None:
        self.schema_version: str = schema_version
        self._lock = threading.RLock()
        self._closed = False
        self._hits = 0
        self._misses = 0
        self._stores = 0
        self._invalidations = 0

        if isinstance(path, str) and path == MEMORY_PATH:
            self.path: Path | str = MEMORY_PATH
            target = MEMORY_PATH
        else:
            resolved = Path(path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            self.path = resolved
            target = str(resolved)

        # check_same_thread=False is safe *because* every statement below runs
        # under self._lock; the connection is never touched unguarded.
        self._conn = sqlite3.connect(target, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            if target != MEMORY_PATH:
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ----------------------------------------------------------------- lookup
    def get(self, key: str) -> str | None:
        """Return the cached response for ``key``, or ``None``.

        A row written under a different ``schema_version`` is a **miss**, not a
        hit: the response was produced under different framing.
        """
        entry = self.get_entry(key)
        return entry.response if entry is not None else None

    def get_entry(self, key: str) -> CacheEntry | None:
        """Like :meth:`get` but returns the full row, incrementing ``hit_count``."""
        with self._guard() as conn:
            row = conn.execute(
                "SELECT key, schema_version, model, response, created_at, "
                "hit_count FROM provider_cache "
                "WHERE key = ? AND schema_version = ?",
                (key, self.schema_version),
            ).fetchone()
            if row is None:
                self._misses += 1
                _CACHE_EVENTS.labels(event="miss").inc()
                return None
            conn.execute(
                "UPDATE provider_cache SET hit_count = hit_count + 1 "
                "WHERE key = ?",
                (key,),
            )
            conn.commit()
            self._hits += 1
            _CACHE_EVENTS.labels(event="hit").inc()
            return CacheEntry(
                key=row["key"],
                schema_version=row["schema_version"],
                model=row["model"],
                response=row["response"],
                created_at=row["created_at"],
                hit_count=int(row["hit_count"]) + 1,
            )

    # ------------------------------------------------------------------ write
    def put(self, key: str, response: str, *, model: str) -> None:
        """Store ``response`` under ``key``, replacing any existing row.

        The stored ``schema_version`` is this cache's, so a re-store after a
        version bump supersedes the stale row rather than accumulating both.
        """
        with self._guard() as conn:
            conn.execute(
                "INSERT INTO provider_cache "
                "(key, schema_version, model, response, created_at, hit_count) "
                "VALUES (?, ?, ?, ?, ?, 0) "
                "ON CONFLICT(key) DO UPDATE SET "
                "schema_version=excluded.schema_version, "
                "model=excluded.model, response=excluded.response, "
                "created_at=excluded.created_at, hit_count=0",
                (key, self.schema_version, model, response, time.time()),
            )
            conn.commit()
            self._stores += 1
            _CACHE_EVENTS.labels(event="store").inc()

    # ------------------------------------------------------------- invalidate
    def invalidate(self, key: str) -> bool:
        """Drop one entry (any schema version).  Returns whether a row went."""
        with self._guard() as conn:
            cursor = conn.execute(
                "DELETE FROM provider_cache WHERE key = ?", (key,)
            )
            conn.commit()
            removed = cursor.rowcount > 0
            if removed:
                self._invalidations += 1
                _CACHE_EVENTS.labels(event="invalidate").inc()
            return removed

    def invalidate_model(self, model: str) -> int:
        """Drop every entry for one model id.  Returns the row count."""
        with self._guard() as conn:
            cursor = conn.execute(
                "DELETE FROM provider_cache WHERE model = ?", (model,)
            )
            conn.commit()
            removed = int(cursor.rowcount)
            if removed:
                self._invalidations += removed
                _CACHE_EVENTS.labels(event="invalidate").inc(removed)
            return removed

    def invalidate_all(self) -> int:
        """Empty the cache.  Returns the row count removed."""
        with self._guard() as conn:
            cursor = conn.execute("DELETE FROM provider_cache")
            conn.commit()
            removed = int(cursor.rowcount)
            if removed:
                self._invalidations += removed
                _CACHE_EVENTS.labels(event="invalidate").inc(removed)
            return removed

    def purge_other_versions(self) -> int:
        """Delete rows written under a different ``schema_version``.

        Housekeeping only — those rows already miss on lookup.  Returns the
        row count removed.
        """
        with self._guard() as conn:
            cursor = conn.execute(
                "DELETE FROM provider_cache WHERE schema_version != ?",
                (self.schema_version,),
            )
            conn.commit()
            return int(cursor.rowcount)

    # ------------------------------------------------------------ inspection
    def __len__(self) -> int:
        """Rows visible at this cache's ``schema_version``."""
        with self._guard() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM provider_cache "
                "WHERE schema_version = ?",
                (self.schema_version,),
            ).fetchone()
            return int(row["n"])

    def total_rows(self) -> int:
        """Rows at **every** schema version — the on-disk footprint."""
        with self._guard() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM provider_cache"
            ).fetchone()
            return int(row["n"])

    @property
    def stats(self) -> CacheStats:
        """Counters for this cache object's lifetime."""
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                stores=self._stores,
                invalidations=self._invalidations,
            )

    # ------------------------------------------------------------- lifecycle
    def close(self) -> None:
        """Close the connection.  Idempotent."""
        with self._lock:
            if not self._closed:
                self._conn.close()
                self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @contextmanager
    def _guard(self) -> Iterator[sqlite3.Connection]:
        """Hold the lock and refuse use after close."""
        with self._lock:
            if self._closed:
                raise RuntimeError("ResponseCache is closed")
            yield self._conn

    def __repr__(self) -> str:
        return (
            f"ResponseCache(path={self.path!r}, "
            f"schema_version={self.schema_version!r})"
        )
