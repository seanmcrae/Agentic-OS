"""
SemanticCache - LLM Response Cache with Semantic Similarity
============================================================

Provides a two-tier cache for LLM responses:
  1. Exact match via SHA-256 hash (O(1) fast path)
  2. Fuzzy match via cosine similarity over stored embeddings (semantic path)

Features:
- TTL-based expiration checked on access
- LRU-style eviction (lowest hit-rate entries removed when capacity is exceeded)
- Thread-safe via asyncio.Lock
- Optional custom embedding function; falls back to deterministic mock
- Detailed statistics via CacheStats
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """A single cached LLM response."""

    query: str
    """The original query text."""

    response: str
    """The cached LLM response."""

    embedding: List[float]
    """Dense embedding of the query used for similarity search."""

    created_at: float = field(default_factory=time.time)
    """Unix timestamp of when the entry was created."""

    hits: int = 0
    """Number of times this entry has been returned as a cache hit."""

    ttl: float = 3600.0
    """Time-to-live in seconds. Non-positive value means no expiration."""

    @property
    def is_expired(self) -> bool:
        """Return True if this entry has exceeded its TTL."""
        if self.ttl <= 0:
            return False
        return (time.time() - self.created_at) > self.ttl

    @property
    def hit_rate(self) -> float:
        """Hits per second of existence (proxy for access frequency)."""
        age = max(time.time() - self.created_at, 1e-9)
        return self.hits / age


@dataclass
class CacheStats:
    """Snapshot of SemanticCache performance statistics."""

    total_entries: int
    """Current number of entries stored in the cache."""

    total_hits: int
    """Cumulative exact + semantic cache hits since last clear."""

    total_misses: int
    """Cumulative cache misses since last clear."""

    hit_rate: float
    """Fraction of accesses that resulted in a cache hit (0.0 – 1.0)."""

    avg_similarity: float
    """Average cosine similarity score of all semantic hits (0.0 – 1.0)."""

    memory_bytes: int
    """Approximate in-memory size of all stored entries in bytes."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Return cosine similarity in [0, 1] between two equal-length vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = _l2_norm(a)
    norm_b = _l2_norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _mock_embedding(text: str, dim: int = 256) -> List[float]:
    """
    Generate a deterministic, hash-seeded unit-vector embedding.

    Not semantically meaningful, but useful for testing without a real
    embedding API.  Re-uses the implementation from dzogchen.memory.vector
    when that module is available, falling back to this local version.
    """
    try:
        from dzogchen.memory.vector import _mock_embedding as _vec_mock
        return _vec_mock(text, dim)
    except ImportError:
        pass

    digest = hashlib.sha256(text.encode()).digest()
    embedding = []
    for i in range(dim):
        byte_val = digest[i % len(digest)]
        val = (byte_val / 255.0) * 2.0 - 1.0  # map to [-1, 1]
        embedding.append(val)

    norm = _l2_norm(embedding)
    if norm > 0:
        embedding = [x / norm for x in embedding]
    return embedding


def _estimate_bytes(entry: CacheEntry) -> int:
    """Rough byte estimate for a CacheEntry."""
    text_bytes = sys.getsizeof(entry.query) + sys.getsizeof(entry.response)
    embedding_bytes = len(entry.embedding) * 8  # float64 = 8 bytes each
    return text_bytes + embedding_bytes + 128  # 128 bytes for dataclass overhead


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------

class SemanticCache:
    """
    Two-tier LLM response cache with exact-hash and semantic-similarity lookup.

    Usage::

        cache = SemanticCache(embed_fn=my_async_embed, similarity_threshold=0.92)

        response = await cache.get("What is the capital of France?")
        if response is None:
            response = await call_llm(...)
            await cache.set("What is the capital of France?", response)

    Args:
        embed_fn: Optional async or sync callable ``(text: str) -> List[float]``.
            When omitted a deterministic mock embedding is used.
        similarity_threshold: Minimum cosine similarity (0–1) required to
            return a semantic hit. Defaults to 0.92.
        max_size: Maximum number of entries before LRU eviction. Defaults to 1000.
        ttl_seconds: Entry lifetime in seconds. Defaults to 3600 (1 hour).
            Pass 0 or a negative value to disable TTL.
        exact_match_first: When True (default), always probe the exact SHA-256
            index before the embedding search.
    """

    def __init__(
        self,
        embed_fn: Optional[Callable] = None,
        similarity_threshold: float = 0.92,
        max_size: int = 1000,
        ttl_seconds: float = 3600.0,
        exact_match_first: bool = True,
    ) -> None:
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be in [0, 1]")
        if max_size < 1:
            raise ValueError("max_size must be >= 1")

        self._embed_fn = embed_fn
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.exact_match_first = exact_match_first

        # Primary storage: arbitrary key -> CacheEntry
        self._store: Dict[str, CacheEntry] = {}

        # Fast path: SHA-256(query) -> store key
        self._hash_index: Dict[str, str] = {}

        # Statistics accumulators
        self._total_hits: int = 0
        self._total_misses: int = 0
        self._similarity_scores: List[float] = []

        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(self, query: str) -> Optional[str]:
        """
        Retrieve a cached response for *query*.

        Checks the exact SHA-256 hash index first (when ``exact_match_first``
        is True), then scans embeddings for a semantically similar match above
        ``similarity_threshold``.

        Returns the cached response string, or ``None`` on a miss.
        """
        async with self._lock:
            # 1. Exact-match fast path
            if self.exact_match_first:
                result = self._exact_get(query)
                if result is not None:
                    self._total_hits += 1
                    logger.debug("Cache exact-hit for query: %.60s…", query)
                    return result

            # 2. Semantic similarity path
            query_embedding = await self._embed(query)
            result, score = self._semantic_get(query_embedding)
            if result is not None:
                self._total_hits += 1
                self._similarity_scores.append(score)
                logger.debug(
                    "Cache semantic-hit (similarity=%.4f) for query: %.60s…",
                    score,
                    query,
                )
                return result

            self._total_misses += 1
            return None

    async def set(self, query: str, response: str) -> None:
        """
        Store *response* as the cached answer for *query*.

        An embedding is generated for *query* and stored alongside the response.
        Eviction is triggered automatically when ``max_size`` is exceeded.
        """
        async with self._lock:
            key = _sha256(query)

            # If key already exists, update the entry in place
            if key in self._store:
                entry = self._store[key]
                entry.response = response
                entry.created_at = time.time()
                entry.hits = 0
                logger.debug("Cache entry updated for key: %s", key[:16])
                return

            embedding = await self._embed(query)
            entry = CacheEntry(
                query=query,
                response=response,
                embedding=embedding,
                ttl=self.ttl_seconds,
            )

            self._store[key] = entry
            self._hash_index[_sha256(query)] = key

            # Evict if over capacity
            if len(self._store) > self.max_size:
                self._evict_one()

            logger.debug("Cache set for key: %s (store size=%d)", key[:16], len(self._store))

    async def invalidate(self, query: str) -> bool:
        """
        Remove the cached entry for an exact *query*.

        Returns ``True`` if an entry was found and removed, ``False`` otherwise.
        """
        async with self._lock:
            key = _sha256(query)
            if key in self._store:
                del self._store[key]
                self._hash_index.pop(key, None)
                logger.debug("Cache invalidated key: %s", key[:16])
                return True
            return False

    async def clear(self) -> None:
        """Remove all entries and reset all statistics."""
        async with self._lock:
            self._store.clear()
            self._hash_index.clear()
            self._total_hits = 0
            self._total_misses = 0
            self._similarity_scores.clear()
            logger.debug("Cache cleared.")

    def get_stats(self) -> CacheStats:
        """Return a point-in-time snapshot of cache statistics."""
        total_accesses = self._total_hits + self._total_misses
        hit_rate = self._total_hits / total_accesses if total_accesses > 0 else 0.0
        avg_similarity = (
            sum(self._similarity_scores) / len(self._similarity_scores)
            if self._similarity_scores
            else 0.0
        )
        memory_bytes = sum(_estimate_bytes(e) for e in self._store.values())
        return CacheStats(
            total_entries=len(self._store),
            total_hits=self._total_hits,
            total_misses=self._total_misses,
            hit_rate=hit_rate,
            avg_similarity=avg_similarity,
            memory_bytes=memory_bytes,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _exact_get(self, query: str) -> Optional[str]:
        """Check the SHA-256 hash index. Returns response or None."""
        key = _sha256(query)
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.is_expired:
            del self._store[key]
            self._hash_index.pop(key, None)
            logger.debug("Cache expired entry evicted: %s", key[:16])
            return None
        entry.hits += 1
        return entry.response

    def _semantic_get(self, query_embedding: List[float]) -> Tuple[Optional[str], float]:
        """
        Linear scan over all stored embeddings for the best cosine match.

        Returns ``(response, similarity)`` if a match above threshold is
        found, otherwise ``(None, 0.0)``.
        """
        best_entry: Optional[CacheEntry] = None
        best_score: float = 0.0
        expired_keys: List[str] = []

        for key, entry in self._store.items():
            if entry.is_expired:
                expired_keys.append(key)
                continue
            score = _cosine_similarity(query_embedding, entry.embedding)
            if score > best_score:
                best_score = score
                best_entry = entry

        # Opportunistic cleanup of expired entries found during scan
        for key in expired_keys:
            self._store.pop(key, None)
            self._hash_index.pop(key, None)

        if best_entry is not None and best_score >= self.similarity_threshold:
            best_entry.hits += 1
            return best_entry.response, best_score

        return None, 0.0

    def _evict_one(self) -> None:
        """
        Remove the single entry with the lowest hit-rate to make room.

        This approximates LRU: entries that have never been accessed and
        are old are evicted first.
        """
        if not self._store:
            return
        victim_key = min(self._store, key=lambda k: self._store[k].hit_rate)
        victim = self._store.pop(victim_key, None)
        if victim:
            self._hash_index.pop(_sha256(victim.query), None)
            logger.debug("Cache evicted low-hit-rate entry: %s", victim_key[:16])

    async def _embed(self, text: str) -> List[float]:
        """Generate an embedding for *text* using the configured function."""
        if self._embed_fn is None:
            return _mock_embedding(text)

        try:
            if asyncio.iscoroutinefunction(self._embed_fn):
                return await self._embed_fn(text)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._embed_fn, text)
        except Exception as exc:
            logger.warning("Embedding function raised an exception: %s; using mock.", exc)
            return _mock_embedding(text)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"SemanticCache(entries={stats.total_entries}, "
            f"hit_rate={stats.hit_rate:.2%}, "
            f"threshold={self.similarity_threshold})"
        )


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    """Return the hex-encoded SHA-256 digest of *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
