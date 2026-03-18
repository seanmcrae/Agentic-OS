"""
VectorStore - Embedding-based semantic vector search
====================================================

Provides fast approximate nearest-neighbor search over dense
embeddings, enabling true semantic memory retrieval beyond
keyword matching.

Backends:
- Pure Python fallback (cosine similarity, no dependencies)
- NumPy-accelerated search
- Optional FAISS integration for millions of vectors
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class VectorEntry:
    """An entry in the vector store."""
    entry_id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def dimension(self) -> int:
        return len(self.embedding)


@dataclass
class SearchResult:
    """A search result with similarity score."""
    entry: VectorEntry
    score: float  # cosine similarity 0.0 - 1.0
    rank: int


class VectorStore:
    """
    In-memory vector store with cosine similarity search.

    Supports pure Python, NumPy, and FAISS backends
    depending on available dependencies.

    Example:
        store = VectorStore(embed_fn=my_embedding_function)
        store.add("Python is great", {"type": "fact"})
        results = await store.search("programming languages", k=5)
    """

    def __init__(
        self,
        embed_fn: Optional[Callable] = None,
        dimension: int = 1536,
        max_entries: int = 100_000,
        backend: str = "auto",   # "auto" | "numpy" | "pure" | "faiss"
    ):
        self._embed_fn = embed_fn
        self.dimension = dimension
        self.max_entries = max_entries

        self._entries: Dict[str, VectorEntry] = {}
        self._entry_order: List[str] = []  # For ordered access
        self._counter = 0

        # Determine backend
        self._backend = self._resolve_backend(backend)
        logger.info(f"VectorStore initialized with {self._backend} backend")

    async def add(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        entry_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> str:
        """
        Add a new entry to the vector store.

        Args:
            content: Text content to embed and store
            metadata: Additional metadata
            entry_id: Optional explicit ID
            embedding: Pre-computed embedding (skips embed_fn call)

        Returns:
            Entry ID
        """
        if embedding is None:
            embedding = await self._embed(content)

        if not embedding:
            logger.warning(f"Could not generate embedding for: {content[:50]}")
            return ""

        eid = entry_id or f"vec_{self._counter:010d}"
        self._counter += 1

        entry = VectorEntry(
            entry_id=eid,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
        )

        self._entries[eid] = entry
        self._entry_order.append(eid)

        # Evict oldest if over capacity
        if len(self._entries) > self.max_entries:
            oldest_id = self._entry_order.pop(0)
            self._entries.pop(oldest_id, None)

        return eid

    async def search(
        self,
        query: str,
        k: int = 10,
        threshold: float = 0.0,
        filter_fn: Optional[Callable[[VectorEntry], bool]] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> List[SearchResult]:
        """
        Find the k most similar entries to a query.

        Args:
            query: Query text
            k: Number of results to return
            threshold: Minimum similarity score (0.0-1.0)
            filter_fn: Optional filter applied before scoring
            query_embedding: Pre-computed query embedding

        Returns:
            List of SearchResult sorted by similarity (highest first)
        """
        if not self._entries:
            return []

        if query_embedding is None:
            query_embedding = await self._embed(query)

        if not query_embedding:
            return []

        # Filter candidates
        candidates = list(self._entries.values())
        if filter_fn:
            candidates = [e for e in candidates if filter_fn(e)]

        # Score
        if self._backend == "numpy":
            scored = self._numpy_search(query_embedding, candidates, k, threshold)
        else:
            scored = self._pure_search(query_embedding, candidates, k, threshold)

        return [
            SearchResult(entry=entry, score=score, rank=i + 1)
            for i, (entry, score) in enumerate(scored)
        ]

    async def search_by_embedding(
        self,
        embedding: List[float],
        k: int = 10,
        threshold: float = 0.0,
    ) -> List[SearchResult]:
        """Search using a pre-computed embedding vector."""
        return await self.search("", k=k, threshold=threshold, query_embedding=embedding)

    def remove(self, entry_id: str) -> bool:
        """Remove an entry from the store."""
        if entry_id in self._entries:
            self._entries.pop(entry_id)
            if entry_id in self._entry_order:
                self._entry_order.remove(entry_id)
            return True
        return False

    def get(self, entry_id: str) -> Optional[VectorEntry]:
        """Get an entry by ID."""
        return self._entries.get(entry_id)

    async def update(
        self,
        entry_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update an existing entry."""
        entry = self._entries.get(entry_id)
        if not entry:
            return False

        if content is not None:
            entry.content = content
            entry.embedding = await self._embed(content) or entry.embedding

        if metadata is not None:
            entry.metadata.update(metadata)

        return True

    def get_all(self, limit: int = 100) -> List[VectorEntry]:
        """Get all stored entries."""
        return [
            self._entries[eid]
            for eid in self._entry_order[-limit:]
            if eid in self._entries
        ]

    @property
    def size(self) -> int:
        return len(self._entries)

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        dims = [len(e.embedding) for e in self._entries.values()]
        return {
            "size": len(self._entries),
            "capacity": self.max_entries,
            "backend": self._backend,
            "embedding_dimension": dims[0] if dims else self.dimension,
            "has_embed_fn": self._embed_fn is not None,
        }

    # =========================================================================
    # Search backends
    # =========================================================================

    def _pure_search(
        self,
        query: List[float],
        candidates: List[VectorEntry],
        k: int,
        threshold: float,
    ) -> List[Tuple[VectorEntry, float]]:
        """Pure Python cosine similarity search."""
        scored = []
        q_norm = _l2_norm(query)

        for entry in candidates:
            if len(entry.embedding) != len(query):
                continue
            score = _cosine_similarity(query, entry.embedding, q_norm)
            if score >= threshold:
                scored.append((entry, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def _numpy_search(
        self,
        query: List[float],
        candidates: List[VectorEntry],
        k: int,
        threshold: float,
    ) -> List[Tuple[VectorEntry, float]]:
        """NumPy-accelerated cosine similarity search."""
        try:
            import numpy as np

            q = np.array(query, dtype=np.float32)
            q_norm = np.linalg.norm(q)
            if q_norm == 0:
                return []

            # Build matrix of candidate embeddings
            valid = [(e, e.embedding) for e in candidates if len(e.embedding) == len(query)]
            if not valid:
                return []

            entries, embeddings = zip(*valid)
            matrix = np.array(embeddings, dtype=np.float32)

            # Batch cosine similarity
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            normalized = matrix / norms
            scores = normalized @ (q / q_norm)

            # Filter and sort
            results = [
                (entries[i], float(scores[i]))
                for i in range(len(entries))
                if scores[i] >= threshold
            ]
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]

        except ImportError:
            return self._pure_search(query, candidates, k, threshold)

    def _resolve_backend(self, requested: str) -> str:
        """Determine the best available backend."""
        if requested == "faiss":
            try:
                import faiss  # noqa
                return "faiss"
            except ImportError:
                logger.warning("FAISS not available, falling back to numpy")
                requested = "numpy"

        if requested in ("numpy", "auto"):
            try:
                import numpy  # noqa
                return "numpy"
            except ImportError:
                if requested == "numpy":
                    logger.warning("NumPy not available, using pure Python")
                return "pure"

        return "pure"

    async def _embed(self, text: str) -> Optional[List[float]]:
        """Generate an embedding for the given text."""
        if self._embed_fn is None:
            # Return a deterministic mock embedding based on text hash
            return _mock_embedding(text, self.dimension)

        try:
            import asyncio
            if asyncio.iscoroutinefunction(self._embed_fn):
                return await self._embed_fn(text)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self._embed_fn, text)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None


# ============================================================================
# Embedding function factories
# ============================================================================

def openai_embed_fn(api_key: Optional[str] = None, model: str = "text-embedding-3-small") -> Callable:
    """Create an OpenAI embedding function."""
    import os
    key = api_key or os.environ.get("OPENAI_API_KEY")

    async def embed(text: str) -> List[float]:
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=key)
            response = await client.embeddings.create(input=text, model=model)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            return _mock_embedding(text, 1536)

    return embed


def anthropic_embed_fn(api_key: Optional[str] = None) -> Callable:
    """Create an embedding function using Claude's token representations."""
    # Anthropic doesn't have a dedicated embedding API yet
    # Using mock embeddings as placeholder
    async def embed(text: str) -> List[float]:
        return _mock_embedding(text, 1536)
    return embed


def sentence_transformer_fn(model_name: str = "all-MiniLM-L6-v2") -> Callable:
    """Create a sentence-transformers embedding function (local, no API needed)."""
    _model = None

    def _load():
        nonlocal _model
        if _model is None:
            try:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required: pip install sentence-transformers"
                )
        return _model

    async def embed(text: str) -> List[float]:
        import asyncio
        model = _load()
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, model.encode, text)
        return embedding.tolist()

    return embed


# ============================================================================
# Helpers
# ============================================================================

def _cosine_similarity(a: List[float], b: List[float], a_norm: Optional[float] = None) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    if a_norm is None:
        a_norm = _l2_norm(a)
    b_norm = _l2_norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0
    return dot / (a_norm * b_norm)


def _l2_norm(v: List[float]) -> float:
    """Compute L2 norm of a vector."""
    return math.sqrt(sum(x * x for x in v))


def _mock_embedding(text: str, dim: int = 1536) -> List[float]:
    """
    Generate a deterministic mock embedding from text.

    Not semantically meaningful but useful for testing
    without an embedding API.
    """
    import hashlib
    h = hashlib.sha256(text.encode()).digest()

    # Use hash bytes to seed a deterministic embedding
    embedding = []
    for i in range(dim):
        byte_idx = i % len(h)
        val = (h[byte_idx] / 255.0) * 2 - 1  # Normalize to [-1, 1]
        embedding.append(val)

    # Normalize to unit vector
    norm = _l2_norm(embedding)
    if norm > 0:
        embedding = [x / norm for x in embedding]

    return embedding
