"""
WorkingMemory - Short-term, high-priority in-context memory
============================================================

Implements cognitive working memory: limited capacity, fast access,
automatic decay, and priority-based eviction. Mirrors how humans
maintain active context while reasoning.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class WorkingMemoryItem:
    """An item held in working memory."""
    key: str
    value: Any
    priority: float = 0.5
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None  # Time-to-live in seconds, None = no expiry
    tags: List[str] = field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    @property
    def activation(self) -> float:
        """ACT-R inspired activation: recency + frequency + priority."""
        recency = 1.0 / max(1.0, time.time() - self.last_accessed)
        frequency = min(1.0, self.access_count / 10.0)
        return (recency * 0.4) + (frequency * 0.3) + (self.priority * 0.3)


class WorkingMemory:
    """
    Capacity-limited working memory with automatic decay.

    Implements a cognitive working memory system with:
    - Strict capacity limits (default: 7 ± 2 items)
    - Activation-based retention (ACT-R inspired)
    - Automatic expiry for time-sensitive items
    - Priority-based eviction when capacity exceeded
    """

    def __init__(self, capacity: int = 7, default_ttl: Optional[float] = None):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._store: OrderedDict[str, WorkingMemoryItem] = OrderedDict()

    def put(
        self,
        key: str,
        value: Any,
        priority: float = 0.5,
        ttl: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Add or update an item in working memory.

        If capacity would be exceeded, evicts the lowest-activation item.
        """
        # Update existing item
        if key in self._store:
            item = self._store[key]
            item.value = value
            item.priority = priority
            item.last_accessed = time.time()
            item.access_count += 1
            return

        # Evict expired items first
        self._evict_expired()

        # Evict lowest-activation item if at capacity
        if len(self._store) >= self.capacity:
            self._evict_lowest_activation()

        self._store[key] = WorkingMemoryItem(
            key=key,
            value=value,
            priority=priority,
            ttl=ttl or self.default_ttl,
            tags=tags or [],
        )
        logger.debug(f"WorkingMemory: stored '{key}' (size={len(self._store)}/{self.capacity})")

    def get(self, key: str) -> Optional[Any]:
        """Retrieve an item, updating its activation."""
        self._evict_expired()

        if key not in self._store:
            return None

        item = self._store[key]
        item.last_accessed = time.time()
        item.access_count += 1
        return item.value

    def peek(self, key: str) -> Optional[Any]:
        """Retrieve an item without updating activation."""
        if key in self._store:
            item = self._store[key]
            if not item.is_expired:
                return item.value
        return None

    def remove(self, key: str) -> Optional[Any]:
        """Remove and return an item from working memory."""
        if key in self._store:
            return self._store.pop(key).value
        return None

    def contains(self, key: str) -> bool:
        """Check if a key exists in working memory."""
        self._evict_expired()
        return key in self._store

    def get_by_tags(self, tags: List[str]) -> Dict[str, Any]:
        """Retrieve all items matching any of the given tags."""
        self._evict_expired()
        result = {}
        for key, item in self._store.items():
            if any(tag in item.tags for tag in tags):
                result[key] = item.value
                item.last_accessed = time.time()
                item.access_count += 1
        return result

    def get_all(self) -> Dict[str, Any]:
        """Retrieve all current items as a dictionary."""
        self._evict_expired()
        return {k: v.value for k, v in self._store.items()}

    def clear(self) -> None:
        """Clear all working memory."""
        self._store.clear()

    def get_activation_map(self) -> Dict[str, float]:
        """Get activation levels for all items."""
        return {k: v.activation for k, v in self._store.items()}

    def get_most_active(self, n: int = 3) -> List[Any]:
        """Get the n most active items."""
        self._evict_expired()
        sorted_items = sorted(self._store.values(), key=lambda x: x.activation, reverse=True)
        for item in sorted_items[:n]:
            item.last_accessed = time.time()
            item.access_count += 1
        return [item.value for item in sorted_items[:n]]

    @property
    def size(self) -> int:
        """Current number of items in working memory."""
        self._evict_expired()
        return len(self._store)

    @property
    def is_full(self) -> bool:
        return self.size >= self.capacity

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about working memory usage."""
        self._evict_expired()
        activations = [item.activation for item in self._store.values()]
        return {
            "size": len(self._store),
            "capacity": self.capacity,
            "utilization": len(self._store) / self.capacity,
            "avg_activation": sum(activations) / len(activations) if activations else 0.0,
            "keys": list(self._store.keys()),
        }

    def _evict_expired(self) -> None:
        """Remove all expired items."""
        expired = [k for k, v in self._store.items() if v.is_expired]
        for key in expired:
            del self._store[key]
            logger.debug(f"WorkingMemory: evicted expired item '{key}'")

    def _evict_lowest_activation(self) -> None:
        """Remove the item with lowest activation."""
        if not self._store:
            return
        min_key = min(self._store, key=lambda k: self._store[k].activation)
        del self._store[min_key]
        logger.debug(f"WorkingMemory: evicted low-activation item '{min_key}'")
