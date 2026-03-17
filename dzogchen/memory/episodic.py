"""
EpisodicMemory - Autobiographical event-based long-term memory
==============================================================

Stores sequences of events with temporal context, enabling
recall of past experiences, conversation history, and
action-outcome pairs for learning.
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """A single episode (event) in episodic memory."""
    episode_id: str
    event_type: str
    content: Any
    context: Dict[str, Any] = field(default_factory=dict)
    outcome: Optional[Any] = None
    emotional_valence: float = 0.0  # -1.0 (negative) to 1.0 (positive)
    importance: float = 0.5
    timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "event_type": self.event_type,
            "content": self.content,
            "context": self.context,
            "outcome": self.outcome,
            "emotional_valence": self.emotional_valence,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        return cls(**data)


class EpisodicMemory:
    """
    Long-term episodic memory with temporal indexing.

    Stores experiences as episodes with rich context,
    supporting temporal queries, pattern recognition across
    episodes, and importance-weighted retrieval.
    """

    def __init__(
        self,
        max_episodes: int = 10_000,
        importance_threshold: float = 0.1,
        persistence_path: Optional[str] = None,
    ):
        self.max_episodes = max_episodes
        self.importance_threshold = importance_threshold
        self.persistence_path = persistence_path

        self._episodes: List[Episode] = []
        self._episode_counter = 0
        self._session_index: Dict[str, List[str]] = {}  # session_id -> [episode_ids]
        self._tag_index: Dict[str, List[str]] = {}  # tag -> [episode_ids]
        self._type_index: Dict[str, List[str]] = {}  # event_type -> [episode_ids]

        # Load from persistence if available
        if persistence_path:
            self._load_from_disk()

    def store(
        self,
        event_type: str,
        content: Any,
        context: Optional[Dict[str, Any]] = None,
        outcome: Optional[Any] = None,
        importance: float = 0.5,
        emotional_valence: float = 0.0,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Episode:
        """
        Store a new episode in memory.

        Args:
            event_type: Type of event (e.g., "conversation", "tool_use", "error")
            content: The event content
            context: Surrounding context
            outcome: What resulted from this event
            importance: How important this episode is (0.0-1.0)
            emotional_valence: Emotional quality (-1.0 to 1.0)
            session_id: Session this episode belongs to
            tags: Descriptive tags

        Returns:
            The stored Episode
        """
        if importance < self.importance_threshold:
            # Still create episode but mark as low importance
            pass

        episode_id = f"ep_{self._episode_counter:08d}"
        self._episode_counter += 1

        episode = Episode(
            episode_id=episode_id,
            event_type=event_type,
            content=content,
            context=context or {},
            outcome=outcome,
            importance=importance,
            emotional_valence=emotional_valence,
            session_id=session_id,
            tags=tags or [],
        )

        self._episodes.append(episode)
        self._update_indices(episode)

        # Consolidate if over capacity
        if len(self._episodes) > self.max_episodes:
            self._consolidate()

        if self.persistence_path:
            self._persist_episode(episode)

        logger.debug(f"Stored episode {episode_id}: {event_type}")
        return episode

    def recall(
        self,
        query: Optional[str] = None,
        event_type: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        limit: int = 20,
        min_importance: float = 0.0,
    ) -> List[Episode]:
        """
        Recall episodes matching the given criteria.

        Args:
            query: Text to search in episode content
            event_type: Filter by event type
            session_id: Filter by session
            tags: Filter by tags (any match)
            time_range: (start_timestamp, end_timestamp)
            limit: Maximum episodes to return
            min_importance: Minimum importance threshold

        Returns:
            List of matching episodes, most recent first
        """
        candidates = self._episodes.copy()

        # Filter by event type
        if event_type and event_type in self._type_index:
            candidate_ids = set(self._type_index[event_type])
            candidates = [e for e in candidates if e.episode_id in candidate_ids]

        # Filter by session
        if session_id and session_id in self._session_index:
            candidate_ids = set(self._session_index[session_id])
            candidates = [e for e in candidates if e.episode_id in candidate_ids]

        # Filter by tags
        if tags:
            tag_ids: set = set()
            for tag in tags:
                tag_ids.update(self._tag_index.get(tag, []))
            if tag_ids:
                candidates = [e for e in candidates if e.episode_id in tag_ids]

        # Filter by time range
        if time_range:
            start, end = time_range
            candidates = [e for e in candidates if start <= e.timestamp <= end]

        # Filter by importance
        if min_importance > 0:
            candidates = [e for e in candidates if e.importance >= min_importance]

        # Filter by query (simple text search)
        if query:
            query_lower = query.lower()
            candidates = [
                e for e in candidates
                if query_lower in str(e.content).lower()
                or any(query_lower in str(v).lower() for v in e.context.values())
            ]

        # Sort by recency and importance
        candidates.sort(
            key=lambda e: e.timestamp * 0.7 + e.importance * 0.3,
            reverse=True,
        )

        return candidates[:limit]

    def get_session_history(self, session_id: str) -> List[Episode]:
        """Get all episodes from a session in chronological order."""
        episode_ids = set(self._session_index.get(session_id, []))
        episodes = [e for e in self._episodes if e.episode_id in episode_ids]
        episodes.sort(key=lambda e: e.timestamp)
        return episodes

    def get_recent(self, n: int = 10, event_type: Optional[str] = None) -> List[Episode]:
        """Get the n most recent episodes."""
        candidates = self._episodes
        if event_type:
            candidates = [e for e in candidates if e.event_type == event_type]
        return sorted(candidates, key=lambda e: e.timestamp, reverse=True)[:n]

    def get_important(self, n: int = 10, min_importance: float = 0.7) -> List[Episode]:
        """Get the most important episodes."""
        important = [e for e in self._episodes if e.importance >= min_importance]
        return sorted(important, key=lambda e: e.importance, reverse=True)[:n]

    def update_outcome(self, episode_id: str, outcome: Any, importance_adjustment: float = 0.0) -> bool:
        """Update the outcome of a stored episode."""
        for episode in self._episodes:
            if episode.episode_id == episode_id:
                episode.outcome = outcome
                episode.importance = max(0.0, min(1.0, episode.importance + importance_adjustment))
                return True
        return False

    def get_patterns(self, event_type: str, n: int = 5) -> List[Dict[str, Any]]:
        """
        Identify patterns in episodes of a given type.
        Returns common context patterns.
        """
        episodes = [e for e in self._episodes if e.event_type == event_type]
        if not episodes:
            return []

        # Simple frequency analysis of context keys
        context_keys: Dict[str, int] = {}
        for ep in episodes:
            for key in ep.context:
                context_keys[key] = context_keys.get(key, 0) + 1

        # Return most common context patterns
        sorted_keys = sorted(context_keys.items(), key=lambda x: x[1], reverse=True)
        return [
            {"key": k, "frequency": v, "rate": v / len(episodes)}
            for k, v in sorted_keys[:n]
        ]

    @property
    def total_episodes(self) -> int:
        return len(self._episodes)

    def get_stats(self) -> Dict[str, Any]:
        """Get episodic memory statistics."""
        if not self._episodes:
            return {"total": 0}

        type_counts = {}
        for ep in self._episodes:
            type_counts[ep.event_type] = type_counts.get(ep.event_type, 0) + 1

        return {
            "total": len(self._episodes),
            "capacity": self.max_episodes,
            "utilization": len(self._episodes) / self.max_episodes,
            "event_types": type_counts,
            "sessions": len(self._session_index),
            "avg_importance": sum(e.importance for e in self._episodes) / len(self._episodes),
            "oldest": self._episodes[0].timestamp if self._episodes else None,
            "newest": self._episodes[-1].timestamp if self._episodes else None,
        }

    def _update_indices(self, episode: Episode) -> None:
        """Update all indices with a new episode."""
        # Session index
        if episode.session_id:
            if episode.session_id not in self._session_index:
                self._session_index[episode.session_id] = []
            self._session_index[episode.session_id].append(episode.episode_id)

        # Tag index
        for tag in episode.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = []
            self._tag_index[tag].append(episode.episode_id)

        # Type index
        if episode.event_type not in self._type_index:
            self._type_index[episode.event_type] = []
        self._type_index[episode.event_type].append(episode.episode_id)

    def _consolidate(self) -> None:
        """Remove least important episodes to stay within capacity."""
        # Keep the most important and most recent episodes
        sorted_episodes = sorted(
            self._episodes,
            key=lambda e: e.importance * 0.6 + (e.timestamp / time.time()) * 0.4,
            reverse=True,
        )
        # Keep 80% capacity worth of episodes
        keep_count = int(self.max_episodes * 0.8)
        removed = self._episodes[keep_count:]
        self._episodes = sorted_episodes[:keep_count]

        # Rebuild indices
        self._session_index.clear()
        self._tag_index.clear()
        self._type_index.clear()
        for ep in self._episodes:
            self._update_indices(ep)

        logger.info(f"EpisodicMemory consolidated: removed {len(removed)} episodes")

    def _persist_episode(self, episode: Episode) -> None:
        """Persist a single episode to disk."""
        try:
            import os
            os.makedirs(self.persistence_path, exist_ok=True)  # type: ignore
            path = f"{self.persistence_path}/{episode.episode_id}.json"
            with open(path, "w") as f:
                json.dump(episode.to_dict(), f)
        except Exception as e:
            logger.warning(f"Failed to persist episode {episode.episode_id}: {e}")

    def _load_from_disk(self) -> None:
        """Load episodes from disk."""
        try:
            import os
            if not os.path.exists(self.persistence_path):  # type: ignore
                return
            for filename in os.listdir(self.persistence_path):  # type: ignore
                if filename.endswith(".json"):
                    path = f"{self.persistence_path}/{filename}"
                    with open(path) as f:
                        data = json.load(f)
                    episode = Episode.from_dict(data)
                    self._episodes.append(episode)
                    self._update_indices(episode)
                    self._episode_counter = max(
                        self._episode_counter,
                        int(episode.episode_id.split("_")[-1]) + 1,
                    )
            logger.info(f"Loaded {len(self._episodes)} episodes from disk")
        except Exception as e:
            logger.warning(f"Failed to load episodes from disk: {e}")
