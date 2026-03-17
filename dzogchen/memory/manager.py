"""
MemoryManager - Unified interface for all memory systems
========================================================

Orchestrates working, episodic, and semantic memory systems,
handles memory consolidation, cross-system queries, and
provides a unified API for storing and retrieving information.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
import logging

from dzogchen.memory.working import WorkingMemory
from dzogchen.memory.episodic import EpisodicMemory, Episode
from dzogchen.memory.semantic import SemanticMemory, KnowledgeEntry

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Unified memory management system.

    Provides a single interface for all three memory systems and handles:
    - Automatic routing of new information to appropriate memory stores
    - Consolidation from working -> episodic -> semantic
    - Cross-system queries that combine results from all stores
    - Memory maintenance (consolidation, pruning, indexing)
    """

    def __init__(
        self,
        working_capacity: int = 7,
        episodic_max: int = 10_000,
        semantic_max: int = 100_000,
        auto_consolidate: bool = True,
        consolidation_interval: float = 300.0,
        persistence_path: Optional[str] = None,
    ):
        self.working = WorkingMemory(capacity=working_capacity)
        self.episodic = EpisodicMemory(
            max_episodes=episodic_max,
            persistence_path=persistence_path,
        )
        self.semantic = SemanticMemory(max_entries=semantic_max)

        self.auto_consolidate = auto_consolidate
        self.consolidation_interval = consolidation_interval
        self._last_consolidation = time.time()
        self._session_id: Optional[str] = None

    def set_session(self, session_id: str) -> None:
        """Set the current session ID for episodic memory tagging."""
        self._session_id = session_id

    # =========================================================================
    # Storage operations
    # =========================================================================

    def remember(
        self,
        key: str,
        value: Any,
        memory_type: str = "working",
        priority: float = 0.5,
        ttl: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Store information in the appropriate memory system.

        Args:
            key: Identifier for the information
            value: The information to store
            memory_type: "working" | "episodic" | "semantic" | "all"
            priority: Priority/importance (0.0-1.0)
            ttl: Time-to-live for working memory
            tags: Tags for retrieval
        """
        if memory_type in ("working", "all"):
            self.working.put(key, value, priority=priority, ttl=ttl, tags=tags or [])

        if memory_type in ("episodic", "all"):
            self.episodic.store(
                event_type="remember",
                content={"key": key, "value": value},
                importance=priority,
                session_id=self._session_id,
                tags=tags or [],
            )

        if memory_type == "semantic":
            self.semantic.store(
                concept=key,
                definition=str(value),
                confidence=priority,
            )

    def store_interaction(
        self,
        user_input: str,
        agent_response: str,
        context: Optional[Dict[str, Any]] = None,
        quality: float = 0.7,
    ) -> Episode:
        """Store a user-agent interaction in episodic memory."""
        # Keep recent interaction in working memory
        self.working.put(
            f"last_interaction_{self._session_id or 'global'}",
            {"input": user_input, "response": agent_response},
            priority=0.8,
            ttl=3600,  # 1 hour
        )

        # Store in episodic memory
        episode = self.episodic.store(
            event_type="interaction",
            content={"input": user_input, "response": agent_response},
            context=context or {},
            importance=quality,
            session_id=self._session_id,
            tags=["conversation"],
        )

        # Maybe trigger consolidation
        if self.auto_consolidate:
            self._maybe_consolidate()

        return episode

    def learn_fact(
        self,
        concept: str,
        definition: str,
        domain: str = "general",
        confidence: float = 0.8,
        source: str = "interaction",
    ) -> KnowledgeEntry:
        """Learn a new fact and store it in semantic memory."""
        entry = self.semantic.store(
            concept=concept,
            definition=definition,
            domain=domain,
            confidence=confidence,
            source=source,
        )

        # Also put in working memory for immediate access
        self.working.put(f"fact:{concept}", definition, priority=0.6, ttl=300)

        return entry

    def store_tool_result(
        self,
        tool_name: str,
        params: Dict[str, Any],
        result: Any,
        success: bool,
        execution_time: float,
    ) -> Episode:
        """Store a tool execution result."""
        importance = 0.7 if success else 0.9  # Failures are more important to remember

        episode = self.episodic.store(
            event_type="tool_execution",
            content={"tool": tool_name, "params": params},
            outcome={"result": result, "success": success, "time": execution_time},
            importance=importance,
            emotional_valence=0.3 if success else -0.3,
            session_id=self._session_id,
            tags=["tool", tool_name],
        )

        return episode

    # =========================================================================
    # Retrieval operations
    # =========================================================================

    def recall(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Comprehensive recall across all memory systems.

        Returns results from working, episodic, and semantic memory.
        """
        results = {
            "working": {},
            "episodic": [],
            "semantic": [],
            "query": query,
        }

        # Working memory: check for direct key match or tag match
        working_items = self.working.get_all()
        relevant_working = {
            k: v for k, v in working_items.items()
            if query.lower() in k.lower() or query.lower() in str(v).lower()
        }
        results["working"] = relevant_working

        # Episodic memory
        episodes = self.episodic.recall(query=query, limit=limit)
        results["episodic"] = [
            {
                "episode_id": ep.episode_id,
                "type": ep.event_type,
                "content": ep.content,
                "timestamp": ep.timestamp,
                "importance": ep.importance,
            }
            for ep in episodes
        ]

        # Semantic memory
        knowledge = self.semantic.search(query=query, limit=limit)
        results["semantic"] = [
            {
                "concept": e.concept,
                "definition": e.definition,
                "domain": e.domain,
                "confidence": e.confidence,
            }
            for e in knowledge
        ]

        return results

    def get_context(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a comprehensive context snapshot for the current state.

        Used to provide agents with full situational awareness.
        """
        sid = session_id or self._session_id

        context = {
            "working_memory": self.working.get_all(),
            "recent_episodes": [],
            "key_facts": [],
            "session_history_count": 0,
        }

        # Recent episodes
        if sid:
            recent = self.episodic.get_session_history(sid)
            context["recent_episodes"] = [
                {"type": ep.event_type, "content": ep.content, "timestamp": ep.timestamp}
                for ep in recent[-10:]
            ]
            context["session_history_count"] = len(recent)
        else:
            recent = self.episodic.get_recent(n=5)
            context["recent_episodes"] = [
                {"type": ep.event_type, "content": ep.content, "timestamp": ep.timestamp}
                for ep in recent
            ]

        # Key facts (high confidence semantic memory)
        key_facts = self.semantic.get_high_confidence(min_confidence=0.9, limit=10)
        context["key_facts"] = [
            {"concept": e.concept, "definition": e.definition}
            for e in key_facts
        ]

        return context

    def get_conversation_history(
        self,
        session_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get conversation history as a list of message pairs."""
        sid = session_id or self._session_id
        if not sid:
            episodes = self.episodic.get_recent(n=limit, event_type="interaction")
        else:
            session_eps = self.episodic.get_session_history(sid)
            episodes = [e for e in session_eps if e.event_type == "interaction"][-limit:]

        return [
            {
                "role": "user",
                "content": ep.content.get("input", "") if isinstance(ep.content, dict) else str(ep.content),
                "timestamp": ep.timestamp,
            }
            for ep in episodes
        ] + [
            {
                "role": "assistant",
                "content": ep.content.get("response", "") if isinstance(ep.content, dict) else "",
                "timestamp": ep.timestamp,
            }
            for ep in episodes
        ]

    # =========================================================================
    # Consolidation
    # =========================================================================

    async def consolidate(self) -> Dict[str, int]:
        """
        Run memory consolidation: promote important episodic memories
        to semantic knowledge.

        Returns counts of consolidated items.
        """
        logger.info("Running memory consolidation")
        consolidated = {"episodic_to_semantic": 0}

        # Find highly important, repeated episodes
        important_episodes = self.episodic.get_important(n=50, min_importance=0.8)

        for episode in important_episodes:
            # Extract facts from interactions
            if episode.event_type == "interaction" and isinstance(episode.content, dict):
                # This would normally use an LLM to extract facts
                # For now, store the episode summary in semantic memory
                key = f"learned_from_{episode.episode_id}"
                if not self.semantic.lookup(key):
                    self.semantic.store(
                        concept=key,
                        definition=str(episode.content),
                        confidence=episode.importance * 0.8,
                        source="episodic_consolidation",
                    )
                    consolidated["episodic_to_semantic"] += 1

        self._last_consolidation = time.time()
        logger.info(f"Consolidation complete: {consolidated}")
        return consolidated

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics."""
        return {
            "working": self.working.get_stats(),
            "episodic": self.episodic.get_stats(),
            "semantic": self.semantic.get_stats(),
            "session_id": self._session_id,
            "last_consolidation": self._last_consolidation,
        }

    def _maybe_consolidate(self) -> None:
        """Trigger consolidation if enough time has passed."""
        if time.time() - self._last_consolidation > self.consolidation_interval:
            asyncio.create_task(self.consolidate())
