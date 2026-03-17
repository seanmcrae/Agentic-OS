"""
SemanticMemory - Conceptual knowledge and fact storage
======================================================

Stores factual knowledge, concepts, relationships, and
general world knowledge. Supports fuzzy retrieval by
semantic similarity using embeddings.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeEntry:
    """A unit of semantic knowledge."""
    entry_id: str
    concept: str
    definition: str
    domain: str = "general"
    confidence: float = 1.0
    source: str = "unknown"
    embedding: Optional[List[float]] = None
    related_concepts: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0


class SemanticMemory:
    """
    Conceptual long-term memory for facts and knowledge.

    Stores structured knowledge entries with optional embedding-based
    similarity search, domain organization, and confidence tracking.
    """

    def __init__(
        self,
        max_entries: int = 100_000,
        embedding_function: Optional[Any] = None,
        similarity_threshold: float = 0.75,
    ):
        self.max_entries = max_entries
        self._embed = embedding_function
        self.similarity_threshold = similarity_threshold

        self._entries: Dict[str, KnowledgeEntry] = {}
        self._concept_index: Dict[str, str] = {}  # normalized concept -> entry_id
        self._domain_index: Dict[str, List[str]] = {}  # domain -> [entry_ids]
        self._entry_counter = 0

    def store(
        self,
        concept: str,
        definition: str,
        domain: str = "general",
        confidence: float = 1.0,
        source: str = "unknown",
        related_concepts: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> KnowledgeEntry:
        """
        Store a knowledge entry.

        If an entry for the concept already exists, it's updated if
        the new confidence is higher.
        """
        concept_key = concept.lower().strip()

        # Update existing entry if concept already known
        if concept_key in self._concept_index:
            existing_id = self._concept_index[concept_key]
            existing = self._entries[existing_id]

            if confidence >= existing.confidence:
                existing.definition = definition
                existing.confidence = confidence
                existing.source = source
                existing.updated_at = time.time()
                if related_concepts:
                    existing.related_concepts = list(set(
                        existing.related_concepts + related_concepts
                    ))
                if properties:
                    existing.properties.update(properties)
                logger.debug(f"SemanticMemory: updated '{concept}'")
                return existing
            else:
                logger.debug(f"SemanticMemory: kept existing '{concept}' (higher confidence)")
                return existing

        # Create new entry
        entry_id = f"sem_{self._entry_counter:010d}"
        self._entry_counter += 1

        entry = KnowledgeEntry(
            entry_id=entry_id,
            concept=concept,
            definition=definition,
            domain=domain,
            confidence=confidence,
            source=source,
            related_concepts=related_concepts or [],
            properties=properties or {},
            embedding=embedding,
        )

        self._entries[entry_id] = entry
        self._concept_index[concept_key] = entry_id

        if domain not in self._domain_index:
            self._domain_index[domain] = []
        self._domain_index[domain].append(entry_id)

        logger.debug(f"SemanticMemory: stored '{concept}' in domain '{domain}'")
        return entry

    def lookup(self, concept: str) -> Optional[KnowledgeEntry]:
        """Exact lookup by concept name."""
        concept_key = concept.lower().strip()
        entry_id = self._concept_index.get(concept_key)
        if entry_id and entry_id in self._entries:
            entry = self._entries[entry_id]
            entry.access_count += 1
            return entry
        return None

    def search(
        self,
        query: str,
        domain: Optional[str] = None,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> List[KnowledgeEntry]:
        """
        Search for knowledge entries by text similarity.

        Uses embedding similarity if available, otherwise falls back
        to keyword matching.
        """
        candidates = list(self._entries.values())

        # Filter by domain
        if domain and domain in self._domain_index:
            domain_ids = set(self._domain_index[domain])
            candidates = [e for e in candidates if e.entry_id in domain_ids]

        # Filter by confidence
        if min_confidence > 0:
            candidates = [e for e in candidates if e.confidence >= min_confidence]

        # Score by relevance
        scored = [(e, self._score_entry(e, query)) for e in candidates]
        scored = [(e, s) for e, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)

        results = [e for e, _ in scored[:limit]]
        for e in results:
            e.access_count += 1
        return results

    def get_related(
        self,
        concept: str,
        depth: int = 1,
        max_results: int = 20,
    ) -> List[KnowledgeEntry]:
        """Get entries related to a concept, up to given depth."""
        entry = self.lookup(concept)
        if not entry:
            return []

        visited = {entry.entry_id}
        results = []
        frontier = list(entry.related_concepts)

        for _ in range(depth):
            next_frontier = []
            for related_concept in frontier:
                related_entry = self.lookup(related_concept)
                if related_entry and related_entry.entry_id not in visited:
                    visited.add(related_entry.entry_id)
                    results.append(related_entry)
                    next_frontier.extend(related_entry.related_concepts)

                    if len(results) >= max_results:
                        return results

            frontier = next_frontier
            if not frontier:
                break

        return results

    def get_domain_knowledge(self, domain: str, limit: int = 50) -> List[KnowledgeEntry]:
        """Get all knowledge in a specific domain."""
        if domain not in self._domain_index:
            return []

        entries = [
            self._entries[eid]
            for eid in self._domain_index[domain]
            if eid in self._entries
        ]
        entries.sort(key=lambda e: e.confidence, reverse=True)
        return entries[:limit]

    def get_high_confidence(self, min_confidence: float = 0.9, limit: int = 20) -> List[KnowledgeEntry]:
        """Get the highest confidence knowledge entries."""
        high_conf = [e for e in self._entries.values() if e.confidence >= min_confidence]
        return sorted(high_conf, key=lambda e: e.confidence, reverse=True)[:limit]

    def forget(self, concept: str) -> bool:
        """Remove a knowledge entry."""
        concept_key = concept.lower().strip()
        if concept_key not in self._concept_index:
            return False

        entry_id = self._concept_index.pop(concept_key)
        if entry_id in self._entries:
            entry = self._entries.pop(entry_id)
            # Remove from domain index
            if entry.domain in self._domain_index:
                self._domain_index[entry.domain] = [
                    eid for eid in self._domain_index[entry.domain]
                    if eid != entry_id
                ]
            return True
        return False

    def update_confidence(self, concept: str, confidence_delta: float) -> bool:
        """Adjust the confidence of a knowledge entry."""
        entry = self.lookup(concept)
        if entry:
            entry.confidence = max(0.0, min(1.0, entry.confidence + confidence_delta))
            entry.updated_at = time.time()
            return True
        return False

    @property
    def total_entries(self) -> int:
        return len(self._entries)

    @property
    def domains(self) -> List[str]:
        return list(self._domain_index.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get semantic memory statistics."""
        if not self._entries:
            return {"total": 0}

        avg_confidence = sum(e.confidence for e in self._entries.values()) / len(self._entries)
        domain_sizes = {d: len(ids) for d, ids in self._domain_index.items()}

        return {
            "total_entries": len(self._entries),
            "capacity": self.max_entries,
            "utilization": len(self._entries) / self.max_entries,
            "domains": domain_sizes,
            "avg_confidence": avg_confidence,
            "most_accessed": sorted(
                self._entries.values(),
                key=lambda e: e.access_count,
                reverse=True,
            )[:5],
        }

    def _score_entry(self, entry: KnowledgeEntry, query: str) -> float:
        """Score an entry's relevance to a query."""
        if not query:
            return entry.confidence

        query_lower = query.lower()
        score = 0.0

        # Exact concept match
        if query_lower == entry.concept.lower():
            score += 1.0

        # Partial concept match
        elif query_lower in entry.concept.lower() or entry.concept.lower() in query_lower:
            score += 0.7

        # Definition match
        if query_lower in entry.definition.lower():
            score += 0.4

        # Related concepts match
        for related in entry.related_concepts:
            if query_lower in related.lower():
                score += 0.2
                break

        # Word-level overlap
        query_words = set(query_lower.split())
        concept_words = set(entry.concept.lower().split())
        definition_words = set(entry.definition.lower().split()[:20])

        word_overlap = len(query_words & concept_words) / max(len(query_words), 1)
        def_overlap = len(query_words & definition_words) / max(len(query_words), 1)

        score += word_overlap * 0.5 + def_overlap * 0.3

        # Weight by confidence
        return score * entry.confidence
