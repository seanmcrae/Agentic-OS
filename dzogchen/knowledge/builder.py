"""
KnowledgeGraphBuilder - Automatically build graphs from text
============================================================

Extracts entities and relationships from text using LLM
to populate the knowledge graph. Also handles:
- Fact extraction from documents
- Entity disambiguation
- Relationship validation
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

from dzogchen.knowledge.graph import KnowledgeGraph, KGNode, KGEdge

logger = logging.getLogger(__name__)


@dataclass
class ExtractedTriple:
    """A subject-predicate-object triple extracted from text."""
    subject: str
    subject_type: str
    predicate: str
    object: str
    object_type: str
    confidence: float = 0.8
    source_text: str = ""


class KnowledgeGraphBuilder:
    """
    Builds and enriches a knowledge graph from natural language text.

    Uses LLM-based extraction to identify entities and relationships,
    then adds them to the graph with appropriate confidence scores.
    """

    EXTRACTION_PROMPT = """Extract all entities and relationships from the following text.

Text: {text}

Return a list of triples in this EXACT format, one per line:
SUBJECT_TYPE | SUBJECT | PREDICATE | OBJECT_TYPE | OBJECT

Rules:
- SUBJECT_TYPE and OBJECT_TYPE should be: Person, Organization, Concept, Event, Place, Technology, Animal, Other
- PREDICATE should be one of: is_a, part_of, created_by, located_in, related_to, causes, uses, has, happened_in, leads_to, contrasts_with, invented_by, founded_by
- Be specific and factual
- Extract at least 3 triples if possible
- Only include information explicitly stated in the text

Example output:
Person | Alan Turing | created | Concept | Turing Machine
Technology | Python | is_a | Concept | Programming Language
Organization | Google | founded_by | Person | Larry Page"""

    DISAMBIGUATION_PROMPT = """Two entity names may refer to the same concept. Determine if they are the same.

Entity A: "{name_a}" (type: {type_a})
Entity B: "{name_b}" (type: {type_b})

Are these the same entity? Reply with:
SAME: [reason]
or
DIFFERENT: [reason]"""

    def __init__(
        self,
        graph: Optional[KnowledgeGraph] = None,
        model_caller: Optional[Callable] = None,
        confidence_threshold: float = 0.6,
    ):
        self.graph = graph or KnowledgeGraph()
        self._model_caller = model_caller
        self.confidence_threshold = confidence_threshold
        self._extraction_count = 0
        self._triple_count = 0

    async def extract_from_text(
        self,
        text: str,
        source: str = "text",
        add_to_graph: bool = True,
    ) -> List[ExtractedTriple]:
        """
        Extract knowledge triples from natural language text.

        Args:
            text: Source text to extract from
            source: Label for the data source
            add_to_graph: Whether to add extracted triples to the graph

        Returns:
            List of extracted triples
        """
        if not text.strip():
            return []

        triples = await self._llm_extract(text, source)

        # Filter by confidence
        triples = [t for t in triples if t.confidence >= self.confidence_threshold]

        if add_to_graph:
            added = self._add_triples_to_graph(triples, source)
            self._triple_count += added

        self._extraction_count += 1
        logger.info(f"Extracted {len(triples)} triples from text ({len(text)} chars)")
        return triples

    async def extract_from_documents(
        self,
        documents: List[Dict[str, str]],
        source: str = "document",
    ) -> Dict[str, Any]:
        """
        Batch extract from multiple documents.

        Args:
            documents: List of {"title": str, "content": str} dicts
            source: Data source label

        Returns:
            Summary statistics
        """
        all_triples = []
        for doc in documents:
            title = doc.get("title", "")
            content = doc.get("content", "")
            doc_source = f"{source}:{title}" if title else source

            triples = await self.extract_from_text(content, source=doc_source)
            all_triples.extend(triples)

            # Brief pause between extractions to avoid rate limits
            await asyncio.sleep(0.1)

        return {
            "documents_processed": len(documents),
            "triples_extracted": len(all_triples),
            "graph_stats": self.graph.get_stats(),
        }

    async def enrich_node(
        self,
        node_name: str,
        additional_context: str = "",
    ) -> List[ExtractedTriple]:
        """
        Enrich a node with additional knowledge by generating related facts.

        Args:
            node_name: Name of the node to enrich
            additional_context: Additional context for enrichment

        Returns:
            Newly added triples
        """
        node = self.graph.find_node(node_name)
        if not node:
            logger.warning(f"Node not found for enrichment: {node_name}")
            return []

        if self._model_caller is None:
            return []

        # Get existing neighbors for context
        neighbors = self.graph.get_neighbors(node, direction="both")
        neighbor_text = ", ".join(f"{e.relation} {n.name}" for n, e in neighbors[:5])

        prompt = f"""Generate 5 important facts about "{node_name}" ({node.label}).

Existing known relationships: {neighbor_text or "none"}
Additional context: {additional_context or "none"}

Format each fact as a triple:
SUBJECT_TYPE | SUBJECT | PREDICATE | OBJECT_TYPE | OBJECT"""

        try:
            response = await self._model_caller(prompt)
            triples = self._parse_triples(response, source=f"enrichment:{node_name}")
            added = self._add_triples_to_graph(triples, source=f"enrichment:{node_name}")
            logger.info(f"Enriched {node_name} with {added} new triples")
            return triples
        except Exception as e:
            logger.error(f"Enrichment failed for {node_name}: {e}")
            return []

    async def find_missing_links(
        self,
        max_candidates: int = 20,
    ) -> List[Tuple[KGNode, KGNode, str, float]]:
        """
        Predict likely missing edges in the graph using link prediction.

        Uses common-neighbor heuristics to suggest probable relationships.

        Returns:
            List of (node_a, node_b, suggested_relation, confidence)
        """
        candidates = []
        nodes = list(self.graph._nodes.values())[:100]  # Limit for performance

        for i, node_a in enumerate(nodes):
            for node_b in nodes[i+1:]:
                # Skip if already connected
                existing = self.graph.get_edges_between(node_a.node_id, node_b.node_id)
                if existing:
                    continue

                # Check common neighbors
                common = self.graph.find_common_neighbors(node_a.name, node_b.name)
                if not common:
                    continue

                # Suggest relation based on node types
                suggested_relation = self._suggest_relation(node_a, node_b)
                confidence = min(0.9, 0.3 + len(common) * 0.15)

                candidates.append((node_a, node_b, suggested_relation, confidence))

                if len(candidates) >= max_candidates:
                    return sorted(candidates, key=lambda x: x[3], reverse=True)

        return sorted(candidates, key=lambda x: x[3], reverse=True)

    async def validate_edges(
        self,
        sample_size: int = 20,
    ) -> Dict[str, Any]:
        """
        Validate existing edges using LLM to flag potentially incorrect ones.

        Returns:
            Validation report with flagged edges
        """
        if self._model_caller is None:
            return {"validated": 0, "flagged": []}

        edges = list(self.graph._edges.values())[:sample_size]
        flagged = []

        for edge in edges:
            src = self.graph.get_node(edge.source_id)
            tgt = self.graph.get_node(edge.target_id)
            if not src or not tgt:
                continue

            prompt = f"""Is this relationship factually correct?
"{src.name}" --[{edge.relation}]--> "{tgt.name}"

Reply with:
VALID: [brief reason]
or
INVALID: [brief reason]
or
UNCERTAIN: [why uncertain]"""

            try:
                response = await self._model_caller(prompt)
                if "INVALID" in response.upper():
                    flagged.append({
                        "edge": edge.to_dict(),
                        "source_name": src.name,
                        "target_name": tgt.name,
                        "reason": response,
                    })
            except Exception:
                pass

        return {
            "validated": len(edges),
            "flagged": flagged,
            "flag_rate": len(flagged) / len(edges) if edges else 0,
        }

    async def _llm_extract(self, text: str, source: str) -> List[ExtractedTriple]:
        """Use LLM to extract triples from text."""
        if self._model_caller is None:
            return self._heuristic_extract(text, source)

        prompt = self.EXTRACTION_PROMPT.format(text=text[:3000])

        try:
            response = await self._model_caller(prompt)
            return self._parse_triples(response, source)
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}. Using heuristics.")
            return self._heuristic_extract(text, source)

    def _parse_triples(self, response: str, source: str) -> List[ExtractedTriple]:
        """Parse LLM response into ExtractedTriple objects."""
        triples = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or "|" not in line:
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 5:
                continue

            subj_type, subj, pred, obj_type, obj = parts

            # Basic validation
            if not subj or not pred or not obj:
                continue
            if len(subj) > 100 or len(obj) > 100:
                continue

            triples.append(ExtractedTriple(
                subject=subj,
                subject_type=subj_type or "Concept",
                predicate=pred,
                object=obj,
                object_type=obj_type or "Concept",
                confidence=0.8,
                source_text=source,
            ))

        return triples

    def _heuristic_extract(self, text: str, source: str) -> List[ExtractedTriple]:
        """Fallback heuristic extraction without LLM."""
        # Very basic: look for "X is a Y", "X is part of Y" patterns
        import re
        triples = []

        patterns = [
            (r"(\w[\w\s]+) is (?:a|an) ([\w\s]+)", "is_a"),
            (r"(\w[\w\s]+) is part of ([\w\s]+)", "part_of"),
            (r"(\w[\w\s]+) created ([\w\s]+)", "created"),
            (r"(\w[\w\s]+) uses ([\w\s]+)", "uses"),
        ]

        for pattern, relation in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subj = match.group(1).strip()[:50]
                obj = match.group(2).strip()[:50]
                if len(subj) > 2 and len(obj) > 2:
                    triples.append(ExtractedTriple(
                        subject=subj,
                        subject_type="Concept",
                        predicate=relation,
                        object=obj,
                        object_type="Concept",
                        confidence=0.5,
                        source_text=source,
                    ))

        return triples[:10]  # Limit heuristic results

    def _add_triples_to_graph(self, triples: List[ExtractedTriple], source: str) -> int:
        """Add extracted triples to the knowledge graph."""
        added = 0
        for triple in triples:
            try:
                src_node = self.graph.add_node(
                    triple.subject_type, triple.subject, source=source
                )
                tgt_node = self.graph.add_node(
                    triple.object_type, triple.object, source=source
                )
                self.graph.add_edge(
                    src_node, tgt_node, triple.predicate,
                    confidence=triple.confidence,
                    source_name=source,
                )
                added += 1
            except Exception as e:
                logger.debug(f"Failed to add triple: {e}")
        return added

    def _suggest_relation(self, node_a: KGNode, node_b: KGNode) -> str:
        """Suggest a likely relation type based on node labels."""
        type_relation_map = {
            ("Person", "Organization"): "member_of",
            ("Person", "Concept"): "specializes_in",
            ("Organization", "Concept"): "works_on",
            ("Technology", "Concept"): "implements",
            ("Event", "Person"): "involves",
            ("Place", "Organization"): "hosts",
        }
        key = (node_a.label, node_b.label)
        return type_relation_map.get(key, "related_to")
