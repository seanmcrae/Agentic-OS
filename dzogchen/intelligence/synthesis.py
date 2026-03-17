"""
KnowledgeSynthesizer - Cross-domain knowledge integration and insight generation
================================================================================

Combines information from multiple sources, identifies patterns,
resolves conflicts, and generates novel insights through synthesis.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SynthesisMode(str, Enum):
    INTEGRATIVE = "integrative"    # Combine compatible information
    COMPARATIVE = "comparative"    # Identify similarities and differences
    CONTRASTIVE = "contrastive"    # Highlight contradictions
    INFERENTIAL = "inferential"    # Draw new conclusions
    CREATIVE = "creative"          # Generate novel combinations


@dataclass
class KnowledgeFragment:
    """A piece of knowledge to be synthesized."""
    content: str
    source: str
    confidence: float = 0.8
    domain: str = "general"
    tags: Set[str] = field(default_factory=set)
    supports: List[str] = field(default_factory=list)   # IDs of fragments this supports
    contradicts: List[str] = field(default_factory=list) # IDs this contradicts


@dataclass
class SynthesisResult:
    """Result of a knowledge synthesis operation."""
    query: str
    mode: SynthesisMode
    fragments_used: List[KnowledgeFragment]
    synthesis: str
    insights: List[str]
    contradictions_resolved: List[str]
    confidence: float
    novel_connections: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeSynthesizer:
    """
    Synthesizes knowledge from multiple fragments into coherent insights.

    Uses various synthesis modes to integrate, compare, contrast, and
    generate novel knowledge from existing information.
    """

    SYNTHESIS_PROMPT = """Synthesize the following knowledge fragments into a coherent response.

Query: {query}
Mode: {mode}

Knowledge Fragments:
{fragments}

Instructions:
1. Integrate compatible information
2. Note any contradictions and resolve them
3. Identify non-obvious connections
4. Generate actionable insights
5. Highlight what's most important

Provide:
SYNTHESIS: [Integrated understanding]
INSIGHTS: [List of key insights, especially non-obvious ones]
CONTRADICTIONS: [Any conflicts found and how they're resolved]
NOVEL CONNECTIONS: [Surprising or unexpected relationships identified]"""

    def __init__(self, model_caller: Optional[Callable] = None):
        self._model_caller = model_caller
        self._fragment_cache: List[KnowledgeFragment] = []
        self._synthesis_history: List[SynthesisResult] = []

    async def synthesize(
        self,
        query: str,
        fragments: List[KnowledgeFragment],
        mode: SynthesisMode = SynthesisMode.INTEGRATIVE,
    ) -> SynthesisResult:
        """
        Synthesize knowledge fragments into a coherent response.

        Args:
            query: The question or topic to synthesize around
            fragments: Knowledge fragments to synthesize
            mode: The synthesis mode to use

        Returns:
            SynthesisResult with integrated knowledge and insights
        """
        logger.info(f"Synthesizing {len(fragments)} fragments in {mode} mode")

        # Preprocess: resolve obvious contradictions
        resolved_fragments, contradictions = self._preprocess_fragments(fragments)

        if self._model_caller and resolved_fragments:
            result = await self._llm_synthesis(query, resolved_fragments, mode, contradictions)
        else:
            result = self._heuristic_synthesis(query, resolved_fragments, mode, contradictions)

        self._synthesis_history.append(result)
        return result

    async def find_connections(
        self,
        fragment_a: KnowledgeFragment,
        fragment_b: KnowledgeFragment,
    ) -> List[str]:
        """Find connections between two knowledge fragments."""
        if self._model_caller is None:
            return self._heuristic_connections(fragment_a, fragment_b)

        prompt = f"""Find non-obvious connections between these two pieces of knowledge:

A: {fragment_a.content}
B: {fragment_b.content}

List 3-5 specific, concrete connections or relationships. Be precise."""

        try:
            response = await self._model_caller(prompt)
            return [
                line.strip().lstrip("0123456789. -")
                for line in response.split("\n")
                if line.strip() and len(line.strip()) > 10
            ][:5]
        except Exception as e:
            logger.error(f"Connection finding failed: {e}")
            return []

    async def resolve_contradiction(
        self,
        fragment_a: KnowledgeFragment,
        fragment_b: KnowledgeFragment,
    ) -> Tuple[str, float]:
        """
        Resolve a contradiction between two knowledge fragments.

        Returns:
            Tuple of (resolution_text, confidence_in_resolution)
        """
        if self._model_caller is None:
            return (
                f"Contradiction between '{fragment_a.source}' and '{fragment_b.source}' - unable to resolve without LLM",
                0.3,
            )

        prompt = f"""These two statements appear to contradict each other. Provide a resolution.

Statement A (source: {fragment_a.source}, confidence: {fragment_a.confidence}):
{fragment_a.content}

Statement B (source: {fragment_b.source}, confidence: {fragment_b.confidence}):
{fragment_b.content}

Possible resolutions:
1. One is correct, one is wrong - explain which and why
2. Both are true in different contexts - explain the contexts
3. They're not actually contradicting - explain the distinction
4. The truth is somewhere in between - synthesize

Provide your resolution with a confidence score (0.0-1.0)."""

        try:
            response = await self._model_caller(prompt)
            # Simple confidence extraction
            confidence = 0.75
            if "confidence:" in response.lower():
                try:
                    conf_idx = response.lower().index("confidence:")
                    conf_str = response[conf_idx + 11:conf_idx + 16].strip()
                    confidence = float(conf_str)
                except Exception:
                    pass
            return response, confidence
        except Exception as e:
            logger.error(f"Contradiction resolution failed: {e}")
            return ("Unable to resolve contradiction", 0.2)

    async def generate_insights(
        self,
        fragments: List[KnowledgeFragment],
        domain: str = "general",
    ) -> List[str]:
        """Generate novel insights from a collection of knowledge fragments."""
        if not fragments:
            return []

        if self._model_caller is None:
            return [f"Pattern identified across {len(fragments)} knowledge fragments in {domain}"]

        combined = "\n".join(f"- [{f.source}] {f.content}" for f in fragments[:20])

        prompt = f"""Analyze these knowledge fragments and generate novel, non-obvious insights.

Domain: {domain}
Fragments:
{combined}

Generate 5 insights that:
1. Are not directly stated in the fragments
2. Reveal surprising patterns or connections
3. Have practical implications
4. Challenge common assumptions
5. Synthesize information across multiple fragments

Format as numbered insights."""

        try:
            response = await self._model_caller(prompt)
            return [
                line.strip().lstrip("0123456789. ")
                for line in response.split("\n")
                if line.strip() and line.strip()[0].isdigit()
            ][:5]
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            return []

    def _preprocess_fragments(
        self,
        fragments: List[KnowledgeFragment],
    ) -> Tuple[List[KnowledgeFragment], List[str]]:
        """Preprocess fragments to identify and handle contradictions."""
        contradictions = []

        # Sort by confidence (highest first)
        sorted_fragments = sorted(fragments, key=lambda f: f.confidence, reverse=True)

        # Simple deduplication
        seen_content: Set[str] = set()
        unique_fragments = []
        for f in sorted_fragments:
            content_key = f.content[:100].lower()
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_fragments.append(f)

        return unique_fragments, contradictions

    async def _llm_synthesis(
        self,
        query: str,
        fragments: List[KnowledgeFragment],
        mode: SynthesisMode,
        contradictions: List[str],
    ) -> SynthesisResult:
        """Use LLM to synthesize knowledge fragments."""
        fragment_text = "\n".join(
            f"[{i+1}] ({f.source}, confidence={f.confidence:.1f}): {f.content}"
            for i, f in enumerate(fragments[:20])
        )

        prompt = self.SYNTHESIS_PROMPT.format(
            query=query,
            mode=mode.value,
            fragments=fragment_text,
        )

        try:
            response = await self._model_caller(prompt)
            return self._parse_synthesis_response(query, mode, fragments, response, contradictions)
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            return self._heuristic_synthesis(query, fragments, mode, contradictions)

    def _parse_synthesis_response(
        self,
        query: str,
        mode: SynthesisMode,
        fragments: List[KnowledgeFragment],
        response: str,
        contradictions: List[str],
    ) -> SynthesisResult:
        """Parse the LLM synthesis response into structured result."""
        synthesis = ""
        insights = []
        novel_connections = []
        contradiction_resolutions = list(contradictions)

        sections = {"SYNTHESIS": "", "INSIGHTS": "", "CONTRADICTIONS": "", "NOVEL CONNECTIONS": ""}
        current_section = None

        for line in response.split("\n"):
            line = line.strip()
            for section_name in sections:
                if line.startswith(section_name + ":"):
                    current_section = section_name
                    content = line[len(section_name) + 1:].strip()
                    if content:
                        sections[section_name] += content + "\n"
                    break
            else:
                if current_section and line:
                    sections[current_section] += line + "\n"

        synthesis = sections["SYNTHESIS"].strip() or response[:500]

        insights = [
            line.strip().lstrip("0123456789. -")
            for line in sections["INSIGHTS"].split("\n")
            if line.strip() and len(line.strip()) > 10
        ]

        novel_connections = [
            line.strip().lstrip("0123456789. -")
            for line in sections["NOVEL CONNECTIONS"].split("\n")
            if line.strip() and len(line.strip()) > 10
        ]

        if sections["CONTRADICTIONS"].strip():
            contradiction_resolutions.append(sections["CONTRADICTIONS"].strip())

        avg_confidence = sum(f.confidence for f in fragments) / len(fragments) if fragments else 0.5

        return SynthesisResult(
            query=query,
            mode=mode,
            fragments_used=fragments,
            synthesis=synthesis,
            insights=insights,
            contradictions_resolved=contradiction_resolutions,
            confidence=avg_confidence,
            novel_connections=novel_connections,
        )

    def _heuristic_synthesis(
        self,
        query: str,
        fragments: List[KnowledgeFragment],
        mode: SynthesisMode,
        contradictions: List[str],
    ) -> SynthesisResult:
        """Fallback heuristic synthesis without LLM."""
        if not fragments:
            return SynthesisResult(
                query=query,
                mode=mode,
                fragments_used=[],
                synthesis="No knowledge fragments available for synthesis.",
                insights=[],
                contradictions_resolved=[],
                confidence=0.0,
                novel_connections=[],
            )

        # Simple concatenation with deduplication
        synthesis_parts = []
        for f in fragments[:5]:
            synthesis_parts.append(f"From {f.source}: {f.content}")

        synthesis = "\n\n".join(synthesis_parts)
        avg_confidence = sum(f.confidence for f in fragments) / len(fragments)

        return SynthesisResult(
            query=query,
            mode=mode,
            fragments_used=fragments,
            synthesis=synthesis,
            insights=[f"Synthesized from {len(fragments)} sources"],
            contradictions_resolved=contradictions,
            confidence=avg_confidence,
            novel_connections=[],
        )

    def _heuristic_connections(
        self,
        fragment_a: KnowledgeFragment,
        fragment_b: KnowledgeFragment,
    ) -> List[str]:
        """Simple heuristic connection finding."""
        connections = []

        # Find shared tags
        shared_tags = fragment_a.tags & fragment_b.tags
        if shared_tags:
            connections.append(f"Both relate to: {', '.join(shared_tags)}")

        # Same domain
        if fragment_a.domain == fragment_b.domain:
            connections.append(f"Both from domain: {fragment_a.domain}")

        # Same source
        if fragment_a.source == fragment_b.source:
            connections.append(f"Same source: {fragment_a.source}")

        return connections
