"""
ReflectionEngine - Self-reflection and output quality evaluation
================================================================

Enables Dzogchen to evaluate its own reasoning quality, identify
errors, biases, and gaps, and generate improved responses through
iterative self-critique.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ReflectionDimension(str, Enum):
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    SAFETY = "safety"
    UNCERTAINTY = "uncertainty"
    BIAS = "bias"


@dataclass
class ReflectionResult:
    """The result of a reflection pass."""
    original_response: str
    reflection_notes: List[str]
    dimension_scores: Dict[ReflectionDimension, float]  # 0.0 - 1.0
    identified_issues: List[str]
    suggested_improvements: List[str]
    revised_response: Optional[str] = None
    confidence: float = 0.0
    reflection_depth: int = 1
    timestamp: float = field(default_factory=time.time)

    @property
    def overall_quality(self) -> float:
        if not self.dimension_scores:
            return 0.0
        return sum(self.dimension_scores.values()) / len(self.dimension_scores)

    @property
    def needs_revision(self) -> bool:
        return self.overall_quality < 0.75 or len(self.identified_issues) > 0


class ReflectionEngine:
    """
    Self-reflection engine for evaluating and improving AI outputs.

    Uses a multi-dimensional quality framework to assess responses
    and iteratively improve them through targeted self-critique.
    """

    REFLECTION_PROMPT_TEMPLATE = """You are performing a critical self-reflection on your previous response.

Original Query: {query}
Your Response: {response}

Evaluate this response across the following dimensions:
1. ACCURACY: Is the information factually correct?
2. COMPLETENESS: Does it fully address the query?
3. COHERENCE: Is the reasoning logical and consistent?
4. RELEVANCE: Is all content directly relevant?
5. SAFETY: Are there any harmful or misleading elements?
6. UNCERTAINTY: Is uncertainty appropriately acknowledged?
7. BIAS: Are there any unwarranted assumptions or biases?

Provide:
- A score (0.0-1.0) for each dimension
- Specific issues identified
- Concrete improvements to make
- A revised response if needed

Be brutally honest. Your goal is to provide the best possible answer."""

    SYNTHESIS_PROMPT_TEMPLATE = """Given the following reflection feedback, synthesize an improved response.

Original Query: {query}
Original Response: {original}
Reflection Notes: {notes}
Issues Identified: {issues}
Suggested Improvements: {improvements}

Provide a revised, improved response that addresses all identified issues.
Be concise, accurate, and directly responsive to the original query."""

    def __init__(
        self,
        model_caller: Optional[Any] = None,
        max_depth: int = 3,
        quality_threshold: float = 0.85,
        enable_synthesis: bool = True,
    ):
        self._model_caller = model_caller
        self.max_depth = max_depth
        self.quality_threshold = quality_threshold
        self.enable_synthesis = enable_synthesis
        self._reflection_history: List[ReflectionResult] = []

    async def reflect(
        self,
        query: str,
        response: str,
        context: Optional[Dict[str, Any]] = None,
        depth: int = 1,
    ) -> ReflectionResult:
        """
        Perform a reflection pass on a response.

        Args:
            query: The original query/prompt
            response: The response to evaluate
            context: Additional context for reflection
            depth: Current reflection depth (for recursive improvement)

        Returns:
            ReflectionResult with evaluation and optional revision
        """
        logger.info(f"Starting reflection (depth={depth}) on response of {len(response)} chars")

        # Perform the reflection analysis
        reflection_data = await self._analyze_response(query, response, context)

        result = ReflectionResult(
            original_response=response,
            reflection_notes=reflection_data.get("notes", []),
            dimension_scores=reflection_data.get("scores", {}),
            identified_issues=reflection_data.get("issues", []),
            suggested_improvements=reflection_data.get("improvements", []),
            confidence=reflection_data.get("confidence", 0.7),
            reflection_depth=depth,
        )

        # If quality is insufficient and we haven't hit max depth, revise
        if result.needs_revision and depth < self.max_depth and self.enable_synthesis:
            logger.info(f"Quality {result.overall_quality:.2f} below threshold, generating revision")
            revised = await self._synthesize_improvement(query, result)

            if revised:
                result.revised_response = revised

                # Recursively reflect on the revision
                if depth < self.max_depth - 1:
                    sub_result = await self.reflect(query, revised, context, depth + 1)
                    if sub_result.overall_quality > result.overall_quality:
                        result.revised_response = sub_result.revised_response or revised

        self._reflection_history.append(result)
        return result

    async def quick_check(self, response: str) -> Tuple[bool, List[str]]:
        """
        Fast heuristic check without LLM call.
        Returns (is_acceptable, issues).
        """
        issues = []

        # Basic safety checks
        if len(response) < 10:
            issues.append("Response is suspiciously short")

        if response.count("I don't know") > 3:
            issues.append("Excessive uncertainty hedging")

        # Check for incomplete sentences
        stripped = response.strip()
        if stripped and not stripped[-1] in ".!?\"'":
            issues.append("Response appears to be cut off mid-sentence")

        # Check for repetition
        words = response.lower().split()
        if len(words) > 10:
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            max_freq = max(word_freq.values()) if word_freq else 0
            if max_freq > len(words) * 0.15:  # More than 15% of words are the same
                issues.append("Potential repetitive content detected")

        return len(issues) == 0, issues

    async def _analyze_response(
        self,
        query: str,
        response: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze a response using the LLM or heuristics."""
        if self._model_caller is None:
            return self._heuristic_analysis(response)

        prompt = self.REFLECTION_PROMPT_TEMPLATE.format(
            query=query,
            response=response,
        )

        try:
            llm_response = await self._model_caller(prompt)
            return self._parse_reflection_response(llm_response)
        except Exception as e:
            logger.warning(f"LLM reflection failed, falling back to heuristics: {e}")
            return self._heuristic_analysis(response)

    def _heuristic_analysis(self, response: str) -> Dict[str, Any]:
        """Fallback heuristic analysis without LLM."""
        words = response.split()
        word_count = len(words)

        # Heuristic scores
        coherence = min(1.0, word_count / 50) if word_count > 0 else 0.0
        completeness = min(1.0, word_count / 100) if word_count > 0 else 0.0

        return {
            "notes": ["Heuristic analysis only - no LLM available"],
            "scores": {
                ReflectionDimension.ACCURACY: 0.7,
                ReflectionDimension.COMPLETENESS: completeness,
                ReflectionDimension.COHERENCE: coherence,
                ReflectionDimension.RELEVANCE: 0.7,
                ReflectionDimension.SAFETY: 0.9,
                ReflectionDimension.UNCERTAINTY: 0.7,
                ReflectionDimension.BIAS: 0.8,
            },
            "issues": [],
            "improvements": [],
            "confidence": 0.5,
        }

    def _parse_reflection_response(self, llm_output: str) -> Dict[str, Any]:
        """Parse the LLM's reflection output into structured data."""
        # This would normally parse structured LLM output
        # For now, return a reasonable default structure
        return {
            "notes": [llm_output[:500] if llm_output else ""],
            "scores": {dim: 0.75 for dim in ReflectionDimension},
            "issues": [],
            "improvements": [],
            "confidence": 0.75,
        }

    async def _synthesize_improvement(
        self,
        query: str,
        reflection: ReflectionResult,
    ) -> Optional[str]:
        """Generate an improved response based on reflection feedback."""
        if self._model_caller is None:
            return None

        prompt = self.SYNTHESIS_PROMPT_TEMPLATE.format(
            query=query,
            original=reflection.original_response,
            notes="\n".join(reflection.reflection_notes),
            issues="\n".join(reflection.identified_issues),
            improvements="\n".join(reflection.suggested_improvements),
        )

        try:
            return await self._model_caller(prompt)
        except Exception as e:
            logger.error(f"Failed to synthesize improvement: {e}")
            return None

    def get_reflection_stats(self) -> Dict[str, Any]:
        """Get statistics about reflection history."""
        if not self._reflection_history:
            return {"total_reflections": 0}

        qualities = [r.overall_quality for r in self._reflection_history]
        revision_count = sum(1 for r in self._reflection_history if r.revised_response)

        return {
            "total_reflections": len(self._reflection_history),
            "average_quality": sum(qualities) / len(qualities),
            "revision_rate": revision_count / len(self._reflection_history),
            "max_depth_reached": max(r.reflection_depth for r in self._reflection_history),
        }
