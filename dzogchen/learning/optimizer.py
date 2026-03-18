"""
PromptOptimizer - Automated prompt engineering via DSPy-inspired optimization
==============================================================================

Automatically improves system prompts through:
- Zero-shot to few-shot promotion
- Chain-of-thought injection
- Instruction optimization via feedback
- A/B testing of prompt variants
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class PromptVariant:
    """A prompt variant being tested."""
    variant_id: str
    prompt: str
    scores: List[float] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def avg_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.5

    @property
    def sample_count(self) -> int:
        return len(self.scores)

    @property
    def confidence_interval(self) -> Tuple[float, float]:
        """95% confidence interval using normal approximation."""
        if self.sample_count < 2:
            return (0.0, 1.0)
        import math
        std = (sum((s - self.avg_score) ** 2 for s in self.scores) / self.sample_count) ** 0.5
        margin = 1.96 * std / math.sqrt(self.sample_count)
        return (
            max(0.0, self.avg_score - margin),
            min(1.0, self.avg_score + margin),
        )


class PromptOptimizer:
    """
    Automatically optimizes prompts through empirical testing.

    Uses a simplified version of the DSPy/OPRO approach:
    1. Start with a base prompt
    2. Generate variants (with different instructions, examples, formats)
    3. A/B test variants against each other
    4. Keep the best-performing variant
    5. Repeat with new variants based on the winner

    This is the same technique used by leading AI labs to
    automatically improve their prompts.
    """

    VARIANT_GENERATION_PROMPT = """You are an expert prompt engineer. Improve this system prompt.

Original prompt:
{base_prompt}

Recent failure examples:
{failures}

Recent success examples:
{successes}

Generate 3 improved variants of this prompt. Each variant should:
1. Be specific and action-oriented
2. Include clear constraints and format requirements
3. Leverage the success patterns while avoiding failure modes

Format:
VARIANT 1:
[prompt text]

VARIANT 2:
[prompt text]

VARIANT 3:
[prompt text]"""

    COT_INJECTION = "\n\nThink step by step before giving your final answer."
    CONFIDENCE_INJECTION = "\n\nFor each claim, indicate your confidence level (high/medium/low)."
    FORMAT_INJECTION = "\n\nStructure your response with: (1) Direct answer, (2) Explanation, (3) Caveats."

    def __init__(
        self,
        base_prompt: str,
        model_caller: Optional[Callable] = None,
        min_samples_per_variant: int = 10,
        max_variants: int = 5,
    ):
        self.base_prompt = base_prompt
        self._model_caller = model_caller
        self.min_samples = min_samples_per_variant
        self.max_variants = max_variants

        # Initialize with base prompt and some augmented variants
        self._variants: Dict[str, PromptVariant] = {}
        self._winner: Optional[str] = None

        self._initialize_variants()

    @property
    def best_prompt(self) -> str:
        """Get the current best-performing prompt variant."""
        if self._winner and self._winner in self._variants:
            return self._variants[self._winner].prompt

        if not self._variants:
            return self.base_prompt

        best = max(self._variants.values(), key=lambda v: v.avg_score)
        return best.prompt

    def select_prompt(self) -> Tuple[str, str]:
        """
        Select a prompt for the next request using Thompson sampling.

        Returns (variant_id, prompt_text).
        Uses exploration/exploitation trade-off to keep testing.
        """
        if not self._variants:
            return ("base", self.base_prompt)

        # Thompson sampling: sample from each variant's distribution
        best_sample = -1.0
        best_id = list(self._variants.keys())[0]

        for vid, variant in self._variants.items():
            if variant.sample_count < self.min_samples:
                # Always sample under-tested variants
                return (vid, variant.prompt)

            # Sample from Beta distribution
            alpha = sum(1 for s in variant.scores if s > 0.7) + 1
            beta = sum(1 for s in variant.scores if s <= 0.7) + 1

            try:
                import random
                # Approximate Beta sample
                sample = random.betavariate(alpha, beta)
                if sample > best_sample:
                    best_sample = sample
                    best_id = vid
            except Exception:
                return (best_id, self._variants[best_id].prompt)

        return (best_id, self._variants[best_id].prompt)

    def record_result(self, variant_id: str, score: float) -> None:
        """Record the quality score for a prompt variant."""
        if variant_id in self._variants:
            self._variants[variant_id].scores.append(score)

        # Update winner if we have enough data
        self._update_winner()

    async def optimize(
        self,
        failures: Optional[List[str]] = None,
        successes: Optional[List[str]] = None,
    ) -> List[PromptVariant]:
        """
        Run one optimization round: generate and add new variants.

        Args:
            failures: Recent failure examples
            successes: Recent success examples

        Returns:
            List of newly created variants
        """
        if self._model_caller is None:
            return []

        current_best = self.best_prompt
        failure_text = "\n".join(failures[:3]) if failures else "None available"
        success_text = "\n".join(successes[:3]) if successes else "None available"

        prompt = self.VARIANT_GENERATION_PROMPT.format(
            base_prompt=current_best,
            failures=failure_text,
            successes=success_text,
        )

        try:
            response = await self._model_caller(prompt)
            new_variants = self._parse_variants(response)

            # Add to rotation (replacing weakest if at capacity)
            added = []
            for text in new_variants:
                if len(self._variants) >= self.max_variants:
                    self._remove_weakest_variant()

                vid = f"v{len(self._variants) + 1}_{int(time.time())}"
                variant = PromptVariant(variant_id=vid, prompt=text)
                self._variants[vid] = variant
                added.append(variant)
                logger.info(f"PromptOptimizer: added variant {vid}")

            return added

        except Exception as e:
            logger.error(f"Prompt optimization failed: {e}")
            return []

    def get_variant_report(self) -> Dict[str, Any]:
        """Get a report on all variants and their performance."""
        return {
            "variant_count": len(self._variants),
            "winner": self._winner,
            "best_prompt_preview": self.best_prompt[:200],
            "variants": [
                {
                    "id": v.variant_id,
                    "avg_score": v.avg_score,
                    "sample_count": v.sample_count,
                    "confidence_interval": v.confidence_interval,
                    "prompt_preview": v.prompt[:100],
                }
                for v in sorted(
                    self._variants.values(),
                    key=lambda v: v.avg_score,
                    reverse=True,
                )
            ],
        }

    def _initialize_variants(self) -> None:
        """Create initial set of variants from the base prompt."""
        variants = [
            ("base", self.base_prompt),
            ("cot", self.base_prompt + self.COT_INJECTION),
            ("confidence", self.base_prompt + self.CONFIDENCE_INJECTION),
            ("structured", self.base_prompt + self.FORMAT_INJECTION),
        ]

        for vid, prompt in variants[:self.max_variants]:
            self._variants[vid] = PromptVariant(variant_id=vid, prompt=prompt)

    def _update_winner(self) -> None:
        """Update the winner based on current data."""
        tested = {
            vid: v for vid, v in self._variants.items()
            if v.sample_count >= self.min_samples
        }

        if not tested:
            return

        # Find the variant whose lower confidence bound is highest
        # (conservative estimate - avoids selecting lucky flukes)
        def lower_bound(v: PromptVariant) -> float:
            lo, _ = v.confidence_interval
            return lo

        best = max(tested.values(), key=lower_bound)
        self._winner = best.variant_id

    def _remove_weakest_variant(self) -> None:
        """Remove the worst-performing variant to make room."""
        if not self._variants:
            return

        tested = [(vid, v) for vid, v in self._variants.items() if v.sample_count >= 3]
        if not tested:
            # Remove oldest if nothing is tested enough
            oldest = min(self._variants, key=lambda k: self._variants[k].created_at)
            del self._variants[oldest]
            return

        weakest_id = min(tested, key=lambda x: x[1].avg_score)[0]
        del self._variants[weakest_id]
        logger.debug(f"Removed weakest variant: {weakest_id}")

    def _parse_variants(self, response: str) -> List[str]:
        """Parse generated variants from LLM response."""
        variants = []
        current = []
        in_variant = False

        for line in response.split("\n"):
            if line.strip().startswith("VARIANT ") and ":" in line:
                if current and in_variant:
                    variant_text = "\n".join(current).strip()
                    if variant_text:
                        variants.append(variant_text)
                current = []
                in_variant = True
            elif in_variant:
                current.append(line)

        # Don't forget last variant
        if current and in_variant:
            variant_text = "\n".join(current).strip()
            if variant_text:
                variants.append(variant_text)

        return variants[:3]  # Return at most 3
