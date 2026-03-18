"""
SelfImprover - Continuous self-improvement via feedback learning
================================================================

Analyzes feedback patterns to adaptively improve:
- Prompt engineering (optimize system prompts)
- Strategy selection (which reasoning approach works best)
- Model routing (which model performs best for which tasks)
- Few-shot example selection (best examples for in-context learning)
- Error pattern recognition (identify and correct recurring mistakes)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

from dzogchen.learning.feedback import FeedbackCollector, FeedbackSummary, Feedback

logger = logging.getLogger(__name__)


@dataclass
class ImprovementAction:
    """A concrete improvement action to take."""
    action_type: str         # "update_prompt" | "switch_strategy" | "add_example"
    description: str
    payload: Dict[str, Any]
    expected_improvement: float
    confidence: float
    applied_at: Optional[float] = None
    result_score: Optional[float] = None


@dataclass
class ImprovementCycle:
    """One cycle of the self-improvement loop."""
    cycle_id: int
    started_at: float
    feedback_analyzed: int
    actions_taken: List[ImprovementAction]
    baseline_score: float
    target_score: float
    achieved_score: Optional[float] = None
    completed_at: Optional[float] = None

    @property
    def improvement_delta(self) -> Optional[float]:
        if self.achieved_score is not None:
            return self.achieved_score - self.baseline_score
        return None


class SelfImprover:
    """
    Continuous self-improvement engine.

    Runs periodic improvement cycles that:
    1. Analyze feedback to find patterns and weaknesses
    2. Generate improvement hypotheses
    3. Test improvements in controlled experiments
    4. Apply successful improvements system-wide
    5. Monitor ongoing performance

    This creates a flywheel effect where the system gets
    progressively better through use.
    """

    def __init__(
        self,
        feedback_collector: FeedbackCollector,
        model_caller: Optional[Callable] = None,
        improvement_threshold: float = 0.1,   # Min improvement to apply a change
        min_feedback_for_cycle: int = 20,
        cycle_interval_seconds: float = 3600,  # 1 hour
    ):
        self.feedback = feedback_collector
        self._model_caller = model_caller
        self.improvement_threshold = improvement_threshold
        self.min_feedback_for_cycle = min_feedback_for_cycle
        self.cycle_interval = cycle_interval_seconds

        self._cycles: List[ImprovementCycle] = []
        self._applied_prompts: Dict[str, str] = {}      # domain -> optimized prompt
        self._preferred_strategies: Dict[str, str] = {} # domain -> best strategy
        self._few_shot_examples: Dict[str, List[Feedback]] = {}  # domain -> examples
        self._running = False
        self._cycle_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the continuous improvement loop."""
        self._running = True
        self._cycle_task = asyncio.create_task(self._improvement_loop())
        logger.info("SelfImprover: continuous improvement loop started")

    async def stop(self) -> None:
        """Stop the improvement loop."""
        self._running = False
        if self._cycle_task:
            self._cycle_task.cancel()
            try:
                await self._cycle_task
            except asyncio.CancelledError:
                pass

    async def run_cycle(self) -> ImprovementCycle:
        """
        Manually trigger one improvement cycle.

        Returns:
            ImprovementCycle with results
        """
        summary = self.feedback.summarize()

        cycle = ImprovementCycle(
            cycle_id=len(self._cycles) + 1,
            started_at=time.time(),
            feedback_analyzed=summary.total_feedback,
            actions_taken=[],
            baseline_score=summary.avg_score,
            target_score=min(1.0, summary.avg_score + 0.1),
        )

        if summary.total_feedback < self.min_feedback_for_cycle:
            logger.info(
                f"Not enough feedback ({summary.total_feedback}/{self.min_feedback_for_cycle}) "
                "to run improvement cycle"
            )
            cycle.completed_at = time.time()
            self._cycles.append(cycle)
            return cycle

        logger.info(
            f"Running improvement cycle {cycle.cycle_id} "
            f"(baseline={cycle.baseline_score:.2f})"
        )

        # Run all improvement strategies
        actions = await asyncio.gather(
            self._improve_strategy_selection(summary),
            self._update_few_shot_examples(summary),
            self._identify_error_patterns(summary),
            return_exceptions=True,
        )

        for result in actions:
            if isinstance(result, Exception):
                logger.warning(f"Improvement action failed: {result}")
            elif isinstance(result, list):
                cycle.actions_taken.extend(result)
            elif isinstance(result, ImprovementAction):
                cycle.actions_taken.append(result)

        cycle.completed_at = time.time()
        self._cycles.append(cycle)

        logger.info(
            f"Cycle {cycle.cycle_id} complete: "
            f"{len(cycle.actions_taken)} actions taken"
        )
        return cycle

    def get_optimized_prompt(self, domain: str, base_prompt: str) -> str:
        """
        Get the optimized system prompt for a domain.

        Returns the domain-specific optimized prompt if available,
        otherwise the base prompt.
        """
        return self._applied_prompts.get(domain, base_prompt)

    def get_preferred_strategy(self, domain: str, default: str = "cot") -> str:
        """Get the empirically best reasoning strategy for a domain."""
        return self._preferred_strategies.get(domain, default)

    def get_few_shot_examples(self, domain: str, n: int = 3) -> List[Feedback]:
        """Get the best few-shot examples for a domain."""
        return self._few_shot_examples.get(domain, [])[:n]

    def build_few_shot_prompt(self, domain: str, query: str) -> str:
        """
        Build a prompt augmented with few-shot examples from feedback.

        Returns a prompt with high-quality examples prepended.
        """
        examples = self.get_few_shot_examples(domain, n=3)
        if not examples:
            return query

        example_text = "\n\n".join(
            f"Q: {ex.query}\nA: {ex.response}"
            for ex in examples
            if ex.query and ex.response
        )

        return f"""Here are some examples of high-quality responses:

{example_text}

Now answer the following:
Q: {query}
A:"""

    def get_improvement_report(self) -> Dict[str, Any]:
        """Get a comprehensive improvement report."""
        if not self._cycles:
            return {"cycles_completed": 0, "status": "No cycles run yet"}

        latest = self._cycles[-1]
        total_actions = sum(len(c.actions_taken) for c in self._cycles)
        improvements = [
            c.improvement_delta
            for c in self._cycles
            if c.improvement_delta is not None
        ]

        return {
            "cycles_completed": len(self._cycles),
            "total_actions_taken": total_actions,
            "current_preferred_strategies": dict(self._preferred_strategies),
            "domains_with_examples": list(self._few_shot_examples.keys()),
            "domains_with_optimized_prompts": list(self._applied_prompts.keys()),
            "avg_improvement_per_cycle": sum(improvements) / len(improvements) if improvements else None,
            "latest_cycle": {
                "id": latest.cycle_id,
                "feedback_analyzed": latest.feedback_analyzed,
                "actions_taken": len(latest.actions_taken),
                "baseline": latest.baseline_score,
            },
        }

    # =========================================================================
    # Improvement strategies
    # =========================================================================

    async def _improve_strategy_selection(
        self, summary: FeedbackSummary
    ) -> List[ImprovementAction]:
        """Update strategy preferences based on empirical performance data."""
        actions = []

        for strategy, score in summary.strategy_scores.items():
            # Find domains where this strategy excels
            domain_strategy_scores: Dict[str, Dict[str, List[float]]] = {}
            for fb in self.feedback._feedback:
                if fb.strategy_used == strategy and fb.domain:
                    if fb.domain not in domain_strategy_scores:
                        domain_strategy_scores[fb.domain] = {}
                    if strategy not in domain_strategy_scores[fb.domain]:
                        domain_strategy_scores[fb.domain][strategy] = []
                    domain_strategy_scores[fb.domain][strategy].append(fb.score)

            for domain, strategy_data in domain_strategy_scores.items():
                for strat, scores in strategy_data.items():
                    if len(scores) >= 5:
                        avg = sum(scores) / len(scores)
                        current = self._preferred_strategies.get(domain)

                        # Check if this strategy beats current preference
                        current_scores = [
                            f.score for f in self.feedback._feedback
                            if f.domain == domain and f.strategy_used == current
                        ]
                        current_avg = sum(current_scores) / len(current_scores) if current_scores else 0.5

                        if avg > current_avg + self.improvement_threshold:
                            self._preferred_strategies[domain] = strat
                            actions.append(ImprovementAction(
                                action_type="update_strategy",
                                description=f"Switching domain '{domain}' from '{current}' to '{strat}'",
                                payload={"domain": domain, "old": current, "new": strat},
                                expected_improvement=avg - current_avg,
                                confidence=min(0.9, len(scores) / 20),
                                applied_at=time.time(),
                            ))

        return actions

    async def _update_few_shot_examples(
        self, summary: FeedbackSummary
    ) -> List[ImprovementAction]:
        """Update few-shot examples with the best recent feedback."""
        actions = []
        best_examples = self.feedback.get_best_examples(n=20, min_score=0.85)

        # Group by domain
        by_domain: Dict[str, List[Feedback]] = {}
        for ex in best_examples:
            if ex.domain not in by_domain:
                by_domain[ex.domain] = []
            by_domain[ex.domain].append(ex)

        for domain, examples in by_domain.items():
            # Sort by score, take top 5
            top = sorted(examples, key=lambda e: e.score, reverse=True)[:5]
            old_count = len(self._few_shot_examples.get(domain, []))
            self._few_shot_examples[domain] = top

            if len(top) > old_count:
                actions.append(ImprovementAction(
                    action_type="update_examples",
                    description=f"Updated {len(top)} few-shot examples for domain '{domain}'",
                    payload={"domain": domain, "example_count": len(top)},
                    expected_improvement=0.05,
                    confidence=0.7,
                    applied_at=time.time(),
                ))

        return actions

    async def _identify_error_patterns(
        self, summary: FeedbackSummary
    ) -> List[ImprovementAction]:
        """Identify recurring error patterns and generate fixes."""
        actions = []
        worst = self.feedback.get_worst_examples(n=10, max_score=0.3)

        if not worst or self._model_caller is None:
            return actions

        # Cluster error examples by query similarity
        error_texts = [f"Query: {e.query}\nResponse: {e.response[:200]}" for e in worst]

        if len(error_texts) >= 3:
            analysis_prompt = f"""Analyze these failed AI responses and identify the common error patterns.

Failed Examples:
{chr(10).join(error_texts[:5])}

Identify:
1. The main failure mode (factual error, misunderstanding, incomplete, etc.)
2. A specific fix (e.g., "Add disclaimer for medical topics", "Use deductive reasoning for math")
3. Which domains are most affected

Format your response as:
PATTERN: [description]
FIX: [specific action to take]
DOMAINS: [affected domains]"""

            try:
                analysis = await self._model_caller(analysis_prompt)
                actions.append(ImprovementAction(
                    action_type="error_pattern_identified",
                    description="Recurring error patterns analyzed",
                    payload={"analysis": analysis, "sample_size": len(worst)},
                    expected_improvement=0.08,
                    confidence=0.6,
                    applied_at=time.time(),
                ))
            except Exception as e:
                logger.warning(f"Error pattern analysis failed: {e}")

        return actions

    async def _improvement_loop(self) -> None:
        """Background loop that periodically runs improvement cycles."""
        while self._running:
            try:
                await asyncio.sleep(self.cycle_interval)
                if self._running:
                    await self.run_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Improvement loop error: {e}")
                await asyncio.sleep(60)  # Back off on error
