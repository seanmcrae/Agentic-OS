"""
MetacognitionModule - Thinking about thinking
==============================================

Implements metacognitive capabilities: monitoring cognitive processes,
recognizing uncertainty, calibrating confidence, detecting reasoning
errors, and adaptively selecting strategies.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CognitiveState(str, Enum):
    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"
    CONFUSED = "confused"
    EXPLORING = "exploring"
    FOCUSED = "focused"
    OVERLOADED = "overloaded"


class ReasoningStrategy(str, Enum):
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    ANALOGICAL = "analogical"
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    CREATIVE = "creative"
    SYSTEMATIC = "systematic"


@dataclass
class CognitiveMeasure:
    """A measurement of cognitive state at a point in time."""
    state: CognitiveState
    confidence: float  # 0.0 - 1.0
    uncertainty_sources: List[str]
    active_strategy: ReasoningStrategy
    cognitive_load: float  # 0.0 - 1.0
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyEvaluation:
    """Evaluation of a reasoning strategy's effectiveness."""
    strategy: ReasoningStrategy
    success_rate: float
    avg_quality: float
    avg_time_seconds: float
    use_count: int
    best_for: List[str]  # task types this strategy excels at


class MetacognitionModule:
    """
    Metacognitive awareness and adaptive strategy selection.

    Monitors the quality of reasoning processes, detects when current
    approaches are failing, and dynamically selects better strategies.
    Maintains calibrated confidence estimates across different domains.
    """

    # Strategy selection heuristics based on task characteristics
    STRATEGY_HEURISTICS: Dict[str, List[ReasoningStrategy]] = {
        "mathematical": [ReasoningStrategy.DEDUCTIVE, ReasoningStrategy.SYSTEMATIC],
        "creative": [ReasoningStrategy.CREATIVE, ReasoningStrategy.ANALOGICAL],
        "causal": [ReasoningStrategy.ABDUCTIVE, ReasoningStrategy.INDUCTIVE],
        "complex": [ReasoningStrategy.TREE_OF_THOUGHT, ReasoningStrategy.CHAIN_OF_THOUGHT],
        "simple": [ReasoningStrategy.CHAIN_OF_THOUGHT, ReasoningStrategy.DEDUCTIVE],
        "uncertain": [ReasoningStrategy.INDUCTIVE, ReasoningStrategy.ABDUCTIVE],
        "analogical": [ReasoningStrategy.ANALOGICAL, ReasoningStrategy.INDUCTIVE],
    }

    def __init__(self):
        self._cognitive_state = CognitiveState.EXPLORING
        self._confidence = 0.7
        self._cognitive_load = 0.3

        # Strategy performance tracking
        self._strategy_history: Dict[ReasoningStrategy, List[Tuple[float, float]]] = defaultdict(list)

        # Domain-specific confidence calibration
        self._domain_confidence: Dict[str, float] = {}

        # Cognitive measures history
        self._measures: List[CognitiveMeasure] = []

        # Uncertainty sources currently active
        self._active_uncertainties: List[str] = []

        # Error pattern detection
        self._error_patterns: Dict[str, int] = defaultdict(int)

    def assess(self, context: Dict[str, Any]) -> CognitiveMeasure:
        """
        Perform a metacognitive assessment of the current cognitive state.

        Args:
            context: Current task context and information

        Returns:
            CognitiveMeasure capturing the current state
        """
        # Update cognitive state based on context
        self._update_cognitive_state(context)

        # Select appropriate strategy
        strategy = self.select_strategy(context)

        measure = CognitiveMeasure(
            state=self._cognitive_state,
            confidence=self._confidence,
            uncertainty_sources=list(self._active_uncertainties),
            active_strategy=strategy,
            cognitive_load=self._cognitive_load,
            context=context,
        )

        self._measures.append(measure)
        return measure

    def select_strategy(self, task_context: Dict[str, Any]) -> ReasoningStrategy:
        """
        Adaptively select the best reasoning strategy for the current task.

        Considers:
        - Task characteristics (complexity, domain, type)
        - Historical strategy performance
        - Current cognitive state
        """
        task_type = task_context.get("type", "complex")
        complexity = task_context.get("complexity", 0.5)
        domain = task_context.get("domain", "general")

        # Get candidate strategies from heuristics
        candidates = self.STRATEGY_HEURISTICS.get(task_type, [ReasoningStrategy.CHAIN_OF_THOUGHT])

        # Factor in complexity
        if complexity > 0.8:
            if ReasoningStrategy.TREE_OF_THOUGHT not in candidates:
                candidates.insert(0, ReasoningStrategy.TREE_OF_THOUGHT)
        elif complexity < 0.3:
            if ReasoningStrategy.CHAIN_OF_THOUGHT not in candidates:
                candidates.insert(0, ReasoningStrategy.CHAIN_OF_THOUGHT)

        # Factor in current cognitive state
        if self._cognitive_state == CognitiveState.OVERLOADED:
            return ReasoningStrategy.SYSTEMATIC  # Fall back to systematic
        elif self._cognitive_state == CognitiveState.CONFUSED:
            return ReasoningStrategy.CHAIN_OF_THOUGHT  # Start from scratch

        # Use performance history to pick best candidate
        best_strategy = candidates[0]
        best_score = 0.0

        for strategy in candidates:
            history = self._strategy_history.get(strategy, [])
            if history:
                avg_score = sum(q for _, q in history) / len(history)
                if avg_score > best_score:
                    best_score = avg_score
                    best_strategy = strategy

        logger.debug(f"Selected strategy: {best_strategy} for task type: {task_type}")
        return best_strategy

    def record_outcome(
        self,
        strategy: ReasoningStrategy,
        quality: float,
        time_taken: float,
        domain: Optional[str] = None,
    ) -> None:
        """
        Record the outcome of a reasoning attempt for future strategy selection.

        Args:
            strategy: The strategy that was used
            quality: Quality score 0.0-1.0
            time_taken: Time in seconds
            domain: Optional domain for calibration
        """
        self._strategy_history[strategy].append((time_taken, quality))

        # Keep only recent history (last 100 outcomes per strategy)
        if len(self._strategy_history[strategy]) > 100:
            self._strategy_history[strategy] = self._strategy_history[strategy][-100:]

        # Update domain confidence
        if domain:
            if domain not in self._domain_confidence:
                self._domain_confidence[domain] = quality
            else:
                # Exponential moving average
                self._domain_confidence[domain] = (
                    0.9 * self._domain_confidence[domain] + 0.1 * quality
                )

        # Update overall confidence
        self._confidence = 0.8 * self._confidence + 0.2 * quality

    def flag_uncertainty(self, source: str, description: str) -> None:
        """Flag a source of uncertainty."""
        uncertainty = f"{source}: {description}"
        if uncertainty not in self._active_uncertainties:
            self._active_uncertainties.append(uncertainty)
        self._cognitive_state = CognitiveState.UNCERTAIN
        logger.debug(f"Uncertainty flagged: {uncertainty}")

    def clear_uncertainty(self, source: Optional[str] = None) -> None:
        """Clear uncertainty flags, optionally for a specific source."""
        if source:
            self._active_uncertainties = [
                u for u in self._active_uncertainties if not u.startswith(source)
            ]
        else:
            self._active_uncertainties.clear()

        if not self._active_uncertainties:
            self._cognitive_state = CognitiveState.CONFIDENT

    def detect_cognitive_bias(self, reasoning_trace: List[str]) -> List[str]:
        """
        Analyze a reasoning trace for common cognitive biases.

        Args:
            reasoning_trace: List of reasoning steps

        Returns:
            List of detected biases with descriptions
        """
        biases = []
        full_text = " ".join(reasoning_trace).lower()

        # Confirmation bias: only considering supporting evidence
        if "confirms" in full_text and "contradicts" not in full_text:
            biases.append("Potential confirmation bias: only confirming evidence considered")

        # Anchoring: over-relying on first information
        if reasoning_trace and len(reasoning_trace) > 3:
            first_step = reasoning_trace[0].lower()
            if sum(first_step in step.lower() for step in reasoning_trace[1:]) > len(reasoning_trace) / 2:
                biases.append("Potential anchoring bias: over-reliance on initial information")

        # Availability heuristic: over-weighting recent/memorable information
        recent_count = sum(1 for r in reasoning_trace if "recent" in r.lower() or "common" in r.lower())
        if recent_count > len(reasoning_trace) * 0.4:
            biases.append("Potential availability heuristic: over-weighting salient information")

        # Overconfidence
        certainty_words = ["definitely", "certainly", "obviously", "clearly", "always", "never"]
        certainty_count = sum(1 for word in certainty_words if word in full_text)
        if certainty_count > 3:
            biases.append(f"Potential overconfidence: {certainty_count} absolute certainty expressions")

        if biases:
            self._error_patterns["cognitive_bias"] += len(biases)

        return biases

    def get_calibration_report(self) -> Dict[str, Any]:
        """Get a confidence calibration report across domains and strategies."""
        strategy_performance = {}
        for strategy, history in self._strategy_history.items():
            if history:
                times, qualities = zip(*history)
                strategy_performance[strategy.value] = {
                    "avg_quality": sum(qualities) / len(qualities),
                    "avg_time": sum(times) / len(times),
                    "use_count": len(history),
                }

        return {
            "current_state": self._cognitive_state.value,
            "overall_confidence": self._confidence,
            "cognitive_load": self._cognitive_load,
            "domain_confidence": dict(self._domain_confidence),
            "strategy_performance": strategy_performance,
            "active_uncertainties": list(self._active_uncertainties),
            "error_pattern_counts": dict(self._error_patterns),
            "total_assessments": len(self._measures),
        }

    def _update_cognitive_state(self, context: Dict[str, Any]) -> None:
        """Update the cognitive state based on context signals."""
        uncertainty_count = len(self._active_uncertainties)
        task_complexity = context.get("complexity", 0.5)
        available_info = context.get("information_richness", 0.5)

        # Determine cognitive load
        self._cognitive_load = min(1.0, task_complexity * 0.6 + uncertainty_count * 0.1)

        if self._cognitive_load > 0.85:
            self._cognitive_state = CognitiveState.OVERLOADED
        elif uncertainty_count > 3:
            self._cognitive_state = CognitiveState.CONFUSED
        elif uncertainty_count > 1:
            self._cognitive_state = CognitiveState.UNCERTAIN
        elif task_complexity > 0.7:
            self._cognitive_state = CognitiveState.EXPLORING
        elif available_info > 0.7 and self._confidence > 0.8:
            self._cognitive_state = CognitiveState.CONFIDENT
        else:
            self._cognitive_state = CognitiveState.FOCUSED
