"""
FeedbackCollector - Collect, store, and analyze user feedback
=============================================================

Tracks quality signals to enable continuous improvement:
- Explicit ratings (thumbs up/down, 1-5 stars)
- Implicit signals (follow-up questions, corrections)
- Task completion rates
- Latency and cost preferences
- Per-domain calibration
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    EXPLICIT_POSITIVE = "explicit_positive"
    EXPLICIT_NEGATIVE = "explicit_negative"
    CORRECTION = "correction"          # User corrected the response
    FOLLOW_UP = "follow_up"            # User asked a follow-up (may indicate confusion)
    TASK_COMPLETE = "task_complete"    # Task was successfully completed
    TASK_FAILED = "task_failed"        # Task was not completed
    PREFERENCE = "preference"          # User expressed a preference


@dataclass
class Feedback:
    """A single piece of feedback for a response."""
    feedback_id: str
    interaction_id: str              # Links to the original interaction
    feedback_type: FeedbackType
    score: float                     # Normalized 0.0 - 1.0
    query: str = ""
    response: str = ""
    correction: Optional[str] = None # What the correct response should be
    comment: Optional[str] = None
    domain: str = "general"
    strategy_used: Optional[str] = None
    model_used: Optional[str] = None
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackSummary:
    """Aggregated feedback statistics."""
    total_feedback: int
    avg_score: float
    positive_rate: float
    correction_rate: float
    domain_scores: Dict[str, float]
    strategy_scores: Dict[str, float]
    model_scores: Dict[str, float]
    trend: str  # "improving" | "stable" | "declining"
    recommendations: List[str]


class FeedbackCollector:
    """
    Collects and analyzes feedback to drive continuous improvement.

    Aggregates signals across interactions, identifies patterns,
    and generates actionable recommendations for the system.
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._feedback: List[Feedback] = []
        self._counter = 0

        # Running statistics
        self._domain_stats: Dict[str, List[float]] = defaultdict(list)
        self._strategy_stats: Dict[str, List[float]] = defaultdict(list)
        self._model_stats: Dict[str, List[float]] = defaultdict(list)

    def record(
        self,
        query: str,
        response: str,
        score: float,
        feedback_type: FeedbackType = FeedbackType.EXPLICIT_POSITIVE,
        correction: Optional[str] = None,
        comment: Optional[str] = None,
        domain: str = "general",
        strategy_used: Optional[str] = None,
        model_used: Optional[str] = None,
        latency_ms: float = 0.0,
        interaction_id: Optional[str] = None,
        **metadata,
    ) -> Feedback:
        """
        Record a piece of feedback.

        Args:
            query: The original user query
            response: The system response
            score: Quality score 0.0 (bad) to 1.0 (excellent)
            feedback_type: Type of feedback signal
            correction: The correct response if this is a correction
            comment: Free-text comment
            domain: Domain category
            strategy_used: Which reasoning strategy was used
            model_used: Which model generated the response
            latency_ms: Response latency

        Returns:
            Recorded Feedback object
        """
        score = max(0.0, min(1.0, score))

        fid = f"fb_{self._counter:08d}"
        self._counter += 1
        iid = interaction_id or fid

        fb = Feedback(
            feedback_id=fid,
            interaction_id=iid,
            feedback_type=feedback_type,
            score=score,
            query=query,
            response=response,
            correction=correction,
            comment=comment,
            domain=domain,
            strategy_used=strategy_used,
            model_used=model_used,
            latency_ms=latency_ms,
            metadata=metadata,
        )

        self._feedback.append(fb)

        # Update running stats
        self._domain_stats[domain].append(score)
        if strategy_used:
            self._strategy_stats[strategy_used].append(score)
        if model_used:
            self._model_stats[model_used].append(score)

        # Maintain sliding window
        if len(self._feedback) > self.window_size:
            oldest = self._feedback.pop(0)
            # Note: rolling stats aren't removed here for simplicity

        logger.debug(f"Feedback recorded: {feedback_type.value} score={score:.2f} domain={domain}")
        return fb

    def thumbs_up(self, query: str, response: str, **kwargs) -> Feedback:
        """Record a positive signal (thumbs up / 5 stars)."""
        return self.record(
            query, response, score=1.0,
            feedback_type=FeedbackType.EXPLICIT_POSITIVE, **kwargs
        )

    def thumbs_down(self, query: str, response: str, **kwargs) -> Feedback:
        """Record a negative signal (thumbs down / 1 star)."""
        return self.record(
            query, response, score=0.0,
            feedback_type=FeedbackType.EXPLICIT_NEGATIVE, **kwargs
        )

    def correction(self, query: str, bad_response: str, good_response: str, **kwargs) -> Feedback:
        """Record a correction - the model got it wrong."""
        return self.record(
            query, bad_response, score=0.1,
            feedback_type=FeedbackType.CORRECTION,
            correction=good_response, **kwargs
        )

    def rate(self, query: str, response: str, stars: int, **kwargs) -> Feedback:
        """Record a 1-5 star rating."""
        score = (stars - 1) / 4.0
        fb_type = (
            FeedbackType.EXPLICIT_POSITIVE if stars >= 4
            else FeedbackType.EXPLICIT_NEGATIVE if stars <= 2
            else FeedbackType.PREFERENCE
        )
        return self.record(query, response, score=score, feedback_type=fb_type, **kwargs)

    def summarize(self, recent_n: Optional[int] = None) -> FeedbackSummary:
        """
        Generate a comprehensive feedback summary.

        Args:
            recent_n: Only analyze the most recent N feedback items

        Returns:
            FeedbackSummary with statistics and recommendations
        """
        data = self._feedback[-recent_n:] if recent_n else self._feedback

        if not data:
            return FeedbackSummary(
                total_feedback=0,
                avg_score=0.0,
                positive_rate=0.0,
                correction_rate=0.0,
                domain_scores={},
                strategy_scores={},
                model_scores={},
                trend="unknown",
                recommendations=["No feedback collected yet"],
            )

        scores = [f.score for f in data]
        avg_score = sum(scores) / len(scores)
        positive_rate = sum(1 for f in data if f.feedback_type == FeedbackType.EXPLICIT_POSITIVE) / len(data)
        correction_rate = sum(1 for f in data if f.feedback_type == FeedbackType.CORRECTION) / len(data)

        # Domain scores
        domain_scores = {}
        for domain, domain_fb in defaultdict(list, {
            f.domain: [] for f in data
        }).items():
            domain_scores[domain] = sum(
                f.score for f in data if f.domain == domain
            ) / max(1, sum(1 for f in data if f.domain == domain))

        # Strategy scores
        strategy_scores = {}
        for f in data:
            if f.strategy_used:
                if f.strategy_used not in strategy_scores:
                    strategy_scores[f.strategy_used] = []
                strategy_scores[f.strategy_used].append(f.score)
        strategy_scores = {
            k: sum(v) / len(v) for k, v in strategy_scores.items()
        }

        # Model scores
        model_scores = {}
        for f in data:
            if f.model_used:
                if f.model_used not in model_scores:
                    model_scores[f.model_used] = []
                model_scores[f.model_used].append(f.score)
        model_scores = {
            k: sum(v) / len(v) for k, v in model_scores.items()
        }

        # Trend (compare first half vs second half)
        trend = "stable"
        if len(scores) >= 10:
            mid = len(scores) // 2
            first_half = sum(scores[:mid]) / mid
            second_half = sum(scores[mid:]) / (len(scores) - mid)
            delta = second_half - first_half
            if delta > 0.05:
                trend = "improving"
            elif delta < -0.05:
                trend = "declining"

        # Generate recommendations
        recommendations = self._generate_recommendations(
            avg_score, correction_rate, strategy_scores, model_scores
        )

        return FeedbackSummary(
            total_feedback=len(data),
            avg_score=avg_score,
            positive_rate=positive_rate,
            correction_rate=correction_rate,
            domain_scores=domain_scores,
            strategy_scores=strategy_scores,
            model_scores=model_scores,
            trend=trend,
            recommendations=recommendations,
        )

    def get_corrections(self, domain: Optional[str] = None) -> List[Tuple[str, str, str]]:
        """
        Get (query, bad_response, correction) tuples for fine-tuning.

        Returns:
            List of (query, bad_response, correct_response) tuples
        """
        corrections = [
            (f.query, f.response, f.correction)
            for f in self._feedback
            if f.feedback_type == FeedbackType.CORRECTION and f.correction
        ]
        if domain:
            corrections = [
                c for f, c in zip(
                    [f for f in self._feedback if f.feedback_type == FeedbackType.CORRECTION],
                    corrections,
                )
                if f.domain == domain
            ]
        return corrections

    def get_best_examples(self, n: int = 10, min_score: float = 0.9) -> List[Feedback]:
        """Get the highest-quality feedback examples (for few-shot learning)."""
        good = [f for f in self._feedback if f.score >= min_score and f.response]
        return sorted(good, key=lambda f: f.score, reverse=True)[:n]

    def get_worst_examples(self, n: int = 10, max_score: float = 0.3) -> List[Feedback]:
        """Get the lowest-quality examples (for identifying failure modes)."""
        bad = [f for f in self._feedback if f.score <= max_score and f.query]
        return sorted(bad, key=lambda f: f.score)[:n]

    def get_domain_weakness(self) -> Optional[str]:
        """Find the domain where performance is worst."""
        if not self._domain_stats:
            return None
        domain_avgs = {
            d: sum(scores) / len(scores)
            for d, scores in self._domain_stats.items()
            if len(scores) >= 3  # Need at least 3 samples
        }
        if not domain_avgs:
            return None
        return min(domain_avgs, key=domain_avgs.get)  # type: ignore

    def _generate_recommendations(
        self,
        avg_score: float,
        correction_rate: float,
        strategy_scores: Dict[str, float],
        model_scores: Dict[str, float],
    ) -> List[str]:
        """Generate actionable recommendations from feedback patterns."""
        recs = []

        if avg_score < 0.6:
            recs.append("Overall quality is low - consider increasing reflection depth")

        if correction_rate > 0.2:
            recs.append(f"High correction rate ({correction_rate:.0%}) - review failure cases and improve factual accuracy")

        if strategy_scores:
            best_strategy = max(strategy_scores, key=strategy_scores.get)  # type: ignore
            worst_strategy = min(strategy_scores, key=strategy_scores.get)  # type: ignore
            if strategy_scores[best_strategy] - strategy_scores.get(worst_strategy, 0) > 0.15:
                recs.append(f"Strategy '{best_strategy}' outperforms '{worst_strategy}' - use it more often")

        if model_scores:
            best_model = max(model_scores, key=model_scores.get)  # type: ignore
            recs.append(f"Best performing model: {best_model} (score={model_scores[best_model]:.2f})")

        weak_domain = self.get_domain_weakness()
        if weak_domain:
            recs.append(f"Weakest domain: '{weak_domain}' - consider domain-specific prompts or training")

        if not recs:
            recs.append("Performance looks good! Keep collecting feedback.")

        return recs
