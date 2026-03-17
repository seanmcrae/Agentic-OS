"""
ModelRouter - Intelligent multi-model routing
=============================================

Routes requests to the optimal model based on:
- Task type and complexity
- Cost constraints
- Performance requirements
- Provider availability
- Load balancing
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from enum import Enum
import logging

from dzogchen.models.base import BaseModelAdapter, ModelMessage, ModelResponse, ToolDefinition

logger = logging.getLogger(__name__)


class RoutingStrategy(str, Enum):
    PERFORMANCE = "performance"    # Best quality, regardless of cost
    ECONOMY = "economy"            # Lowest cost that meets requirements
    BALANCED = "balanced"          # Balance quality and cost
    FASTEST = "fastest"            # Lowest latency
    FALLBACK = "fallback"          # Primary with automatic fallback


@dataclass
class ModelScore:
    """Score for a model candidate on a routing decision."""
    adapter: BaseModelAdapter
    quality_score: float
    cost_score: float
    speed_score: float
    availability: bool = True

    @property
    def weighted_score(self) -> float:
        if not self.availability:
            return 0.0
        return (self.quality_score * 0.5 + self.cost_score * 0.3 + self.speed_score * 0.2)


@dataclass
class RoutingDecision:
    """A routing decision with rationale."""
    selected_adapter: BaseModelAdapter
    strategy: RoutingStrategy
    scores: List[ModelScore]
    rationale: str
    estimated_cost: float
    timestamp: float = field(default_factory=time.time)


class ModelRouter:
    """
    Intelligent router that selects the best model for each request.

    Tracks model performance, costs, and availability to make
    optimal routing decisions. Supports fallback chains for reliability.
    """

    def __init__(
        self,
        primary: Optional[BaseModelAdapter] = None,
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        max_cost_per_request: Optional[float] = None,
        fallback_chain: Optional[List[BaseModelAdapter]] = None,
    ):
        self._adapters: List[BaseModelAdapter] = []
        self._primary = primary
        self.strategy = strategy
        self.max_cost_per_request = max_cost_per_request
        self._fallback_chain: List[BaseModelAdapter] = fallback_chain or []

        if primary:
            self._adapters.append(primary)
        if fallback_chain:
            self._adapters.extend(fallback_chain)

        # Performance tracking per model
        self._model_stats: Dict[str, Dict[str, float]] = {}
        self._routing_history: List[RoutingDecision] = []
        self._failure_counts: Dict[str, int] = {}
        self._circuit_breaker_until: Dict[str, float] = {}

    def add_model(self, adapter: BaseModelAdapter, is_primary: bool = False) -> None:
        """Register a model adapter."""
        if adapter not in self._adapters:
            self._adapters.append(adapter)
        if is_primary:
            self._primary = adapter
        logger.info(f"Added model: {adapter.model_id}")

    async def route(
        self,
        messages: List[ModelMessage],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tools: Optional[List[ToolDefinition]] = None,
        task_type: Optional[str] = None,
        required_quality: float = 0.7,
        stream: bool = False,
    ) -> ModelResponse:
        """
        Route a request to the optimal model and execute it.

        Args:
            messages: Conversation messages
            system_prompt: System instructions
            max_tokens: Maximum output tokens
            temperature: Sampling temperature
            tools: Available tools
            task_type: Type of task for routing hints
            required_quality: Minimum quality requirement (0.0-1.0)
            stream: Whether to stream the response

        Returns:
            Model response
        """
        # Select best adapter
        adapter, decision = self._select_adapter(task_type, required_quality)
        self._routing_history.append(decision)

        logger.debug(f"Routing to {adapter.model_id} (strategy={self.strategy.value})")

        # Try primary, fall back to chain on failure
        for attempt_adapter in [adapter] + self._get_fallback_chain(adapter):
            if self._is_circuit_broken(attempt_adapter.model_id):
                logger.warning(f"Circuit breaker open for {attempt_adapter.model_id}, skipping")
                continue

            try:
                response = await attempt_adapter.generate(
                    messages=messages,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    tools=tools,
                )
                self._record_success(attempt_adapter.model_id, response)
                return response

            except Exception as e:
                logger.warning(f"Model {attempt_adapter.model_id} failed: {e}")
                self._record_failure(attempt_adapter.model_id)

        raise RuntimeError(
            f"All models failed for request. Tried: {adapter.model_id} + {len(self._fallback_chain)} fallbacks"
        )

    async def stream_route(
        self,
        messages: List[ModelMessage],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        task_type: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Route and stream a response."""
        adapter, _ = self._select_adapter(task_type, required_quality=0.5)

        try:
            async for chunk in adapter.stream_generate(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Streaming failed for {adapter.model_id}: {e}")
            raise

    async def simple_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        task_type: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Simple prompt routing - returns just the text."""
        from dzogchen.models.base import MessageRole
        messages = [ModelMessage(role=MessageRole.USER, content=prompt)]
        response = await self.route(
            messages=messages,
            system_prompt=system_prompt,
            task_type=task_type,
            **kwargs,
        )
        return response.content

    def _select_adapter(
        self,
        task_type: Optional[str],
        required_quality: float,
    ) -> Tuple[BaseModelAdapter, RoutingDecision]:
        """Select the best adapter based on strategy and context."""
        if not self._adapters:
            raise ValueError("No model adapters registered")

        if len(self._adapters) == 1:
            adapter = self._adapters[0]
            decision = RoutingDecision(
                selected_adapter=adapter,
                strategy=self.strategy,
                scores=[],
                rationale="Only one model available",
                estimated_cost=0.0,
            )
            return adapter, decision

        # Score all available adapters
        scores = [self._score_adapter(a, task_type, required_quality) for a in self._adapters]

        # Filter by circuit breaker
        available_scores = [s for s in scores if s.availability and not self._is_circuit_broken(s.adapter.model_id)]

        if not available_scores:
            logger.warning("All models are circuit-broken, using primary")
            available_scores = scores[:1]

        # Select based on strategy
        if self.strategy == RoutingStrategy.PERFORMANCE:
            best = max(available_scores, key=lambda s: s.quality_score)
        elif self.strategy == RoutingStrategy.ECONOMY:
            best = max(available_scores, key=lambda s: s.cost_score)
        elif self.strategy == RoutingStrategy.FASTEST:
            best = max(available_scores, key=lambda s: s.speed_score)
        else:  # BALANCED or FALLBACK
            best = max(available_scores, key=lambda s: s.weighted_score)

        rationale = (
            f"Selected {best.adapter.model_id} with quality={best.quality_score:.2f}, "
            f"cost_score={best.cost_score:.2f}, speed={best.speed_score:.2f}"
        )

        decision = RoutingDecision(
            selected_adapter=best.adapter,
            strategy=self.strategy,
            scores=scores,
            rationale=rationale,
            estimated_cost=0.0,
        )

        return best.adapter, decision

    def _score_adapter(
        self,
        adapter: BaseModelAdapter,
        task_type: Optional[str],
        required_quality: float,
    ) -> ModelScore:
        """Score an adapter for the given request."""
        model_id = adapter.model_id

        # Quality scores based on known model capabilities
        quality_map = {
            "claude-opus-4-6": 0.98,
            "claude-sonnet-4-6": 0.92,
            "claude-haiku-4-5-20251001": 0.82,
            "gpt-4o": 0.93,
            "gpt-4o-mini": 0.80,
            "o1-preview": 0.97,
        }

        # Cost efficiency (higher = cheaper)
        cost_map = {
            "claude-haiku-4-5-20251001": 0.95,
            "gpt-4o-mini": 0.93,
            "claude-sonnet-4-6": 0.70,
            "gpt-4o": 0.65,
            "claude-opus-4-6": 0.40,
            "o1-preview": 0.30,
        }

        # Speed scores (higher = faster)
        speed_map = {
            "claude-haiku-4-5-20251001": 0.95,
            "gpt-4o-mini": 0.90,
            "claude-sonnet-4-6": 0.80,
            "gpt-4o": 0.75,
            "claude-opus-4-6": 0.65,
        }

        # Use historical data if available
        stats = self._model_stats.get(model_id, {})
        quality = stats.get("quality", quality_map.get(model_id, 0.75))

        return ModelScore(
            adapter=adapter,
            quality_score=quality,
            cost_score=cost_map.get(model_id, 0.5),
            speed_score=speed_map.get(model_id, 0.5),
            availability=quality >= required_quality,
        )

    def _get_fallback_chain(self, primary: BaseModelAdapter) -> List[BaseModelAdapter]:
        """Get the fallback chain excluding the primary adapter."""
        if self._fallback_chain:
            return [a for a in self._fallback_chain if a != primary]
        return [a for a in self._adapters if a != primary]

    def _record_success(self, model_id: str, response: ModelResponse) -> None:
        """Record a successful request for performance tracking."""
        if model_id not in self._model_stats:
            self._model_stats[model_id] = {"quality": 0.75, "latency": 1000, "success_rate": 1.0}

        stats = self._model_stats[model_id]
        # Exponential moving average
        stats["latency"] = 0.9 * stats["latency"] + 0.1 * response.latency_ms
        stats["success_rate"] = 0.95 * stats.get("success_rate", 1.0) + 0.05

        # Reset failure count on success
        self._failure_counts[model_id] = 0

    def _record_failure(self, model_id: str) -> None:
        """Record a failure and potentially open circuit breaker."""
        self._failure_counts[model_id] = self._failure_counts.get(model_id, 0) + 1

        # Open circuit breaker after 3 consecutive failures
        if self._failure_counts[model_id] >= 3:
            # Exponential backoff: 30s, 60s, 120s
            backoff = 30 * (2 ** min(self._failure_counts[model_id] - 3, 3))
            self._circuit_breaker_until[model_id] = time.time() + backoff
            logger.warning(f"Circuit breaker opened for {model_id} for {backoff}s")

    def _is_circuit_broken(self, model_id: str) -> bool:
        """Check if circuit breaker is open for a model."""
        until = self._circuit_breaker_until.get(model_id, 0)
        if time.time() > until:
            if until > 0:
                logger.info(f"Circuit breaker closed for {model_id}")
                del self._circuit_breaker_until[model_id]
            return False
        return True

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        if not self._routing_history:
            return {"total_decisions": 0}

        model_selections: Dict[str, int] = {}
        for decision in self._routing_history:
            mid = decision.selected_adapter.model_id
            model_selections[mid] = model_selections.get(mid, 0) + 1

        return {
            "total_decisions": len(self._routing_history),
            "model_selections": model_selections,
            "registered_models": [a.model_id for a in self._adapters],
            "strategy": self.strategy.value,
            "model_stats": self._model_stats,
            "circuit_breakers": {
                k: v for k, v in self._circuit_breaker_until.items()
                if v > time.time()
            },
        }
