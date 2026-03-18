"""Cost tracking and budget management for LLM API usage."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class AlertType(Enum):
    APPROACHING_LIMIT = "approaching_limit"   # >= 80 % utilization
    LIMIT_REACHED = "limit_reached"           # >= 100 % utilization
    RATE_EXCEEDED = "rate_exceeded"


@dataclass
class Budget:
    name: str
    limit_usd: float
    period: str = "session"  # "session" | "hourly" | "daily" | "monthly"
    alert_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.8, 1.0])
    current_spend: float = 0.0
    alert_callbacks: List[Callable] = field(default_factory=list)

    @property
    def utilization(self) -> float:
        if self.limit_usd == 0:
            return 0.0
        return self.current_spend / self.limit_usd

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit_usd - self.current_spend)

    @property
    def is_exhausted(self) -> bool:
        return self.current_spend >= self.limit_usd


@dataclass
class CostRecord:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: float
    operation: str
    session_id: str


@dataclass
class BudgetAlert:
    alert_type: AlertType
    budget_name: str
    utilization: float
    remaining_usd: float
    timestamp: float


# Pricing table: (input_price_per_million, output_price_per_million)
_MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # Anthropic
    "claude-3-opus":           (15.0,  75.0),
    "claude-3-sonnet":         (3.0,   15.0),
    "claude-3-haiku":          (0.25,  1.25),
    "claude-3-5-sonnet":       (3.0,   15.0),
    "claude-3-5-haiku":        (0.8,   4.0),
    # OpenAI
    "gpt-4":                   (30.0,  60.0),
    "gpt-4-turbo":             (10.0,  30.0),
    "gpt-3.5-turbo":           (0.5,   1.5),
    "gpt-3.5":                 (0.5,   1.5),
    # Google
    "gemini-pro":              (0.5,   1.5),
    "gemini-1.5-pro":          (3.5,   10.5),
    "gemini-1.5-flash":        (0.35,  1.05),
}

_DEFAULT_PRICING: Tuple[float, float] = (1.0, 3.0)


class BudgetManager:
    """Tracks API costs and enforces spend budgets with threshold alerts."""

    def __init__(
        self,
        default_budget_usd: float = 1.0,
        alert_callbacks: Optional[List[Callable]] = None,
    ) -> None:
        self._budgets: Dict[str, Budget] = {}
        self._records: List[CostRecord] = []
        self._global_callbacks: List[Callable] = list(alert_callbacks or [])
        self._fired_thresholds: Dict[str, set] = defaultdict(set)

        # Create the default budget immediately
        self.create_budget("default", default_budget_usd, "session")

    # ---- budget management ---------------------------------------------------

    def create_budget(self, name: str, limit_usd: float, period: str = "session") -> Budget:
        budget = Budget(
            name=name,
            limit_usd=limit_usd,
            period=period,
            alert_callbacks=list(self._global_callbacks),
        )
        self._budgets[name] = budget
        return budget

    def add_callback(self, callback: Callable) -> None:
        """Add an alert callback that fires for all budgets."""
        self._global_callbacks.append(callback)
        for budget in self._budgets.values():
            if callback not in budget.alert_callbacks:
                budget.alert_callbacks.append(callback)

    def reset_budget(self, name: str) -> None:
        """Reset spend counter and fired-threshold tracking for a budget."""
        budget = self._budgets.get(name)
        if budget is not None:
            budget.current_spend = 0.0
        self._fired_thresholds.pop(name, None)

    # ---- cost recording ------------------------------------------------------

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        operation: str = "generate",
        session_id: str = "default",
    ) -> CostRecord:
        cost = self._calculate_cost(model, input_tokens, output_tokens)
        record = CostRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            timestamp=time.time(),
            operation=operation,
            session_id=session_id,
        )
        self._records.append(record)

        # Update all budgets
        for budget in self._budgets.values():
            budget.current_spend += cost
            # Fire alerts asynchronously (best-effort; caller can await if needed)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._fire_alerts(budget, cost))
                else:
                    loop.run_until_complete(self._fire_alerts(budget, cost))
            except RuntimeError:
                # No event loop available; skip async alerts
                pass

        return record

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD using the pricing table."""
        # Exact match first, then prefix match
        pricing = _MODEL_PRICING.get(model)
        if pricing is None:
            for key in _MODEL_PRICING:
                if model.startswith(key) or key in model:
                    pricing = _MODEL_PRICING[key]
                    break
        if pricing is None:
            pricing = _DEFAULT_PRICING

        input_price_per_m, output_price_per_m = pricing
        cost = (input_tokens / 1_000_000) * input_price_per_m + (output_tokens / 1_000_000) * output_price_per_m
        return round(cost, 8)

    # ---- budget checks -------------------------------------------------------

    def check_budget(
        self,
        budget_name: str = "default",
        estimated_cost: float = 0.0,
    ) -> Tuple[bool, str]:
        """
        Check whether a proposed operation can proceed within budget.

        Returns:
            (can_proceed: bool, reason: str)
        """
        budget = self._budgets.get(budget_name)
        if budget is None:
            return True, f"Budget '{budget_name}' not found; proceeding without constraint."

        if budget.is_exhausted:
            return (
                False,
                f"Budget '{budget_name}' is exhausted "
                f"(spent ${budget.current_spend:.4f} of ${budget.limit_usd:.4f}).",
            )

        projected = budget.current_spend + estimated_cost
        if projected > budget.limit_usd:
            return (
                False,
                f"Estimated cost ${estimated_cost:.4f} would exceed budget '{budget_name}' "
                f"(${budget.remaining:.4f} remaining).",
            )

        utilization_after = projected / budget.limit_usd if budget.limit_usd else 0.0
        if utilization_after >= 0.8:
            return (
                True,
                f"Warning: budget '{budget_name}' will be at "
                f"{utilization_after * 100:.1f}% utilization after this operation.",
            )

        return True, "OK"

    # ---- reporting -----------------------------------------------------------

    def get_session_cost(self, session_id: str = "default") -> float:
        return sum(r.cost_usd for r in self._records if r.session_id == session_id)

    def get_cost_breakdown(self) -> Dict[str, Any]:
        """Return cost aggregated by model, operation, and session."""
        by_model: Dict[str, float] = defaultdict(float)
        by_operation: Dict[str, float] = defaultdict(float)
        by_session: Dict[str, float] = defaultdict(float)
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0

        for r in self._records:
            by_model[r.model] += r.cost_usd
            by_operation[r.operation] += r.cost_usd
            by_session[r.session_id] += r.cost_usd
            total_input_tokens += r.input_tokens
            total_output_tokens += r.output_tokens
            total_cost += r.cost_usd

        return {
            "total_cost_usd": round(total_cost, 6),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "by_model": {k: round(v, 6) for k, v in by_model.items()},
            "by_operation": {k: round(v, 6) for k, v in by_operation.items()},
            "by_session": {k: round(v, 6) for k, v in by_session.items()},
            "record_count": len(self._records),
            "budgets": {
                name: {
                    "limit_usd": b.limit_usd,
                    "current_spend": round(b.current_spend, 6),
                    "utilization": round(b.utilization, 4),
                    "remaining": round(b.remaining, 6),
                    "period": b.period,
                    "is_exhausted": b.is_exhausted,
                }
                for name, b in self._budgets.items()
            },
        }

    def get_forecast(self, session_id: str = "default") -> Dict[str, float]:
        """Extrapolate hourly and daily cost based on current session spend rate."""
        session_records = [r for r in self._records if r.session_id == session_id]
        if not session_records:
            return {"hourly_usd": 0.0, "daily_usd": 0.0, "weekly_usd": 0.0, "monthly_usd": 0.0}

        total_cost = sum(r.cost_usd for r in session_records)
        earliest = min(r.timestamp for r in session_records)
        elapsed_seconds = max(1.0, time.time() - earliest)

        cost_per_second = total_cost / elapsed_seconds
        hourly = cost_per_second * 3600
        daily = hourly * 24

        return {
            "hourly_usd": round(hourly, 6),
            "daily_usd": round(daily, 6),
            "weekly_usd": round(daily * 7, 6),
            "monthly_usd": round(daily * 30, 6),
        }

    # ---- alert firing --------------------------------------------------------

    async def _fire_alerts(self, budget: Budget, new_cost: float) -> None:
        """Fire threshold-based alerts for the given budget after adding new_cost."""
        utilization = budget.utilization
        fired = self._fired_thresholds[budget.name]

        for threshold in sorted(budget.alert_thresholds):
            if utilization >= threshold and threshold not in fired:
                fired.add(threshold)

                if threshold >= 1.0:
                    alert_type = AlertType.LIMIT_REACHED
                elif threshold >= 0.8:
                    alert_type = AlertType.APPROACHING_LIMIT
                else:
                    alert_type = AlertType.APPROACHING_LIMIT

                alert = BudgetAlert(
                    alert_type=alert_type,
                    budget_name=budget.name,
                    utilization=utilization,
                    remaining_usd=budget.remaining,
                    timestamp=time.time(),
                )

                callbacks = list(budget.alert_callbacks)
                for cb in callbacks:
                    try:
                        result = cb(alert)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:
                        pass
