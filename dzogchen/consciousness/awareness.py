"""
ConsciousnessLayer - Global Workspace Theory inspired awareness system
=======================================================================

Implements a Global Workspace Theory (GWT) inspired consciousness layer
that maintains a unified, coherent understanding of the current context,
goals, and state across all subsystems.

Key concepts:
- Global workspace: shared information accessible to all cognitive modules
- Attention: selective focus on the most relevant information
- Integration: combining signals from multiple sources into coherent awareness
- Broadcasting: disseminating relevant information to active processes
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class AwarenessSignal:
    """A signal entering the global workspace."""
    source: str
    content: Any
    salience: float  # 0.0 - 1.0; higher = more likely to enter workspace
    timestamp: float = field(default_factory=time.time)
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "AwarenessSignal") -> bool:
        return self.salience < other.salience


@dataclass
class WorkspaceState:
    """The current contents of the global workspace."""
    primary_focus: Optional[AwarenessSignal] = None
    active_signals: List[AwarenessSignal] = field(default_factory=list)
    background_context: Dict[str, Any] = field(default_factory=dict)
    current_goals: List[str] = field(default_factory=list)
    active_concerns: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class ConsciousnessLayer:
    """
    Global Workspace Theory-inspired consciousness layer.

    Acts as the central integration point for all cognitive processes,
    maintaining coherent awareness across all subsystems and enabling
    coordinated, purposeful behavior.
    """

    def __init__(
        self,
        workspace_capacity: int = 7,  # Miller's magic number
        attention_threshold: float = 0.3,
        broadcast_interval: float = 0.1,
    ):
        self.workspace_capacity = workspace_capacity
        self.attention_threshold = attention_threshold
        self.broadcast_interval = broadcast_interval

        self._workspace = WorkspaceState()
        self._signal_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._subscribers: Dict[str, List[Callable]] = {}
        self._signal_history: deque = deque(maxlen=1000)
        self._running = False
        self._broadcast_task: Optional[asyncio.Task] = None

        # Attention weights - which signals from which sources matter most
        self._source_weights: Dict[str, float] = {
            "user_input": 1.0,
            "error": 0.95,
            "goal_completion": 0.9,
            "reasoning": 0.7,
            "memory_retrieval": 0.6,
            "tool_result": 0.65,
            "background_process": 0.3,
        }

    async def start(self) -> None:
        """Start the consciousness broadcast loop."""
        self._running = True
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        logger.info("ConsciousnessLayer started")

    async def stop(self) -> None:
        """Stop the consciousness layer."""
        self._running = False
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
        logger.info("ConsciousnessLayer stopped")

    async def attend(self, signal: AwarenessSignal) -> bool:
        """
        Submit a signal for potential entry into the global workspace.
        Returns True if the signal was admitted to the workspace.
        """
        # Apply source weight
        source_weight = self._source_weights.get(signal.source, 0.5)
        effective_salience = signal.salience * source_weight

        if effective_salience < self.attention_threshold:
            logger.debug(f"Signal from {signal.source} below attention threshold")
            return False

        # Priority queue uses (priority, item) tuples; negate salience for max-heap behavior
        await self._signal_queue.put((-effective_salience, signal))
        self._signal_history.append(signal)
        return True

    async def perceive(self, source: str, content: Any, salience: float = 0.5, tags: Optional[Set[str]] = None) -> bool:
        """Convenience method to create and submit an awareness signal."""
        signal = AwarenessSignal(
            source=source,
            content=content,
            salience=salience,
            tags=tags or set(),
        )
        return await self.attend(signal)

    def get_workspace_state(self) -> WorkspaceState:
        """Get the current global workspace state."""
        return self._workspace

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Subscribe to workspace broadcast events."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def set_goal(self, goal: str) -> None:
        """Set or update the current primary goal."""
        if goal not in self._workspace.current_goals:
            self._workspace.current_goals.insert(0, goal)
            # Keep goals manageable
            self._workspace.current_goals = self._workspace.current_goals[:5]

    def add_concern(self, concern: str) -> None:
        """Register an active concern that should influence processing."""
        if concern not in self._workspace.active_concerns:
            self._workspace.active_concerns.append(concern)

    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conscious context."""
        state = self._workspace
        return {
            "primary_focus": state.primary_focus.content if state.primary_focus else None,
            "active_signal_count": len(state.active_signals),
            "current_goals": state.current_goals,
            "active_concerns": state.active_concerns,
            "background_keys": list(state.background_context.keys()),
            "timestamp": state.timestamp,
        }

    def get_signal_history(self, limit: int = 50) -> List[AwarenessSignal]:
        """Retrieve recent signal history."""
        return list(self._signal_history)[-limit:]

    async def _broadcast_loop(self) -> None:
        """Main broadcast loop - integrates new signals into workspace."""
        while self._running:
            try:
                await self._integrate_signals()
                await self._broadcast_to_subscribers()
                await asyncio.sleep(self.broadcast_interval)
            except Exception as e:
                logger.error(f"Error in consciousness broadcast loop: {e}")
                await asyncio.sleep(self.broadcast_interval)

    async def _integrate_signals(self) -> None:
        """Integrate queued signals into the global workspace."""
        new_signals = []

        # Drain the queue
        while not self._signal_queue.empty():
            try:
                _, signal = self._signal_queue.get_nowait()
                new_signals.append(signal)
            except asyncio.QueueEmpty:
                break

        if not new_signals:
            return

        # Sort by salience
        new_signals.sort(key=lambda s: s.salience, reverse=True)

        # Admit top signals to workspace (limited by capacity)
        self._workspace.active_signals = new_signals[:self.workspace_capacity]
        self._workspace.timestamp = time.time()

        # Set primary focus to highest salience signal
        if self._workspace.active_signals:
            self._workspace.primary_focus = self._workspace.active_signals[0]

        # Update background context
        for signal in new_signals:
            if signal.tags:
                for tag in signal.tags:
                    self._workspace.background_context[tag] = signal.content

    async def _broadcast_to_subscribers(self) -> None:
        """Broadcast current workspace state to all subscribers."""
        if not self._workspace.primary_focus:
            return

        state = self._workspace

        # Notify "all" subscribers
        for callback in self._subscribers.get("all", []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(state)
                else:
                    callback(state)
            except Exception as e:
                logger.error(f"Error in consciousness subscriber callback: {e}")

        # Notify type-specific subscribers
        if state.primary_focus:
            for tag in state.primary_focus.tags:
                for callback in self._subscribers.get(tag, []):
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(state)
                        else:
                            callback(state)
                    except Exception as e:
                        logger.error(f"Error in tag subscriber callback: {e}")
