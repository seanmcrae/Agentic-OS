"""
BaseAgent - Foundation for all Dzogchen agents
===============================================

Provides core agent capabilities:
- Autonomous reasoning loop
- Tool use and execution
- Memory integration
- Goal-directed behavior
- Self-monitoring and reflection
"""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class AgentMessage:
    """A message in an agent's conversation."""
    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """The result of an agent's task execution."""
    agent_id: str
    task: str
    result: Any
    success: bool
    state: AgentState
    messages: List[AgentMessage]
    tools_used: List[str]
    iterations: int
    execution_time: float
    confidence: float = 0.8
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base agent with full reasoning and tool use capabilities.

    Implements the standard agent loop:
    1. Observe: understand current state and goal
    2. Think: reason about what to do
    3. Act: execute a tool or respond
    4. Reflect: evaluate the result
    5. Repeat until goal achieved or max iterations reached
    """

    SYSTEM_PROMPT_TEMPLATE = """You are {name}, a highly capable AI agent powered by Dzogchen.

Your goal: {goal}

Capabilities:
{capabilities}

Available tools:
{tools}

Guidelines:
- Think step by step before acting
- Use tools when they would help
- Be precise and efficient
- Acknowledge uncertainty when present
- Stop when the goal is achieved"""

    def __init__(
        self,
        name: str,
        model_caller: Optional[Callable] = None,
        tools: Optional[Dict[str, Callable]] = None,
        memory: Optional[Any] = None,
        max_iterations: int = 20,
        reflection_enabled: bool = True,
        agent_id: Optional[str] = None,
    ):
        self.name = name
        self.agent_id = agent_id or str(uuid.uuid4())[:8]
        self._model_caller = model_caller
        self._tools: Dict[str, Callable] = tools or {}
        self._memory = memory
        self.max_iterations = max_iterations
        self.reflection_enabled = reflection_enabled

        self._state = AgentState.IDLE
        self._messages: List[AgentMessage] = []
        self._current_task: Optional[str] = None
        self._iteration = 0
        self._start_time: Optional[float] = None
        self._tools_used: List[str] = []

        # Event hooks
        self._on_state_change: Optional[Callable] = None
        self._on_tool_use: Optional[Callable] = None
        self._on_message: Optional[Callable] = None

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def is_busy(self) -> bool:
        return self._state not in (AgentState.IDLE, AgentState.COMPLETED, AgentState.FAILED)

    def add_tool(self, name: str, func: Callable, description: str = "") -> None:
        """Register a tool for the agent to use."""
        self._tools[name] = func
        logger.debug(f"Agent {self.name}: added tool '{name}'")

    def on_state_change(self, callback: Callable) -> None:
        """Register a state change callback."""
        self._on_state_change = callback

    async def run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> AgentResult:
        """
        Execute a task using the agent's reasoning loop.

        Args:
            task: The task to accomplish
            context: Additional context
            stream: Whether to stream thinking steps

        Returns:
            AgentResult with outcome and execution trace
        """
        self._current_task = task
        self._start_time = time.time()
        self._iteration = 0
        self._tools_used = []
        self._messages = []

        await self._set_state(AgentState.THINKING)
        logger.info(f"Agent {self.name} ({self.agent_id}): starting task '{task[:60]}...'")

        # Add initial context to messages
        system_prompt = self._build_system_prompt(task)
        self._add_message("system", system_prompt)
        self._add_message("user", task)

        # Run the agent loop
        final_answer = None
        error = None
        success = False

        try:
            async for step_result in self._agent_loop(task, context):
                if isinstance(step_result, str):
                    final_answer = step_result

            if final_answer:
                success = True
                await self._set_state(AgentState.COMPLETED)
            else:
                await self._set_state(AgentState.FAILED)

        except asyncio.CancelledError:
            await self._set_state(AgentState.PAUSED)
            error = "Task cancelled"
        except Exception as e:
            error = str(e)
            await self._set_state(AgentState.FAILED)
            logger.error(f"Agent {self.name} failed: {e}")

        execution_time = time.time() - self._start_time

        return AgentResult(
            agent_id=self.agent_id,
            task=task,
            result=final_answer,
            success=success,
            state=self._state,
            messages=self._messages.copy(),
            tools_used=self._tools_used.copy(),
            iterations=self._iteration,
            execution_time=execution_time,
            error=error,
        )

    async def think(self, prompt: str) -> str:
        """Generate a reasoning step."""
        if self._model_caller is None:
            return f"[Simulated thinking about: {prompt[:50]}]"

        await self._set_state(AgentState.THINKING)
        try:
            response = await self._model_caller(prompt)
            self._add_message("assistant", response)
            return response
        except Exception as e:
            logger.error(f"Agent {self.name} thinking error: {e}")
            raise

    async def use_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a registered tool."""
        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_name}. Available: {list(self._tools.keys())}")

        await self._set_state(AgentState.EXECUTING)
        self._tools_used.append(tool_name)

        if self._on_tool_use:
            await self._maybe_call(self._on_tool_use, tool_name, kwargs)

        try:
            tool_func = self._tools[tool_name]
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**kwargs)
            else:
                result = tool_func(**kwargs)

            self._add_message("tool", f"Tool {tool_name} result: {str(result)[:500]}")
            return result

        except Exception as e:
            error_msg = f"Tool {tool_name} failed: {e}"
            self._add_message("tool", error_msg)
            raise RuntimeError(error_msg) from e

    async def reflect(self, task: str, result: Any) -> str:
        """Reflect on whether the task has been completed satisfactorily."""
        if not self.reflection_enabled or self._model_caller is None:
            return "Task complete."

        await self._set_state(AgentState.REFLECTING)

        prompt = f"""Task: {task}

Result achieved: {str(result)[:500]}

Is this task fully and correctly completed?
If yes, say "COMPLETE: [brief summary]"
If no, say "INCOMPLETE: [what's missing]"
If there's an error, say "ERROR: [what went wrong]"."""

        try:
            reflection = await self._model_caller(prompt)
            self._add_message("assistant", f"[Reflection] {reflection}")
            return reflection
        except Exception as e:
            return f"Reflection failed: {e}"

    def _build_system_prompt(self, goal: str) -> str:
        """Build the system prompt for this agent."""
        capabilities = self._get_capabilities_description()
        tools_desc = "\n".join(
            f"  - {name}: {getattr(func, '__doc__', 'No description')}"
            for name, func in self._tools.items()
        ) or "  No tools available"

        return self.SYSTEM_PROMPT_TEMPLATE.format(
            name=self.name,
            goal=goal,
            capabilities=capabilities,
            tools=tools_desc,
        )

    def _get_capabilities_description(self) -> str:
        """Get a description of this agent's capabilities."""
        return "- Reasoning and analysis\n- Tool use\n- Memory access\n- Goal-directed behavior"

    def _add_message(self, role: str, content: str, **metadata) -> AgentMessage:
        """Add a message to the conversation history."""
        msg = AgentMessage(role=role, content=content, metadata=metadata)
        self._messages.append(msg)
        if self._on_message:
            asyncio.create_task(self._maybe_call(self._on_message, msg))
        return msg

    async def _set_state(self, new_state: AgentState) -> None:
        """Update agent state and notify callbacks."""
        old_state = self._state
        self._state = new_state
        if self._on_state_change and old_state != new_state:
            await self._maybe_call(self._on_state_change, old_state, new_state)

    @abstractmethod
    async def _agent_loop(
        self,
        task: str,
        context: Optional[Dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """The main agent reasoning loop. Must be implemented by subclasses."""
        ...

    async def _maybe_call(self, func: Callable, *args, **kwargs) -> Any:
        """Call a function, handling both sync and async."""
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Callback error: {e}")
            return None

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "state": self._state.value,
            "task": self._current_task,
            "iteration": self._iteration,
            "tools_available": list(self._tools.keys()),
            "message_count": len(self._messages),
        }
