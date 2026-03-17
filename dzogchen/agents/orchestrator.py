"""
AgentOrchestrator - Multi-agent coordination and task delegation
================================================================

Manages a pool of specialized agents, routes tasks to the most
suitable agent, enables collaboration between agents, and tracks
overall system performance.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum
import logging

from dzogchen.agents.base import BaseAgent, AgentResult, AgentState

logger = logging.getLogger(__name__)


class DelegationMode(str, Enum):
    SEQUENTIAL = "sequential"    # Agents run one after another
    PARALLEL = "parallel"        # Agents run simultaneously
    PIPELINE = "pipeline"        # Output of one feeds into next
    COMPETITIVE = "competitive"  # Multiple agents compete, best wins
    CONSENSUS = "consensus"      # Multiple agents, majority vote


@dataclass
class OrchestratedTask:
    """A task managed by the orchestrator."""
    task_id: str
    original_task: str
    delegated_tasks: Dict[str, str]  # agent_id -> task
    mode: DelegationMode
    results: Dict[str, AgentResult] = field(default_factory=dict)
    final_result: Optional[Any] = None
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


class ReActAgent(BaseAgent):
    """
    ReAct (Reason + Act) agent implementation.

    Uses the ReAct pattern: alternate between reasoning (Thought)
    and acting (Action) until the task is complete.
    """

    async def _agent_loop(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """ReAct loop: Think → Act → Observe → Repeat."""
        observation = f"Task received: {task}"
        if context:
            observation += f"\nContext: {context}"

        while self._iteration < self.max_iterations:
            self._iteration += 1

            # Think
            thought_prompt = f"""Observation: {observation}

Available tools: {list(self._tools.keys())}

Think step by step. What should I do next?
Format:
Thought: [your reasoning]
Action: [tool_name or "final_answer"]
Action Input: [tool arguments as JSON, or final answer text]"""

            thought = await self.think(thought_prompt)

            # Parse the action
            action, action_input = self._parse_react_response(thought)

            # Check for final answer
            if action == "final_answer":
                if self.reflection_enabled:
                    reflection = await self.reflect(task, action_input)
                    if "COMPLETE" in reflection:
                        yield action_input
                        return
                    elif "INCOMPLETE" in reflection and self._iteration < self.max_iterations - 1:
                        observation = f"Incomplete: {reflection}. Previous answer: {action_input}"
                        continue
                yield action_input
                return

            # Execute tool
            if action in self._tools:
                try:
                    import json
                    if isinstance(action_input, str):
                        try:
                            kwargs = json.loads(action_input)
                        except Exception:
                            kwargs = {"input": action_input}
                    else:
                        kwargs = action_input or {}

                    result = await self.use_tool(action, **kwargs)
                    observation = f"Tool '{action}' result: {str(result)[:1000]}"
                except Exception as e:
                    observation = f"Tool '{action}' failed: {e}. Try a different approach."
            else:
                observation = f"Unknown tool '{action}'. Available: {list(self._tools.keys())}"

        # Max iterations reached
        final_thought = await self.think(
            f"Max iterations reached. Summarize what was accomplished for task: {task}"
        )
        yield final_thought

    def _parse_react_response(self, response: str) -> Tuple[str, Any]:
        """Parse a ReAct format response into (action, action_input)."""
        action = "final_answer"
        action_input: Any = response

        lines = response.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("Action:"):
                action = line[7:].strip()
            elif line.startswith("Action Input:"):
                action_input = line[13:].strip()
                # Check for multi-line input
                for next_line in lines[i+1:]:
                    if next_line.startswith("Thought:") or next_line.startswith("Action:"):
                        break
                    action_input += "\n" + next_line

        return action, action_input


class AgentOrchestrator:
    """
    Multi-agent orchestration system.

    Manages agent lifecycle, routes tasks to appropriate agents,
    enables inter-agent communication, and aggregates results.
    """

    def __init__(
        self,
        model_caller: Optional[Callable] = None,
        max_parallel_agents: int = 10,
    ):
        self._model_caller = model_caller
        self.max_parallel_agents = max_parallel_agents
        self._agents: Dict[str, BaseAgent] = {}
        self._agent_specializations: Dict[str, List[str]] = {}  # agent_id -> [task_types]
        self._task_history: List[OrchestratedTask] = []
        self._running_tasks: Set[str] = set()

    def register_agent(
        self,
        agent: BaseAgent,
        specializations: Optional[List[str]] = None,
    ) -> None:
        """Register an agent with the orchestrator."""
        self._agents[agent.agent_id] = agent
        self._agent_specializations[agent.agent_id] = specializations or []
        logger.info(f"Registered agent: {agent.name} ({agent.agent_id})")

    def create_react_agent(
        self,
        name: str,
        tools: Optional[Dict[str, Callable]] = None,
        specializations: Optional[List[str]] = None,
    ) -> ReActAgent:
        """Create and register a new ReAct agent."""
        agent = ReActAgent(
            name=name,
            model_caller=self._model_caller,
            tools=tools,
        )
        self.register_agent(agent, specializations)
        return agent

    async def execute(
        self,
        task: str,
        mode: DelegationMode = DelegationMode.SEQUENTIAL,
        context: Optional[Dict[str, Any]] = None,
        preferred_agents: Optional[List[str]] = None,
    ) -> OrchestratedTask:
        """
        Execute a task using the best available agent(s).

        Args:
            task: The task to execute
            mode: How to use multiple agents
            context: Additional context
            preferred_agents: Specific agent IDs to use

        Returns:
            OrchestratedTask with all results
        """
        task_id = str(uuid.uuid4())[:8]
        orchestrated = OrchestratedTask(
            task_id=task_id,
            original_task=task,
            delegated_tasks={},
            mode=mode,
        )

        self._running_tasks.add(task_id)

        try:
            if mode == DelegationMode.PARALLEL:
                await self._execute_parallel(orchestrated, context, preferred_agents)
            elif mode == DelegationMode.COMPETITIVE:
                await self._execute_competitive(orchestrated, context, preferred_agents)
            elif mode == DelegationMode.CONSENSUS:
                await self._execute_consensus(orchestrated, context, preferred_agents)
            else:
                # Sequential or Pipeline
                await self._execute_sequential(orchestrated, context, preferred_agents)

            orchestrated.status = "completed"
            orchestrated.completed_at = time.time()

        except Exception as e:
            orchestrated.status = "failed"
            logger.error(f"Orchestrated task {task_id} failed: {e}")

        finally:
            self._running_tasks.discard(task_id)

        self._task_history.append(orchestrated)
        return orchestrated

    async def broadcast(
        self,
        message: str,
        sender_id: Optional[str] = None,
        recipient_ids: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Broadcast a message to agents and collect responses.

        Returns dict of agent_id -> response.
        """
        targets = recipient_ids or list(self._agents.keys())
        responses = {}

        for agent_id in targets:
            if agent_id == sender_id:
                continue
            agent = self._agents.get(agent_id)
            if agent:
                try:
                    result = await agent.run(message)
                    responses[agent_id] = result.result or ""
                except Exception as e:
                    responses[agent_id] = f"Error: {e}"

        return responses

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get a registered agent by ID."""
        return self._agents.get(agent_id)

    def get_available_agents(self) -> List[BaseAgent]:
        """Get all agents that are not currently busy."""
        return [a for a in self._agents.values() if not a.is_busy]

    def get_agents_for_task(self, task_type: str) -> List[BaseAgent]:
        """Get agents specialized for a given task type."""
        specialized = [
            self._agents[aid]
            for aid, specs in self._agent_specializations.items()
            if task_type in specs and aid in self._agents
        ]
        return specialized or list(self._agents.values())

    async def _execute_sequential(
        self,
        orchestrated: OrchestratedTask,
        context: Optional[Dict[str, Any]],
        preferred_agents: Optional[List[str]],
    ) -> None:
        """Execute task sequentially, passing results between agents."""
        agents = self._get_agents(preferred_agents, count=1)
        if not agents:
            raise ValueError("No agents available")

        agent = agents[0]
        orchestrated.delegated_tasks[agent.agent_id] = orchestrated.original_task
        result = await agent.run(orchestrated.original_task, context)
        orchestrated.results[agent.agent_id] = result
        orchestrated.final_result = result.result

    async def _execute_parallel(
        self,
        orchestrated: OrchestratedTask,
        context: Optional[Dict[str, Any]],
        preferred_agents: Optional[List[str]],
    ) -> None:
        """Execute task with multiple agents in parallel."""
        agents = self._get_agents(preferred_agents, count=min(3, len(self._agents)))
        if not agents:
            raise ValueError("No agents available")

        tasks = []
        for agent in agents:
            orchestrated.delegated_tasks[agent.agent_id] = orchestrated.original_task
            tasks.append(agent.run(orchestrated.original_task, context))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for agent, result in zip(agents, results):
            if isinstance(result, Exception):
                logger.warning(f"Agent {agent.name} failed in parallel execution: {result}")
            else:
                orchestrated.results[agent.agent_id] = result  # type: ignore

        # Use first successful result
        for result in orchestrated.results.values():
            if result.success:
                orchestrated.final_result = result.result
                break

    async def _execute_competitive(
        self,
        orchestrated: OrchestratedTask,
        context: Optional[Dict[str, Any]],
        preferred_agents: Optional[List[str]],
    ) -> None:
        """Race multiple agents, use the best result."""
        await self._execute_parallel(orchestrated, context, preferred_agents)

        # Find highest confidence result
        best_result = None
        best_confidence = 0.0

        for result in orchestrated.results.values():
            if result.success and result.confidence > best_confidence:
                best_confidence = result.confidence
                best_result = result

        if best_result:
            orchestrated.final_result = best_result.result

    async def _execute_consensus(
        self,
        orchestrated: OrchestratedTask,
        context: Optional[Dict[str, Any]],
        preferred_agents: Optional[List[str]],
    ) -> None:
        """Get consensus from multiple agents via majority vote."""
        await self._execute_parallel(orchestrated, context, preferred_agents)

        # Collect all results
        answers = [r.result for r in orchestrated.results.values() if r.success and r.result]

        if not answers:
            return

        # Simple frequency-based consensus
        answer_counts: Dict[str, int] = {}
        for answer in answers:
            key = str(answer)[:100]
            answer_counts[key] = answer_counts.get(key, 0) + 1

        most_common = max(answer_counts, key=answer_counts.get)  # type: ignore
        orchestrated.final_result = most_common

    def _get_agents(
        self,
        preferred_ids: Optional[List[str]],
        count: int = 1,
    ) -> List[BaseAgent]:
        """Get agents, preferring specified IDs and available agents."""
        if preferred_ids:
            preferred = [self._agents[aid] for aid in preferred_ids if aid in self._agents]
            if preferred:
                return preferred[:count]

        available = self.get_available_agents()
        return available[:count] or list(self._agents.values())[:count]

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "registered_agents": len(self._agents),
            "running_tasks": len(self._running_tasks),
            "completed_tasks": len(self._task_history),
            "agent_states": {
                a.name: a.state.value for a in self._agents.values()
            },
        }
