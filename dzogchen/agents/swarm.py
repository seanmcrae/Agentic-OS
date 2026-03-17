"""
SwarmIntelligence - Emergent collective intelligence through agent cooperation
==============================================================================

Implements swarm intelligence patterns:
- Stigmergy: indirect coordination through shared environment
- Ant colony optimization for task routing
- Particle swarm for exploration
- Collective problem solving with emergent solutions
"""

from __future__ import annotations

import asyncio
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

from dzogchen.agents.base import BaseAgent, AgentResult, AgentState, AgentMessage
from dzogchen.agents.orchestrator import ReActAgent

logger = logging.getLogger(__name__)


@dataclass
class Pheromone:
    """Digital pheromone for ant-colony-style coordination."""
    path: str
    strength: float
    task_type: str
    deposited_at: float = field(default_factory=time.time)
    evaporation_rate: float = 0.1  # per second

    @property
    def current_strength(self) -> float:
        age = time.time() - self.deposited_at
        return self.strength * math.exp(-self.evaporation_rate * age)


@dataclass
class SwarmParticle:
    """A particle in the swarm exploration space."""
    agent: "SwarmAgent"
    position: List[float]  # Abstract position in solution space
    velocity: List[float]
    best_position: List[float]
    best_score: float = 0.0


class SwarmAgent(ReActAgent):
    """
    An agent that participates in swarm intelligence.

    Has awareness of other swarm agents, can share discoveries,
    and coordinates through stigmergy (pheromone-like signals).
    """

    def __init__(self, *args, swarm_id: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.swarm_id = swarm_id
        self._shared_memory: Dict[str, Any] = {}
        self._discoveries: List[Dict[str, Any]] = []
        self._peer_ids: List[str] = []

    def connect_to_peers(self, peer_ids: List[str]) -> None:
        """Connect this agent to its swarm peers."""
        self._peer_ids = peer_ids

    def share_discovery(self, key: str, value: Any, confidence: float = 0.8) -> None:
        """Share a discovery with the swarm."""
        self._discoveries.append({
            "key": key,
            "value": value,
            "confidence": confidence,
            "agent_id": self.agent_id,
            "timestamp": time.time(),
        })
        # Update shared memory
        self._shared_memory[key] = value

    def receive_discovery(self, discovery: Dict[str, Any]) -> None:
        """Receive a discovery from a peer agent."""
        key = discovery.get("key")
        if key:
            # Only update if received discovery has higher confidence
            current_conf = self._shared_memory.get(f"__conf_{key}", 0.0)
            new_conf = discovery.get("confidence", 0.5)
            if new_conf > current_conf:
                self._shared_memory[key] = discovery.get("value")
                self._shared_memory[f"__conf_{key}"] = new_conf

    async def _agent_loop(self, task: str, context: Optional[Dict[str, Any]] = None):
        """Enhanced agent loop with swarm coordination."""
        # Merge shared memory into context
        enhanced_context = {**(context or {}), "swarm_knowledge": self._shared_memory}

        async for result in super()._agent_loop(task, enhanced_context):
            # Share findings with swarm
            if result:
                self.share_discovery(f"solution:{task[:50]}", result)
            yield result


class SwarmIntelligence:
    """
    Manages a swarm of cooperating agents for collective problem solving.

    Uses bio-inspired algorithms:
    - Ant colony: task routing via virtual pheromones
    - Particle swarm: exploration/exploitation balance
    - Bee algorithm: division of labor and recruitment
    """

    def __init__(
        self,
        swarm_size: int = 5,
        model_caller: Optional[Callable] = None,
        evaporation_rate: float = 0.05,
        pheromone_weight: float = 0.7,
        exploration_rate: float = 0.3,
    ):
        self.swarm_size = swarm_size
        self._model_caller = model_caller
        self.evaporation_rate = evaporation_rate
        self.pheromone_weight = pheromone_weight
        self.exploration_rate = exploration_rate

        self.swarm_id = str(uuid.uuid4())[:8]
        self._agents: List[SwarmAgent] = []
        self._pheromones: List[Pheromone] = []
        self._shared_memory: Dict[str, Any] = {}
        self._collective_knowledge: List[Dict[str, Any]] = []

        # Initialize swarm
        self._initialize_swarm()

    def _initialize_swarm(self) -> None:
        """Create the initial swarm of agents."""
        agent_roles = [
            "Explorer",    # Explores novel solutions
            "Exploiter",   # Refines known good solutions
            "Critic",      # Evaluates and validates
            "Synthesizer", # Combines partial solutions
            "Scout",       # Broad search, rapid assessment
        ]

        for i in range(self.swarm_size):
            role = agent_roles[i % len(agent_roles)]
            agent = SwarmAgent(
                name=f"Swarm-{role}-{i+1}",
                model_caller=self._model_caller,
                swarm_id=self.swarm_id,
            )
            self._agents.append(agent)

        # Connect peers
        all_ids = [a.agent_id for a in self._agents]
        for agent in self._agents:
            peers = [pid for pid in all_ids if pid != agent.agent_id]
            agent.connect_to_peers(peers)

        logger.info(f"SwarmIntelligence {self.swarm_id}: initialized {len(self._agents)} agents")

    async def solve(
        self,
        problem: str,
        iterations: int = 3,
        convergence_threshold: float = 0.9,
        tools: Optional[Dict[str, Callable]] = None,
    ) -> Dict[str, Any]:
        """
        Apply collective swarm intelligence to solve a problem.

        Args:
            problem: The problem to solve
            iterations: Number of collective refinement iterations
            convergence_threshold: Stop early if consensus >= threshold
            tools: Tools available to agents

        Returns:
            Dict with solution, confidence, and swarm statistics
        """
        logger.info(f"Swarm {self.swarm_id}: beginning collective problem solving")

        # Register tools with all agents
        if tools:
            for agent in self._agents:
                for tool_name, tool_func in tools.items():
                    agent.add_tool(tool_name, tool_func)

        all_solutions = []
        best_solution = None
        best_confidence = 0.0

        for iteration in range(iterations):
            logger.info(f"Swarm iteration {iteration + 1}/{iterations}")

            # Parallel exploration
            tasks = [
                agent.run(problem, context={"swarm_knowledge": self._shared_memory})
                for agent in self._agents
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect valid results
            iteration_solutions = []
            for agent, result in zip(self._agents, results):
                if isinstance(result, Exception):
                    logger.warning(f"Agent {agent.name} failed: {result}")
                    continue

                if result.success and result.result:  # type: ignore
                    solution_entry = {
                        "agent": agent.name,
                        "solution": result.result,  # type: ignore
                        "confidence": result.confidence,  # type: ignore
                        "iteration": iteration,
                    }
                    iteration_solutions.append(solution_entry)
                    all_solutions.append(solution_entry)

                    # Deposit pheromone on successful paths
                    self._deposit_pheromone(
                        f"iteration_{iteration}",
                        result.confidence,  # type: ignore
                        "problem_solving",
                    )

            # Share discoveries between agents
            for agent in self._agents:
                for discovery in agent._discoveries:
                    for other_agent in self._agents:
                        if other_agent.agent_id != discovery["agent_id"]:
                            other_agent.receive_discovery(discovery)

            # Update shared memory with iteration findings
            if iteration_solutions:
                best_this_iter = max(iteration_solutions, key=lambda s: s["confidence"])
                self._shared_memory[f"best_iteration_{iteration}"] = best_this_iter["solution"]

                if best_this_iter["confidence"] > best_confidence:
                    best_confidence = best_this_iter["confidence"]
                    best_solution = best_this_iter["solution"]

            # Check for convergence
            consensus = self._measure_consensus(iteration_solutions)
            if consensus >= convergence_threshold:
                logger.info(f"Swarm converged at iteration {iteration + 1} (consensus={consensus:.2f})")
                break

            # Evaporate old pheromones
            self._evaporate_pheromones()

        # Synthesize final answer
        final_solution = await self._synthesize_collective(problem, all_solutions)

        return {
            "solution": final_solution or best_solution,
            "confidence": best_confidence,
            "swarm_id": self.swarm_id,
            "iterations": iterations,
            "solutions_explored": len(all_solutions),
            "agents": len(self._agents),
            "collective_knowledge_size": len(self._shared_memory),
        }

    async def explore_parallel(
        self,
        problems: List[str],
        tools: Optional[Dict[str, Callable]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Have the swarm explore multiple problems simultaneously.
        Each agent handles a different problem.
        """
        if tools:
            for agent in self._agents:
                for name, func in tools.items():
                    agent.add_tool(name, func)

        # Distribute problems across agents
        tasks = []
        assignments = []

        for i, problem in enumerate(problems):
            agent = self._agents[i % len(self._agents)]
            tasks.append(agent.run(problem))
            assignments.append((problem, agent.name))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [
            {
                "problem": problem,
                "agent": agent_name,
                "result": r.result if not isinstance(r, Exception) and r.success else None,  # type: ignore
                "success": not isinstance(r, Exception) and r.success,  # type: ignore
                "error": str(r) if isinstance(r, Exception) else r.error,  # type: ignore
            }
            for (problem, agent_name), r in zip(assignments, results)
        ]

    def _deposit_pheromone(self, path: str, strength: float, task_type: str) -> None:
        """Deposit a virtual pheromone on a solution path."""
        # Check if pheromone already exists on this path
        for ph in self._pheromones:
            if ph.path == path:
                ph.strength = min(1.0, ph.strength + strength * 0.3)
                ph.deposited_at = time.time()
                return

        self._pheromones.append(Pheromone(
            path=path,
            strength=strength,
            task_type=task_type,
            evaporation_rate=self.evaporation_rate,
        ))

    def _evaporate_pheromones(self) -> None:
        """Evaporate weak pheromones."""
        self._pheromones = [
            ph for ph in self._pheromones
            if ph.current_strength > 0.01
        ]

    def _measure_consensus(self, solutions: List[Dict[str, Any]]) -> float:
        """Measure how much consensus exists among solutions."""
        if not solutions or len(solutions) < 2:
            return 1.0 if solutions else 0.0

        # Simple similarity measurement based on string overlap
        solution_texts = [str(s.get("solution", ""))[:200] for s in solutions]
        total_similarity = 0.0
        comparisons = 0

        for i in range(len(solution_texts)):
            for j in range(i + 1, len(solution_texts)):
                words_a = set(solution_texts[i].lower().split())
                words_b = set(solution_texts[j].lower().split())

                if words_a or words_b:
                    similarity = len(words_a & words_b) / max(len(words_a | words_b), 1)
                    total_similarity += similarity
                    comparisons += 1

        return total_similarity / comparisons if comparisons > 0 else 0.0

    async def _synthesize_collective(
        self,
        problem: str,
        all_solutions: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Synthesize a final answer from all collected solutions."""
        if not all_solutions:
            return None

        if self._model_caller is None:
            # Return best solution without synthesis
            best = max(all_solutions, key=lambda s: s.get("confidence", 0))
            return best.get("solution")

        solutions_text = "\n".join(
            f"Agent {s['agent']} (confidence={s['confidence']:.2f}):\n{str(s['solution'])[:300]}"
            for s in sorted(all_solutions, key=lambda s: s.get("confidence", 0), reverse=True)[:5]
        )

        synthesis_prompt = f"""A swarm of {len(self._agents)} AI agents explored this problem:

Problem: {problem}

Their solutions:
{solutions_text}

Synthesize the best elements from these solutions into a single, optimal answer.
Combine complementary insights, resolve conflicts, and produce the definitive response."""

        try:
            return await self._model_caller(synthesis_prompt)
        except Exception as e:
            logger.error(f"Swarm synthesis failed: {e}")
            best = max(all_solutions, key=lambda s: s.get("confidence", 0))
            return best.get("solution")

    def get_swarm_stats(self) -> Dict[str, Any]:
        """Get swarm performance statistics."""
        return {
            "swarm_id": self.swarm_id,
            "size": len(self._agents),
            "agent_states": {a.name: a.state.value for a in self._agents},
            "active_pheromones": len(self._pheromones),
            "shared_knowledge_items": len(self._shared_memory),
            "total_discoveries": sum(len(a._discoveries) for a in self._agents),
        }
