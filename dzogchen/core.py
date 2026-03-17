"""
Dzogchen - The central intelligence orchestrator
=================================================

The primary entry point for the Dzogchen framework.
Integrates all subsystems into a unified, coherent intelligence.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional
import logging

from dzogchen.config import DzogchenConfig
from dzogchen.consciousness.awareness import ConsciousnessLayer
from dzogchen.consciousness.reflection import ReflectionEngine
from dzogchen.consciousness.metacognition import MetacognitionModule
from dzogchen.intelligence.reasoning import ReasoningEngine
from dzogchen.intelligence.planning import PlanningEngine
from dzogchen.intelligence.synthesis import KnowledgeSynthesizer, KnowledgeFragment, SynthesisMode
from dzogchen.memory.manager import MemoryManager
from dzogchen.models.router import ModelRouter, RoutingStrategy
from dzogchen.agents.orchestrator import AgentOrchestrator, DelegationMode
from dzogchen.tools.registry import ToolRegistry
from dzogchen.tools.executor import ToolExecutor
from dzogchen.tools.builtin import get_builtin_tools

logger = logging.getLogger(__name__)


class Dzogchen:
    """
    Dzogchen - The Great Perfection AI Framework

    A unified super-intelligent system that integrates:
    - Multi-model intelligence routing
    - Consciousness and metacognition
    - Multi-tier memory (working/episodic/semantic)
    - Autonomous reasoning (CoT/ToT/self-consistency)
    - Self-reflection and quality improvement
    - Multi-agent coordination and swarm intelligence
    - Dynamic tool use and creation
    - Knowledge graph synthesis

    Usage:
        from dzogchen import Dzogchen

        dz = Dzogchen()

        # Simple query
        response = await dz.think("What is consciousness?")

        # Agent mode
        result = await dz.act("Research the latest AI breakthroughs")

        # Swarm mode
        result = await dz.swarm("Solve this complex optimization problem...")
    """

    def __init__(self, config: Optional[DzogchenConfig] = None):
        self.config = config or DzogchenConfig.from_env()
        self.session_id = str(uuid.uuid4())

        # Initialize logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.value),
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )

        logger.info(f"Initializing Dzogchen v{self.config.version} (session={self.session_id})")

        # Initialize subsystems
        self._initialize_models()
        self._initialize_memory()
        self._initialize_tools()
        self._initialize_consciousness()
        self._initialize_intelligence()
        self._initialize_agents()

        self._started = False
        self._start_time: Optional[float] = None

        logger.info("Dzogchen initialized successfully")

    # =========================================================================
    # Primary Interface
    # =========================================================================

    async def think(
        self,
        query: str,
        strategy: str = "cot",
        context: Optional[Dict[str, Any]] = None,
        reflect: bool = True,
    ) -> str:
        """
        Apply multi-step reasoning to answer a query.

        Args:
            query: The question or problem to reason about
            strategy: Reasoning strategy: "cot" | "tot" | "self_consistency"
            context: Additional context
            reflect: Whether to self-reflect and improve the answer

        Returns:
            Reasoned response
        """
        await self._ensure_started()

        # Signal consciousness layer
        await self.consciousness.perceive(
            "user_input", query, salience=0.9, tags={"query", "reasoning"}
        )

        # Retrieve relevant memories
        memory_context = self.memory.recall(query, limit=5)

        # Metacognitive assessment
        task_context = {"type": "reasoning", "complexity": self._estimate_complexity(query)}
        cognitive_measure = self.metacognition.assess(task_context)

        # Use cognitive measure to override strategy if needed
        if cognitive_measure.cognitive_load > 0.8 and strategy == "cot":
            strategy = "self_consistency"
            logger.debug("High cognitive load: upgrading to self_consistency strategy")

        # Reason
        full_context = {
            **(context or {}),
            "memory": memory_context,
            "cognitive_state": cognitive_measure.state.value,
        }

        trace = await self.reasoning.reason(query, strategy=strategy, context=full_context)
        answer = trace.final_answer or "I was unable to reach a conclusion."

        # Self-reflection
        if reflect and self.config.enable_consciousness_layer:
            reflection = await self.reflection.reflect(query, answer)
            if reflection.revised_response:
                answer = reflection.revised_response
                logger.debug(f"Reflection improved answer (quality={reflection.overall_quality:.2f})")

        # Store interaction in memory
        self.memory.store_interaction(query, answer, context=full_context)

        return answer

    async def act(
        self,
        task: str,
        tools: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        max_iterations: int = 20,
    ) -> Dict[str, Any]:
        """
        Execute a task autonomously using agents and tools.

        Args:
            task: The task to accomplish
            tools: Specific tool names to make available (None = all built-ins)
            context: Additional context
            max_iterations: Maximum agent iterations

        Returns:
            Dict with result, tools_used, and execution details
        """
        await self._ensure_started()

        # Signal consciousness
        await self.consciousness.perceive(
            "user_input", task, salience=0.95, tags={"task", "action"}
        )
        self.consciousness.set_goal(task)

        # Select tools
        available_tools = self._select_tools(tools)

        # Create agent for this task
        agent = self.orchestrator.create_react_agent(
            name="TaskAgent",
            tools=available_tools,
        )

        # Execute
        result = await agent.run(task, context=context)

        # Store result
        self.memory.store_tool_result(
            "agent_task", {"task": task}, result.result,
            success=result.success, execution_time=result.execution_time,
        )

        return {
            "result": result.result,
            "success": result.success,
            "tools_used": result.tools_used,
            "iterations": result.iterations,
            "execution_time": result.execution_time,
            "agent_id": result.agent_id,
        }

    async def swarm(
        self,
        problem: str,
        swarm_size: int = 3,
        iterations: int = 2,
        tools: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Apply swarm intelligence to a complex problem.

        Multiple cooperative agents explore solution space collectively,
        sharing discoveries and synthesizing a final answer.

        Args:
            problem: The problem to solve
            swarm_size: Number of swarm agents
            iterations: Collective refinement iterations
            tools: Available tools

        Returns:
            Swarm solution with confidence and statistics
        """
        await self._ensure_started()

        from dzogchen.agents.swarm import SwarmIntelligence
        available_tools = self._select_tools(tools)

        swarm = SwarmIntelligence(
            swarm_size=swarm_size,
            model_caller=self._get_model_caller(),
        )

        result = await swarm.solve(problem, iterations=iterations, tools=available_tools)
        return result

    async def plan(
        self,
        goal: str,
        execute: bool = False,
        tools: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Create (and optionally execute) a multi-step plan to achieve a goal.

        Args:
            goal: The goal to plan for
            execute: Whether to execute the plan after creating it
            tools: Available tools for plan execution

        Returns:
            Plan details and optional execution results
        """
        await self._ensure_started()

        available_tools = self._select_tools(tools)
        tool_names = list(available_tools.keys())

        # Create plan
        execution_plan = await self.planner.plan(
            goal=goal,
            available_tools=tool_names,
            context=self.memory.get_context(self.session_id),
        )

        result = {
            "goal": goal,
            "steps": [
                {
                    "id": step.step_id,
                    "goal": step.goal,
                    "actions": [a.name for a in step.actions],
                    "dependencies": step.dependencies,
                }
                for step in execution_plan.steps
            ],
            "estimated_time": execution_plan.total_estimated_time,
            "status": execution_plan.status.value,
        }

        if execute:
            async def action_executor(action):
                tool_func = available_tools.get(action.name)
                if tool_func:
                    return await self.tool_executor.execute(action.name, action.tool_params)
                return {"simulated": True, "action": action.name}

            executed_plan = await self.planner.execute(
                execution_plan,
                action_executor=action_executor,
            )
            result["execution_status"] = executed_plan.status.value
            result["progress"] = executed_plan.get_progress()

        return result

    async def learn(
        self,
        concept: str,
        definition: str,
        domain: str = "general",
        source: str = "user",
    ) -> None:
        """Teach Dzogchen a new fact or concept."""
        await self._ensure_started()
        self.memory.learn_fact(concept, definition, domain=domain, source=source)
        logger.info(f"Learned: {concept}")

    async def remember(self, query: str) -> Dict[str, Any]:
        """Recall information across all memory systems."""
        await self._ensure_started()
        return self.memory.recall(query)

    async def synthesize(
        self,
        query: str,
        fragments: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Synthesize knowledge from multiple sources."""
        await self._ensure_started()

        # If no fragments provided, use memory
        if not fragments:
            memory_results = self.memory.recall(query)
            fragments = []
            for ep in memory_results.get("episodic", []):
                fragments.append({
                    "content": str(ep.get("content", "")),
                    "source": ep.get("type", "memory"),
                })
            for fact in memory_results.get("semantic", []):
                fragments.append({
                    "content": fact.get("definition", ""),
                    "source": fact.get("concept", "knowledge"),
                })

        knowledge_fragments = [
            KnowledgeFragment(
                content=f.get("content", ""),
                source=f.get("source", "unknown"),
                confidence=f.get("confidence", 0.7),
            )
            for f in fragments
            if f.get("content")
        ]

        result = await self.synthesizer.synthesize(
            query=query,
            fragments=knowledge_fragments,
            mode=SynthesisMode.INTEGRATIVE,
        )

        return result.synthesis

    async def stream_think(
        self,
        query: str,
    ) -> AsyncGenerator[str, None]:
        """Stream reasoning steps as they're generated."""
        await self._ensure_started()

        async for step in self.reasoning.stream_reasoning(query):
            yield f"Step {step.step_number}: {step.thought}\n"

    def get_status(self) -> Dict[str, Any]:
        """Get complete system status."""
        return {
            "version": self.config.version,
            "session_id": self.session_id,
            "started": self._started,
            "uptime_seconds": time.time() - self._start_time if self._start_time else 0,
            "model": self.config.primary_model.model_id,
            "memory": self.memory.get_stats(),
            "tools": self.tool_registry.get_stats(),
            "agents": self.orchestrator.get_stats(),
            "reasoning": self.reasoning.get_reasoning_stats(),
            "consciousness": self.consciousness.get_context_summary(),
            "metacognition": self.metacognition.get_calibration_report(),
            "routing": self.model_router.get_routing_stats(),
        }

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> None:
        """Start the Dzogchen consciousness layer."""
        if not self._started:
            await self.consciousness.start()
            self.memory.set_session(self.session_id)
            self._started = True
            self._start_time = time.time()
            logger.info(f"Dzogchen started (session={self.session_id})")

    async def stop(self) -> None:
        """Gracefully stop Dzogchen."""
        if self._started:
            await self.consciousness.stop()
            self._started = False
            logger.info("Dzogchen stopped")

    async def __aenter__(self) -> "Dzogchen":
        await self.start()
        return self

    async def __aexit__(self, *args) -> None:
        await self.stop()

    # =========================================================================
    # Internal initialization
    # =========================================================================

    def _initialize_models(self) -> None:
        """Set up model router and adapters."""
        from dzogchen.models.claude import ClaudeAdapter

        self.model_router = ModelRouter(strategy=RoutingStrategy.BALANCED)

        if self.config.primary_model.api_key:
            primary_adapter = ClaudeAdapter(
                model_id=self.config.primary_model.model_id,
                api_key=self.config.primary_model.api_key,
            )
            self.model_router.add_model(primary_adapter, is_primary=True)

        # Add fallback models if configured
        for fallback_config in self.config.fallback_models:
            if fallback_config.provider.value == "openai":
                from dzogchen.models.openai import OpenAIAdapter
                adapter = OpenAIAdapter(
                    model_id=fallback_config.model_id,
                    api_key=fallback_config.api_key,
                )
                self.model_router.add_model(adapter)

    def _initialize_memory(self) -> None:
        """Set up the multi-tier memory system."""
        mem_config = self.config.memory
        self.memory = MemoryManager(
            working_capacity=mem_config.working_memory_size,
            episodic_max=mem_config.episodic_memory_size,
            semantic_max=mem_config.semantic_memory_size,
            auto_consolidate=True,
            consolidation_interval=mem_config.consolidation_interval,
            persistence_path=mem_config.persistence_path,
        )

    def _initialize_tools(self) -> None:
        """Set up the tool registry and executor."""
        self.tool_registry = ToolRegistry()
        builtin = get_builtin_tools()
        for name, func in builtin.items():
            self.tool_registry.register(func, name=name)

        self.tool_executor = ToolExecutor(
            registry=self.tool_registry,
            default_timeout=self.config.max_tool_execution_time,
        )

    def _initialize_consciousness(self) -> None:
        """Set up the consciousness layer."""
        self.consciousness = ConsciousnessLayer()
        self.metacognition = MetacognitionModule()

        caller = self._get_model_caller()
        self.reflection = ReflectionEngine(
            model_caller=caller,
            max_depth=self.config.agent.reflection_depth,
            enable_synthesis=True,
        )

    def _initialize_intelligence(self) -> None:
        """Set up reasoning, planning, and synthesis engines."""
        caller = self._get_model_caller()
        self.reasoning = ReasoningEngine(model_caller=caller)
        self.planner = PlanningEngine(model_caller=caller)
        self.synthesizer = KnowledgeSynthesizer(model_caller=caller)

    def _initialize_agents(self) -> None:
        """Set up the agent orchestrator."""
        self.orchestrator = AgentOrchestrator(
            model_caller=self._get_model_caller(),
            max_parallel_agents=self.config.agent.max_parallel_agents,
        )

    def _get_model_caller(self) -> Optional[Callable]:
        """Get a simple callable for the primary model."""
        async def caller(prompt: str) -> str:
            return await self.model_router.simple_prompt(prompt)

        adapters = list(self.model_router._adapters)
        if not adapters:
            return None

        return caller

    def _select_tools(self, tool_names: Optional[List[str]]) -> Dict[str, Callable]:
        """Select tools by name, or return all if None."""
        if tool_names is None:
            return self.tool_registry.get_as_dict()
        return {
            name: self.tool_registry.get_func(name)
            for name in tool_names
            if self.tool_registry.get_func(name)
        }

    async def _ensure_started(self) -> None:
        """Ensure the system is started before use."""
        if not self._started:
            await self.start()

    def _estimate_complexity(self, text: str) -> float:
        """Estimate text complexity on 0.0-1.0 scale."""
        words = text.split()
        length_score = min(1.0, len(words) / 100)
        question_count = text.count("?") + text.count("how") + text.count("why")
        question_score = min(1.0, question_count / 5)
        return (length_score + question_score) / 2
