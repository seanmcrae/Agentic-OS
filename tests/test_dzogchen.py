"""
Comprehensive test suite for Dzogchen
======================================
"""

import asyncio
import pytest
import time
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_model_caller():
    """A mock model caller that returns predictable responses."""
    async def caller(prompt: str) -> str:
        if "step by step" in prompt.lower() or "think" in prompt.lower():
            return """Thought: Let me analyze this carefully.
Action: final_answer
Action Input: This is a test response demonstrating intelligent reasoning.

Answer: The answer is 42."""
        elif "rate" in prompt.lower() or "score" in prompt.lower():
            return "0.8"
        elif "plan" in prompt.lower() or "step" in prompt.lower():
            return """STEP 1: Analyze the problem
  - Prerequisite: none
  - Actions: Use analyze tool to understand the problem
  - Success criteria: Understanding is achieved

STEP 2: Execute solution
  - Prerequisite: step_0
  - Actions: Implement the solution
  - Success criteria: Solution is implemented"""
        else:
            return f"Intelligent response to: {prompt[:100]}"

    return caller


@pytest.fixture
def dzogchen_config():
    """A test configuration without real API keys."""
    from dzogchen.config import DzogchenConfig, ModelConfig, ModelProvider
    config = DzogchenConfig()
    config.primary_model = ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-opus-4-6",
        api_key=None,  # No real API key in tests
    )
    return config


# ============================================================================
# Memory Tests
# ============================================================================

class TestWorkingMemory:
    def test_basic_put_and_get(self):
        from dzogchen.memory.working import WorkingMemory
        wm = WorkingMemory(capacity=5)
        wm.put("key1", "value1")
        assert wm.get("key1") == "value1"

    def test_capacity_limit(self):
        from dzogchen.memory.working import WorkingMemory
        wm = WorkingMemory(capacity=3)
        for i in range(5):
            wm.put(f"key{i}", f"value{i}", priority=float(i) / 10)
        assert wm.size <= 3

    def test_ttl_expiry(self):
        from dzogchen.memory.working import WorkingMemory
        import time
        wm = WorkingMemory(capacity=10)
        wm.put("expires_soon", "value", ttl=0.01)  # 10ms TTL
        time.sleep(0.05)
        assert wm.get("expires_soon") is None

    def test_priority_eviction(self):
        from dzogchen.memory.working import WorkingMemory
        wm = WorkingMemory(capacity=2)
        wm.put("low_priority", "low", priority=0.1)
        wm.put("high_priority", "high", priority=0.9)
        wm.put("new_item", "new", priority=0.5)
        # Low priority item should be evicted
        assert wm.contains("high_priority")

    def test_activation_update_on_access(self):
        from dzogchen.memory.working import WorkingMemory
        wm = WorkingMemory(capacity=5)
        wm.put("item", "value")
        wm.get("item")  # Access should increase activation
        wm.get("item")
        stats = wm.get_stats()
        assert stats["size"] == 1

    def test_tags(self):
        from dzogchen.memory.working import WorkingMemory
        wm = WorkingMemory(capacity=10)
        wm.put("task1", "data1", tags=["important", "urgent"])
        wm.put("task2", "data2", tags=["important"])
        wm.put("other", "data3", tags=["background"])
        result = wm.get_by_tags(["important"])
        assert "task1" in result
        assert "task2" in result
        assert "other" not in result


class TestEpisodicMemory:
    def test_store_and_recall(self):
        from dzogchen.memory.episodic import EpisodicMemory
        em = EpisodicMemory()
        episode = em.store(
            event_type="interaction",
            content={"input": "hello", "response": "hi there"},
            importance=0.8,
        )
        assert episode.episode_id is not None

        results = em.recall(query="hello")
        assert len(results) > 0

    def test_session_history(self):
        from dzogchen.memory.episodic import EpisodicMemory
        em = EpisodicMemory()
        for i in range(5):
            em.store("interaction", f"event_{i}", session_id="test_session")

        history = em.get_session_history("test_session")
        assert len(history) == 5

    def test_importance_filter(self):
        from dzogchen.memory.episodic import EpisodicMemory
        em = EpisodicMemory()
        em.store("low", "unimportant", importance=0.1)
        em.store("high", "important", importance=0.9)

        important = em.get_important(n=5, min_importance=0.8)
        assert len(important) == 1
        assert important[0].event_type == "high"

    def test_type_filter(self):
        from dzogchen.memory.episodic import EpisodicMemory
        em = EpisodicMemory()
        em.store("conversation", "msg1")
        em.store("tool_use", "tool1")
        em.store("conversation", "msg2")

        convos = em.recall(event_type="conversation")
        assert all(e.event_type == "conversation" for e in convos)

    def test_capacity_consolidation(self):
        from dzogchen.memory.episodic import EpisodicMemory
        em = EpisodicMemory(max_episodes=10)
        for i in range(15):
            em.store("test", f"content_{i}", importance=float(i) / 15)
        assert em.total_episodes <= 10  # Should be within capacity


class TestSemanticMemory:
    def test_store_and_lookup(self):
        from dzogchen.memory.semantic import SemanticMemory
        sm = SemanticMemory()
        sm.store(
            concept="Python",
            definition="A high-level programming language",
            domain="computing",
        )
        entry = sm.lookup("Python")
        assert entry is not None
        assert "programming" in entry.definition

    def test_search(self):
        from dzogchen.memory.semantic import SemanticMemory
        sm = SemanticMemory()
        sm.store("Machine Learning", "A subset of AI using statistical methods", domain="ai")
        sm.store("Deep Learning", "Neural network-based machine learning", domain="ai")
        sm.store("Cooking", "The art of preparing food", domain="food")

        results = sm.search("neural network", domain="ai")
        assert len(results) > 0
        assert results[0].domain == "ai"

    def test_confidence_update(self):
        from dzogchen.memory.semantic import SemanticMemory
        sm = SemanticMemory()
        sm.store("fact", "original definition", confidence=0.7)
        sm.update_confidence("fact", 0.2)
        entry = sm.lookup("fact")
        assert entry.confidence == pytest.approx(0.9, abs=0.01)

    def test_higher_confidence_overwrites(self):
        from dzogchen.memory.semantic import SemanticMemory
        sm = SemanticMemory()
        sm.store("concept", "old definition", confidence=0.5)
        sm.store("concept", "new definition", confidence=0.9)
        entry = sm.lookup("concept")
        assert entry.definition == "new definition"

    def test_forget(self):
        from dzogchen.memory.semantic import SemanticMemory
        sm = SemanticMemory()
        sm.store("to_forget", "will be deleted")
        assert sm.lookup("to_forget") is not None
        sm.forget("to_forget")
        assert sm.lookup("to_forget") is None


class TestMemoryManager:
    def test_unified_recall(self):
        from dzogchen.memory.manager import MemoryManager
        mm = MemoryManager()
        mm.remember("test_key", "test_value", memory_type="working")
        mm.learn_fact("Python", "A programming language")
        results = mm.recall("Python")
        assert "semantic" in results

    def test_store_interaction(self):
        from dzogchen.memory.manager import MemoryManager
        mm = MemoryManager()
        mm.set_session("test_session")
        episode = mm.store_interaction("Hello", "Hi there!", quality=0.9)
        assert episode is not None
        history = mm.get_conversation_history()
        assert len(history) > 0

    def test_context_snapshot(self):
        from dzogchen.memory.manager import MemoryManager
        mm = MemoryManager()
        mm.remember("current_task", "writing tests", priority=0.9)
        context = mm.get_context()
        assert "working_memory" in context
        assert "current_task" in context["working_memory"]


# ============================================================================
# Consciousness Tests
# ============================================================================

class TestConsciousnessLayer:
    @pytest.mark.asyncio
    async def test_perceive_above_threshold(self):
        from dzogchen.consciousness.awareness import ConsciousnessLayer
        cl = ConsciousnessLayer(attention_threshold=0.3)
        result = await cl.perceive("test", "content", salience=0.8)
        assert result is True

    @pytest.mark.asyncio
    async def test_perceive_below_threshold(self):
        from dzogchen.consciousness.awareness import ConsciousnessLayer
        cl = ConsciousnessLayer(attention_threshold=0.3)
        result = await cl.perceive("background", "noise", salience=0.1)
        assert result is False

    @pytest.mark.asyncio
    async def test_goal_setting(self):
        from dzogchen.consciousness.awareness import ConsciousnessLayer
        cl = ConsciousnessLayer()
        cl.set_goal("achieve great things")
        state = cl.get_workspace_state()
        assert "achieve great things" in state.current_goals

    @pytest.mark.asyncio
    async def test_broadcast_loop(self):
        from dzogchen.consciousness.awareness import ConsciousnessLayer
        cl = ConsciousnessLayer()
        await cl.start()

        received = []
        cl.subscribe("all", lambda state: received.append(state))

        await cl.perceive("test", "important event", salience=0.9)
        await asyncio.sleep(0.2)  # Wait for broadcast

        await cl.stop()

    def test_signal_history(self):
        from dzogchen.consciousness.awareness import ConsciousnessLayer, AwarenessSignal
        cl = ConsciousnessLayer()
        # Add some signals to history manually
        import time as time_module
        signal = AwarenessSignal(
            source="test",
            content="test content",
            salience=0.5,
        )
        cl._signal_history.append(signal)
        history = cl.get_signal_history(limit=10)
        assert len(history) >= 1


class TestReflectionEngine:
    @pytest.mark.asyncio
    async def test_quick_check_short_response(self):
        from dzogchen.consciousness.reflection import ReflectionEngine
        re = ReflectionEngine()
        is_ok, issues = await re.quick_check("ok")
        assert not is_ok
        assert len(issues) > 0

    @pytest.mark.asyncio
    async def test_quick_check_good_response(self):
        from dzogchen.consciousness.reflection import ReflectionEngine
        re = ReflectionEngine()
        is_ok, issues = await re.quick_check(
            "This is a well-formed response that answers the question clearly and completely."
        )
        assert is_ok

    @pytest.mark.asyncio
    async def test_reflect_without_model(self):
        from dzogchen.consciousness.reflection import ReflectionEngine
        re = ReflectionEngine(model_caller=None)
        result = await re.reflect("What is 2+2?", "4")
        assert result is not None
        assert result.overall_quality > 0


class TestMetacognitionModule:
    def test_cognitive_state_assessment(self):
        from dzogchen.consciousness.metacognition import MetacognitionModule, CognitiveState
        mc = MetacognitionModule()
        measure = mc.assess({"type": "simple", "complexity": 0.2})
        assert measure.state is not None
        assert 0.0 <= measure.confidence <= 1.0

    def test_strategy_selection(self):
        from dzogchen.consciousness.metacognition import MetacognitionModule, ReasoningStrategy
        mc = MetacognitionModule()
        strategy = mc.select_strategy({"type": "mathematical", "complexity": 0.5})
        assert strategy in ReasoningStrategy

    def test_flag_and_clear_uncertainty(self):
        from dzogchen.consciousness.metacognition import MetacognitionModule, CognitiveState
        mc = MetacognitionModule()
        mc.flag_uncertainty("source", "unsure about X")
        assert mc._cognitive_state == CognitiveState.UNCERTAIN
        mc.clear_uncertainty("source")
        assert len(mc._active_uncertainties) == 0

    def test_bias_detection(self):
        from dzogchen.consciousness.metacognition import MetacognitionModule
        mc = MetacognitionModule()
        biased_trace = [
            "This definitely confirms my hypothesis",
            "Another definitive confirmation",
            "The evidence clearly and obviously confirms it",
        ]
        biases = mc.detect_cognitive_bias(biased_trace)
        assert len(biases) > 0  # Should detect overconfidence


# ============================================================================
# Intelligence Tests
# ============================================================================

class TestReasoningEngine:
    @pytest.mark.asyncio
    async def test_chain_of_thought_without_model(self):
        from dzogchen.intelligence.reasoning import ReasoningEngine
        re = ReasoningEngine(model_caller=None)
        trace = await re.reason("What is 2 + 2?", strategy="cot")
        assert trace is not None
        assert trace.query == "What is 2 + 2?"

    @pytest.mark.asyncio
    async def test_chain_of_thought_with_mock(self, mock_model_caller):
        from dzogchen.intelligence.reasoning import ReasoningEngine
        re = ReasoningEngine(model_caller=mock_model_caller)
        trace = await re.reason("What is the capital of France?", strategy="cot")
        assert trace is not None
        assert len(trace.steps) > 0

    @pytest.mark.asyncio
    async def test_reasoning_stats(self, mock_model_caller):
        from dzogchen.intelligence.reasoning import ReasoningEngine
        re = ReasoningEngine(model_caller=mock_model_caller)
        await re.reason("Test query 1", strategy="cot")
        await re.reason("Test query 2", strategy="cot")
        stats = re.get_reasoning_stats()
        assert stats["total_traces"] == 2


class TestPlanningEngine:
    @pytest.mark.asyncio
    async def test_simple_decompose(self):
        from dzogchen.intelligence.planning import PlanningEngine
        pe = PlanningEngine(model_caller=None)
        plan = await pe.plan("Build a house")
        assert len(plan.steps) > 0
        assert plan.goal == "Build a house"

    @pytest.mark.asyncio
    async def test_plan_with_mock(self, mock_model_caller):
        from dzogchen.intelligence.planning import PlanningEngine
        pe = PlanningEngine(model_caller=mock_model_caller)
        plan = await pe.plan("Solve the trolley problem")
        assert plan is not None

    @pytest.mark.asyncio
    async def test_plan_execution(self):
        from dzogchen.intelligence.planning import PlanningEngine, PlanStatus
        pe = PlanningEngine(model_caller=None)
        plan = await pe.plan("Simple task")

        executed = await pe.execute(plan)
        # Should complete even with simulated actions
        assert executed.status in (PlanStatus.COMPLETED, PlanStatus.FAILED)


# ============================================================================
# Model Tests
# ============================================================================

class TestModelRouter:
    def test_register_and_route_fallback(self):
        from dzogchen.models.router import ModelRouter

        mock_adapter = MagicMock()
        mock_adapter.model_id = "test-model"
        mock_adapter.is_busy = False

        router = ModelRouter()
        router.add_model(mock_adapter, is_primary=True)
        assert len(router._adapters) == 1

    def test_circuit_breaker(self):
        from dzogchen.models.router import ModelRouter
        router = ModelRouter()
        router._failure_counts["bad-model"] = 3
        router._circuit_breaker_until["bad-model"] = time.time() + 60
        assert router._is_circuit_broken("bad-model") is True

    def test_circuit_breaker_reset(self):
        from dzogchen.models.router import ModelRouter
        router = ModelRouter()
        router._circuit_breaker_until["old-model"] = time.time() - 1  # Expired
        assert router._is_circuit_broken("old-model") is False

    def test_routing_stats(self):
        from dzogchen.models.router import ModelRouter
        router = ModelRouter()
        stats = router.get_routing_stats()
        assert "total_decisions" in stats


# ============================================================================
# Tool Tests
# ============================================================================

class TestToolRegistry:
    def test_register_function(self):
        from dzogchen.tools.registry import ToolRegistry

        registry = ToolRegistry()

        def my_tool(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y

        t = registry.register(my_tool, name="add", description="Add two numbers")
        assert registry.get("add") is not None
        assert registry.get("add").metadata.name == "add"

    def test_list_tools_by_category(self):
        from dzogchen.tools.registry import ToolRegistry

        registry = ToolRegistry()

        def tool_a():
            """Tool A."""
            pass

        def tool_b():
            """Tool B."""
            pass

        registry.register(tool_a, name="tool_a", category="math")
        registry.register(tool_b, name="tool_b", category="text")

        math_tools = registry.list_tools(category="math")
        assert len(math_tools) == 1
        assert math_tools[0].metadata.name == "tool_a"

    def test_to_model_definitions(self):
        from dzogchen.tools.registry import ToolRegistry

        registry = ToolRegistry()

        def search_tool(query: str) -> str:
            """Search for information."""
            return "results"

        registry.register(search_tool, name="search")
        definitions = registry.to_model_definitions()
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search"


class TestToolExecutor:
    @pytest.mark.asyncio
    async def test_execute_sync_tool(self):
        from dzogchen.tools.registry import ToolRegistry
        from dzogchen.tools.executor import ToolExecutor

        registry = ToolRegistry()

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        registry.register(add, name="add")
        executor = ToolExecutor(registry)
        result = await executor.execute("add", {"a": 5, "b": 3})
        assert result == 8

    @pytest.mark.asyncio
    async def test_execute_async_tool(self):
        from dzogchen.tools.registry import ToolRegistry
        from dzogchen.tools.executor import ToolExecutor

        registry = ToolRegistry()

        async def async_tool(msg: str) -> str:
            """Async tool."""
            return f"Processed: {msg}"

        registry.register(async_tool, name="async_test")
        executor = ToolExecutor(registry)
        result = await executor.execute("async_test", {"msg": "hello"})
        assert "Processed: hello" == result

    @pytest.mark.asyncio
    async def test_unknown_tool_raises(self):
        from dzogchen.tools.registry import ToolRegistry
        from dzogchen.tools.executor import ToolExecutor

        registry = ToolRegistry()
        executor = ToolExecutor(registry)

        with pytest.raises(ValueError, match="Unknown tool"):
            await executor.execute("nonexistent_tool")

    @pytest.mark.asyncio
    async def test_timeout_enforcement(self):
        from dzogchen.tools.registry import ToolRegistry
        from dzogchen.tools.executor import ToolExecutor

        registry = ToolRegistry()

        async def slow_tool() -> str:
            """A very slow tool."""
            await asyncio.sleep(100)
            return "done"

        registry.register(slow_tool, name="slow_tool")
        executor = ToolExecutor(registry, default_timeout=1, max_retries=0)

        with pytest.raises((TimeoutError, RuntimeError)):
            await executor.execute("slow_tool", timeout=1)


class TestBuiltinTools:
    def test_calculate(self):
        from dzogchen.tools.builtin import calculate
        result = calculate("2 + 2")
        assert result["result"] == 4
        assert result["success"] is True

    def test_calculate_complex(self):
        from dzogchen.tools.builtin import calculate
        result = calculate("sqrt(144)")
        assert result["result"] == 12.0

    def test_calculate_unsafe(self):
        from dzogchen.tools.builtin import calculate
        result = calculate("import os")
        assert not result.get("success", True) or result.get("error")

    def test_text_stats(self):
        from dzogchen.tools.builtin import text_stats
        stats = text_stats("Hello world. This is a test.")
        assert stats["word_count"] == 6
        assert stats["char_count"] > 0

    def test_json_parse_and_format(self):
        from dzogchen.tools.builtin import json_parse, json_format
        data = {"key": "value", "num": 42}
        formatted = json_format(data)
        parsed = json_parse(formatted)
        assert parsed == data

    def test_get_current_time(self):
        from dzogchen.tools.builtin import get_current_time
        result = get_current_time()
        assert "datetime" in result
        assert "unix_timestamp" in result


# ============================================================================
# Agent Tests
# ============================================================================

class TestBaseAgent:
    @pytest.mark.asyncio
    async def test_react_agent_simple_task(self, mock_model_caller):
        from dzogchen.agents.orchestrator import ReActAgent

        tool_called = []

        def simple_tool(text: str) -> str:
            """A simple test tool."""
            tool_called.append(text)
            return f"Processed: {text}"

        agent = ReActAgent(
            name="TestAgent",
            model_caller=mock_model_caller,
            tools={"simple_tool": simple_tool},
        )

        result = await agent.run("Use the simple tool to process 'hello'")
        assert result is not None
        assert result.agent_id is not None
        assert result.iterations > 0

    @pytest.mark.asyncio
    async def test_agent_state_transitions(self, mock_model_caller):
        from dzogchen.agents.orchestrator import ReActAgent
        from dzogchen.agents.base import AgentState

        agent = ReActAgent(name="StateTestAgent", model_caller=mock_model_caller)
        assert agent.state == AgentState.IDLE

        result = await agent.run("Simple task")
        assert agent.state in (AgentState.COMPLETED, AgentState.FAILED)


class TestAgentOrchestrator:
    @pytest.mark.asyncio
    async def test_create_and_execute_agent(self, mock_model_caller):
        from dzogchen.agents.orchestrator import AgentOrchestrator

        orchestrator = AgentOrchestrator(model_caller=mock_model_caller)
        agent = orchestrator.create_react_agent("Worker")
        assert agent in orchestrator._agents.values()

    @pytest.mark.asyncio
    async def test_sequential_execution(self, mock_model_caller):
        from dzogchen.agents.orchestrator import AgentOrchestrator, DelegationMode

        orchestrator = AgentOrchestrator(model_caller=mock_model_caller)
        orchestrator.create_react_agent("Worker1")

        result = await orchestrator.execute(
            "Analyze this problem",
            mode=DelegationMode.SEQUENTIAL,
        )
        assert result is not None
        assert result.status in ("completed", "failed")

    def test_orchestrator_stats(self, mock_model_caller):
        from dzogchen.agents.orchestrator import AgentOrchestrator
        orchestrator = AgentOrchestrator(model_caller=mock_model_caller)
        orchestrator.create_react_agent("Agent1")
        stats = orchestrator.get_stats()
        assert stats["registered_agents"] == 1


# ============================================================================
# Integration Tests
# ============================================================================

class TestDzogchenIntegration:
    @pytest.mark.asyncio
    async def test_think_without_api_key(self):
        """Dzogchen should work (with limited capability) without API key."""
        from dzogchen.core import Dzogchen
        from dzogchen.config import DzogchenConfig

        config = DzogchenConfig()
        config.primary_model.api_key = None

        dz = Dzogchen(config=config)
        await dz.start()

        # Should not crash even without model
        status = dz.get_status()
        assert "version" in status
        assert "session_id" in status

        await dz.stop()

    @pytest.mark.asyncio
    async def test_memory_operations(self):
        """Test that memory operations work correctly."""
        from dzogchen.core import Dzogchen
        from dzogchen.config import DzogchenConfig

        config = DzogchenConfig()
        config.primary_model.api_key = None

        async with Dzogchen(config=config) as dz:
            await dz.learn("pytest", "A Python testing framework", domain="programming")
            results = await dz.remember("pytest")
            assert "semantic" in results

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager protocol."""
        from dzogchen.core import Dzogchen
        from dzogchen.config import DzogchenConfig

        config = DzogchenConfig()
        config.primary_model.api_key = None

        async with Dzogchen(config=config) as dz:
            assert dz._started is True

        assert dz._started is False

    @pytest.mark.asyncio
    async def test_tool_availability(self):
        """Test that built-in tools are available."""
        from dzogchen.core import Dzogchen
        from dzogchen.config import DzogchenConfig

        config = DzogchenConfig()
        config.primary_model.api_key = None

        async with Dzogchen(config=config) as dz:
            tools = dz.tool_registry.list_tools()
            tool_names = [t.metadata.name for t in tools]
            assert "calculate" in tool_names
            assert "get_current_time" in tool_names
