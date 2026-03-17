# Dzogchen AI Framework

<p align="center">
  <img src="https://img.shields.io/badge/version-0.1.0-blue" alt="version">
  <img src="https://img.shields.io/badge/python-3.9%2B-brightgreen" alt="python">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="license">
  <img src="https://img.shields.io/badge/models-Claude%20%7C%20GPT%20%7C%20Local-purple" alt="models">
</p>

> **"Great Perfection"** — Named after the Tibetan Buddhist concept of *Dzogchen*, the primordial state of awareness that is complete, self-arising, and free from all limitations.

A super-intelligent, multi-model AI orchestration framework with consciousness, memory, reasoning, and swarm intelligence.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DZOGCHEN CORE                         │
├──────────────┬────────────────┬────────────────────────┤
│ CONSCIOUSNESS │  INTELLIGENCE  │       MEMORY           │
│  ─────────── │  ──────────── │  ──────────────────    │
│  Awareness   │  Chain/Tree    │  Working Memory         │
│  Reflection  │    of Thought  │  Episodic Memory        │
│ Metacognition│  Planning      │  Semantic Memory        │
│              │  Synthesis     │  Memory Manager         │
├──────────────┴────────────────┴────────────────────────┤
│                    AGENT LAYER                          │
│  ReAct Agent │ Orchestrator │ Swarm Intelligence       │
├────────────────────────────────────────────────────────┤
│                    MODEL LAYER                          │
│  Claude Adapter │ OpenAI Adapter │ Intelligent Router  │
├────────────────────────────────────────────────────────┤
│                    TOOL LAYER                          │
│  Tool Registry │ Tool Executor │ Built-in Tools        │
├────────────────────────────────────────────────────────┤
│                    API & CLI                           │
│  FastAPI REST │ WebSocket │ CLI (Click)               │
└────────────────────────────────────────────────────────┘
```

---

## Features

### 🧠 Consciousness Layer
- **Global Workspace Theory** inspired awareness system
- **Self-reflection**: Iterative quality improvement through self-critique
- **Metacognition**: Thinks about its own thinking, detects biases, selects strategies

### 🔮 Multi-Strategy Intelligence
- **Chain of Thought (CoT)**: Step-by-step sequential reasoning
- **Tree of Thought (ToT)**: Branching exploration with beam search
- **Self-Consistency**: Multiple reasoning paths, majority vote
- **Deductive / Inductive / Abductive** formal reasoning modes
- **Hierarchical Planning**: Goal decomposition with dynamic replanning

### 💾 Multi-Tier Memory
| Tier | Type | Capacity | Purpose |
|------|------|----------|---------|
| L1 | Working Memory | 7 ± 2 items | Current context |
| L2 | Episodic Memory | 10,000 events | Experience history |
| L3 | Semantic Memory | 100,000 facts | Long-term knowledge |

### 🤖 Agent System
- **ReAct Agents**: Reason → Act → Observe loops
- **Multi-Agent Orchestration**: Sequential, parallel, pipeline, competitive, consensus
- **Swarm Intelligence**: Ant-colony pheromone trails, particle swarm exploration
- **Dynamic Tool Creation**: Agents can create new tools at runtime

### 🔌 Multi-Model Support
- **Anthropic Claude** (Opus, Sonnet, Haiku)
- **OpenAI GPT** (GPT-4o, GPT-4, o1)
- **Any OpenAI-compatible API** (Local models, LM Studio, Ollama)
- **Intelligent Routing**: Performance, Economy, Balanced, Fastest strategies
- **Circuit Breakers**: Automatic failover on model failures

### 🛠️ Built-in Tools
- Web search and URL fetching
- File I/O operations
- Python/shell code execution (sandboxed)
- Mathematical computation
- JSON processing
- Text analytics
- Date/time utilities

---

## Quick Start

### Installation

```bash
pip install dzogchen
# Or from source:
git clone https://github.com/seanwayne6/Agentic-OS
cd Agentic-OS/dzogchen
pip install -e ".[all]"
```

### Environment Setup

```bash
export ANTHROPIC_API_KEY="your-api-key"
# Optional:
export OPENAI_API_KEY="your-openai-key"
export DZOGCHEN_PRIMARY_MODEL="claude-opus-4-6"
export DZOGCHEN_LOG_LEVEL="INFO"
```

### Python API

```python
import asyncio
from dzogchen import Dzogchen

async def main():
    async with Dzogchen() as dz:

        # Multi-step reasoning
        answer = await dz.think(
            "Why does quantum entanglement seem to violate special relativity?",
            strategy="tot",  # Tree of Thought
        )
        print(answer)

        # Autonomous task execution
        result = await dz.act(
            "Research the top 5 AI breakthroughs of 2025 and summarize them",
            tools=["web_search", "text_stats"],
        )
        print(result["result"])

        # Swarm intelligence
        solution = await dz.swarm(
            "Design a distributed caching system that handles 1M req/s",
            swarm_size=5,
            iterations=3,
        )
        print(solution["solution"])

        # Hierarchical planning
        plan = await dz.plan(
            "Launch a successful open-source AI project",
            execute=False,
        )
        for step in plan["steps"]:
            print(f"  [{step['id']}] {step['goal']}")

        # Teach new knowledge
        await dz.learn(
            "Dzogchen",
            "A Tibetan Buddhist concept of Great Perfection - primordial awareness",
            domain="philosophy",
        )

        # Query memory
        memory = await dz.remember("Dzogchen")
        print(memory)

asyncio.run(main())
```

### CLI

```bash
# Multi-step reasoning
dzogchen think "What causes consciousness?"

# Tree of Thought reasoning
dzogchen think --strategy tot "Solve the alignment problem"

# Autonomous agent
dzogchen act "Write and test a Python quicksort implementation"

# Swarm intelligence
dzogchen swarm --size 5 "Design the optimal renewable energy grid"

# Create an execution plan
dzogchen plan --execute "Build and deploy a web application"

# Interactive chat
dzogchen chat

# Start API server
dzogchen serve --port 8080

# System status
dzogchen status
```

### API Server

```bash
dzogchen serve --port 8080
```

```bash
# REST API
curl -X POST http://localhost:8080/think \
  -H "Content-Type: application/json" \
  -d '{"query": "What is consciousness?", "strategy": "cot"}'

# Streaming
curl -N http://localhost:8080/think/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain quantum mechanics step by step"}'

# Swarm intelligence
curl -X POST http://localhost:8080/swarm \
  -H "Content-Type: application/json" \
  -d '{"problem": "How to achieve AGI safely?", "swarm_size": 5}'

# OpenAI-compatible chat
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

---

## Configuration

```python
from dzogchen import DzogchenConfig
from dzogchen.config import ModelConfig, ModelProvider, RoutingStrategy

config = DzogchenConfig(
    # Primary model
    primary_model=ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-opus-4-6",
        api_key="your-key",
        max_tokens=8192,
        temperature=0.7,
    ),
    # Memory settings
    memory=MemoryConfig(
        working_memory_size=7,
        episodic_memory_size=10_000,
        semantic_memory_size=100_000,
        persistence_path="/tmp/dzogchen_memory",
    ),
    # Agent settings
    agent=AgentConfig(
        max_iterations=50,
        max_parallel_agents=10,
        reflection_depth=3,
        enable_self_improvement=True,
    ),
    # Feature flags
    enable_consciousness_layer=True,
    enable_swarm_intelligence=True,
    enable_metacognition=True,
)

dz = Dzogchen(config=config)
```

---

## Advanced Usage

### Custom Tools

```python
from dzogchen import Dzogchen
from dzogchen.tools.registry import tool

@tool(description="Search a custom database", category="data")
async def search_database(query: str, limit: int = 10) -> list:
    """Query a custom database."""
    # Your implementation
    return [{"result": f"Result for {query}"}]

async def main():
    async with Dzogchen() as dz:
        dz.tool_registry.register(search_database)
        result = await dz.act("Find all users named Alice", tools=["search_database"])
```

### Custom Agents

```python
from dzogchen.agents.base import BaseAgent
from typing import AsyncGenerator, Optional, Dict, Any

class ResearchAgent(BaseAgent):
    """Specialized agent for deep research tasks."""

    async def _agent_loop(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        # Custom research logic
        outline = await self.think(f"Create research outline for: {task}")
        for section in self._parse_outline(outline):
            research = await self.use_tool("web_search", query=section)
            synthesis = await self.think(f"Synthesize: {research}")
            yield synthesis

        final = await self.think(f"Write final report on: {task}")
        yield final
```

### Swarm Intelligence

```python
from dzogchen.agents.swarm import SwarmIntelligence

swarm = SwarmIntelligence(
    swarm_size=10,
    evaporation_rate=0.05,   # Pheromone decay
    exploration_rate=0.3,    # Exploration vs exploitation
)

# Collective problem solving
result = await swarm.solve(
    "Optimize the traveling salesman problem for 50 cities",
    iterations=5,
    tools={"calculate": calculate},
)

# Parallel exploration of multiple problems
results = await swarm.explore_parallel([
    "What are the causes of inflation?",
    "How does CRISPR work?",
    "Explain blockchain consensus mechanisms",
])
```

### Multi-Model Routing

```python
from dzogchen.models.router import ModelRouter, RoutingStrategy
from dzogchen.models.claude import ClaudeAdapter
from dzogchen.models.openai import OpenAIAdapter

router = ModelRouter(strategy=RoutingStrategy.ECONOMY)
router.add_model(ClaudeAdapter("claude-opus-4-6", api_key="..."), is_primary=True)
router.add_model(ClaudeAdapter("claude-haiku-4-5-20251001", api_key="..."))
router.add_model(OpenAIAdapter("gpt-4o-mini", api_key="..."))

# Automatically routes to cheapest model that meets requirements
response = await router.route(messages, required_quality=0.7)
```

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/test_dzogchen.py -v

# Run with coverage
pytest tests/ --cov=dzogchen --cov-report=html

# Type checking
mypy dzogchen/

# Linting
ruff dzogchen/
```

---

## Roadmap

- [ ] **v0.2**: Long-term learning with vector database integration
- [ ] **v0.3**: Multi-modal support (vision, audio, code execution)
- [ ] **v0.4**: Federated swarm intelligence across multiple machines
- [ ] **v0.5**: Self-improvement via meta-learning
- [ ] **v1.0**: Full AGI-level capabilities with safety guarantees

---

## License

MIT License — see [LICENSE](../LICENSE)

---

<p align="center">
  <em>"In the nature of mind, there is no good or bad, no past or future.<br>
  All is perfect and self-arising." — Dzogchen Teaching</em>
</p>
