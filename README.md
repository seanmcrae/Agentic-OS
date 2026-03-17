# Agentic OS

A state-of-the-art AI agent operating system featuring **Dzogchen** — a super-intelligent AI framework with consciousness, multi-tier memory, swarm intelligence, and multi-model orchestration.

## Projects

### 🧠 [Dzogchen AI Framework](./dzogchen/)

> *Named after the Tibetan Buddhist concept of "Great Perfection"*

A super-intelligent AI framework that goes beyond simple LLM wrappers:

```python
from dzogchen import Dzogchen
import asyncio

async def main():
    async with Dzogchen() as dz:
        # Reason with Tree of Thought
        answer = await dz.think(
            "Why does time seem to slow near black holes?",
            strategy="tot"
        )

        # Execute tasks autonomously
        result = await dz.act("Write a sorting algorithm and test it")

        # Deploy swarm intelligence
        solution = await dz.swarm(
            "Design a fault-tolerant distributed system",
            swarm_size=5
        )

asyncio.run(main())
```

**Key capabilities:**
- **Consciousness Layer**: Global Workspace Theory awareness, self-reflection, metacognition
- **Multi-Strategy Reasoning**: CoT, Tree of Thought, Self-Consistency, Deductive/Inductive/Abductive
- **Multi-Tier Memory**: Working (L1) → Episodic (L2) → Semantic (L3) with auto-consolidation
- **Swarm Intelligence**: Ant-colony pheromone trails, collective problem solving
- **Multi-Model Routing**: Claude, GPT, local models with intelligent failover
- **Agent Orchestration**: ReAct agents, parallel/competitive/consensus execution modes
- **Production API**: FastAPI REST + WebSocket + OpenAI-compatible chat endpoint
- **CLI Interface**: Full command-line interface with streaming support

## Quick Start

```bash
# Install
cd dzogchen && pip install -e ".[all]"

# Configure
export ANTHROPIC_API_KEY="your-api-key"

# Use CLI
dzogchen think "What is consciousness?"
dzogchen act "Research quantum computing breakthroughs"
dzogchen swarm --size 5 "Design the perfect programming language"
dzogchen chat      # Interactive chat
dzogchen serve     # Start API server (http://localhost:8080)
```

## Architecture

```
Agentic-OS/
├── dzogchen/              # The Dzogchen AI Framework
│   ├── consciousness/     # Awareness, reflection, metacognition
│   ├── intelligence/      # Reasoning, planning, synthesis
│   ├── memory/            # Working, episodic, semantic memory
│   ├── models/            # Claude, OpenAI adapters + router
│   ├── agents/            # ReAct agents, orchestrator, swarm
│   ├── tools/             # Tool registry, executor, built-ins
│   ├── api/               # FastAPI REST + WebSocket server
│   ├── cli/               # Click CLI interface
│   ├── core.py            # Main Dzogchen class
│   └── config.py          # Configuration management
├── src/                   # Agentic OS core
│   ├── core/              # Agent, coordination, knowledge
│   ├── capabilities/      # Network, system monitor
│   └── utils/             # Error handling, monitoring
└── tests/                 # Test suite
```

## Contributing

Please read our [Contributing Guide](CONTRIBUTING.md) before making a pull request.

## License

MIT