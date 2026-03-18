"""
Swarm Intelligence Demo
========================
Demonstrates Dzogchen's swarm intelligence capabilities:
- Collective problem solving
- Parallel exploration
- Ant-colony pheromone coordination
- Emergent consensus

Run:
    export ANTHROPIC_API_KEY="your-key"  # Optional - works without API key too
    python dzogchen/examples/swarm_demo.py
"""

import asyncio
from dzogchen.agents.swarm import SwarmIntelligence
from dzogchen.tools.builtin import calculate, get_current_time, text_stats


async def demo_collective_solving():
    """Collective problem solving without an API key (simulated)."""
    print("\n[1] Collective Problem Solving")
    print("-" * 50)

    swarm = SwarmIntelligence(
        swarm_size=3,
        model_caller=None,  # No API key needed for this demo
    )

    print(f"  Swarm ID: {swarm.swarm_id}")
    print(f"  Agents: {len(swarm._agents)}")
    for agent in swarm._agents:
        print(f"    - {agent.name}")

    # Show swarm stats
    stats = swarm.get_swarm_stats()
    print(f"\n  Active pheromones: {stats['active_pheromones']}")
    print(f"  Shared knowledge: {stats['shared_knowledge_items']} items")


async def demo_parallel_exploration():
    """Parallel exploration of multiple problems."""
    print("\n[2] Parallel Exploration")
    print("-" * 50)

    swarm = SwarmIntelligence(swarm_size=3, model_caller=None)
    tools = {
        "calculate": calculate,
        "get_current_time": get_current_time,
    }

    problems = [
        "What is 144 + 256?",
        "What time is it right now?",
        "What is the square root of 625?",
    ]

    print(f"  Exploring {len(problems)} problems with {len(swarm._agents)} agents...")
    results = await swarm.explore_parallel(problems, tools=tools)

    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} [{r['agent']}] {r['problem'][:40]}...")
        if r["result"]:
            print(f"      -> {str(r['result'])[:80]}")


async def demo_pheromone_mechanics():
    """Show pheromone-based coordination mechanics."""
    print("\n[3] Pheromone Coordination")
    print("-" * 50)

    swarm = SwarmIntelligence(
        swarm_size=4,
        evaporation_rate=0.02,
        exploration_rate=0.3,
    )

    # Simulate pheromone deposition
    swarm._deposit_pheromone("path_a", strength=0.9, task_type="reasoning")
    swarm._deposit_pheromone("path_b", strength=0.4, task_type="reasoning")
    swarm._deposit_pheromone("path_a", strength=0.8, task_type="reasoning")  # Reinforce

    print(f"  Active pheromone trails: {len(swarm._pheromones)}")
    for ph in swarm._pheromones:
        print(f"    [{ph.path}] strength={ph.current_strength:.2f} type={ph.task_type}")

    # Simulate evaporation
    await asyncio.sleep(0.1)
    swarm._evaporate_pheromones()
    print(f"\n  After evaporation: {len(swarm._pheromones)} trails")
    for ph in swarm._pheromones:
        print(f"    [{ph.path}] strength={ph.current_strength:.2f}")


async def demo_discovery_sharing():
    """Show how swarm agents share discoveries."""
    print("\n[4] Discovery Sharing")
    print("-" * 50)

    from dzogchen.agents.swarm import SwarmAgent

    # Create a mini swarm
    agent1 = SwarmAgent(name="Scout-1")
    agent2 = SwarmAgent(name="Scout-2")
    agent3 = SwarmAgent(name="Scout-3")

    agents = [agent1, agent2, agent3]
    all_ids = [a.agent_id for a in agents]

    # Connect agents
    for agent in agents:
        agent.connect_to_peers([aid for aid in all_ids if aid != agent.agent_id])

    # Agent 1 makes a discovery
    agent1.share_discovery("optimal_path", "Use deductive reasoning first", confidence=0.85)
    print(f"  Agent 1 shared {len(agent1._discoveries)} discovery")

    # Broadcast to others
    for discovery in agent1._discoveries:
        for other in [agent2, agent3]:
            other.receive_discovery(discovery)

    # Check shared knowledge
    for agent in agents:
        has_knowledge = "optimal_path" in agent._shared_memory
        print(f"  {agent.name}: has shared knowledge = {has_knowledge}")


async def demo_consensus():
    """Demonstrate consensus measurement."""
    print("\n[5] Consensus Measurement")
    print("-" * 50)

    swarm = SwarmIntelligence(swarm_size=3)

    # High consensus solutions
    high_consensus = [
        {"solution": "Python is a great programming language for beginners", "confidence": 0.9},
        {"solution": "Python is an excellent programming language for beginners", "confidence": 0.85},
        {"solution": "Python is a wonderful programming language for learning", "confidence": 0.88},
    ]

    # Low consensus solutions
    low_consensus = [
        {"solution": "Go is the best language", "confidence": 0.7},
        {"solution": "Rust is superior in all ways", "confidence": 0.8},
        {"solution": "Python dominates the field", "confidence": 0.75},
    ]

    high = swarm._measure_consensus(high_consensus)
    low = swarm._measure_consensus(low_consensus)

    print(f"  High consensus solutions: {high:.2%} agreement")
    print(f"  Low consensus solutions:  {low:.2%} agreement")
    print(f"\n  {'✓' if high > low else '✗'} Consensus detection working correctly")


async def main():
    print("=" * 60)
    print("  Dzogchen Swarm Intelligence Demo")
    print("=" * 60)

    await demo_collective_solving()
    await demo_parallel_exploration()
    await demo_pheromone_mechanics()
    await demo_discovery_sharing()
    await demo_consensus()

    print("\n" + "=" * 60)
    print("  Swarm Demo complete!")
    print("  To use with real LLMs: pass model_caller=your_caller")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
