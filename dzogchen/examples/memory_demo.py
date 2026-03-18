"""
Memory Systems Demo
====================
Demonstrates Dzogchen's multi-tier memory architecture:
- Working Memory (L1): fast, capacity-limited, activation-based
- Episodic Memory (L2): autobiographical, temporal, searchable
- Semantic Memory (L3): facts, concepts, knowledge
- Vector Store: embedding-based similarity search

Run:
    python dzogchen/examples/memory_demo.py
"""

import asyncio
import time
from dzogchen.memory import WorkingMemory, EpisodicMemory, SemanticMemory, VectorStore
from dzogchen.memory.manager import MemoryManager
from dzogchen.memory.vector import _mock_embedding


def demo_working_memory():
    """Demonstrate L1 Working Memory."""
    print("\n[L1] Working Memory")
    print("-" * 50)

    wm = WorkingMemory(capacity=5)

    # Store items with different priorities
    items = [
        ("current_task", "debugging the neural network", 0.9),
        ("user_name", "Alice", 0.7),
        ("context_topic", "machine learning", 0.6),
        ("last_tool", "calculate", 0.5),
        ("session_goal", "answer user questions", 0.8),
    ]

    for key, value, priority in items:
        wm.put(key, value, priority=priority)

    print(f"  Stored {wm.size}/{wm.capacity} items")
    print(f"  Most active: {wm.get_most_active(n=3)}")

    # Access an item (increases its activation)
    _ = wm.get("current_task")
    _ = wm.get("current_task")

    activations = wm.get_activation_map()
    print("\n  Activation levels:")
    for key, activation in sorted(activations.items(), key=lambda x: -x[1]):
        print(f"    {key:20s}: {activation:.3f}")

    # TTL expiry demo
    wm.put("temp_data", "expires soon", ttl=0.05)
    print(f"\n  Before TTL: 'temp_data' = {wm.get('temp_data')}")
    time.sleep(0.1)
    print(f"  After TTL:  'temp_data' = {wm.get('temp_data')} (expired)")

    # Overflow eviction
    print("\n  Adding 3 more items (capacity={wm.capacity}):")
    for i in range(3):
        wm.put(f"overflow_{i}", f"value_{i}", priority=0.2)
    print(f"  Size after overflow: {wm.size}/{wm.capacity} (eviction occurred)")


def demo_episodic_memory():
    """Demonstrate L2 Episodic Memory."""
    print("\n[L2] Episodic Memory")
    print("-" * 50)

    em = EpisodicMemory(max_episodes=1000)

    # Store a conversation history
    session_id = "demo_session"
    interactions = [
        ("interaction", {"input": "What is Python?", "response": "Python is a high-level programming language."}),
        ("interaction", {"input": "Who created it?", "response": "Guido van Rossum created Python in 1991."}),
        ("tool_execution", {"tool": "calculate", "params": {"expr": "2+2"}}),
        ("error", {"type": "TimeoutError", "tool": "web_search", "message": "Request timed out"}),
        ("interaction", {"input": "What are Python's uses?", "response": "Python is used in ML, web dev, automation..."}),
    ]

    for event_type, content in interactions:
        importance = 0.9 if event_type == "error" else 0.7
        em.store(event_type, content, importance=importance, session_id=session_id)

    print(f"  Stored {em.total_episodes} episodes")

    # Retrieve conversation
    history = em.get_session_history(session_id)
    print(f"  Session history: {len(history)} events")

    # Search
    results = em.recall(query="Python", limit=5)
    print(f"  Search 'Python': {len(results)} results")

    # Get important events
    important = em.get_important(n=5, min_importance=0.85)
    print(f"  High-importance events: {len(important)}")
    for ep in important:
        print(f"    [{ep.event_type}] importance={ep.importance:.2f}")

    # Patterns
    patterns = em.get_patterns("interaction", n=3)
    print(f"\n  Interaction context patterns: {len(patterns)}")


def demo_semantic_memory():
    """Demonstrate L3 Semantic Memory."""
    print("\n[L3] Semantic Memory")
    print("-" * 50)

    sm = SemanticMemory(max_entries=10000)

    # Store knowledge
    knowledge = [
        ("Python", "A high-level, general-purpose programming language emphasizing readability", "programming"),
        ("Machine Learning", "A field of AI that enables systems to learn from data", "ai"),
        ("Neural Network", "A computational model inspired by biological neural networks", "ai"),
        ("Transformer", "An attention-based neural network architecture for sequence modeling", "ai"),
        ("Docker", "A platform for containerizing applications", "devops"),
        ("REST API", "An architectural style for distributed hypermedia systems", "architecture"),
        ("Recursion", "A technique where a function calls itself", "programming"),
        ("Gradient Descent", "An optimization algorithm for minimizing a loss function", "ai"),
    ]

    for concept, definition, domain in knowledge:
        sm.store(concept, definition, domain=domain, confidence=0.95)

    print(f"  Stored {sm.total_entries} knowledge entries")
    print(f"  Domains: {sm.domains}")

    # Exact lookup
    entry = sm.lookup("Python")
    if entry:
        print(f"\n  Lookup 'Python': {entry.definition[:60]}...")
        print(f"    Domain: {entry.domain}, Confidence: {entry.confidence}")

    # Semantic search
    results = sm.search("attention mechanism neural", domain="ai")
    print(f"\n  Search 'attention mechanism neural' in AI domain:")
    for r in results[:3]:
        print(f"    [{r.confidence:.2f}] {r.concept}: {r.definition[:50]}...")

    # Related concepts
    related = sm.get_related("Machine Learning", depth=1)
    print(f"\n  Related to 'Machine Learning': {[r.concept for r in related]}")

    # Confidence update (learning from corrections)
    sm.update_confidence("Python", 0.05)
    updated = sm.lookup("Python")
    print(f"\n  Python confidence after correction: {updated.confidence:.2f}")


async def demo_vector_store():
    """Demonstrate the Vector Store."""
    print("\n[Vector] Embedding-based Similarity Search")
    print("-" * 50)

    # Use mock embeddings (no API key needed)
    store = VectorStore(embed_fn=None, dimension=64, backend="pure")

    # Add some entries
    facts = [
        ("Python is a high-level programming language", {"type": "fact", "domain": "programming"}),
        ("Machine learning uses statistical methods", {"type": "fact", "domain": "ai"}),
        ("Neural networks are inspired by brains", {"type": "fact", "domain": "ai"}),
        ("Docker containers are lightweight VMs", {"type": "fact", "domain": "devops"}),
        ("REST APIs use HTTP verbs", {"type": "fact", "domain": "web"}),
        ("Git tracks source code changes", {"type": "fact", "domain": "devops"}),
        ("Transformers use attention mechanisms", {"type": "fact", "domain": "ai"}),
    ]

    for content, metadata in facts:
        await store.add(content, metadata=metadata)

    print(f"  Stored {store.size} entries")

    # Similarity search (using mock embeddings - not truly semantic)
    results = await store.search("deep learning AI", k=3)
    print(f"\n  Search 'deep learning AI': {len(results)} results")
    for r in results:
        print(f"    [{r.score:.3f}] {r.entry.content}")

    # Filtered search
    ai_results = await store.search(
        "computation",
        k=3,
        filter_fn=lambda e: e.metadata.get("domain") == "ai",
    )
    print(f"\n  Filtered to AI domain: {len(ai_results)} results")
    for r in ai_results:
        print(f"    [{r.score:.3f}] {r.entry.content}")

    print(f"\n  Store stats: {store.get_stats()}")


async def demo_memory_manager():
    """Demonstrate the unified MemoryManager."""
    print("\n[Manager] Unified Memory Manager")
    print("-" * 50)

    mm = MemoryManager(
        working_capacity=5,
        episodic_max=1000,
        semantic_max=10000,
    )
    mm.set_session("demo")

    # Cross-system storage
    mm.remember("current_topic", "AI consciousness", priority=0.8)
    mm.learn_fact("consciousness", "The state of being aware of one's existence", domain="philosophy")
    episode = mm.store_interaction(
        "What is consciousness?",
        "Consciousness is the state of subjective awareness...",
        quality=0.9,
    )

    # Cross-system recall
    results = mm.recall("consciousness")
    print(f"  Working memory hits: {len(results['working'])}")
    print(f"  Episodic memory hits: {len(results['episodic'])}")
    print(f"  Semantic memory hits: {len(results['semantic'])}")

    # Context snapshot
    ctx = mm.get_context()
    print(f"\n  Context snapshot:")
    print(f"    Working items: {len(ctx['working_memory'])}")
    print(f"    Recent episodes: {len(ctx['recent_episodes'])}")
    print(f"    Key facts: {len(ctx['key_facts'])}")

    # Conversation history
    history = mm.get_conversation_history(limit=5)
    print(f"\n  Conversation history: {len(history)} messages")

    # Full stats
    stats = mm.get_stats()
    print(f"\n  Memory system stats:")
    print(f"    Working:  {stats['working']['size']}/{stats['working']['capacity']} items")
    print(f"    Episodic: {stats['episodic']['total']} episodes")
    print(f"    Semantic: {stats['semantic']['total_entries']} facts")


async def main():
    print("=" * 60)
    print("  Dzogchen Memory Systems Demo")
    print("=" * 60)

    demo_working_memory()
    demo_episodic_memory()
    demo_semantic_memory()
    await demo_vector_store()
    await demo_memory_manager()

    print("\n" + "=" * 60)
    print("  Memory Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
