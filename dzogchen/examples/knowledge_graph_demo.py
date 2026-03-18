"""
Knowledge Graph Demo
=====================
Demonstrates Dzogchen's knowledge graph capabilities:
- Building a graph from facts
- Graph traversal and pathfinding
- Inference and pattern queries
- LLM-powered knowledge extraction

Run:
    python dzogchen/examples/knowledge_graph_demo.py
"""

import asyncio
from dzogchen.knowledge.graph import KnowledgeGraph
from dzogchen.knowledge.builder import KnowledgeGraphBuilder


def demo_basic_graph():
    """Build a knowledge graph about AI and tech."""
    print("\n[1] Building a Technology Knowledge Graph")
    print("-" * 50)

    kg = KnowledgeGraph()

    # Add nodes
    python = kg.add_node("Language", "Python", properties={"year": 1991, "paradigm": "multi-paradigm"})
    guido = kg.add_node("Person", "Guido van Rossum")
    llm = kg.add_node("Concept", "Large Language Model")
    transformer = kg.add_node("Concept", "Transformer Architecture")
    attention = kg.add_node("Concept", "Attention Mechanism")
    vaswani = kg.add_node("Person", "Ashish Vaswani")
    google = kg.add_node("Organization", "Google")
    openai = kg.add_node("Organization", "OpenAI")
    gpt4 = kg.add_node("Technology", "GPT-4")
    claude = kg.add_node("Technology", "Claude")
    anthropic = kg.add_node("Organization", "Anthropic")
    ml = kg.add_node("Concept", "Machine Learning")
    deep_learning = kg.add_node("Concept", "Deep Learning")

    # Add edges
    kg.add_edge(guido, python, "created")
    kg.add_edge(transformer, attention, "uses")
    kg.add_edge(vaswani, transformer, "invented")
    kg.add_edge(llm, transformer, "uses")
    kg.add_edge(gpt4, llm, "is_a")
    kg.add_edge(claude, llm, "is_a")
    kg.add_edge(openai, gpt4, "created_by")
    kg.add_edge(anthropic, claude, "created_by")
    kg.add_edge(deep_learning, ml, "is_a")
    kg.add_edge(llm, deep_learning, "uses")
    kg.add_edge(google, transformer, "invented_by")
    kg.add_edge(vaswani, google, "works_at")

    stats = kg.get_stats()
    print(f"  Nodes: {stats['node_count']}, Edges: {stats['edge_count']}")
    print(f"  Labels: {list(stats['labels'].keys())}")
    print(f"  Relations: {list(stats['relations'].keys())}")

    return kg


def demo_traversal(kg: KnowledgeGraph):
    """Demonstrate graph traversal and pathfinding."""
    print("\n[2] Graph Traversal & Pathfinding")
    print("-" * 50)

    # Shortest path
    path = kg.shortest_path("Ashish Vaswani", "Claude")
    if path:
        print(f"  Path from Vaswani to Claude ({path.length} hops):")
        print(f"    {path.to_text()}")
    else:
        print("  No path found between Vaswani and Claude")

    # BFS traversal
    print("\n  BFS from 'Large Language Model' (depth=2):")
    reachable = kg.bfs("Large Language Model", max_depth=2)
    for node in reachable[:8]:
        print(f"    - {node.label}: {node.name}")

    # All paths
    paths = kg.all_paths("Ashish Vaswani", "GPT-4", max_hops=4)
    print(f"\n  All paths from Vaswani to GPT-4: {len(paths)} found")
    for p in paths[:3]:
        print(f"    [{p.length} hops] {p.to_text()}")


def demo_queries(kg: KnowledgeGraph):
    """Demonstrate pattern matching queries."""
    print("\n[3] Pattern Queries")
    print("-" * 50)

    # Find all things that are a type of "Machine Learning"
    result = kg.query("? is_a Machine Learning")
    print(f"  '? is_a Machine Learning': {[n.name for n in result.nodes]}")

    # Find what LLMs use
    result = kg.query("Large Language Model uses ?")
    print(f"  'LLM uses ?': {[n.name for n in result.nodes]}")

    # Find common neighbors
    common = kg.find_common_neighbors("GPT-4", "Claude")
    print(f"  Common neighbors of GPT-4 and Claude: {[n.name for n in common]}")


def demo_inference(kg: KnowledgeGraph):
    """Demonstrate graph inference."""
    print("\n[4] Graph Inference")
    print("-" * 50)

    # Transitive closure of is_a
    transitives = kg.infer_transitive("is_a", "GPT-4", max_depth=5)
    print("  Transitive 'is_a' from GPT-4:")
    for node, depth in transitives:
        print(f"    {'  ' * depth}-> {node.name}")

    # Infer inverse relations
    inverses = kg.infer_inverse("created", "was_created_by")
    print(f"\n  Inferred {len(inverses)} inverse 'was_created_by' edges")

    # Degree statistics
    stats = kg.get_stats()
    print(f"\n  Avg out-degree: {stats['avg_out_degree']:.2f}")
    print(f"  Max degree: {stats['max_degree']}")


async def demo_builder():
    """Demonstrate LLM-powered knowledge extraction."""
    print("\n[5] Automatic Knowledge Extraction")
    print("-" * 50)

    kg = KnowledgeGraph()
    builder = KnowledgeGraphBuilder(graph=kg)  # No model_caller = heuristic extraction

    text = """
    Python was created by Guido van Rossum in 1991. It is a high-level programming language
    that emphasizes code readability. Python is part of the Open Source community and is used
    extensively in machine learning and data science.

    TensorFlow is a machine learning framework created by Google. PyTorch is a competing
    framework developed by Meta (formerly Facebook). Both TensorFlow and PyTorch use Python
    as their primary interface and support GPU acceleration.
    """

    triples = await builder.extract_from_text(text, source="demo_text", add_to_graph=True)
    print(f"  Extracted {len(triples)} triples from text")
    for t in triples[:5]:
        print(f"    [{t.subject_type}] {t.subject} --{t.predicate}--> [{t.object_type}] {t.object}")

    print(f"\n  Graph now has {len(kg)} nodes")

    # Find missing links
    suggestions = await builder.find_missing_links(max_candidates=5)
    print(f"\n  Missing link suggestions ({len(suggestions)}):")
    for node_a, node_b, relation, confidence in suggestions[:3]:
        print(f"    {node_a.name} --[{relation}]--> {node_b.name} (conf={confidence:.2f})")


def demo_serialization(kg: KnowledgeGraph):
    """Demonstrate graph export and import."""
    print("\n[6] Serialization")
    print("-" * 50)

    # Export
    data = kg.to_dict()
    print(f"  Exported: {data['metadata']['node_count']} nodes, {data['metadata']['edge_count']} edges")

    # Import into new graph
    kg2 = KnowledgeGraph.from_dict(data)
    print(f"  Imported into new graph: {len(kg2)} nodes")

    # Verify data integrity
    node = kg2.find_node("Python")
    if node:
        print(f"  'Python' node intact: label={node.label}, year={node.properties.get('year')}")


async def main():
    print("=" * 60)
    print("  Dzogchen Knowledge Graph Demo")
    print("=" * 60)

    kg = demo_basic_graph()
    demo_traversal(kg)
    demo_queries(kg)
    demo_inference(kg)
    await demo_builder()
    demo_serialization(kg)

    print("\n" + "=" * 60)
    print("  Knowledge Graph Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
