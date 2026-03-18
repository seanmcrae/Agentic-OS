"""
Dzogchen Quickstart
====================
The fastest way to get started with the Dzogchen AI framework.

Run:
    export ANTHROPIC_API_KEY="your-key"
    python dzogchen/examples/quickstart.py
"""

import asyncio
from dzogchen import Dzogchen


async def main():
    print("=" * 60)
    print("  Dzogchen AI Framework - Quickstart")
    print("=" * 60)

    async with Dzogchen() as dz:

        # ── 1. Simple reasoning ──────────────────────────────────
        print("\n[1] Chain of Thought Reasoning")
        print("-" * 40)
        answer = await dz.think(
            "What makes a programming language 'good'?",
            strategy="cot",
            reflect=True,
        )
        print(answer)

        # ── 2. Teach a fact ──────────────────────────────────────
        print("\n[2] Teaching a Fact")
        print("-" * 40)
        await dz.learn(
            "Dzogchen",
            "A Tibetan Buddhist concept meaning 'Great Perfection' - "
            "the primordial state of awareness, complete and self-arising.",
            domain="philosophy",
        )
        print("Taught: Dzogchen (philosophy domain)")

        # ── 3. Memory recall ─────────────────────────────────────
        print("\n[3] Memory Recall")
        print("-" * 40)
        memories = await dz.remember("Dzogchen")
        semantic_hits = memories.get("semantic", [])
        if semantic_hits:
            print(f"Found: {semantic_hits[0]['concept']} -> {semantic_hits[0]['definition'][:80]}...")

        # ── 4. System status ─────────────────────────────────────
        print("\n[4] System Status")
        print("-" * 40)
        status = dz.get_status()
        print(f"  Version:   {status['version']}")
        print(f"  Session:   {status['session_id']}")
        print(f"  Model:     {status['model']}")
        mem = status["memory"]
        print(f"  Working:   {mem.get('working', {}).get('size', 0)} items")
        print(f"  Episodic:  {mem.get('episodic', {}).get('total', 0)} episodes")
        print(f"  Semantic:  {mem.get('semantic', {}).get('total_entries', 0)} entries")

        # ── 5. Knowledge synthesis ───────────────────────────────
        print("\n[5] Knowledge Synthesis")
        print("-" * 40)
        synthesis = await dz.synthesize(
            "What is Dzogchen?",
            fragments=[
                {"content": "Dzogchen is a teaching tradition in Tibetan Buddhism", "source": "wikipedia"},
                {"content": "The term means 'Great Perfection' or 'Great Completeness'", "source": "dictionary"},
                {"content": "It refers to the primordial state of every human being", "source": "lama"},
            ]
        )
        print(synthesis[:400] + "..." if len(synthesis) > 400 else synthesis)

    print("\n" + "=" * 60)
    print("  Quickstart complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
