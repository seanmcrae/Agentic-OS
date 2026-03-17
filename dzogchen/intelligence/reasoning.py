"""
ReasoningEngine - Multi-strategy reasoning with chain/tree of thought
=====================================================================

Implements multiple reasoning paradigms:
- Chain of Thought (CoT): step-by-step sequential reasoning
- Tree of Thought (ToT): branching exploration with backtracking
- Self-consistency: multiple reasoning paths, majority vote
- Analogical reasoning: problem-solving by analogy
- Deductive/Inductive/Abductive reasoning modes
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""
    step_number: int
    thought: str
    conclusion: Optional[str] = None
    confidence: float = 0.7
    alternatives_considered: List[str] = field(default_factory=list)
    evidence_used: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReasoningTrace:
    """Complete trace of a reasoning process."""
    query: str
    strategy: str
    steps: List[ReasoningStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    confidence: float = 0.0
    reasoning_time: float = 0.0
    tokens_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, thought: str, conclusion: Optional[str] = None, confidence: float = 0.7) -> ReasoningStep:
        step = ReasoningStep(
            step_number=len(self.steps) + 1,
            thought=thought,
            conclusion=conclusion,
            confidence=confidence,
        )
        self.steps.append(step)
        return step

    def summarize(self) -> str:
        """Summarize the reasoning trace."""
        lines = [f"Query: {self.query}", f"Strategy: {self.strategy}", "Steps:"]
        for step in self.steps:
            lines.append(f"  {step.step_number}. {step.thought}")
            if step.conclusion:
                lines.append(f"     -> {step.conclusion}")
        if self.final_answer:
            lines.append(f"Answer: {self.final_answer}")
        lines.append(f"Confidence: {self.confidence:.2f}")
        return "\n".join(lines)


@dataclass
class ThoughtNode:
    """A node in the Tree of Thought search tree."""
    thought: str
    value: float = 0.0
    children: List["ThoughtNode"] = field(default_factory=list)
    parent: Optional["ThoughtNode"] = None
    depth: int = 0
    is_terminal: bool = False
    answer: Optional[str] = None

    def add_child(self, thought: str) -> "ThoughtNode":
        child = ThoughtNode(thought=thought, parent=self, depth=self.depth + 1)
        self.children.append(child)
        return child

    def get_path(self) -> List[str]:
        """Get the path from root to this node."""
        path = []
        node = self
        while node is not None:
            path.insert(0, node.thought)
            node = node.parent
        return path


class ReasoningEngine:
    """
    Multi-strategy reasoning engine with adaptive strategy selection.

    Supports Chain of Thought, Tree of Thought, Self-Consistency,
    and various formal reasoning modes.
    """

    COT_SYSTEM_PROMPT = """You are a precise, rigorous thinker. Break down complex problems
step by step, showing your reasoning clearly. For each step:
1. State what you're thinking
2. Show your reasoning
3. Draw an intermediate conclusion
End with a clear final answer."""

    TOT_EVALUATION_PROMPT = """Rate the following reasoning step on a scale of 0.0 to 1.0
for how likely it leads to a correct solution. Consider:
- Logical soundness
- Progress toward the goal
- Avoidance of dead ends
Reply with just a number between 0.0 and 1.0."""

    def __init__(
        self,
        model_caller: Optional[Callable] = None,
        max_steps: int = 20,
        tot_branching_factor: int = 3,
        tot_beam_width: int = 5,
        self_consistency_samples: int = 5,
    ):
        self._model_caller = model_caller
        self.max_steps = max_steps
        self.tot_branching_factor = tot_branching_factor
        self.tot_beam_width = tot_beam_width
        self.self_consistency_samples = self_consistency_samples
        self._reasoning_history: List[ReasoningTrace] = []

    async def reason(
        self,
        query: str,
        strategy: str = "cot",
        context: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> ReasoningTrace:
        """
        Apply the specified reasoning strategy to answer a query.

        Args:
            query: The question or problem to reason about
            strategy: "cot" | "tot" | "self_consistency" | "deductive" | "inductive"
            context: Additional context information
            stream: If True, yield intermediate steps as they're generated

        Returns:
            Complete ReasoningTrace with all steps and final answer
        """
        start_time = time.time()

        if strategy == "cot":
            trace = await self._chain_of_thought(query, context)
        elif strategy == "tot":
            trace = await self._tree_of_thought(query, context)
        elif strategy == "self_consistency":
            trace = await self._self_consistency(query, context)
        elif strategy == "deductive":
            trace = await self._deductive_reasoning(query, context)
        elif strategy == "inductive":
            trace = await self._inductive_reasoning(query, context)
        elif strategy == "abductive":
            trace = await self._abductive_reasoning(query, context)
        else:
            trace = await self._chain_of_thought(query, context)

        trace.reasoning_time = time.time() - start_time
        self._reasoning_history.append(trace)
        return trace

    async def stream_reasoning(
        self,
        query: str,
        strategy: str = "cot",
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[ReasoningStep, None]:
        """Stream reasoning steps as they're generated."""
        trace = ReasoningTrace(query=query, strategy=strategy)

        # For CoT streaming
        if strategy == "cot":
            async for step in self._stream_cot_steps(query, context, trace):
                yield step
        else:
            # Non-streaming fallback
            completed_trace = await self.reason(query, strategy, context)
            for step in completed_trace.steps:
                yield step

    async def _chain_of_thought(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningTrace:
        """Standard Chain of Thought reasoning."""
        trace = ReasoningTrace(query=query, strategy="chain_of_thought")

        if self._model_caller is None:
            trace.add_step(
                f"Analyzing: {query}",
                "Unable to reason without model",
                0.3,
            )
            trace.final_answer = "Model not available"
            trace.confidence = 0.0
            return trace

        prompt = f"""{self.COT_SYSTEM_PROMPT}

Question: {query}

Context: {context or 'None provided'}

Let's think step by step:"""

        response = await self._model_caller(prompt)

        # Parse the response into steps
        lines = response.split("\n")
        step_num = 0
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                step_num += 1
                trace.add_step(line, confidence=0.8)

        # Extract final answer
        if "Answer:" in response:
            final_idx = response.index("Answer:")
            trace.final_answer = response[final_idx + 7:].strip()
        elif step_num > 0:
            trace.final_answer = response.split("\n")[-1].strip()

        trace.confidence = 0.8
        return trace

    async def _tree_of_thought(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningTrace:
        """Tree of Thought reasoning with BFS/beam search."""
        trace = ReasoningTrace(query=query, strategy="tree_of_thought")

        root = ThoughtNode(thought=f"Problem: {query}")
        beam = [root]

        for depth in range(min(self.max_steps // self.tot_branching_factor, 5)):
            next_beam = []

            for node in beam:
                # Generate branches from this node
                branches = await self._generate_thought_branches(
                    query, node.get_path(), self.tot_branching_factor
                )

                for branch_thought in branches:
                    child = node.add_child(branch_thought)

                    # Evaluate this thought
                    value = await self._evaluate_thought(query, child.get_path())
                    child.value = value

                    # Check if terminal (contains an answer)
                    if any(marker in branch_thought.lower() for marker in ["answer:", "conclusion:", "therefore:"]):
                        child.is_terminal = True
                        child.answer = branch_thought

                    next_beam.append(child)

                    # Add step to trace
                    trace.add_step(
                        branch_thought,
                        conclusion=f"Evaluated: {value:.2f}",
                        confidence=value,
                    )

            # Keep best candidates (beam search)
            next_beam.sort(key=lambda n: n.value, reverse=True)
            beam = next_beam[:self.tot_beam_width]

            # Check for terminal nodes in beam
            terminal_nodes = [n for n in beam if n.is_terminal]
            if terminal_nodes:
                best_terminal = max(terminal_nodes, key=lambda n: n.value)
                trace.final_answer = best_terminal.answer
                trace.confidence = best_terminal.value
                return trace

        # No terminal found - use best path
        if beam:
            best_node = max(beam, key=lambda n: n.value)
            path = best_node.get_path()
            trace.final_answer = path[-1] if path else "Unable to reach conclusion"
            trace.confidence = best_node.value

        return trace

    async def _self_consistency(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningTrace:
        """Self-consistency: run CoT multiple times, take majority vote."""
        trace = ReasoningTrace(query=query, strategy="self_consistency")

        if self._model_caller is None:
            return await self._chain_of_thought(query, context)

        # Run multiple CoT samples
        samples = await asyncio.gather(*[
            self._chain_of_thought(query, context)
            for _ in range(self.self_consistency_samples)
        ])

        # Collect answers
        answers = [s.final_answer for s in samples if s.final_answer]

        # Find most common answer (simple majority vote)
        if not answers:
            trace.final_answer = None
            trace.confidence = 0.0
            return trace

        answer_counts: Dict[str, int] = {}
        for answer in answers:
            answer_counts[answer] = answer_counts.get(answer, 0) + 1

        best_answer = max(answer_counts, key=answer_counts.get)  # type: ignore
        vote_ratio = answer_counts[best_answer] / len(answers)

        # Add summary step
        trace.add_step(
            f"Ran {self.self_consistency_samples} independent reasoning paths",
            f"Majority answer ({vote_ratio:.0%} agreement): {best_answer}",
            vote_ratio,
        )

        trace.final_answer = best_answer
        trace.confidence = vote_ratio

        # Include all sample traces
        for i, sample in enumerate(samples):
            trace.metadata[f"sample_{i}"] = sample.summarize()

        return trace

    async def _deductive_reasoning(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningTrace:
        """Formal deductive reasoning from premises to conclusions."""
        trace = ReasoningTrace(query=query, strategy="deductive")

        if self._model_caller is None:
            return await self._chain_of_thought(query, context)

        prompt = f"""Apply formal deductive reasoning to answer this question.

Question: {query}
Context: {context or 'None'}

Format your response as:
PREMISES:
1. [stated fact or assumption]
2. [stated fact or assumption]
...

DEDUCTIONS:
1. From premises X and Y, we can deduce: [logical conclusion]
2. From deduction 1 and premise Z: [further conclusion]
...

CONCLUSION:
[Final definitive answer]"""

        response = await self._model_caller(prompt)
        trace.add_step("Applied deductive reasoning", response[:200], 0.85)
        trace.final_answer = self._extract_conclusion(response)
        trace.confidence = 0.85
        return trace

    async def _inductive_reasoning(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningTrace:
        """Inductive reasoning from observations to generalizations."""
        trace = ReasoningTrace(query=query, strategy="inductive")

        if self._model_caller is None:
            return await self._chain_of_thought(query, context)

        prompt = f"""Apply inductive reasoning to answer this question.

Question: {query}
Context: {context or 'None'}

Format your response as:
OBSERVATIONS:
1. [specific observation or data point]
2. [specific observation or data point]
...

PATTERNS:
1. [pattern identified from observations]
...

GENERALIZATION:
[General principle inferred from patterns]

ANSWER:
[Answer to the question based on inductive reasoning, with confidence level]"""

        response = await self._model_caller(prompt)
        trace.add_step("Applied inductive reasoning", response[:200], 0.75)
        trace.final_answer = self._extract_answer(response)
        trace.confidence = 0.75  # Induction is inherently probabilistic
        return trace

    async def _abductive_reasoning(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningTrace:
        """Abductive reasoning - inference to best explanation."""
        trace = ReasoningTrace(query=query, strategy="abductive")

        if self._model_caller is None:
            return await self._chain_of_thought(query, context)

        prompt = f"""Apply abductive reasoning (inference to best explanation).

Question/Observation: {query}
Context: {context or 'None'}

Format your response as:
OBSERVATIONS TO EXPLAIN:
[List what we know and need to explain]

CANDIDATE EXPLANATIONS:
1. [Hypothesis A]: [Why this could explain the observations]
2. [Hypothesis B]: [Why this could explain the observations]
3. [Hypothesis C]: [Why this could explain the observations]

EVALUATION:
[Compare explanations on simplicity, scope, fit with evidence]

BEST EXPLANATION:
[Most plausible explanation and why]"""

        response = await self._model_caller(prompt)
        trace.add_step("Applied abductive reasoning", response[:200], 0.78)
        trace.final_answer = self._extract_best_explanation(response)
        trace.confidence = 0.78
        return trace

    async def _generate_thought_branches(
        self,
        query: str,
        current_path: List[str],
        n_branches: int,
    ) -> List[str]:
        """Generate thought branches for Tree of Thought."""
        if self._model_caller is None:
            return [f"Branch {i}: Consider approach {i}" for i in range(n_branches)]

        path_str = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(current_path))
        prompt = f"""Problem: {query}

Reasoning so far:
{path_str}

Generate {n_branches} different next reasoning steps to explore.
Each step should be a distinct approach or consideration.
Format as numbered list."""

        try:
            response = await self._model_caller(prompt)
            branches = [
                line.strip().lstrip("0123456789. ")
                for line in response.split("\n")
                if line.strip() and line.strip()[0].isdigit()
            ]
            return branches[:n_branches] or [response[:200]]
        except Exception as e:
            logger.warning(f"Failed to generate branches: {e}")
            return [f"Step: {i}" for i in range(n_branches)]

    async def _evaluate_thought(self, query: str, path: List[str]) -> float:
        """Evaluate how promising a thought path is (0.0 - 1.0)."""
        if self._model_caller is None:
            return 0.5

        path_str = "\n".join(path)
        prompt = f"""Problem: {query}
Reasoning path:
{path_str}

Rate this reasoning path from 0.0 to 1.0. Reply with only a decimal number."""

        try:
            response = await self._model_caller(prompt)
            value = float(response.strip())
            return max(0.0, min(1.0, value))
        except Exception:
            return 0.5

    async def _stream_cot_steps(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
        trace: ReasoningTrace,
    ) -> AsyncGenerator[ReasoningStep, None]:
        """Stream Chain of Thought steps."""
        completed = await self._chain_of_thought(query, context)
        for step in completed.steps:
            yield step
            await asyncio.sleep(0)  # Allow other coroutines to run

    def _extract_conclusion(self, text: str) -> Optional[str]:
        """Extract the CONCLUSION section from deductive reasoning output."""
        if "CONCLUSION:" in text:
            idx = text.index("CONCLUSION:")
            return text[idx + 11:].strip().split("\n")[0]
        return text.split("\n")[-1].strip()

    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract the ANSWER section from reasoning output."""
        if "ANSWER:" in text:
            idx = text.index("ANSWER:")
            return text[idx + 7:].strip().split("\n")[0]
        return text.split("\n")[-1].strip()

    def _extract_best_explanation(self, text: str) -> Optional[str]:
        """Extract the BEST EXPLANATION section from abductive reasoning output."""
        if "BEST EXPLANATION:" in text:
            idx = text.index("BEST EXPLANATION:")
            return text[idx + 17:].strip()
        return text.split("\n")[-1].strip()

    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get statistics about reasoning history."""
        if not self._reasoning_history:
            return {"total_traces": 0}

        strategy_counts: Dict[str, int] = {}
        for trace in self._reasoning_history:
            strategy_counts[trace.strategy] = strategy_counts.get(trace.strategy, 0) + 1

        avg_confidence = sum(t.confidence for t in self._reasoning_history) / len(self._reasoning_history)
        avg_steps = sum(len(t.steps) for t in self._reasoning_history) / len(self._reasoning_history)

        return {
            "total_traces": len(self._reasoning_history),
            "strategy_distribution": strategy_counts,
            "average_confidence": avg_confidence,
            "average_steps_per_trace": avg_steps,
            "average_reasoning_time": sum(
                t.reasoning_time for t in self._reasoning_history
            ) / len(self._reasoning_history),
        }
