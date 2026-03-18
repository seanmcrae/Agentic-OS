"""Context window compression strategies for managing token budgets."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CompressionStrategy(Enum):
    SLIDING_WINDOW = "sliding_window"
    SUMMARIZE = "summarize"
    SELECTIVE = "selective"
    HYBRID = "hybrid"


@dataclass
class TokenBudget:
    total: int
    used: int
    reserved_for_output: int = 512

    @property
    def available(self) -> int:
        return self.total - self.used - self.reserved_for_output


@dataclass
class CompressedContext:
    messages: List[dict]
    original_count: int
    compressed_count: int
    tokens_saved: int
    strategy_used: CompressionStrategy
    summary: Optional[str] = None


class ContextCompressor:
    """Compresses conversation context to fit within token budgets."""

    def __init__(
        self,
        model_caller=None,
        max_tokens: int = 8000,
        strategy: CompressionStrategy = CompressionStrategy.HYBRID,
        keep_system: bool = True,
        keep_recent: int = 4,
        summarize_threshold: float = 0.8,
    ) -> None:
        self.model_caller = model_caller
        self.max_tokens = max_tokens
        self.strategy = strategy
        self.keep_system = keep_system
        self.keep_recent = keep_recent
        self.summarize_threshold = summarize_threshold

        self._tiktoken_enc = None
        self._tiktoken_available = False
        self._try_load_tiktoken()

    def _try_load_tiktoken(self) -> None:
        try:
            import tiktoken  # type: ignore

            self._tiktoken_enc = tiktoken.get_encoding("cl100k_base")
            self._tiktoken_available = True
        except Exception:
            self._tiktoken_available = False

    def count_tokens(self, text: str) -> int:
        """Count tokens in a string using tiktoken if available, else estimate."""
        if self._tiktoken_available and self._tiktoken_enc is not None:
            try:
                return len(self._tiktoken_enc.encode(text))
            except Exception:
                pass
        return len(text) // 4

    def count_messages(self, messages: List[dict]) -> int:
        """Count total tokens across all messages."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count_tokens(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total += self.count_tokens(str(block.get("text", "")))
            role = msg.get("role", "")
            total += self.count_tokens(role) + 4  # per-message overhead
        return total

    async def compress(
        self,
        messages: List[dict],
        budget: Optional[TokenBudget] = None,
    ) -> CompressedContext:
        """Compress messages to fit within the token budget using the configured strategy."""
        if budget is None:
            used = self.count_messages(messages)
            budget = TokenBudget(total=self.max_tokens, used=0)

        original_tokens = self.count_messages(messages)

        dispatch = {
            CompressionStrategy.SLIDING_WINDOW: self._sliding_window,
            CompressionStrategy.SUMMARIZE: self._summarize,
            CompressionStrategy.SELECTIVE: self._selective,
            CompressionStrategy.HYBRID: self._hybrid,
        }
        handler = dispatch.get(self.strategy, self._sliding_window)
        result = await handler(messages, budget)

        compressed_tokens = self.count_messages(result.messages)
        tokens_saved = max(0, original_tokens - compressed_tokens)

        return CompressedContext(
            messages=result.messages,
            original_count=len(messages),
            compressed_count=len(result.messages),
            tokens_saved=tokens_saved,
            strategy_used=result.strategy_used,
            summary=result.summary,
        )

    async def _sliding_window(
        self, messages: List[dict], budget: TokenBudget
    ) -> CompressedContext:
        """Keep system message and the last N non-system messages."""
        system_messages = [m for m in messages if m.get("role") == "system"] if self.keep_system else []
        non_system = [m for m in messages if m.get("role") != "system"]

        kept = non_system[-self.keep_recent:] if len(non_system) > self.keep_recent else non_system
        result = system_messages + kept

        return CompressedContext(
            messages=result,
            original_count=len(messages),
            compressed_count=len(result),
            tokens_saved=0,
            strategy_used=CompressionStrategy.SLIDING_WINDOW,
        )

    async def _summarize(
        self, messages: List[dict], budget: TokenBudget
    ) -> CompressedContext:
        """Summarize older messages via LLM, keep recent messages intact."""
        if self.model_caller is None:
            return await self._sliding_window(messages, budget)

        system_messages = [m for m in messages if m.get("role") == "system"] if self.keep_system else []
        non_system = [m for m in messages if m.get("role") != "system"]

        if len(non_system) <= self.keep_recent:
            return CompressedContext(
                messages=messages,
                original_count=len(messages),
                compressed_count=len(messages),
                tokens_saved=0,
                strategy_used=CompressionStrategy.SUMMARIZE,
            )

        to_summarize = non_system[: -self.keep_recent]
        recent = non_system[-self.keep_recent:]

        summary_prompt = (
            "Please provide a concise summary of the following conversation, "
            "preserving all important context, decisions, and information:\n\n"
        )
        for msg in to_summarize:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
            summary_prompt += f"{role}: {content}\n"

        try:
            summary_text = await self.model_caller(
                [{"role": "user", "content": summary_prompt}]
            )
        except Exception:
            return await self._sliding_window(messages, budget)

        summary_message = {
            "role": "system",
            "content": f"[Conversation summary]: {summary_text}",
        }
        result = system_messages + [summary_message] + recent

        return CompressedContext(
            messages=result,
            original_count=len(messages),
            compressed_count=len(result),
            tokens_saved=0,
            strategy_used=CompressionStrategy.SUMMARIZE,
            summary=summary_text,
        )

    async def _selective(
        self, messages: List[dict], budget: TokenBudget
    ) -> CompressedContext:
        """Keep system messages, most recent messages, and highest-importance messages."""
        system_messages = [m for m in messages if m.get("role") == "system"] if self.keep_system else []
        non_system = [m for m in messages if m.get("role") != "system"]

        if len(non_system) <= self.keep_recent:
            return CompressedContext(
                messages=messages,
                original_count=len(messages),
                compressed_count=len(messages),
                tokens_saved=0,
                strategy_used=CompressionStrategy.SELECTIVE,
            )

        recent = non_system[-self.keep_recent:]
        older = non_system[: -self.keep_recent]

        # Score and sort older messages by importance
        scored = [(self._estimate_importance(m), i, m) for i, m in enumerate(older)]
        scored.sort(key=lambda x: x[0], reverse=True)

        # Greedily add important older messages within budget
        available_tokens = budget.available - self.count_messages(system_messages + recent)
        selected_older: List[tuple[int, dict]] = []

        for _, original_idx, msg in scored:
            msg_tokens = self.count_messages([msg])
            if available_tokens - msg_tokens >= 0:
                selected_older.append((original_idx, msg))
                available_tokens -= msg_tokens
            if available_tokens <= 0:
                break

        # Restore original order for selected older messages
        selected_older.sort(key=lambda x: x[0])
        ordered_older = [m for _, m in selected_older]

        result = system_messages + ordered_older + recent

        return CompressedContext(
            messages=result,
            original_count=len(messages),
            compressed_count=len(result),
            tokens_saved=0,
            strategy_used=CompressionStrategy.SELECTIVE,
        )

    async def _hybrid(
        self, messages: List[dict], budget: TokenBudget
    ) -> CompressedContext:
        """Apply selective compression first; summarize if still over budget."""
        selective_result = await self._selective(messages, budget)

        current_tokens = self.count_messages(selective_result.messages)
        threshold_tokens = int(budget.available * self.summarize_threshold)

        if current_tokens <= threshold_tokens:
            return CompressedContext(
                messages=selective_result.messages,
                original_count=len(messages),
                compressed_count=len(selective_result.messages),
                tokens_saved=0,
                strategy_used=CompressionStrategy.HYBRID,
                summary=selective_result.summary,
            )

        summarized_result = await self._summarize(selective_result.messages, budget)

        return CompressedContext(
            messages=summarized_result.messages,
            original_count=len(messages),
            compressed_count=len(summarized_result.messages),
            tokens_saved=0,
            strategy_used=CompressionStrategy.HYBRID,
            summary=summarized_result.summary,
        )

    def _estimate_importance(self, message: dict) -> float:
        """Estimate message importance on a 0–1 scale using heuristics."""
        content = message.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
        if not isinstance(content, str):
            content = str(content)

        score = 0.0

        # Length heuristic: longer messages tend to carry more information
        length = len(content)
        if length > 500:
            score += 0.3
        elif length > 200:
            score += 0.2
        elif length > 50:
            score += 0.1

        # Questions are often pivotal
        question_count = content.count("?")
        if question_count > 0:
            score += min(0.2, question_count * 0.05)

        # High-importance keywords
        keywords = ["important", "must", "critical", "remember", "always", "never",
                    "required", "error", "warning", "fail", "urgent"]
        lower_content = content.lower()
        keyword_hits = sum(1 for kw in keywords if kw in lower_content)
        score += min(0.3, keyword_hits * 0.1)

        # Assistant messages responding to critical context are more valuable
        if message.get("role") == "assistant" and length > 100:
            score += 0.1

        return min(1.0, score)
