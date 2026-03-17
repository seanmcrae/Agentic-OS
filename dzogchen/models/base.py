"""
BaseModelAdapter - Abstract interface for all model providers
============================================================

Defines the contract that all model adapters must fulfill,
enabling seamless swapping between providers.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional
from enum import Enum


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ModelMessage:
    """A single message in a conversation."""
    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class ToolDefinition:
    """Definition of a tool available to the model."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


@dataclass
class ToolCall:
    """A tool call made by the model."""
    tool_call_id: str
    tool_name: str
    arguments: Dict[str, Any]


@dataclass
class ModelResponse:
    """Response from a model."""
    content: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw_response: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        """Estimated cost (if cost data is in metadata)."""
        cost_per_1k_input = self.metadata.get("cost_per_1k_input", 0.0)
        cost_per_1k_output = self.metadata.get("cost_per_1k_output", 0.0)
        return (
            (self.input_tokens / 1000) * cost_per_1k_input
            + (self.output_tokens / 1000) * cost_per_1k_output
        )


class BaseModelAdapter(ABC):
    """Abstract base class for all model provider adapters."""

    def __init__(self, model_id: str, api_key: Optional[str] = None, **kwargs):
        self.model_id = model_id
        self.api_key = api_key
        self._request_count = 0
        self._total_tokens = 0
        self._total_latency = 0.0

    @abstractmethod
    async def generate(
        self,
        messages: List[ModelMessage],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tools: Optional[List[ToolDefinition]] = None,
        stream: bool = False,
    ) -> ModelResponse:
        """Generate a response from the model."""
        ...

    @abstractmethod
    async def stream_generate(
        self,
        messages: List[ModelMessage],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Stream a response from the model."""
        ...

    async def simple_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> str:
        """Simple single-turn prompt. Returns just the text content."""
        messages = [ModelMessage(role=MessageRole.USER, content=prompt)]
        response = await self.generate(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self._track_usage(response)
        return response.content

    def _track_usage(self, response: ModelResponse) -> None:
        """Track usage statistics."""
        self._request_count += 1
        self._total_tokens += response.total_tokens
        self._total_latency += response.latency_ms

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this adapter."""
        return {
            "model": self.model_id,
            "request_count": self._request_count,
            "total_tokens": self._total_tokens,
            "avg_latency_ms": (
                self._total_latency / self._request_count
                if self._request_count > 0 else 0.0
            ),
        }

    def _start_timer(self) -> float:
        return time.time()

    def _end_timer(self, start: float) -> float:
        return (time.time() - start) * 1000  # Convert to ms
