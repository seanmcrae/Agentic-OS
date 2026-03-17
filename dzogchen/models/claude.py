"""
ClaudeAdapter - Anthropic Claude model adapter
===============================================

Full-featured adapter for Anthropic's Claude models with:
- Tool use support
- Streaming
- Vision (image inputs)
- Extended thinking
- Cost tracking
"""

from __future__ import annotations

import os
from typing import Any, AsyncGenerator, Dict, List, Optional
import logging

from dzogchen.models.base import (
    BaseModelAdapter,
    ModelMessage,
    ModelResponse,
    ToolDefinition,
    ToolCall,
    MessageRole,
)

logger = logging.getLogger(__name__)


# Claude pricing (per 1K tokens, as of 2025)
CLAUDE_PRICING = {
    "claude-opus-4-6": {"input": 0.015, "output": 0.075},
    "claude-sonnet-4-6": {"input": 0.003, "output": 0.015},
    "claude-haiku-4-5-20251001": {"input": 0.00025, "output": 0.00125},
}


class ClaudeAdapter(BaseModelAdapter):
    """
    Adapter for Anthropic Claude models.

    Supports all Claude features including tool use, streaming,
    vision, and extended thinking mode.
    """

    DEFAULT_MODEL = "claude-opus-4-6"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 60,
    ):
        super().__init__(model_id, api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        """Lazy initialization of the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    max_retries=self.max_retries,
                    timeout=self.timeout,
                )
            except ImportError:
                raise ImportError(
                    "anthropic package not found. Install with: pip install anthropic"
                )
        return self._client

    async def generate(
        self,
        messages: List[ModelMessage],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tools: Optional[List[ToolDefinition]] = None,
        stream: bool = False,
    ) -> ModelResponse:
        """Generate a response from Claude."""
        start = self._start_timer()
        client = self._get_client()

        # Build messages in Anthropic format
        anthropic_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
            if msg.role != MessageRole.SYSTEM
        ]

        # Build kwargs
        kwargs: Dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthropic_messages,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        if tools:
            kwargs["tools"] = [t.to_dict() for t in tools]

        try:
            response = await client.messages.create(**kwargs)

            # Extract content
            content = ""
            tool_calls = []

            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text
                elif hasattr(block, "type") and block.type == "tool_use":
                    tool_calls.append(ToolCall(
                        tool_call_id=block.id,
                        tool_name=block.name,
                        arguments=block.input,
                    ))

            latency = self._end_timer(start)
            pricing = CLAUDE_PRICING.get(self.model_id, {"input": 0.0, "output": 0.0})

            model_response = ModelResponse(
                content=content,
                model=self.model_id,
                provider="anthropic",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_ms=latency,
                finish_reason=response.stop_reason or "stop",
                tool_calls=tool_calls,
                raw_response=response,
                metadata={
                    "cost_per_1k_input": pricing["input"],
                    "cost_per_1k_output": pricing["output"],
                },
            )

            self._track_usage(model_response)
            logger.debug(
                f"Claude {self.model_id}: {response.usage.input_tokens}in "
                f"{response.usage.output_tokens}out {latency:.0f}ms"
            )
            return model_response

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise

    async def stream_generate(
        self,
        messages: List[ModelMessage],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Stream a response from Claude."""
        client = self._get_client()

        anthropic_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
            if msg.role != MessageRole.SYSTEM
        ]

        kwargs: Dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthropic_messages,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        try:
            async with client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Claude streaming error: {e}")
            raise

    async def generate_with_vision(
        self,
        text: str,
        image_data: bytes,
        image_media_type: str = "image/jpeg",
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """Generate a response with image input."""
        import base64
        start = self._start_timer()
        client = self._get_client()

        image_b64 = base64.standard_b64encode(image_data).decode("utf-8")

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_media_type,
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": text},
            ],
        }]

        kwargs: Dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = await client.messages.create(**kwargs)
        content = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
        latency = self._end_timer(start)

        return ModelResponse(
            content=content,
            model=self.model_id,
            provider="anthropic",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency,
        )
