"""
OpenAIAdapter - OpenAI GPT model adapter
=========================================

Adapter for OpenAI's GPT models with tool use and streaming support.
Also compatible with any OpenAI-API-compatible endpoint.
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


OPENAI_PRICING = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "o1-preview": {"input": 0.015, "output": 0.06},
    "o1-mini": {"input": 0.003, "output": 0.012},
}


class OpenAIAdapter(BaseModelAdapter):
    """
    Adapter for OpenAI GPT models (and compatible APIs).

    Works with OpenAI, Azure OpenAI, local LM Studio servers,
    Ollama, and any OpenAI-compatible API endpoint.
    """

    DEFAULT_MODEL = "gpt-4o"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 60,
    ):
        super().__init__(model_id, api_key or os.environ.get("OPENAI_API_KEY"))
        self.base_url = base_url
        self.organization = organization
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        """Lazy initialization of the OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    organization=self.organization,
                    max_retries=self.max_retries,
                    timeout=self.timeout,
                )
            except ImportError:
                raise ImportError(
                    "openai package not found. Install with: pip install openai"
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
        """Generate a response from GPT."""
        start = self._start_timer()
        client = self._get_client()

        # Build messages in OpenAI format
        openai_messages: List[Dict[str, Any]] = []

        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            openai_messages.append({"role": msg.role.value, "content": msg.content})

        kwargs: Dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": openai_messages,
        }

        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
            kwargs["tool_choice"] = "auto"

        try:
            response = await client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            message = choice.message

            content = message.content or ""
            tool_calls = []

            if message.tool_calls:
                import json
                for tc in message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except Exception:
                        args = {"raw": tc.function.arguments}

                    tool_calls.append(ToolCall(
                        tool_call_id=tc.id,
                        tool_name=tc.function.name,
                        arguments=args,
                    ))

            latency = self._end_timer(start)
            pricing = OPENAI_PRICING.get(self.model_id, {"input": 0.0, "output": 0.0})

            model_response = ModelResponse(
                content=content,
                model=self.model_id,
                provider="openai",
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
                latency_ms=latency,
                finish_reason=choice.finish_reason or "stop",
                tool_calls=tool_calls,
                raw_response=response,
                metadata={
                    "cost_per_1k_input": pricing["input"],
                    "cost_per_1k_output": pricing["output"],
                },
            )

            self._track_usage(model_response)
            return model_response

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def stream_generate(
        self,
        messages: List[ModelMessage],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Stream a response from GPT."""
        client = self._get_client()

        openai_messages: List[Dict[str, Any]] = []
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})
        for msg in messages:
            openai_messages.append({"role": msg.role.value, "content": msg.content})

        try:
            stream = await client.chat.completions.create(
                model=self.model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=openai_messages,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise
