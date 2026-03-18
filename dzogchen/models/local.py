"""
LocalAdapter - Adapter for local models via Ollama / LM Studio / llama.cpp
===========================================================================

Connects to any OpenAI-compatible local server:
- Ollama (http://localhost:11434)
- LM Studio (http://localhost:1234)
- llama.cpp server
- Text Generation WebUI
- vLLM
- Anything with an /v1/chat/completions endpoint
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


class OllamaAdapter(BaseModelAdapter):
    """
    Adapter for Ollama local model server.

    Ollama provides a local REST API that runs open-weight models
    like Llama, Mistral, Gemma, CodeLlama, and many others.

    Usage:
        # Start Ollama: `ollama serve`
        # Pull a model: `ollama pull llama3.2`
        adapter = OllamaAdapter(model_id="llama3.2")
    """

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        model_id: str = "llama3.2",
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 120,
        num_ctx: int = 4096,         # Context window size
        num_predict: int = -1,       # -1 = unlimited
        temperature: float = 0.7,
        repeat_penalty: float = 1.1,
    ):
        super().__init__(model_id, api_key=None)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self._default_temperature = temperature
        self.repeat_penalty = repeat_penalty

    async def generate(
        self,
        messages: List[ModelMessage],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tools: Optional[List[ToolDefinition]] = None,
        stream: bool = False,
    ) -> ModelResponse:
        """Generate via Ollama's /api/chat endpoint."""
        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp required: pip install aiohttp")

        start = self._start_timer()

        # Build messages
        ollama_messages = []
        if system_prompt:
            ollama_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            ollama_messages.append({"role": msg.role.value, "content": msg.content})

        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": self.num_ctx,
                "repeat_penalty": self.repeat_penalty,
            },
        }

        if max_tokens > 0:
            payload["options"]["num_predict"] = max_tokens

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"Ollama error {resp.status}: {text}")

                    data = await resp.json()

            content = data.get("message", {}).get("content", "")
            latency = self._end_timer(start)

            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)

            response = ModelResponse(
                content=content,
                model=self.model_id,
                provider="ollama",
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                latency_ms=latency,
                metadata={"base_url": self.base_url},
            )
            self._track_usage(response)
            return response

        except Exception as e:
            logger.error(f"Ollama error: {e}")
            raise

    async def stream_generate(
        self,
        messages: List[ModelMessage],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Stream from Ollama."""
        try:
            import aiohttp
            import json as json_lib
        except ImportError:
            raise ImportError("aiohttp required: pip install aiohttp")

        ollama_messages = []
        if system_prompt:
            ollama_messages.append({"role": "system", "content": system_prompt})
        for msg in messages:
            ollama_messages.append({"role": msg.role.value, "content": msg.content})

        payload = {
            "model": self.model_id,
            "messages": ollama_messages,
            "stream": True,
            "options": {"temperature": temperature, "num_ctx": self.num_ctx},
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                async for line in resp.content:
                    if line:
                        try:
                            data = json_lib.loads(line.decode("utf-8"))
                            if chunk := data.get("message", {}).get("content", ""):
                                yield chunk
                        except Exception:
                            pass

    async def list_models(self) -> List[str]:
        """List models available in Ollama."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as resp:
                    data = await resp.json()
                    return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []

    async def pull_model(self, model_name: str) -> bool:
        """Pull/download a model from the Ollama registry."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model_name},
                    timeout=aiohttp.ClientTimeout(total=600),
                ) as resp:
                    return resp.status == 200
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False


class LMStudioAdapter(BaseModelAdapter):
    """
    Adapter for LM Studio's OpenAI-compatible local server.

    LM Studio provides a GUI for running local LLMs with an
    OpenAI-compatible API at http://localhost:1234.

    Usage:
        # Start LM Studio server, load a model
        adapter = LMStudioAdapter(model_id="your-loaded-model")
    """

    DEFAULT_BASE_URL = "http://localhost:1234/v1"

    def __init__(
        self,
        model_id: str = "local-model",
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 120,
    ):
        super().__init__(model_id, api_key="not-needed")
        self.base_url = base_url
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(
                    base_url=self.base_url,
                    api_key="not-needed",
                    timeout=self.timeout,
                )
            except ImportError:
                raise ImportError("openai package required: pip install openai")
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
        """Generate via LM Studio's OpenAI-compatible API."""
        start = self._start_timer()
        client = self._get_client()

        oai_messages: List[Dict[str, Any]] = []
        if system_prompt:
            oai_messages.append({"role": "system", "content": system_prompt})
        for msg in messages:
            oai_messages.append({"role": msg.role.value, "content": msg.content})

        try:
            response = await client.chat.completions.create(
                model=self.model_id,
                messages=oai_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            choice = response.choices[0]
            content = choice.message.content or ""
            latency = self._end_timer(start)

            model_response = ModelResponse(
                content=content,
                model=self.model_id,
                provider="lmstudio",
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
                latency_ms=latency,
                finish_reason=choice.finish_reason or "stop",
                metadata={"base_url": self.base_url},
            )
            self._track_usage(model_response)
            return model_response

        except Exception as e:
            logger.error(f"LM Studio error: {e}")
            raise

    async def stream_generate(
        self,
        messages: List[ModelMessage],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Stream from LM Studio."""
        client = self._get_client()

        oai_messages: List[Dict[str, Any]] = []
        if system_prompt:
            oai_messages.append({"role": "system", "content": system_prompt})
        for msg in messages:
            oai_messages.append({"role": msg.role.value, "content": msg.content})

        stream = await client.chat.completions.create(
            model=self.model_id,
            messages=oai_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
