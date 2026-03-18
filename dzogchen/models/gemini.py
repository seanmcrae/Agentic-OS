"""
GeminiAdapter - Google Gemini model adapter
===========================================

Full adapter for Google's Gemini models with:
- Gemini 1.5 Pro / Flash / Ultra support
- Function calling / tool use
- Multi-modal inputs (text + images)
- Streaming
- Safety settings control
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

GEMINI_PRICING = {
    "gemini-2.0-flash": {"input": 0.000075, "output": 0.0003},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    "gemini-1.0-pro": {"input": 0.0005, "output": 0.0015},
}


class GeminiAdapter(BaseModelAdapter):
    """
    Adapter for Google Gemini models via the google-generativeai SDK.

    Supports Gemini 2.0 Flash, 1.5 Pro, 1.5 Flash, and Gemini Ultra.
    Compatible with Google AI Studio and Vertex AI endpoints.
    """

    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        use_vertex: bool = False,
        safety_settings: Optional[Dict[str, str]] = None,
    ):
        super().__init__(model_id, api_key or os.environ.get("GOOGLE_API_KEY"))
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = location
        self.use_vertex = use_vertex
        self.safety_settings = safety_settings
        self._client = None

    def _get_client(self):
        """Lazy-init the Gemini client."""
        if self._client is not None:
            return self._client

        if self.use_vertex:
            try:
                import vertexai
                from vertexai.generative_models import GenerativeModel
                vertexai.init(project=self.project_id, location=self.location)
                self._client = GenerativeModel(self.model_id)
            except ImportError:
                raise ImportError(
                    "vertexai not found. Install with: pip install google-cloud-aiplatform"
                )
        else:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model_id)
            except ImportError:
                raise ImportError(
                    "google-generativeai not found. Install with: pip install google-generativeai"
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
        """Generate a response from Gemini."""
        import asyncio

        start = self._start_timer()
        client = self._get_client()

        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                continue  # System handled separately
            role = "user" if msg.role == MessageRole.USER else "model"
            gemini_messages.append({
                "role": role,
                "parts": [{"text": msg.content}],
            })

        # Build generation config
        gen_config = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }

        # Build tools if provided
        gemini_tools = None
        if tools:
            gemini_tools = [
                {
                    "function_declarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                        for t in tools
                    ]
                }
            ]

        try:
            # Run in executor since google SDK is sync
            loop = asyncio.get_event_loop()

            def _call():
                kwargs: Dict[str, Any] = {
                    "contents": gemini_messages,
                    "generation_config": gen_config,
                }
                if system_prompt:
                    kwargs["system_instruction"] = system_prompt
                if gemini_tools:
                    kwargs["tools"] = gemini_tools
                return client.generate_content(**kwargs)

            response = await loop.run_in_executor(None, _call)

            content = ""
            tool_calls = []

            for part in response.parts:
                if hasattr(part, "text") and part.text:
                    content += part.text
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_calls.append(ToolCall(
                        tool_call_id=fc.name,
                        tool_name=fc.name,
                        arguments=dict(fc.args),
                    ))

            latency = self._end_timer(start)
            pricing = GEMINI_PRICING.get(self.model_id, {"input": 0.0, "output": 0.0})

            # Extract token counts from usage metadata
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, "usage_metadata"):
                input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
                output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)

            model_response = ModelResponse(
                content=content,
                model=self.model_id,
                provider="google",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency,
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
            logger.error(f"Gemini API error: {e}")
            raise

    async def stream_generate(
        self,
        messages: List[ModelMessage],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Stream a response from Gemini."""
        import asyncio

        client = self._get_client()
        gemini_messages = [
            {
                "role": "user" if msg.role == MessageRole.USER else "model",
                "parts": [{"text": msg.content}],
            }
            for msg in messages
            if msg.role != MessageRole.SYSTEM
        ]

        gen_config = {"max_output_tokens": max_tokens, "temperature": temperature}

        def _stream():
            kwargs: Dict[str, Any] = {
                "contents": gemini_messages,
                "generation_config": gen_config,
                "stream": True,
            }
            if system_prompt:
                kwargs["system_instruction"] = system_prompt
            return client.generate_content(**kwargs)

        loop = asyncio.get_event_loop()
        response_stream = await loop.run_in_executor(None, _stream)

        for chunk in response_stream:
            if chunk.parts:
                for part in chunk.parts:
                    if hasattr(part, "text") and part.text:
                        yield part.text

    async def generate_with_image(
        self,
        prompt: str,
        image_data: bytes,
        mime_type: str = "image/jpeg",
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """Generate a response from an image + text prompt."""
        import asyncio
        import base64

        start = self._start_timer()
        client = self._get_client()

        def _call():
            return client.generate_content([
                {"mime_type": mime_type, "data": image_data},
                prompt,
            ])

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _call)
        content = response.text or ""
        latency = self._end_timer(start)

        return ModelResponse(
            content=content,
            model=self.model_id,
            provider="google",
            latency_ms=latency,
        )
