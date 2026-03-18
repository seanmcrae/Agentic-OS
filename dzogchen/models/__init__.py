"""Model adapters for multiple AI providers and intelligent routing."""

from dzogchen.models.base import BaseModelAdapter, ModelResponse, ModelMessage
from dzogchen.models.claude import ClaudeAdapter
from dzogchen.models.openai import OpenAIAdapter
from dzogchen.models.gemini import GeminiAdapter
from dzogchen.models.local import OllamaAdapter, LMStudioAdapter
from dzogchen.models.router import ModelRouter

__all__ = [
    "BaseModelAdapter",
    "ModelResponse",
    "ModelMessage",
    "ClaudeAdapter",
    "OpenAIAdapter",
    "GeminiAdapter",
    "OllamaAdapter",
    "LMStudioAdapter",
    "ModelRouter",
]
