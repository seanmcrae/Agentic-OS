"""
Dzogchen Configuration
======================
Central configuration management with environment variable support,
validation, and dynamic updates.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    COHERE = "cohere"
    LOCAL = "local"


@dataclass
class ModelConfig:
    """Configuration for a specific model provider."""
    provider: ModelProvider
    model_id: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 8192
    temperature: float = 0.7
    timeout: int = 60
    rate_limit_rpm: int = 60
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0


@dataclass
class MemoryConfig:
    """Configuration for the memory subsystem."""
    working_memory_size: int = 100
    episodic_memory_size: int = 10_000
    semantic_memory_size: int = 100_000
    embedding_model: str = "text-embedding-3-small"
    similarity_threshold: float = 0.75
    consolidation_interval: int = 300  # seconds
    persistence_path: Optional[str] = None


@dataclass
class AgentConfig:
    """Configuration for agent behavior."""
    max_iterations: int = 50
    max_parallel_agents: int = 10
    reflection_depth: int = 3
    planning_horizon: int = 10
    confidence_threshold: float = 0.7
    enable_self_improvement: bool = True
    enable_tool_creation: bool = True


@dataclass
class KnowledgeConfig:
    """Configuration for the knowledge graph."""
    graph_backend: str = "in_memory"  # "in_memory" | "neo4j" | "networkx"
    neo4j_uri: Optional[str] = None
    neo4j_user: Optional[str] = None
    neo4j_password: Optional[str] = None
    synthesis_batch_size: int = 50
    max_graph_nodes: int = 1_000_000


@dataclass
class APIConfig:
    """Configuration for the API server."""
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 4
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    enable_auth: bool = False
    api_key_header: str = "X-Dzogchen-Key"
    rate_limit_per_minute: int = 100
    websocket_timeout: int = 300


@dataclass
class DzogchenConfig:
    """
    Master configuration for the Dzogchen framework.

    Supports loading from environment variables and YAML/JSON config files.
    """
    # Core identity
    name: str = "Dzogchen"
    version: str = "0.1.0"
    log_level: LogLevel = LogLevel.INFO

    # Subsystem configs
    primary_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-opus-4-6",
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        max_tokens=8192,
        temperature=0.7,
    ))
    fallback_models: List[ModelConfig] = field(default_factory=list)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    api: APIConfig = field(default_factory=APIConfig)

    # Feature flags
    enable_consciousness_layer: bool = True
    enable_swarm_intelligence: bool = True
    enable_metacognition: bool = True
    enable_knowledge_synthesis: bool = True
    enable_streaming: bool = True

    # Cost & safety controls
    max_cost_per_session_usd: float = 10.0
    max_tool_execution_time: int = 30
    sandbox_tool_execution: bool = True
    human_in_the_loop: bool = False

    @classmethod
    def from_env(cls) -> "DzogchenConfig":
        """Load configuration from environment variables."""
        config = cls()

        if api_key := os.environ.get("ANTHROPIC_API_KEY"):
            config.primary_model.api_key = api_key

        if model_id := os.environ.get("DZOGCHEN_PRIMARY_MODEL"):
            config.primary_model.model_id = model_id

        if port := os.environ.get("DZOGCHEN_PORT"):
            config.api.port = int(port)

        if log_level := os.environ.get("DZOGCHEN_LOG_LEVEL"):
            config.log_level = LogLevel(log_level.upper())

        if mem_path := os.environ.get("DZOGCHEN_MEMORY_PATH"):
            config.memory.persistence_path = mem_path

        return config

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DzogchenConfig":
        """Load configuration from a dictionary."""
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration to dictionary (redacting secrets)."""
        return {
            "name": self.name,
            "version": self.version,
            "log_level": self.log_level.value,
            "primary_model": {
                "provider": self.primary_model.provider.value,
                "model_id": self.primary_model.model_id,
                "api_key": "***" if self.primary_model.api_key else None,
                "max_tokens": self.primary_model.max_tokens,
                "temperature": self.primary_model.temperature,
            },
            "features": {
                "consciousness_layer": self.enable_consciousness_layer,
                "swarm_intelligence": self.enable_swarm_intelligence,
                "metacognition": self.enable_metacognition,
                "knowledge_synthesis": self.enable_knowledge_synthesis,
            },
        }
