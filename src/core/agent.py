"""Enhanced Agent Core Implementation for Agentic OS
Provides improved capabilities, memory management, and error handling."""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import asyncio
import json
from .memory import Memory
from .capabilities import Capability
from .processing import Pipeline
from ..utils.error_handling import AgentError, handle_error
from ..utils.monitoring import monitor

@dataclass
class AgentConfig:
    """Configuration for Agent initialization"""
    name: str
    capabilities: List[str]
    memory_size: int = 1000
    processing_threads: int = 4
    logging_level: str = "INFO"
    timeout: int = 30

class Agent:
    """Enhanced Agent class with improved capabilities and error handling"""
    def __init__(self, config: AgentConfig):
        self.name = config.name
        self.memory = Memory(max_size=config.memory_size)
        self.capabilities = self._load_capabilities(config.capabilities)
        self.pipeline = Pipeline(threads=config.processing_threads)
        self.logger = self._setup_logging(config.logging_level)
        self.timeout = config.timeout
        self._initialize()

    @monitor
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.logger.info(f"Processing input: {json.dumps(input_data, default=str)}")
            self._validate_input(input_data)
            context = self._create_context(input_data)
            async with asyncio.timeout(self.timeout):
                result = await self.pipeline.process(context)
            self.memory.add(context, result)
            return self._format_result(result)
        except asyncio.TimeoutError:
            error = AgentError("Processing timeout", "TIMEOUT")
            return handle_error(error, input_data)
        except Exception as e:
            error = AgentError(str(e), "PROCESSING_ERROR")
            return handle_error(error, input_data)

    async def add_capability(self, capability: Capability) -> None:
        try:
            await capability.initialize()
            self.capabilities[capability.name] = capability
            self.logger.info(f"Added capability: {capability.name}")
        except Exception as e:
            raise AgentError(f"Failed to add capability: {str(e)}", "CAPABILITY_ERROR")

    def _initialize(self) -> None:
        self.logger.info(f"Initializing agent: {self.name}")
        self._setup_monitoring()
        self._register_error_handlers()
        self.memory.initialize()
        self.pipeline.initialize()

    def _load_capabilities(self, capability_names: List[str]) -> Dict[str, Capability]:
        capabilities = {}
        for name in capability_names:
            try:
                capability = Capability.load(name)
                capabilities[name] = capability
            except Exception as e:
                self.logger.error(f"Failed to load capability {name}: {str(e)}")
        return capabilities