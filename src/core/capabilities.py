"""Enhanced Capabilities System for Agentic OS
Provides a flexible framework for implementing and managing agent capabilities."""
from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod
import asyncio
import json
import logging
from dataclasses import dataclass

from ..utils.error_handling import CapabilityError
from ..utils.monitoring import monitor

@dataclass
class CapabilityMetadata:
    """Metadata for capability configuration"""
    name: str
    version: str
    description: str
    requirements: List[str]
    parameters: Dict[str, Any]

class Capability(ABC):
    """Base class for all agent capabilities"""
    
    def __init__(self, metadata: CapabilityMetadata):
        self.metadata = metadata
        self.logger = logging.getLogger(f"capability.{metadata.name}")
        self._handlers: Dict[str, Callable] = {}
        self._initialized = False

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def version(self) -> str:
        return self.metadata.version

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize capability resources"""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up capability resources"""
        pass