import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import MagicMock

from src.core.agent import Agent, AgentConfig
from src.core.capabilities import Capability, CapabilityMetadata
from src.core.memory import Memory

@pytest.fixture
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_capability():
    """Create a mock capability for testing"""
    metadata = CapabilityMetadata(
        name="test_capability",
        version="1.0.0",
        description="Test capability",
        requirements=[],
        parameters={}
    )
    
    class TestCapability(Capability):
        async def initialize(self):
            self._initialized = True
            
        async def shutdown(self):
            pass
            
        async def execute(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
            return {"status": "success", "result": "test_result"}
    
    return TestCapability(metadata)