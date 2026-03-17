"""
Dzogchen - Super-Intelligent AI Framework
==========================================

Named after the Tibetan Buddhist concept of "Great Perfection" - the primordial
state of awareness that is complete, self-arising, and free from all limitations.

Dzogchen is a multi-model AI orchestration framework that provides:
- Multi-tier memory (working, episodic, semantic)
- Autonomous reasoning and planning
- Self-reflection and metacognition
- Multi-agent swarm intelligence
- Dynamic tool creation and execution
- Cross-model intelligence routing
- Knowledge graph synthesis

Example usage:
    from dzogchen import Dzogchen

    dz = Dzogchen()
    response = await dz.think("What is the nature of consciousness?")
    print(response)
"""

__version__ = "0.1.0"
__author__ = "Dzogchen Contributors"
__license__ = "MIT"

from dzogchen.core import Dzogchen
from dzogchen.config import DzogchenConfig
from dzogchen.agents.base import BaseAgent
from dzogchen.models.router import ModelRouter

__all__ = [
    "Dzogchen",
    "DzogchenConfig",
    "BaseAgent",
    "ModelRouter",
    "__version__",
]
