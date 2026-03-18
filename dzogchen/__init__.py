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

__version__ = "0.3.0"
__author__ = "Dzogchen Contributors"
__license__ = "MIT"

from dzogchen.core import Dzogchen
from dzogchen.config import DzogchenConfig
from dzogchen.agents.base import BaseAgent
from dzogchen.models.router import ModelRouter

# Phase 2
from dzogchen.knowledge.graph import KnowledgeGraph
from dzogchen.memory.vector import VectorStore
from dzogchen.learning.feedback import FeedbackCollector
from dzogchen.learning.optimizer import PromptOptimizer

# Phase 3 — State of the art
from dzogchen.cache.semantic import SemanticCache
from dzogchen.structured.output import StructuredGenerator, OutputSchema
from dzogchen.safety.constitutional import ConstitutionalAI, PrincipleSet
from dzogchen.context.compressor import ContextCompressor, CompressionStrategy
from dzogchen.observability.telemetry import Telemetry
from dzogchen.budget.manager import BudgetManager
from dzogchen.pipelines.builder import Pipeline, pipe
from dzogchen.multimodal.processor import MultimodalProcessor, MultimodalInput, ModalityType

__all__ = [
    # Core
    "Dzogchen",
    "DzogchenConfig",
    "BaseAgent",
    "ModelRouter",
    "__version__",
    # Phase 2
    "KnowledgeGraph",
    "VectorStore",
    "FeedbackCollector",
    "PromptOptimizer",
    # Phase 3
    "SemanticCache",
    "StructuredGenerator",
    "OutputSchema",
    "ConstitutionalAI",
    "PrincipleSet",
    "ContextCompressor",
    "CompressionStrategy",
    "Telemetry",
    "BudgetManager",
    "Pipeline",
    "pipe",
    "MultimodalProcessor",
    "MultimodalInput",
    "ModalityType",
]
