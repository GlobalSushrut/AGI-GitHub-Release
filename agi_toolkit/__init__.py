"""
AGI Toolkit
===========

A unified toolkit for ASI and MOCK-LLM integration.

This package provides tools for leveraging advanced AI capabilities
through a simplified, stable API without touching core implementation details.

Main Components:
- AGIAPI: Main entry point for all functionality
- ASIInterface: Interface for ASI-specific functionality
- MOCKLLMInterface: Interface for MOCK-LLM-specific functionality
- UnifiedMemory: Access to the shared memory system
"""

from .unified_api.api import AGIAPI
from .asi_integration.interface import ASIInterface
from .mock_llm_integration.interface import MOCKLLMInterface
from .unified_api.memory import UnifiedMemory

__version__ = "1.0.0"
__all__ = ["AGIAPI", "ASIInterface", "MOCKLLMInterface", "UnifiedMemory"]
