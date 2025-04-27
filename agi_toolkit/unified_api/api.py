"""
Unified API for ASI and MOCK-LLM Integration
===========================================

Main entry point for the AGI Toolkit, providing a simplified,
stable interface for all system functionality.
"""

import os
import sys
import json
import logging
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import numpy as np

from ..asi_integration.interface import ASIInterface
from ..mock_llm_integration.interface import MOCKLLMInterface
from .memory import UnifiedMemory


class AGIAPI:
    """
    Main API for ASI and MOCK-LLM functionality.
    
    This class serves as a clean interface for external applications to use
    the unified ASI and MOCK-LLM system without touching the core code.
    
    Example:
        ```python
        from agi_toolkit import AGIAPI
        
        # Initialize the API
        api = AGIAPI()
        
        # Check component availability
        print(f"ASI available: {api.has_asi}")
        print(f"MOCK-LLM available: {api.has_mock_llm}")
        
        # Generate text with MOCK-LLM
        response = api.generate_text("Explain quantum computing in simple terms")
        print(response)
        
        # Process data with ASI
        result = api.process_with_asi({"query": "Analyze market trends for AI in 2025"})
        print(result)
        ```
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the API.
        
        Args:
            config_path: Optional path to a custom configuration file.
                         If not provided, the default configuration will be used.
        """
        # Setup logging
        self.logger = logging.getLogger("AGIAPI")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info("Initializing AGI Toolkit API")
        
        # Check for globally initialized components first
        try:
            import builtins
            global_asi_available = hasattr(builtins, 'ASI_INSTANCE')
            global_llm_available = hasattr(builtins, 'MOCK_LLM_INSTANCE')
            
            if global_asi_available or global_llm_available:
                self.logger.info("Found globally initialized components")
        except ImportError:
            global_asi_available = False
            global_llm_available = False
        
        # Force ASI components to be available if global instance exists
        if global_asi_available:
            self.logger.info("Using globally initialized ASI components")
            # Make a special environment flag to signal real components should be used
            os.environ['USE_REAL_ASI'] = 'true'
        
        # Initialize component interfaces
        self.asi = ASIInterface()
        self.mock_llm = MOCKLLMInterface()
        self.memory = UnifiedMemory()
        
        # Store availability flags - override if global components exist
        self.has_asi = global_asi_available or self.asi.is_available
        self.has_mock_llm = global_llm_available or self.mock_llm.is_available
        
        self.logger.info(f"ASI available: {self.has_asi}")
        self.logger.info(f"MOCK-LLM available: {self.has_mock_llm}")
        self.logger.info("API initialized successfully")
    
    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        """
        Generate text using the MOCK-LLM model.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            
        Returns:
            Generated text
        """
        if not self.has_mock_llm:
            return "Error: MOCK-LLM components not available"
        
        return self.mock_llm.generate_text(prompt, max_length)
    
    def process_with_asi(self, input_data: Any) -> Dict[str, Any]:
        """
        Process data with the ASI system.
        
        Args:
            input_data: Input data for ASI processing
            
        Returns:
            Processing results
        """
        if not self.has_asi:
            return {"success": False, "error": "ASI components not available"}
        
        return self.asi.process_data(input_data)
    
    def store_data(self, key: str, value: Any, metadata: Dict[str, Any] = None) -> bool:
        """
        Store data in the unified memory system.
        
        Args:
            key: Memory key
            value: Data to store (can be tensor, string, or structured data)
            metadata: Additional metadata for the stored data
            
        Returns:
            Success status
        """
        return self.memory.store(key, value, metadata)
    
    def retrieve_data(self, key: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Retrieve data from the unified memory system.
        
        Args:
            key: Memory key
            
        Returns:
            Tuple of (retrieved_data, metadata)
        """
        return self.memory.retrieve(key)
    
    def train_model(self, 
                   dataset_path: str, 
                   model_type: str = "unified",
                   num_epochs: int = 3,
                   learning_rate: float = 5e-5,
                   batch_size: int = 16) -> Dict[str, Any]:
        """
        Train a model on a dataset.
        
        Args:
            dataset_path: Path to the dataset
            model_type: Type of model to train ("asi", "mock_llm", or "unified")
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            
        Returns:
            Training results
        """
        # Select the appropriate component based on model type
        if model_type == "asi":
            if not self.has_asi:
                return {"success": False, "error": "ASI components not available"}
            return self.asi.train_model(dataset_path, num_epochs, learning_rate, batch_size)
        
        elif model_type == "mock_llm":
            if not self.has_mock_llm:
                return {"success": False, "error": "MOCK-LLM components not available"}
            return self.mock_llm.train_model(dataset_path, num_epochs, learning_rate, batch_size)
        
        else:  # unified
            # Train both if available
            results = {"success": False, "error": "No components available for training"}
            
            if self.has_asi:
                asi_results = self.asi.train_model(
                    dataset_path, num_epochs, learning_rate, batch_size
                )
                results = asi_results
            
            if self.has_mock_llm:
                mock_llm_results = self.mock_llm.train_model(
                    dataset_path, num_epochs, learning_rate, batch_size
                )
                results = mock_llm_results
                
                # If both were trained, combine results
                if self.has_asi:
                    results = {
                        "success": True,
                        "asi_results": asi_results,
                        "mock_llm_results": mock_llm_results
                    }
            
            return results
    
    def evaluate_model(self, 
                      test_data_path: str,
                      model_type: str = "unified") -> Dict[str, Any]:
        """
        Evaluate a model on test data.
        
        Args:
            test_data_path: Path to the test data
            model_type: Type of model to evaluate ("asi", "mock_llm", or "unified")
            
        Returns:
            Evaluation results
        """
        # Select the appropriate component based on model type
        if model_type == "asi":
            if not self.has_asi:
                return {"success": False, "error": "ASI components not available"}
            return self.asi.evaluate_model(test_data_path)
        
        elif model_type == "mock_llm":
            if not self.has_mock_llm:
                return {"success": False, "error": "MOCK-LLM components not available"}
            return self.mock_llm.evaluate_model(test_data_path)
        
        else:  # unified
            # Evaluate both if available
            results = {"success": False, "error": "No components available for evaluation"}
            
            if self.has_asi:
                asi_results = self.asi.evaluate_model(test_data_path)
                results = asi_results
            
            if self.has_mock_llm:
                mock_llm_results = self.mock_llm.evaluate_model(test_data_path)
                results = mock_llm_results
                
                # If both were evaluated, combine results
                if self.has_asi:
                    results = {
                        "success": True,
                        "asi_results": asi_results,
                        "mock_llm_results": mock_llm_results
                    }
            
            return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current system status.
        
        Returns:
            Dictionary containing system status information
        """
        memory_stats = self.memory.get_stats()
        
        status = {
            "version": "1.0.0",
            "components": {
                "asi": {
                    "available": self.has_asi,
                    "status": self.asi.get_status() if self.has_asi else "unavailable"
                },
                "mock_llm": {
                    "available": self.has_mock_llm,
                    "status": self.mock_llm.get_status() if self.has_mock_llm else "unavailable"
                }
            },
            "memory": memory_stats
        }
        
        return status
