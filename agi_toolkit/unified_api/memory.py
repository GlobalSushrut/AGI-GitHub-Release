"""
Unified Memory Interface
=====================

This module provides an interface to the unified memory system
shared between ASI and MOCK-LLM components.
"""

import os
import sys
import json
import logging
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import numpy as np


class UnifiedMemory:
    """
    Interface to the unified memory system.
    
    This class provides methods for storing, retrieving, and managing data
    in the unified memory system shared between ASI and MOCK-LLM components.
    
    Example:
        ```python
        from agi_toolkit import UnifiedMemory
        
        # Initialize memory
        memory = UnifiedMemory()
        
        # Store data
        memory.store("customer_profile", {"name": "John", "id": 12345})
        
        # Retrieve data
        profile, metadata = memory.retrieve("customer_profile")
        print(profile)
        ```
    """
    
    def __init__(self):
        """Initialize the unified memory interface."""
        self.logger = logging.getLogger("UnifiedMemory")
        
        # Try to import the actual unified memory system
        try:
            from unified_training_system.memory.shared_memory import UnifiedMemoryManager
            self.real_memory = UnifiedMemoryManager(
                non_euclidean_dimension=11,
                shared_memory_size=10000
            )
            self.using_real_memory = True
            self.logger.info("Initialized with real unified memory system")
        except ImportError:
            # If not available, use a simple dictionary-based memory system
            self.memory_store = {}
            self.using_real_memory = False
            self.logger.info("Initialized with simulated memory system")
    
    @property
    def is_available(self) -> bool:
        """Check if real memory system is available."""
        return self.using_real_memory
    
    def store(self, key: str, value: Any, metadata: Dict[str, Any] = None) -> bool:
        """
        Store data in the memory system.
        
        Args:
            key: Memory key
            value: Data to store (can be tensor, string, or structured data)
            metadata: Additional metadata for the stored data
            
        Returns:
            Success status
        """
        try:
            # Set default metadata if not provided
            if metadata is None:
                metadata = {}
            
            # Add source to metadata
            metadata["source"] = "agi_toolkit"
            
            if self.using_real_memory:
                # Process the value to ensure it's in the correct format for real memory
                if not isinstance(value, torch.Tensor):
                    # Convert structured data to JSON string first
                    if isinstance(value, (dict, list)):
                        value_str = json.dumps(value)
                        value = torch.tensor([ord(c) for c in value_str], dtype=torch.float32)
                    # Convert string to tensor
                    elif isinstance(value, str):
                        value = torch.tensor([ord(c) for c in value], dtype=torch.float32)
                    # Convert simple types to tensor
                    elif isinstance(value, (int, float, bool)):
                        value = torch.tensor([float(value)], dtype=torch.float32)
                    else:
                        # Try direct conversion or fail
                        try:
                            value = torch.tensor(value, dtype=torch.float32)
                        except:
                            self.logger.error(f"Cannot convert {type(value)} to tensor")
                            return False
                
                # Store in real memory system
                self.real_memory.store(key, value, metadata)
            else:
                # For simulated memory, serialize the data
                if isinstance(value, torch.Tensor):
                    value_to_store = value.tolist()
                else:
                    value_to_store = value
                
                # Store in dictionary with metadata
                self.memory_store[key] = {
                    "value": value_to_store,
                    "metadata": metadata,
                    "type": type(value).__name__
                }
            
            self.logger.info(f"Stored data with key: {key}")
            return True
        except Exception as e:
            self.logger.error(f"Error storing data: {str(e)}")
            traceback.print_exc()
            return False
    
    def retrieve(self, key: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Retrieve data from the memory system.
        
        Args:
            key: Memory key
            
        Returns:
            Tuple of (retrieved_data, metadata)
        """
        try:
            if self.using_real_memory:
                # Retrieve from real memory system
                data, metadata = self.real_memory.retrieve(key)
                
                # Try to convert tensor back to original format if needed
                if isinstance(data, torch.Tensor):
                    # Check if this might be string data
                    if len(data.shape) == 1 and data.dtype == torch.float32:
                        try:
                            # Convert tensor to string
                            string_data = ''.join([chr(int(x)) for x in data])
                            
                            # Try to parse as JSON if it looks like JSON
                            if string_data.strip().startswith('{') or string_data.strip().startswith('['):
                                try:
                                    parsed = json.loads(string_data)
                                    return parsed, metadata
                                except:
                                    # Not valid JSON, return as string
                                    return string_data, metadata
                            else:
                                return string_data, metadata
                        except:
                            # Not a string, return as tensor
                            pass
                
                return data, metadata
            else:
                # Check if key exists in simulated memory
                if key not in self.memory_store:
                    self.logger.warning(f"Key not found: {key}")
                    return None, {}
                
                # Get the stored data
                stored = self.memory_store[key]
                value = stored["value"]
                metadata = stored["metadata"]
                value_type = stored["type"]
                
                # Convert back to original type if possible
                if value_type == "dict" or value_type == "list":
                    try:
                        if isinstance(value, str):
                            value = json.loads(value)
                    except:
                        pass
                elif value_type == "Tensor":
                    value = torch.tensor(value)
                
                return value, metadata
        except Exception as e:
            self.logger.error(f"Error retrieving data: {str(e)}")
            return None, {"error": str(e)}
    
    def list_keys(self) -> List[str]:
        """
        List all keys in the memory system.
        
        Returns:
            List of memory keys
        """
        try:
            if self.using_real_memory:
                # Real memory might not have this function
                # This is a placeholder that would need to be implemented
                # based on the actual capabilities of the real memory system
                return []
            else:
                return list(self.memory_store.keys())
        except Exception as e:
            self.logger.error(f"Error listing keys: {str(e)}")
            return []
    
    def clear(self, key: Optional[str] = None) -> bool:
        """
        Clear memory from the system.
        
        Args:
            key: Specific key to clear, or None to clear all memory
            
        Returns:
            Success status
        """
        try:
            if self.using_real_memory:
                # This is a placeholder that would need to be implemented
                # based on the actual capabilities of the real memory system
                return False
            else:
                if key is not None:
                    if key in self.memory_store:
                        del self.memory_store[key]
                        return True
                    return False
                else:
                    self.memory_store.clear()
                    return True
        except Exception as e:
            self.logger.error(f"Error clearing memory: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system.
        
        Returns:
            Dictionary containing memory statistics
        """
        try:
            if self.using_real_memory:
                # Get stats from real memory system
                return self.real_memory.get_memory_stats()
            else:
                # Return basic stats for simulated memory
                return {
                    "system": "simulated",
                    "keys_count": len(self.memory_store),
                    "keys": list(self.memory_store.keys())
                }
        except Exception as e:
            self.logger.error(f"Error getting memory stats: {str(e)}")
            return {"error": str(e)}
