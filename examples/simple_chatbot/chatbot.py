#!/usr/bin/env python
"""
Simple Chatbot Example
---------------------

This application demonstrates how to build a conversational interface
using the AGI Toolkit without modifying any core code.

Features:
- Text generation with MOCK-LLM
- Memory persistence for conversation history
- Context-aware responses
- Graceful fallback when components are missing
"""

import os
import sys
import logging
import json
import time
from typing import Dict, List, Any

# Add the parent directory to path so we can import the package
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

# Import the AGI Toolkit
from agi_toolkit import AGIAPI


class SimpleChatbot:
    """A simple chatbot built with AGI Toolkit."""
    
    def __init__(self, name="AGI Assistant"):
        """
        Initialize the chatbot.
        
        Args:
            name: The name of the chatbot
        """
        # Setup logging
        self.logger = logging.getLogger("SimpleChatbot")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"Initializing {name}")
        
        self.name = name
        self.history = []
        
        # Initialize the API
        self.api = AGIAPI()
        
        # Check component availability
        self.logger.info(f"ASI available: {self.api.has_asi}")
        self.logger.info(f"MOCK-LLM available: {self.api.has_mock_llm}")
        
        # Store conversation context in memory
        self.api.store_data("chatbot_context", {
            "name": name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "capabilities": {
                "text_generation": self.api.has_mock_llm,
                "reasoning": self.api.has_asi
            }
        })
        
        self.logger.info("Chatbot initialized successfully")
    
    def respond(self, user_input: str) -> str:
        """
        Generate a response to user input.
        
        Args:
            user_input: The user's message
            
        Returns:
            Chatbot's response
        """
        self.logger.info(f"Received input: {user_input[:50]}...")
        
        # Add to history
        self.history.append({
            "role": "user",
            "message": user_input,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Create a prompt with conversation context
        context = self._build_context(user_input)
        
        # Generate response
        if self.api.has_mock_llm:
            response = self.api.generate_text(context)
        else:
            # Fallback response if MOCK-LLM is not available
            response = "I'm sorry, my text generation capabilities are currently offline."
        
        # Use ASI for enhanced understanding if available
        if self.api.has_asi and len(user_input) > 10:
            try:
                # Process with ASI to get deeper understanding
                asi_analysis = self.api.process_with_asi({
                    "input": user_input,
                    "conversation_history": self.history[-5:] if len(self.history) > 5 else self.history
                })
                
                # If ASI provided insights, incorporate them
                if asi_analysis.get("success", False) and isinstance(asi_analysis.get("result"), dict):
                    insights = asi_analysis["result"]
                    if "priority" in insights and insights["priority"] == "high":
                        # For high priority inputs, enhance the response
                        response = f"{response} This seems important, so let me add that {insights.get('additional_info', '')}"
            except Exception as e:
                self.logger.error(f"Error using ASI for enhancement: {str(e)}")
        
        # Add to history
        self.history.append({
            "role": "assistant",
            "message": response,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Store updated conversation history
        self.api.store_data("conversation_history", self.history)
        
        return response
    
    def _build_context(self, current_input: str) -> str:
        """
        Build a context string for response generation.
        
        Args:
            current_input: The current user input
            
        Returns:
            Context string for the model
        """
        # Get recent conversation history (last 5 turns)
        recent_history = self.history[-10:] if len(self.history) > 10 else self.history
        
        # Format history as a conversation
        conversation = ""
        for entry in recent_history:
            role = entry["role"]
            message = entry["message"]
            conversation += f"{role.capitalize()}: {message}\n"
        
        # Add current input if not already in history
        if not self.history or self.history[-1]["message"] != current_input:
            conversation += f"User: {current_input}\n"
        
        # Add instruction for the assistant
        context = f"""
The following is a conversation with {self.name}, an AI assistant.
The assistant is helpful, creative, and friendly.

{conversation}
{self.name}:"""
        
        return context
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the conversation history.
        
        Returns:
            List of conversation entries
        """
        return self.history
    
    def save_conversation(self, filepath: str) -> bool:
        """
        Save the conversation history to a file.
        
        Args:
            filepath: Path to save the conversation
            
        Returns:
            Success status
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.history, f, indent=2)
            
            self.logger.info(f"Conversation saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving conversation: {str(e)}")
            return False
    
    def load_conversation(self, filepath: str) -> bool:
        """
        Load conversation history from a file.
        
        Args:
            filepath: Path to load the conversation from
            
        Returns:
            Success status
        """
        try:
            with open(filepath, 'r') as f:
                self.history = json.load(f)
            
            # Store loaded conversation in memory
            self.api.store_data("conversation_history", self.history)
            
            self.logger.info(f"Conversation loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading conversation: {str(e)}")
            return False


def interactive_mode(chatbot: SimpleChatbot):
    """Run the chatbot in interactive mode."""
    print(f"\nWelcome to {chatbot.name}!")
    print("Type 'exit' to quit, 'save' to save the conversation, or 'load' to load a conversation.")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == "exit":
            print(f"\n{chatbot.name}: Goodbye!")
            break
        
        elif user_input.lower().startswith("save"):
            # Extract filename or use default
            parts = user_input.split(" ", 1)
            filepath = parts[1] if len(parts) > 1 else "conversation.json"
            
            if chatbot.save_conversation(filepath):
                print(f"\n{chatbot.name}: Conversation saved to {filepath}")
            else:
                print(f"\n{chatbot.name}: Sorry, I couldn't save the conversation.")
        
        elif user_input.lower().startswith("load"):
            # Extract filename or use default
            parts = user_input.split(" ", 1)
            filepath = parts[1] if len(parts) > 1 else "conversation.json"
            
            if chatbot.load_conversation(filepath):
                print(f"\n{chatbot.name}: Conversation loaded from {filepath}")
            else:
                print(f"\n{chatbot.name}: Sorry, I couldn't load the conversation.")
        
        else:
            # Generate response
            response = chatbot.respond(user_input)
            print(f"\n{chatbot.name}: {response}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Chatbot Example")
    parser.add_argument("--name", type=str, default="AGI Assistant", help="Name of the chatbot")
    parser.add_argument("--input", type=str, help="Single input to get a response (non-interactive mode)")
    parser.add_argument("--load", type=str, help="Load conversation from file")
    parser.add_argument("--save", type=str, help="Save conversation to file after completion")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the chatbot
    chatbot = SimpleChatbot(name=args.name)
    
    # Load conversation if specified
    if args.load:
        chatbot.load_conversation(args.load)
    
    # Run in different modes based on arguments
    if args.input:
        # Single response mode
        response = chatbot.respond(args.input)
        print(f"\n{chatbot.name}: {response}")
    else:
        # Interactive mode
        interactive_mode(chatbot)
    
    # Save conversation if specified
    if args.save:
        chatbot.save_conversation(args.save)


if __name__ == "__main__":
    main()
