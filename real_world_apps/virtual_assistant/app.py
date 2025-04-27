#!/usr/bin/env python3
"""
Virtual Assistant
---------------

A real-world application that demonstrates how to use the AGI Toolkit
to build a virtual assistant capable of handling various tasks.

Features:
- Natural language understanding
- Task management
- Information retrieval
- Persistent memory
- Contextual responses
"""

import os
import sys
import logging
import argparse
import json
import re
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# Add the parent directory to path so we can import the AGI Toolkit
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the ASI helper module and AGI Toolkit
from real_world_apps.asi_helper import initialize_asi_components
from agi_toolkit import AGIAPI


class VirtualAssistant:
    """A virtual assistant using AGI Toolkit."""
    
    def __init__(self, name: str = "Aria"):
        """
        Initialize the virtual assistant.
        
        Args:
            name: Name of the assistant
        """
        # Configure logging
        self.logger = logging.getLogger("VirtualAssistant")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"Initializing Virtual Assistant '{name}'")
        
        # Initialize real ASI components
        initialize_asi_components()
        
        # Set environment variable to ensure interface uses real components
        os.environ['USE_REAL_ASI'] = 'true'
        
        # Initialize the AGI Toolkit API
        self.api = AGIAPI()
        
        # Check component availability
        self.logger.info(f"ASI available: {self.api.has_asi}")
        self.logger.info(f"MOCK-LLM available: {self.api.has_mock_llm}")
        
        # Assistant properties
        self.name = name
        self.context = {}
        self.conversation_history = []
        
        # Initialize task system
        self.tasks = {}
        self.load_tasks()
        
        # Initialize knowledge base
        self.knowledge = {}
        self.load_knowledge()
        
        self.logger.info(f"Virtual Assistant '{name}' initialized")
    
    def load_tasks(self):
        """Load tasks from memory."""
        try:
            memory_key = "virtual_assistant_tasks"
            tasks_data = self.api.retrieve_data(memory_key)
            
            if tasks_data and isinstance(tasks_data, dict):
                self.tasks = tasks_data
                self.logger.info(f"Loaded {len(self.tasks)} tasks from memory")
            else:
                self.logger.info("No tasks found in memory, starting fresh")
        except Exception as e:
            self.logger.error(f"Error loading tasks: {str(e)}")
    
    def save_tasks(self):
        """Save tasks to memory."""
        try:
            memory_key = "virtual_assistant_tasks"
            self.api.store_data(memory_key, self.tasks)
            self.logger.info(f"Saved {len(self.tasks)} tasks to memory")
        except Exception as e:
            self.logger.error(f"Error saving tasks: {str(e)}")
    
    def load_knowledge(self):
        """Load knowledge base from memory."""
        try:
            memory_key = "virtual_assistant_knowledge"
            knowledge_data = self.api.retrieve_data(memory_key)
            
            if knowledge_data and isinstance(knowledge_data, dict):
                self.knowledge = knowledge_data
                self.logger.info(f"Loaded {len(self.knowledge)} knowledge items from memory")
            else:
                self.logger.info("No knowledge found in memory, starting fresh")
        except Exception as e:
            self.logger.error(f"Error loading knowledge: {str(e)}")
    
    def save_knowledge(self):
        """Save knowledge base to memory."""
        try:
            memory_key = "virtual_assistant_knowledge"
            self.api.store_data(memory_key, self.knowledge)
            self.logger.info(f"Saved {len(self.knowledge)} knowledge items to memory")
        except Exception as e:
            self.logger.error(f"Error saving knowledge: {str(e)}")
    
    def process_message(self, message: str) -> str:
        """
        Process a user message and generate a response.
        
        Args:
            message: User message
            
        Returns:
            Assistant response
        """
        self.logger.info(f"Processing message: {message[:50]}{'...' if len(message) > 50 else ''}")
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Ensure history doesn't get too large
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        # Detect intent and entities
        intent, entities = self._detect_intent(message)
        
        # Generate response based on intent
        response = None
        
        if intent == "task_add":
            response = self._handle_task_add(entities)
        elif intent == "task_list":
            response = self._handle_task_list(entities)
        elif intent == "task_delete":
            response = self._handle_task_delete(entities)
        elif intent == "task_update":
            response = self._handle_task_update(entities)
        elif intent == "knowledge_learn":
            response = self._handle_knowledge_learn(entities)
        elif intent == "knowledge_retrieve":
            response = self._handle_knowledge_retrieve(entities)
        elif intent == "greeting":
            response = self._handle_greeting(entities)
        elif intent == "farewell":
            response = self._handle_farewell(entities)
        elif intent == "help":
            response = self._handle_help(entities)
        
        # If still no response, generate general response
        if not response:
            response = self._generate_general_response(message)
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return response
    
    def _detect_intent(self, message: str) -> Tuple[str, Dict]:
        """
        Detect the intent and entities from the message.
        
        Args:
            message: User message
            
        Returns:
            Tuple of (intent, entities)
        """
        message = message.lower()
        entities = {}
        
        # If we have ASI or MOCK-LLM, use it for intent detection
        if self.api.has_asi or self.api.has_mock_llm:
            prompt = f"""
            Detect the intent and entities from the following message:
            
            Message: {message}
            
            Intents can be:
            - task_add: Adding a new task
            - task_list: Listing tasks
            - task_delete: Deleting a task
            - task_update: Updating a task
            - knowledge_learn: Storing information
            - knowledge_retrieve: Retrieving information
            - greeting: Greeting the assistant
            - farewell: Saying goodbye
            - help: Asking for help
            - general: General conversation
            
            Respond with JSON format:
            {{
                "intent": "detected_intent",
                "entities": {{
                    "entity_name": "entity_value"
                }}
            }}
            """
            
            try:
                if self.api.has_mock_llm:
                    response = self.api.generate_text(prompt)
                else:
                    # Use real ASI for intent detection with proper formatting
                    response = self.api.process_with_asi({
                        "task": "intent_detection",
                        "message": message
                    })
                    
                    # Process the ASI response
                    if isinstance(response, dict) and 'result' in response:
                        result_data = response['result']
                        # Extract intent and entities directly if available
                        if isinstance(result_data, dict):
                            if 'intent' in result_data:
                                return result_data['intent'], result_data.get('entities', {})
                    
                    # Try to parse JSON response as fallback
                    try:
                        # JSON might be in the response text or we might have a string result
                        response_text = response
                        if isinstance(response, dict) and 'text' in response:
                            response_text = response['text']
                        elif isinstance(response, dict) and 'result' in response and isinstance(response['result'], str):
                            response_text = response['result']
                            
                        # Extract JSON from response (might be surrounded by text)
                        json_match = re.search(r'({.*})', str(response_text), re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                            result = json.loads(json_str)
                            
                            if "intent" in result:
                                return result["intent"], result.get("entities", {})
                    except Exception as e:
                        self.logger.warning(f"Error parsing JSON from response: {str(e)}")
            except Exception as e:
                self.logger.error(f"Error parsing intent detection response: {str(e)}")
        
        # Fallback intent detection using simple rules
        if re.search(r'add\s+(?:a\s+)?task|new\s+task|create\s+(?:a\s+)?task', message):
            intent = "task_add"
            # Extract task description
            match = re.search(r'(?:add|new|create)(?:\s+a)?\s+task(?:\s+to)?\s+(.+)', message)
            if match:
                entities["description"] = match.group(1).strip()
            
        elif re.search(r'list\s+(?:my\s+)?tasks|show\s+(?:my\s+)?tasks|what\s+(?:are\s+)?(?:my\s+)?tasks', message):
            intent = "task_list"
            # Extract any filters
            if "today" in message:
                entities["timeframe"] = "today"
            elif "tomorrow" in message:
                entities["timeframe"] = "tomorrow"
            elif "this week" in message:
                entities["timeframe"] = "this_week"
            
        elif re.search(r'delete\s+(?:a\s+)?task|remove\s+(?:a\s+)?task', message):
            intent = "task_delete"
            # Extract task identifier
            match = re.search(r'(?:delete|remove)(?:\s+a)?\s+task(?:\s+to)?\s+(.+)', message)
            if match:
                entities["description"] = match.group(1).strip()
            
        elif re.search(r'update\s+(?:a\s+)?task|modify\s+(?:a\s+)?task|change\s+(?:a\s+)?task', message):
            intent = "task_update"
            # Extract task identifier and new details
            match = re.search(r'(?:update|modify|change)(?:\s+a)?\s+task(?:\s+to)?\s+(.+)', message)
            if match:
                entities["description"] = match.group(1).strip()
            
        elif re.search(r'remember\s+that|note\s+that|store\s+that|learn\s+that', message):
            intent = "knowledge_learn"
            # Extract the information to remember
            match = re.search(r'(?:remember|note|store|learn)\s+that\s+(.+)', message)
            if match:
                entities["information"] = match.group(1).strip()
            
        elif re.search(r'what\s+(?:is|are|was|were)|tell\s+me\s+about|do\s+you\s+know\s+about', message):
            intent = "knowledge_retrieve"
            # Extract the query
            match = re.search(r'(?:what\s+(?:is|are|was|were)|tell\s+me\s+about|do\s+you\s+know\s+about)\s+(.+)', message)
            if match:
                entities["query"] = match.group(1).strip().rstrip('?')
            
        elif re.search(r'^(?:hello|hi|hey|greetings|good\s+(?:morning|afternoon|evening))(?:\s+|$)', message):
            intent = "greeting"
            
        elif re.search(r'^(?:bye|goodbye|see\s+you|later|farewell)(?:\s+|$)', message):
            intent = "farewell"
            
        elif re.search(r'^(?:help|assist|support|guide)(?:\s+|$)', message):
            intent = "help"
            
        else:
            intent = "general"
        
        return intent, entities
    
    def _handle_task_add(self, entities: Dict) -> str:
        """Handle adding a new task."""
        if "description" not in entities:
            return f"I'd be happy to add a task for you. What task would you like me to add?"
        
        description = entities["description"]
        task_id = f"task_{len(self.tasks) + 1}"
        
        # Extract due date if present
        due_date = None
        due_match = re.search(r'(?:due|by|on)\s+(\w+)', description)
        if due_match:
            date_text = due_match.group(1).lower()
            
            if date_text == "today":
                due_date = datetime.now().date().isoformat()
            elif date_text == "tomorrow":
                due_date = (datetime.now() + timedelta(days=1)).date().isoformat()
            elif date_text in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
                # Simplified handling for demo purposes
                due_date = "next " + date_text
        
        # Add the task
        self.tasks[task_id] = {
            "id": task_id,
            "description": description,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "due_date": due_date
        }
        
        self.save_tasks()
        
        if due_date:
            return f"I've added your task: \"{description}\". It's due on {due_date}."
        else:
            return f"I've added your task: \"{description}\"."
    
    def _handle_task_list(self, entities: Dict) -> str:
        """Handle listing tasks."""
        if not self.tasks:
            return "You don't have any tasks at the moment."
        
        # Apply filters if specified
        filtered_tasks = self.tasks.values()
        
        if "timeframe" in entities:
            timeframe = entities["timeframe"]
            today = datetime.now().date().isoformat()
            tomorrow = (datetime.now() + timedelta(days=1)).date().isoformat()
            
            if timeframe == "today":
                filtered_tasks = [t for t in filtered_tasks if t.get("due_date") == today]
            elif timeframe == "tomorrow":
                filtered_tasks = [t for t in filtered_tasks if t.get("due_date") == tomorrow]
            elif timeframe == "this_week":
                # Simplified for demo
                filtered_tasks = [t for t in filtered_tasks if t.get("due_date") and t.get("due_date") >= today]
        
        # Filter by status if specified
        if "status" in entities:
            status = entities["status"]
            filtered_tasks = [t for t in filtered_tasks if t.get("status") == status]
        
        # Sort by creation date
        sorted_tasks = sorted(filtered_tasks, key=lambda t: t.get("created_at", ""))
        
        # Create response
        if not sorted_tasks:
            if "timeframe" in entities:
                return f"You don't have any tasks for {entities['timeframe'].replace('_', ' ')}."
            elif "status" in entities:
                return f"You don't have any {entities['status']} tasks."
            else:
                return "You don't have any tasks at the moment."
        
        response = "Here are your tasks:\n\n"
        
        for i, task in enumerate(sorted_tasks, 1):
            status = "✓" if task.get("status") == "completed" else "☐"
            due_str = f" (Due: {task.get('due_date')})" if task.get("due_date") else ""
            response += f"{i}. {status} {task.get('description')}{due_str}\n"
        
        return response
    
    def _handle_task_delete(self, entities: Dict) -> str:
        """Handle deleting a task."""
        if not self.tasks:
            return "You don't have any tasks to delete."
        
        if "description" not in entities:
            return "Which task would you like me to delete?"
        
        description = entities["description"].lower()
        
        # Look for task that matches the description
        for task_id, task in list(self.tasks.items()):
            if description in task.get("description", "").lower():
                deleted_desc = task.get("description")
                del self.tasks[task_id]
                self.save_tasks()
                return f"I've deleted the task: \"{deleted_desc}\"."
        
        return f"I couldn't find a task matching \"{description}\". Please try again with a different description."
    
    def _handle_task_update(self, entities: Dict) -> str:
        """Handle updating a task."""
        if not self.tasks:
            return "You don't have any tasks to update."
        
        if "description" not in entities:
            return "Which task would you like me to update?"
        
        description = entities["description"].lower()
        
        # Check if there's a status update
        new_status = None
        if "complete" in description or "done" in description or "finished" in description:
            new_status = "completed"
        elif "pending" in description or "open" in description or "active" in description:
            new_status = "pending"
        
        # Look for task that matches the description
        for task_id, task in self.tasks.items():
            task_desc = task.get("description", "").lower()
            
            # Check if task description contains any word from the update description
            words = description.split()
            if any(word in task_desc for word in words if len(word) > 3):
                # Update the status if specified
                if new_status:
                    task["status"] = new_status
                    self.save_tasks()
                    return f"I've marked the task \"{task.get('description')}\" as {new_status}."
                
                # If no status update but found task, ask for clarification
                return f"What would you like to update about the task \"{task.get('description')}\"?"
        
        return f"I couldn't find a task matching your description. Please try again with a different description."
    
    def _handle_knowledge_learn(self, entities: Dict) -> str:
        """Handle learning new information."""
        if "information" not in entities:
            return "What information would you like me to remember?"
        
        information = entities["information"]
        
        # Extract topic/category if present
        topic = "general"
        topic_match = re.search(r'(?:about|regarding|concerning|on)\s+(\w+)', information)
        if topic_match:
            topic = topic_match.group(1).lower()
        
        # Store the information
        if topic not in self.knowledge:
            self.knowledge[topic] = []
        
        self.knowledge[topic].append({
            "information": information,
            "stored_at": datetime.now().isoformat()
        })
        
        self.save_knowledge()
        
        return f"I'll remember that {information}."
    
    def _handle_knowledge_retrieve(self, entities: Dict) -> str:
        """Handle retrieving information from knowledge base."""
        if "query" not in entities:
            return "What information would you like me to recall?"
        
        query = entities["query"].lower()
        
        # Search in all topics
        found_info = []
        
        for topic, items in self.knowledge.items():
            for item in items:
                info = item.get("information", "").lower()
                if query in info or any(word in info for word in query.split() if len(word) > 3):
                    found_info.append(item.get("information"))
        
        if found_info:
            if len(found_info) == 1:
                return found_info[0]
            else:
                response = f"Here's what I know about {query}:\n\n"
                for i, info in enumerate(found_info, 1):
                    response += f"{i}. {info}\n"
                return response
        
        # Try to use ASI to generate a response based on the query
        try:
            if self.api.has_asi:
                # Import ASI helper for knowledge retrieval
                from real_world_apps.asi_helper import process_with_asi
                
                result = process_with_asi(self.api, {
                    "task": "answer_question",
                    "question": query,
                    "context": str(self.knowledge)  # Provide knowledge as context
                })
                
                if isinstance(result, dict) and result.get('success', False):
                    if 'result' in result:
                        if isinstance(result['result'], dict) and 'answer' in result['result']:
                            return result['result']['answer']
                        elif isinstance(result['result'], dict) and 'text' in result['result']:
                            return result['result']['text']
                        elif isinstance(result['result'], str) and len(result['result']) > 0:
                            return result['result']
            
            # Fall back to MOCK-LLM if ASI didn't produce a usable result
            if self.api.has_mock_llm:
                prompt = f"Provide a brief answer to: '{query}'"
                generated = self.api.generate_text(prompt)
                return generated
        except Exception as e:
            self.logger.error(f"Error using ASI for question answering: {str(e)}")
        
        return f"I don't have any information about {query}."
    
    def _handle_greeting(self, entities: Dict) -> str:
        """Handle greeting the user."""
        current_hour = datetime.now().hour
        
        if 5 <= current_hour < 12:
            time_greeting = "Good morning"
        elif 12 <= current_hour < 17:
            time_greeting = "Good afternoon"
        else:
            time_greeting = "Good evening"
        
        return f"{time_greeting}! I'm {self.name}, your virtual assistant. How can I help you today?"
    
    def _handle_farewell(self, entities: Dict) -> str:
        """Handle user saying goodbye."""
        return f"Goodbye! Feel free to ask for my assistance anytime you need it."
    
    def _handle_help(self, entities: Dict) -> str:
        """Handle request for help."""
        return f"""
I'm {self.name}, your virtual assistant. Here are some things I can help you with:

Tasks:
- Add a task: "Add a task to call John tomorrow"
- List tasks: "Show my tasks" or "What are my tasks for today?"
- Complete a task: "Mark the task call John as complete"
- Delete a task: "Delete the task call John"

Knowledge:
- Store information: "Remember that my meeting is at 3 PM"
- Retrieve information: "What time is my meeting?"

You can also just chat with me about anything!
"""
    
    def _generate_general_response(self, message: str) -> str:
        """Generate a general response for the message."""
        # Try using ASI for generating a response first
        try:
            if self.api.has_asi:
                # Format conversation history for context
                history_text = ""
                for entry in self.conversation_history[-5:]:  # Last 5 entries
                    role = "User" if entry.get("role") == "user" else self.name
                    history_text += f"{role}: {entry.get('content')}\n"
                
                # Import ASI helper for text generation
                from real_world_apps.asi_helper import process_with_asi
                
                result = process_with_asi(self.api, {
                    "task": "generate_response",
                    "message": message,
                    "context": history_text,
                    "assistant_name": self.name
                })
                
                if isinstance(result, dict) and result.get('success', False):
                    if 'result' in result:
                        if isinstance(result['result'], dict) and 'response' in result['result']:
                            return result['result']['response']
                        elif isinstance(result['result'], dict) and 'text' in result['result']:
                            return result['result']['text']
                        elif isinstance(result['result'], str) and len(result['result']) > 0:
                            return result['result']
            
            # Fall back to MOCK-LLM if ASI didn't produce a usable result
            if self.api.has_mock_llm:
                # Format conversation history for context
                history_text = ""
                for entry in self.conversation_history[-5:]:  # Last 5 entries
                    role = "User" if entry.get("role") == "user" else self.name
                    history_text += f"{role}: {entry.get('content')}\n"
                
                prompt = f"""
                {history_text}
                
                {self.name}: 
                """
                
                return self.api.generate_text(prompt)
        except Exception as e:
            self.logger.error(f"Error using ASI for response generation: {str(e)}")
        
        # Fallback responses for demo
        fallback_responses = [
            "I understand. Is there anything specific you'd like me to help you with?",
            "That's interesting. How can I assist you further?",
            "I see. Would you like me to help you with any tasks or information?",
            "I'm here to help. What would you like me to do for you?",
            "I'm not sure I understood completely. Could you please clarify?",
            "I'm always learning. How else can I assist you today?"
        ]
        
        return random.choice(fallback_responses)


def interactive_session(assistant: VirtualAssistant):
    """Run an interactive session with the virtual assistant."""
    print(f"\nWelcome to {assistant.name}, your virtual assistant!")
    print("You can start chatting, or type 'exit' to end the session.\n")
    
    # Greeting
    greeting_response = assistant.process_message("hello")
    print(f"{assistant.name}: {greeting_response}\n")
    
    while True:
        try:
            user_input = input("You: ")
            
            if user_input.lower() in ["exit", "quit", "bye"]:
                farewell_response = assistant.process_message("goodbye")
                print(f"\n{assistant.name}: {farewell_response}")
                break
            
            response = assistant.process_message(user_input)
            print(f"\n{assistant.name}: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")


def demo_session(assistant: VirtualAssistant):
    """Run a demonstration session with predefined interactions."""
    print("\n" + "="*80)
    print(f"VIRTUAL ASSISTANT DEMO: {assistant.name}".center(80))
    print("="*80 + "\n")
    
    # Pre-defined interactions to demonstrate functionality
    interactions = [
        "Hello!",
        "Add a task to call John tomorrow",
        "Add a task to prepare presentation for meeting",
        "What are my tasks?",
        "Mark the task call John as complete",
        "Show my tasks",
        "Delete the task prepare presentation",
        "Remember that my favorite color is blue",
        "What is my favorite color?",
        "Thank you for your help"
    ]
    
    for message in interactions:
        print(f"You: {message}")
        response = assistant.process_message(message)
        print(f"\n{assistant.name}: {response}\n")
        print("-" * 80)
    
    print("\n" + "="*80)
    print("END OF DEMONSTRATION".center(80))
    print("="*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Virtual Assistant Application")
    parser.add_argument("--name", type=str, default="Aria", help="Name of the assistant")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--demo", action="store_true", help="Run a demonstration session")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Ensure USE_REAL_ASI is set to true
    os.environ['USE_REAL_ASI'] = 'true'
    
    # Initialize the virtual assistant
    assistant = VirtualAssistant(name=args.name)
    
    # Run the appropriate session
    if args.interactive:
        interactive_session(assistant)
    elif args.demo:
        demo_session(assistant)
    else:
        # Default to demo mode
        demo_session(assistant)


if __name__ == "__main__":
    main()
