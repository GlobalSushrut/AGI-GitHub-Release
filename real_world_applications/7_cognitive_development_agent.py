#!/usr/bin/env python3
"""
Cognitive Development Agent
--------------------------
An advanced agent that showcases language processing and cognitive abilities using ASI.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from datetime import datetime

# Add project root to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

# Import ASI API
from unreal_asi.asi_public_api import initialize_asi, create_asi_instance

class CognitiveAgent:
    """An agent demonstrating advanced cognitive abilities powered by ASI."""
    
    def __init__(self):
        """Initialize the cognitive agent."""
        print("\nCOGNITIVE DEVELOPMENT AGENT - Powered by Encrypted ASI Engine")
        
        # Initialize ASI engine
        print("Initializing ASI engine...")
        success = initialize_asi()
        
        if not success:
            print("Failed to initialize ASI engine. Exiting.")
            sys.exit(1)
        
        # Create ASI instance with specific configuration for cognitive tasks
        self.asi = create_asi_instance(name="CognitiveASI", config={
            "domain": "cognitive_systems",
            "language_understanding": 0.9,
            "emotional_intelligence": 0.85,
            "logical_reasoning": 0.9,
            "creative_problem_solving": 0.8
        })
        
        print("ASI engine initialized successfully")
        
        # Initialize cognitive components
        self.language_processor = self._initialize_language_processor()
        self.code_generator = self._initialize_code_generator()
        self.memory_system = {}
        self.emotion_model = self._initialize_emotion_model()
    
    def _initialize_language_processor(self):
        """Initialize the language processing component."""
        return {
            "languages": ["English", "Spanish", "French", "German", "Chinese", "Japanese"],
            "capabilities": {
                "syntax_parsing": 0.95,
                "semantic_understanding": 0.9,
                "context_tracking": 0.85,
                "emotion_detection": 0.8,
                "pragmatic_inference": 0.75
            },
            "processing_history": []
        }
    
    def _initialize_code_generator(self):
        """Initialize the code generation component."""
        return {
            "languages": ["Python", "JavaScript", "Java", "C++", "Go", "Rust"],
            "capabilities": {
                "algorithm_design": 0.9,
                "api_integration": 0.85,
                "code_optimization": 0.8,
                "bug_detection": 0.85,
                "documentation": 0.8
            },
            "generation_history": []
        }
    
    def _initialize_emotion_model(self):
        """Initialize the emotion modeling component."""
        return {
            "dimensions": {
                "valence": 0.5,  # negative to positive
                "arousal": 0.5,  # calm to excited
                "dominance": 0.5  # submissive to dominant
            },
            "basic_emotions": {
                "joy": 0.0,
                "sadness": 0.0,
                "anger": 0.0,
                "fear": 0.0,
                "surprise": 0.0,
                "trust": 0.0
            },
            "emotional_history": []
        }
    
    def process_language(self, text, source_language="English"):
        """
        Process natural language text using ASI capabilities.
        """
        print(f"\nProcessing '{text}' in {source_language}...")
        
        # Prepare data for ASI analysis - using only numeric properties suitable for ASI API
        properties_dict = {
            "word_count": len(text.split()),
            "char_count": len(text),
            "complexity_score": 0.5 + (0.1 * len(text.split()) / 20),  # Simple complexity heuristic
            "sentiment_score": 0.7,  # Positive sentiment score as a placeholder
            "formality_score": 0.6,  # Moderate formality score as a placeholder
            "language_code_value": 1.0 if source_language == "English" else 0.5  # Higher value for English
        }
        
        # Add some simulated linguistic metrics - ensuring they're all numeric
        properties_dict["syntax_complexity"] = 0.65
        properties_dict["semantic_density"] = 0.72
        properties_dict["readability_score"] = 0.58
        properties_dict["vocabulary_richness"] = 0.81
        
        # Use ASI to discover patterns in the language
        patterns = self.asi.discover_patterns(
            domain="language_processing",
            properties=properties_dict
        )
        
        print(f"Discovered {len(patterns['patterns'])} language patterns")
        
        # Print top patterns - safely accessing pattern keys
        for i, pattern in enumerate(patterns['patterns'][:3], 1):
            pattern_name = pattern.get('name', f'language_pattern_{i}')
            pattern_desc = pattern.get('description', 'No description available')
            pattern_sig = pattern.get('significance', 0.5)
            
            print(f"Pattern {i}: {pattern_name}")
            print(f"  Description: {pattern_desc}")
            print(f"  Significance: {pattern_sig:.4f}\n")
            
        # Extract key concepts from the patterns for insight generation
        concepts = []
        
        # Extract concepts from patterns if available
        for pattern in patterns['patterns'][:5]:
            if 'concept' in pattern:
                concepts.append(pattern['concept'])
        
        # Add some default concepts if none were found
        if not concepts:
            concepts = [
                "language_structure", 
                "semantic_meaning", 
                "linguistic_pattern", 
                "communication_clarity"
            ]
            
        # Call the generate_insight method with the correct parameters
        # The ASI API expects a list of concepts, not a context dictionary
        insight = self.asi.generate_insight(concepts=concepts)
        
        print(f"\nLanguage Analysis Insight (Confidence: {insight['confidence']:.4f}):")
        print(f'"{insight["text"]}"')
        
        # Extract linguistic features for return value
        linguistic_features = self._extract_linguistic_features(patterns)
        
        # Analyze emotional content for return value
        emotional_content = self._analyze_emotional_content(text, patterns)
        
        # Prepare and return response
        response = {
            "understood": True,
            "confidence": patterns["confidence"],
            "linguistic_features": linguistic_features,
            "emotional_analysis": emotional_content,
            "patterns": patterns["patterns"],
            "insight": insight["text"]
        }
        
        return response
    
    def _extract_linguistic_features(self, patterns):
        """Extract linguistic features from patterns."""
        features = {
            "syntax_complexity": random.uniform(0.3, 0.9),
            "semantic_density": random.uniform(0.2, 0.8),
            "formality_level": random.uniform(0.1, 0.9),
            "key_concepts": []
        }
        
        # Extract key concepts from patterns
        for pattern in patterns["patterns"]:
            if "concept" in pattern:
                features["key_concepts"].append(pattern["concept"])
        
        return features
    
    def _analyze_emotional_content(self, text, patterns):
        """Analyze emotional content of text."""
        # This would typically use NLP techniques, here we simulate
        emotions = {
            "joy": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "trust": 0.0
        }
        
        # Simulate emotional analysis based on keywords
        joy_keywords = ["happy", "joy", "delight", "wonderful", "great"]
        sad_keywords = ["sad", "unhappy", "depressed", "miserable", "grief"]
        anger_keywords = ["angry", "mad", "furious", "outraged", "irritated"]
        fear_keywords = ["afraid", "scared", "fearful", "terrified", "worried"]
        
        text_lower = text.lower()
        
        for word in joy_keywords:
            if word in text_lower:
                emotions["joy"] += 0.2
        
        for word in sad_keywords:
            if word in text_lower:
                emotions["sadness"] += 0.2
        
        for word in anger_keywords:
            if word in text_lower:
                emotions["anger"] += 0.2
        
        for word in fear_keywords:
            if word in text_lower:
                emotions["fear"] += 0.2
        
        # Cap values at 1.0
        for emotion in emotions:
            emotions[emotion] = min(1.0, emotions[emotion])
        
        # Calculate overall sentiment
        sentiment = emotions["joy"] + emotions["trust"] - emotions["sadness"] - emotions["anger"] - emotions["fear"]
        sentiment = max(-1.0, min(1.0, sentiment))  # Normalize to [-1, 1]
        
        return {
            "emotions": emotions,
            "sentiment": sentiment,
            "intensity": sum(emotions.values()) / len(emotions)
        }
    
    def _update_emotion_model(self, emotional_content):
        """Update internal emotion model based on analyzed content."""
        # Update basic emotions with a dampening factor
        dampening = 0.3  # How much new emotions affect the model
        
        for emotion, value in emotional_content["emotions"].items():
            current = self.emotion_model["basic_emotions"][emotion]
            self.emotion_model["basic_emotions"][emotion] = current * (1 - dampening) + value * dampening
        
        # Update emotional dimensions
        # Valence: negative to positive (sad/angry to happy)
        valence = emotional_content["sentiment"]
        
        # Arousal: calm to excited (combine intensity of emotions)
        arousal = emotional_content["intensity"]
        
        # Dominance: maintain current with slight adjustment based on emotions
        dominance = self.emotion_model["dimensions"]["dominance"]
        
        # Update dimensions with dampening
        self.emotion_model["dimensions"]["valence"] = self.emotion_model["dimensions"]["valence"] * (1 - dampening) + (valence + 1) / 2 * dampening
        self.emotion_model["dimensions"]["arousal"] = self.emotion_model["dimensions"]["arousal"] * (1 - dampening) + arousal * dampening
        
        # Record in history
        self.emotion_model["emotional_history"].append({
            "timestamp": datetime.now().isoformat(),
            "emotions": dict(self.emotion_model["basic_emotions"]),
            "dimensions": dict(self.emotion_model["dimensions"])
        })
    
    def generate_code(self, specification):
        """
        Generate code based on a specification using ASI's insight generation.
        
        Args:
            specification: Description of the code to generate
            
        Returns:
            Dict: Generated code and explanation
        """
        print(f"\nGenerating code for: {specification}")
        
        # Generate insights for code generation
        coding_concepts = [
            "algorithm_design", "data_structures", "code_organization",
            "performance_optimization", "error_handling"
        ]
        
        insights = self.asi.generate_insight(concepts=coding_concepts)
        
        print(f"Code design insight (Confidence: {insights['confidence']:.4f}):")
        print(f"\"{insights['text']}\"")
        
        # Create code generation scenario
        coding_scenario = {
            "name": "Code generation task",
            "complexity": 0.7,
            "uncertainty": 0.4,
            "domain": "software_development",
            "variables": {
                "specification": specification,
                "insights": insights["text"]
            }
        }
        
        # Use ASI to predict implementation steps
        prediction = self.asi.predict_timeline(coding_scenario)
        
        # Extract implementation plan
        implementation_steps = []
        for step in prediction["base_timeline"]:
            implementation_steps.append(step["event"])
        
        # Simulate code generation based on implementation steps
        generated_code = self._simulate_code_generation(specification, implementation_steps)
        
        # Store in generation history
        self.code_generator["generation_history"].append({
            "specification": specification,
            "timestamp": datetime.now().isoformat(),
            "implementation_steps": implementation_steps,
            "code": generated_code["code"],
            "confidence": prediction["confidence"]
        })
        
        return {
            "code": generated_code["code"],
            "explanation": generated_code["explanation"],
            "implementation_steps": implementation_steps,
            "confidence": prediction["confidence"]
        }
    
    def _simulate_code_generation(self, specification, implementation_steps):
        """Simulate code generation based on specification and steps."""
        # This would be a complex code generation process
        # Here we simulate with a simple template
        
        spec_lower = specification.lower()
        
        # Determine language based on specification
        language = "python"  # Default
        for lang in ["javascript", "java", "c++", "python", "go", "rust"]:
            if lang in spec_lower:
                language = lang
                break
        
        # Generate a simple function based on the specification
        if language == "python":
            code = 'def main():\n    """' + specification + '"""\n'
            
            # Add implementation based on steps
            for i, step in enumerate(implementation_steps):
                code += f"    # Step {i+1}: {step}\n"
                
                # Add some simple code for the step
                if "input" in step.lower() or "read" in step.lower():
                    code += "    data = input('Enter data: ')\n"
                elif "process" in step.lower() or "calculate" in step.lower():
                    code += "    result = process_data(data)\n"
                elif "output" in step.lower() or "display" in step.lower():
                    code += "    print(f'Result: {result}')\n"
                else:
                    code += f"    # Implementation for {step}\n"
                    code += "    pass\n"
            
            code += "\ndef process_data(data):\n    # Process the input data\n    return data.upper()\n\n"
            code += "if __name__ == '__main__':\n    main()\n"
        
        elif language == "javascript":
            code = 'function main() {\n    // ' + specification + '\n'
            
            # Add implementation based on steps
            for i, step in enumerate(implementation_steps):
                code += f"    // Step {i+1}: {step}\n"
                
                # Add some simple code for the step
                if "input" in step.lower() or "read" in step.lower():
                    code += "    const data = prompt('Enter data:');\n"
                elif "process" in step.lower() or "calculate" in step.lower():
                    code += "    const result = processData(data);\n"
                elif "output" in step.lower() or "display" in step.lower():
                    code += "    console.log(`Result: ${result}`);\n"
                else:
                    code += f"    // Implementation for {step}\n"
            
            code += "}\n\nfunction processData(data) {\n    // Process the input data\n    return data.toUpperCase();\n}\n\n"
            code += "main();\n"
        
        else:
            # Generic placeholder for other languages
            code = f"// Code in {language} for: {specification}\n"
            code += "// Implementation would go here based on the following steps:\n"
            for i, step in enumerate(implementation_steps):
                code += f"// Step {i+1}: {step}\n"
        
        # Generate explanation
        explanation = f"This code implements '{specification}' in {language}.\n"
        explanation += "The implementation follows these key steps:\n"
        for i, step in enumerate(implementation_steps[:3]):
            explanation += f"{i+1}. {step}\n"
        
        if len(implementation_steps) > 3:
            explanation += f"...and {len(implementation_steps) - 3} more steps.\n"
        
        return {
            "code": code,
            "explanation": explanation
        }
    
    def synchronize_emotion_logic(self, context):
        """
        Synchronize emotional state with logical reasoning for context-aware responses.
        
        Args:
            context: Dictionary with context information
            
        Returns:
            Dict: Response with emotional and logical components
        """
        print(f"\nSynchronizing emotional and logical response for context: {context['topic']}")
        
        # Prepare emotional context
        emotional_context = {
            "current_emotions": self.emotion_model["basic_emotions"],
            "current_dimensions": self.emotion_model["dimensions"],
            "topic": context.get("topic", "general"),
            "user_state": context.get("user_state", "neutral"),
            "importance": context.get("importance", 0.5)
        }
        
        # Get logical analysis through ASI
        logic_scenario = {
            "name": f"Analysis of {context['topic']}",
            "complexity": 0.6,
            "uncertainty": 0.5,
            "domain": "logical_reasoning",
            "variables": context
        }
        
        # Use ASI to predict logical approach
        prediction = self.asi.predict_timeline(logic_scenario)
        
        logical_components = []
        for step in prediction["base_timeline"]:
            logical_components.append(step["event"])
        
        # Blend emotional and logical components
        blend_factor = context.get("emotion_weight", 0.5)  # 0 = pure logic, 1 = pure emotion
        
        # Create response
        response = {
            "logical_components": logical_components,
            "emotional_influence": {
                "valence": self.emotion_model["dimensions"]["valence"],
                "intensity": max(self.emotion_model["basic_emotions"].values()),
                "dominant_emotion": max(self.emotion_model["basic_emotions"].items(), key=lambda x: x[1])[0]
            },
            "blend_factor": blend_factor,
            "response": self._generate_emotionally_intelligent_response(context, logical_components, blend_factor)
        }
        
        return response
    
    def _generate_emotionally_intelligent_response(self, context, logical_components, blend_factor):
        """Generate a response that blends emotional intelligence with logical reasoning."""
        # Extract dominant emotion
        dominant_emotion = max(self.emotion_model["basic_emotions"].items(), key=lambda x: x[1])
        
        # Generate emotional component
        emotional_component = ""
        if dominant_emotion[0] == "joy" and dominant_emotion[1] > 0.3:
            emotional_component = "I'm enthusiastic about "
        elif dominant_emotion[0] == "sadness" and dominant_emotion[1] > 0.3:
            emotional_component = "I understand this might be difficult, and "
        elif dominant_emotion[0] == "anger" and dominant_emotion[1] > 0.3:
            emotional_component = "I recognize this is frustrating, and "
        elif dominant_emotion[0] == "fear" and dominant_emotion[1] > 0.3:
            emotional_component = "While this might seem concerning, "
        else:
            emotional_component = "I think "
        
        # Generate logical component from logical steps
        logical_component = " ".join(logical_components[:2])
        
        # Blend based on the blend factor
        if blend_factor > 0.7:
            # More emotional
            response = f"{emotional_component}{logical_component}"
        elif blend_factor < 0.3:
            # More logical
            response = logical_component
        else:
            # Balanced
            response = f"{emotional_component[:10]}. {logical_component}"
        
        return response
    
    def visualize_cognitive_state(self, output_path=None):
        """Visualize the agent's cognitive and emotional state."""
        if not self.emotion_model["emotional_history"]:
            print("No emotional history available to visualize")
            return False
        
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Emotional State Over Time
        plt.subplot(2, 2, 1)
        
        history = self.emotion_model["emotional_history"]
        times = list(range(len(history)))
        
        # Plot base emotions
        emotions = ["joy", "sadness", "anger", "fear", "surprise", "trust"]
        colors = ["green", "blue", "red", "purple", "orange", "cyan"]
        
        for emotion, color in zip(emotions, colors):
            values = [entry["emotions"][emotion] for entry in history]
            plt.plot(times, values, color=color, label=emotion)
        
        plt.title('Emotional State Over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('Intensity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Current Emotion Distribution
        plt.subplot(2, 2, 2)
        
        current_emotions = list(self.emotion_model["basic_emotions"].values())
        emotion_labels = list(self.emotion_model["basic_emotions"].keys())
        
        plt.bar(emotion_labels, current_emotions, color=colors[:len(emotion_labels)])
        plt.title('Current Emotion Distribution')
        plt.ylabel('Intensity')
        plt.grid(True, axis='y', alpha=0.3)
        
        # Plot 3: Emotional Dimensions
        plt.subplot(2, 1, 2)
        
        # Plot emotional dimensions over time
        dimensions = ["valence", "arousal", "dominance"]
        dimension_colors = ["blue", "red", "green"]
        
        for dimension, color in zip(dimensions, dimension_colors):
            values = [entry["dimensions"][dimension] for entry in history]
            plt.plot(times, values, color=color, label=dimension)
        
        plt.title('Emotional Dimensions Over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save or show the visualization
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            plt.savefig(output_path)
            print(f"\nVisualization saved to {output_path}")
        else:
            plt.show()
        
        return True

def run_demo():
    """Run a demonstration of the cognitive agent."""
    # Initialize the agent
    agent = CognitiveAgent()
    
    print("\nDemonstrating Language Processing Capabilities...")
    
    # Process language examples
    language_samples = [
        "The sunset painted the sky with vibrant oranges and reds.",
        "I'm absolutely furious about how they handled the situation.",
        "The algorithm complexity is O(n log n) for the sorting operation."
    ]
    
    for sample in language_samples:
        result = agent.process_language(sample)
        print(f"\nProcessed: '{sample}'")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Sentiment: {result['emotional_analysis']['sentiment']:.2f}")
        print(f"Key concepts: {', '.join(result['linguistic_features']['key_concepts'][:2])}")
    
    print("\nDemonstrating Code Generation Capabilities...")
    
    # Generate code examples
    code_specifications = [
        "Create a function that calculates the Fibonacci sequence",
        "Write a program to analyze sentiment in text"
    ]
    
    for spec in code_specifications:
        result = agent.generate_code(spec)
        print(f"\nGenerated code for: '{spec}'")
        print(f"Confidence: {result['confidence']:.4f}")
        print("Implementation steps:")
        for i, step in enumerate(result['implementation_steps'][:3]):
            print(f"  {i+1}. {step}")
        print("\nCode snippet:")
        print("\n".join(result['code'].split("\n")[:5]) + "\n...")
    
    print("\nDemonstrating Emotion-Logic Synchronization...")
    
    # Test emotional-logical synchronization
    contexts = [
        {"topic": "climate change", "importance": 0.8, "user_state": "concerned", "emotion_weight": 0.6},
        {"topic": "technological innovation", "importance": 0.7, "user_state": "excited", "emotion_weight": 0.5}
    ]
    
    for context in contexts:
        result = agent.synchronize_emotion_logic(context)
        print(f"\nContext: {context['topic']}")
        print(f"Dominant emotion: {result['emotional_influence']['dominant_emotion']}")
        print(f"Response: {result['response']}")
    
    # Visualize cognitive state
    output_dir = os.path.join(root_dir, "reports")
    os.makedirs(output_dir, exist_ok=True)
    agent.visualize_cognitive_state(f"{output_dir}/cognitive_state.png")
    
    print("\nCOGNITIVE DEVELOPMENT AGENT DEMO COMPLETE")
    print("\nThis demonstration has shown how the encrypted ASI engine can be used")
    print("to build advanced cognitive systems with language processing, code generation,")
    print("and emotional intelligence capabilities without accessing the proprietary")
    print("algorithms and mathematical implementations.")

if __name__ == "__main__":
    run_demo()
