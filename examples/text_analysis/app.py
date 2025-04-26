#!/usr/bin/env python
"""
Text Analysis Application Example
--------------------------------

This application demonstrates how to use the AGI Toolkit
to build a text analysis tool without modifying core code.

Features:
- Sentiment analysis with MOCK-LLM
- Complex reasoning with ASI
- Content generation
- Memory persistence
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, Any, List

# Add the parent directory to path so we can import the package
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

# Import the AGI Toolkit
from agi_toolkit import AGIAPI


class TextAnalysisApp:
    """Text analysis application built with AGI Toolkit."""
    
    def __init__(self):
        """Initialize the application."""
        # Setup logging
        self.logger = logging.getLogger("TextAnalysisApp")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info("Initializing Text Analysis Application")
        
        # Initialize the API
        self.api = AGIAPI()
        
        # Check component availability
        self.logger.info(f"ASI available: {self.api.has_asi}")
        self.logger.info(f"MOCK-LLM available: {self.api.has_mock_llm}")
        
        # Initialize history
        self.history = []
        
        self.logger.info("Application initialized successfully")
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text using available components.
        
        Args:
            text: Text to analyze
            
        Returns:
            Analysis results
        """
        self.logger.info(f"Analyzing text: {text[:50]}...")
        
        # Store input in history
        self.history.append({
            "type": "input",
            "text": text
        })
        
        results = {
            "input_text": text,
            "length": len(text),
            "word_count": len(text.split())
        }
        
        # Use ASI for advanced analysis if available
        if self.api.has_asi:
            self.logger.info("Using ASI for advanced analysis")
            asi_results = self.api.process_with_asi({"text": text})
            results["asi_analysis"] = asi_results
        
        # Use MOCK-LLM for text generation if available
        if self.api.has_mock_llm:
            self.logger.info("Using MOCK-LLM for text generation")
            response = self.api.generate_text(f"Analyze this text: {text}")
            results["generated_response"] = response
            
            # Get classification from MOCK-LLM
            try:
                from agi_toolkit import MOCKLLMInterface
                mock_llm = MOCKLLMInterface()
                classification = mock_llm.classify_text(text)
                results["classification"] = classification
            except:
                # If direct interface access fails, continue without classification
                pass
        
        # Add basic analysis regardless of component availability
        results["basic_analysis"] = self._basic_analysis(text)
        
        # Store results in memory
        memory_key = f"analysis_{len(self.history)}"
        self.api.store_data(memory_key, results)
        
        # Add to history
        self.history.append({
            "type": "analysis",
            "key": memory_key
        })
        
        return results
    
    def generate_content(self, prompt: str) -> str:
        """
        Generate content based on a prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated content
        """
        self.logger.info(f"Generating content for prompt: {prompt[:50]}...")
        
        # Store prompt in history
        self.history.append({
            "type": "prompt",
            "text": prompt
        })
        
        # Generate with MOCK-LLM if available
        if self.api.has_mock_llm:
            content = self.api.generate_text(prompt)
        else:
            content = "Content generation not available (MOCK-LLM components missing)"
        
        # Store result in history
        self.history.append({
            "type": "generated",
            "text": content
        })
        
        return content
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the application history.
        
        Returns:
            List of history items
        """
        return self.history
    
    def save_state(self, filepath: str) -> bool:
        """
        Save the application state to a file.
        
        Args:
            filepath: Path to save the state
            
        Returns:
            Success status
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.history, f, indent=2)
            
            self.logger.info(f"State saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")
            return False
    
    def load_state(self, filepath: str) -> bool:
        """
        Load the application state from a file.
        
        Args:
            filepath: Path to load the state from
            
        Returns:
            Success status
        """
        try:
            with open(filepath, 'r') as f:
                self.history = json.load(f)
            
            self.logger.info(f"State loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading state: {str(e)}")
            return False
    
    def _basic_analysis(self, text: str) -> Dict[str, Any]:
        """Perform basic text analysis."""
        # Sentiment analysis
        positive_words = ["good", "great", "excellent", "awesome", "like", "love"]
        negative_words = ["bad", "terrible", "awful", "hate", "dislike", "poor"]
        
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Word frequency
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Reading level estimate
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        if avg_word_length > 6:
            reading_level = "advanced"
        elif avg_word_length > 5:
            reading_level = "intermediate"
        else:
            reading_level = "basic"
        
        return {
            "sentiment": sentiment,
            "top_words": dict(top_words),
            "avg_word_length": round(avg_word_length, 2),
            "reading_level": reading_level
        }


def display_results(results: Dict[str, Any]):
    """Display analysis results in a user-friendly format."""
    print("\n" + "="*60)
    print(f"TEXT ANALYSIS: {results['input_text'][:50]}...")
    print("="*60)
    
    print(f"\nWord Count: {results['word_count']}")
    print(f"Length: {results['length']} characters")
    
    # Print basic analysis
    basic = results['basic_analysis']
    print("\nBASIC ANALYSIS:")
    print(f"Sentiment: {basic['sentiment']}")
    print(f"Reading Level: {basic['reading_level']} (avg word length: {basic['avg_word_length']})")
    print("\nTop Words:")
    for word, count in list(basic['top_words'].items())[:5]:
        print(f"  {word}: {count}")
    
    # Print ASI analysis if available
    if 'asi_analysis' in results and results['asi_analysis'].get('success', False):
        print("\nASI ANALYSIS:")
        asi_result = results['asi_analysis'].get('result', {})
        
        if isinstance(asi_result, dict):
            for key, value in asi_result.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {asi_result}")
        
        print(f"Confidence: {results['asi_analysis'].get('confidence', 'N/A')}")
    
    # Print classification if available
    if 'classification' in results:
        print("\nCLASSIFICATION:")
        for label, prob in results['classification'].items():
            if isinstance(prob, float):
                print(f"  {label}: {prob:.2f}")
    
    # Print generated response if available
    if 'generated_response' in results:
        print("\nGENERATED RESPONSE:")
        print(f"  {results['generated_response']}")
    
    print("\n" + "="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Text Analysis Application")
    parser.add_argument("--analyze", type=str, help="Text to analyze")
    parser.add_argument("--generate", type=str, help="Prompt for text generation")
    parser.add_argument("--save", type=str, help="Save state to file")
    parser.add_argument("--load", type=str, help="Load state from file")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the application
    app = TextAnalysisApp()
    
    # Load state if specified
    if args.load:
        app.load_state(args.load)
    
    # Process based on arguments
    if args.analyze:
        results = app.analyze_text(args.analyze)
        display_results(results)
    
    elif args.generate:
        text = app.generate_content(args.generate)
        print("\n" + "="*60)
        print("GENERATED CONTENT:")
        print("="*60)
        print(text)
        print("="*60)
    
    elif not args.load and not args.save:
        # If no specific action, use interactive mode
        print("\nText Analysis Application")
        print("="*60)
        print("1. Analyze text")
        print("2. Generate content")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            text = input("\nEnter text to analyze: ")
            results = app.analyze_text(text)
            display_results(results)
        
        elif choice == "2":
            prompt = input("\nEnter prompt for generation: ")
            text = app.generate_content(prompt)
            print("\n" + "="*60)
            print("GENERATED CONTENT:")
            print("="*60)
            print(text)
            print("="*60)
    
    # Save state if specified
    if args.save:
        app.save_state(args.save)


if __name__ == "__main__":
    main()
