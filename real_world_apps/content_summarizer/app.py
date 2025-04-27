#!/usr/bin/env python3
"""
Content Summarization Tool
-------------------------

A real-world application that demonstrates how to use the AGI Toolkit
to build a content summarization tool that can process articles,
documents, and web content.

Features:
- Automatic text summarization
- Key points extraction
- Reading time estimation
- Topic detection
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Any, Optional

# Add the parent directory to path so we can import the AGI Toolkit
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the AGI Toolkit and ASI helper
from agi_toolkit import AGIAPI
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asi_helper import initialize_asi_components, process_with_asi, generate_summary, extract_key_points

class ContentSummarizer:
    """A tool for summarizing content using AGI Toolkit."""
    
    def __init__(self):
        """Initialize the summarizer."""
        # Configure logging
        self.logger = logging.getLogger("ContentSummarizer")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info("Initializing Content Summarizer")
        
        # Initialize real ASI components if env var is set
        if os.environ.get('USE_REAL_ASI') == 'true':
            # Initialize the real ASI components
            initialize_asi_components()
        
        # Initialize the AGI Toolkit API
        self.api = AGIAPI()
        
        # Check component availability
        self.logger.info(f"ASI available: {self.api.has_asi}")
        self.logger.info(f"MOCK-LLM available: {self.api.has_mock_llm}")
        
        self.logger.info("Content Summarizer initialized")
    
    def summarize(self, text: str, length: str = "medium") -> Dict[str, Any]:
        """
        Summarize the provided content.
        
        Args:
            text: The text content to summarize
            length: Summary length - 'short', 'medium', or 'long'
            
        Returns:
            Dictionary containing summary and metadata
        """
        self.logger.info(f"Summarizing content ({len(text)} characters, {length} summary)")
        
        # Store in memory for context
        self.api.store_data("current_text", {
            "content": text,
            "length": len(text),
            "summary_length": length
        })
        
        # Basic text statistics
        words = text.split()
        word_count = len(words)
        estimated_reading_time = word_count // 200  # Average reading speed: 200 words per minute
        
        # Generate prompt based on desired length
        if length == "short":
            target_length = "2-3 sentences"
        elif length == "medium":
            target_length = "4-5 sentences"
        else:  # long
            target_length = "7-8 sentences"
        
        prompt = f"""Summarize the following text in {target_length}:

{text[:2000]}...  # Truncate if too long for prompt

Focus on the main points and key information.
"""
        
        # Try to generate summary using real ASI first if available, otherwise MOCK-LLM
        if self.api.has_asi:
            self.logger.info("Using real ASI for summary generation")
            
            # Create a request for summary generation
            summary_request = {
                "task": "generate_summary",
                "content": text[:5000],  # Limit content size
                "length": length
            }
            
            # Use our optimized asi_helper function if available
            if os.environ.get('USE_REAL_ASI') == 'true' and 'generate_summary' in globals():
                result = generate_summary(text[:5000], length)
            else:
                # Fallback to regular ASI processing
                result = self.api.process_with_asi(summary_request)
            
            if result.get("success", False) and "result" in result:
                points_data = result["result"]
                
                # Try to extract a summary from the ASI output
                if isinstance(points_data, dict):
                    if "summary" in points_data:  # Our optimized format
                        self.logger.info("Found summary in ASI output")
                        summary = points_data["summary"]
                    elif "text" in points_data:  # Direct text output
                        self.logger.info("Found text in ASI output")
                        summary = points_data["text"]
                    elif "insight" in points_data:  # Insight text
                        self.logger.info("Found insight in ASI output")
                        summary = points_data["insight"]
                    elif "points" in points_data:  # Points-based output
                        # Join points to form a summary
                        self.logger.info("Found points in ASI output")
                        summary = ". ".join(points_data["points"])
                    else:
                        # Try to find any string values in the dictionary
                        str_values = [v for v in points_data.values() if isinstance(v, str) and len(v) > 30]
                        if str_values:
                            self.logger.info("Using string value from ASI output")
                            summary = str_values[0]  # Use the first substantial string
                        else:
                            # Fallback to default summarization
                            self.logger.info("No usable summary in ASI output, using fallback")
                            summary = self._fallback_summarize(text, length)
                else:
                    # If we got a string directly
                    if isinstance(points_data, str) and len(points_data) > 30:
                        summary = points_data
                    else:
                        # Fallback to default summarization
                        summary = self._fallback_summarize(text, length)
            else:
                # Fallback to MOCK-LLM or default summarization
                if self.api.has_mock_llm:
                    summary = self.api.generate_text(prompt)
                else:
                    summary = self._fallback_summarize(text, length)
        elif self.api.has_mock_llm:
            summary = self.api.generate_text(prompt)
        else:
            # Fallback summarization if neither ASI nor MOCK-LLM is available
            summary = self._fallback_summarize(text, length)
        
        # Extract key points using ASI if available
        key_points = []
        if self.api.has_asi:
            # Format input suitable for ASI pattern discovery or insight generation
            text_sample = text[:5000]  # Limit content size
            
            self.logger.info("Using real ASI for key point extraction")
            
            # Try insight generation approach with optimized format for ASI
            result = self.api.process_with_asi({
                "task": "extract_key_points",
                "content": text_sample,
                "max_points": 5
            })
            
            self.logger.info(f"ASI result: {result}")
            
            if result.get("success", False) and "result" in result:
                # Handle the response from real ASI components
                points_data = result["result"]
                
                # Different output formats based on which ASI function was used
                if isinstance(points_data, dict):
                    if "points" in points_data:  # Our optimized format
                        self.logger.info("Found points in expected format")
                        key_points = points_data["points"]
                    elif "patterns" in points_data:  # Pattern discovery output
                        self.logger.info("Found patterns in ASI output")
                        # Extract patterns as key points
                        patterns = points_data.get("patterns", [])
                        for pattern in patterns:
                            if isinstance(pattern, dict):
                                # Try different fields that might contain useful text
                                if "description" in pattern:
                                    key_points.append(pattern["description"])
                                elif "concept" in pattern:
                                    key_points.append(pattern["concept"])
                    elif "text" in points_data:  # Insight generation output
                        self.logger.info("Found text in ASI output")
                        insight_text = points_data.get("text", "")
                        # Split insight into sentences
                        sentences = insight_text.split(". ")
                        for sentence in sentences:
                            if sentence and len(sentence) > 10:  # Only substantial sentences
                                key_points.append(sentence)
                    elif "insight" in points_data:  # Another insight format
                        self.logger.info("Found insight in ASI output")
                        insight_text = points_data.get("insight", "")
                        # Split insight into sentences
                        sentences = insight_text.split(". ")
                        for sentence in sentences:
                            if sentence and len(sentence) > 10:  # Only substantial sentences
                                key_points.append(sentence)
                elif isinstance(points_data, list):
                    self.logger.info("Found list in ASI output")
                    key_points = points_data
                else:
                    # Try to extract something useful from whatever we got
                    text_repr = str(points_data)
                    sentences = text_repr.split(". ")
                    for sentence in sentences:
                        if sentence and len(sentence) > 10:  # Only substantial sentences
                            key_points.append(sentence)
                    
                    if not key_points:  # If we still don't have points
                        key_points = [f"Key point from ASI analysis: {str(points_data)[:100]}"]  
            
            # Ensure we have at least some key points
            if not key_points:
                self.logger.warning("No key points extracted from ASI output, using fallback")
                key_points = self._extract_key_points(text)
        else:
            # Fallback key points extraction
            key_points = self._extract_key_points(text)
        
        # Identify main topics
        topics = self._identify_topics(text)
        
        # Create the result
        result = {
            "summary": summary,
            "key_points": key_points,
            "stats": {
                "word_count": word_count,
                "reading_time_minutes": max(1, estimated_reading_time),
                "character_count": len(text)
            },
            "topics": topics
        }
        
        # Store the result in memory
        self.api.store_data("last_summary", result)
        
        return result
    
    def _fallback_summarize(self, text: str, length: str) -> str:
        """Fallback summarization method when MOCK-LLM is not available."""
        self.logger.info("Using fallback summarization method")
        
        sentences = text.split('.')
        
        # Determine number of sentences based on length
        if length == "short":
            num_sentences = min(3, len(sentences))
        elif length == "medium":
            num_sentences = min(5, len(sentences))
        else:  # long
            num_sentences = min(8, len(sentences))
        
        # Get first n sentences
        summary_sentences = sentences[:num_sentences]
        summary = '. '.join(summary_sentences).strip() + '.'
        
        return f"Summary (fallback method): {summary}"
    
    def _extract_key_points(self, text: str) -> List[str]:
        """Fallback method to extract key points from text."""
        sentences = text.split('.')
        
        # Simple heuristic: sentences with important-sounding phrases
        important_phrases = [
            "importantly", "significant", "key", "critical", "essential",
            "crucial", "main", "primarily", "chief", "vital"
        ]
        
        key_points = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if sentence contains important phrases
            if any(phrase in sentence.lower() for phrase in important_phrases):
                key_points.append(sentence + '.')
            
            # Limit to 5 key points
            if len(key_points) >= 5:
                break
        
        # If we couldn't find enough key points, just take first few sentences
        if len(key_points) < 3:
            for sentence in sentences[:5]:
                sentence = sentence.strip()
                if sentence and sentence not in key_points:
                    key_points.append(sentence + '.')
                    if len(key_points) >= 3:
                        break
        
        return key_points
    
    def _identify_topics(self, text: str) -> List[str]:
        """Identify main topics in the text."""
        # Common topics to check for
        potential_topics = [
            "technology", "science", "politics", "business", 
            "health", "education", "environment", "sports",
            "entertainment", "art", "history", "philosophy",
            "medicine", "finance", "economy", "social", "culture"
        ]
        
        text_lower = text.lower()
        
        # Count occurrences of potential topics
        topic_counts = {}
        for topic in potential_topics:
            count = text_lower.count(topic)
            if count > 0:
                topic_counts[topic] = count
        
        # Sort by count and take top 5
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        top_topics = [topic for topic, _ in sorted_topics[:5]]
        
        return top_topics


def display_summary(result: Dict[str, Any]):
    """Display the summary in a user-friendly format."""
    print("\n" + "="*80)
    print("CONTENT SUMMARY".center(80))
    print("="*80 + "\n")
    
    stats = result["stats"]
    print(f"Word Count: {stats['word_count']} | Reading Time: {stats['reading_time_minutes']} min")
    
    if result["topics"]:
        print(f"Topics: {', '.join(result['topics'])}\n")
    
    print("SUMMARY:")
    print("-" * 80)
    print(result["summary"])
    print("-" * 80)
    
    if result["key_points"]:
        print("\nKEY POINTS:")
        for i, point in enumerate(result["key_points"], 1):
            print(f"{i}. {point}")
    
    print("\n" + "="*80)


def read_file(file_path: str) -> str:
    """Read content from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Content Summarization Tool")
    parser.add_argument("--file", type=str, help="Path to a text file to summarize")
    parser.add_argument("--text", type=str, help="Text content to summarize")
    parser.add_argument("--length", type=str, choices=["short", "medium", "long"], 
                       default="medium", help="Summary length")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the summarizer
    summarizer = ContentSummarizer()
    
    # Get content to summarize
    content = ""
    if args.file:
        try:
            content = read_file(args.file)
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            return
    elif args.text:
        content = args.text
    else:
        # Sample text for demo
        content = """
Artificial intelligence (AI) is revolutionizing industries across the globe, 
from healthcare to finance, transportation to education. Machine learning, 
a subset of AI, uses algorithms to parse data, learn from it, and make informed 
decisions based on what it has learned. Deep learning, a specialized form of 
machine learning, employs neural networks with many layers to analyze various 
factors of data.

The applications of AI are vast and growing. In healthcare, AI helps in diagnosing 
diseases, developing new medicines, and personalizing treatment plans. In finance, 
it powers algorithmic trading, fraud detection, and risk assessment. Autonomous 
vehicles use AI to navigate roads and make split-second decisions to ensure safety.

Despite the benefits, there are concerns about AI's impact on employment, privacy, 
and security. As machines become more capable, some jobs may be automated, leading 
to workforce displacement. Additionally, the vast amounts of data used to train AI 
systems raise privacy concerns, while the potential for AI to be used in cyber attacks 
or autonomous weapons raises security concerns.

Ethical considerations in AI development and deployment include issues of bias, 
fairness, transparency, accountability, and harm prevention. These considerations 
are particularly important as AI systems become more autonomous and are used in 
high-stakes applications like criminal justice, healthcare, and hiring.

The future of AI holds both promise and challenges. As the technology continues to 
evolve, it's crucial for developers, policymakers, and society at large to ensure 
that AI is developed and used in ways that are ethical, beneficial, and respectful 
of human rights and values.
"""
    
    if not content:
        print("Error: No content provided to summarize.")
        return
    
    # Summarize the content
    result = summarizer.summarize(content, args.length)
    
    # Display the summary
    display_summary(result)


if __name__ == "__main__":
    main()
