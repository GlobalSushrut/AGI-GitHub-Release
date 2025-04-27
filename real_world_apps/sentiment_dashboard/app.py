#!/usr/bin/env python3
"""
Sentiment Analysis Dashboard
---------------------------

A real-world application that demonstrates how to use the AGI Toolkit
to build a sentiment analysis dashboard for monitoring customer feedback,
social media mentions, product reviews, etc.

Features:
- Text sentiment analysis (positive, negative, neutral)
- Emotion detection (joy, anger, sadness, etc.)
- Aspect-based sentiment analysis
- Trend visualization
"""

import os
import sys
import argparse
import logging
import json
import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add the parent directory to path so we can import the AGI Toolkit
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the ASI helper module and AGI Toolkit
from real_world_apps.asi_helper import initialize_asi_components, analyze_sentiment
from agi_toolkit import AGIAPI

class SentimentAnalyzer:
    """A sentiment analysis tool using AGI Toolkit."""
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        # Configure logging
        self.logger = logging.getLogger("SentimentAnalyzer")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info("Initializing Sentiment Analyzer")
        
        # Initialize real ASI components
        initialize_asi_components()
        
        # Set environment variable to ensure interface uses real components
        os.environ['USE_REAL_ASI'] = 'true'
        
        # Initialize the AGI Toolkit API
        self.api = AGIAPI()
        
        # Check component availability
        self.logger.info(f"ASI available: {self.api.has_asi}")
        self.logger.info(f"MOCK-LLM available: {self.api.has_mock_llm}")
        
        # Initialize history storage
        self.history = []
        
        self.logger.info("Sentiment Analyzer initialized")
    
    def analyze_sentiment(self, text: str, source: str = "unknown", timestamp: str = None) -> Dict[str, Any]:
        """
        Analyze sentiment of the provided text.
        
        Args:
            text: The text to analyze
            source: Source of the text (e.g., "twitter", "customer_review")
            timestamp: Timestamp of the text (defaults to current time)
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        self.logger.info(f"Analyzing sentiment of text from {source}")
        
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Try to use real ASI for sentiment analysis first
        sentiment_score = 0.0
        sentiment_label = "neutral"
        confidence = 0.5
        
        if self.api.has_asi:
            try:
                # Use the sentiment analysis helper
                sentiment_result = analyze_sentiment(self.api, text)
                
                if isinstance(sentiment_result, dict):
                    # Extract values from the result
                    sentiment_score = sentiment_result.get('score', 0.0)
                    sentiment_label = sentiment_result.get('label', 'neutral')
                    confidence = sentiment_result.get('confidence', 0.5) if 'confidence' in sentiment_result else 0.8
                    
                    self.logger.info(f"ASI sentiment analysis: {sentiment_label} (score: {sentiment_score})")
            except Exception as e:
                self.logger.error(f"Error using ASI for sentiment analysis: {str(e)}")
        
        # Use MOCK-LLM as fallback if ASI failed or is not available
        if (sentiment_label == "neutral" and abs(sentiment_score) < 0.2) and self.api.has_mock_llm:
            classification_prompt = f"""Classify the sentiment of the following text:

{text}

Return only one of these labels: positive, negative, or neutral
"""
            response = self.api.generate_text(classification_prompt)
            sentiment_label = self._parse_sentiment_label(response)
            
            # Map label to score
            if sentiment_label == "positive":
                sentiment_score = 0.8
                confidence = 0.75
            elif sentiment_label == "negative":
                sentiment_score = -0.7
                confidence = 0.7
            else:  # neutral
                sentiment_score = 0.1
                confidence = 0.6
        elif not self.api.has_asi and not self.api.has_mock_llm:
            # Use fallback sentiment analysis if neither ASI nor MOCK-LLM are available
            sentiment_score, sentiment_label, confidence = self._analyze_sentiment_fallback(text)
        
        # Use ASI for more advanced analysis if available
        emotions = {}
        aspects = {}
        
        if self.api.has_asi:
            try:
                # Import ASI helper for process_with_asi
                from real_world_apps.asi_helper import process_with_asi
                
                # Emotion detection
                emotion_result = process_with_asi(self.api, {
                    "task": "detect_emotions",
                    "text": text
                })
                
                if isinstance(emotion_result, dict) and emotion_result.get("success", False) and "result" in emotion_result:
                    emotions_data = emotion_result["result"]
                    if isinstance(emotions_data, dict):
                        # Check for different possible formats of emotion data
                        if "emotions" in emotions_data:
                            emotions = emotions_data["emotions"]
                        elif any(emotion in emotions_data for emotion in ["joy", "anger", "sadness", "fear", "surprise"]):
                            emotions = emotions_data
                        elif isinstance(emotions_data, dict):
                            # Use any dict that looks like emotion data
                            emotions = {k: v for k, v in emotions_data.items() 
                                       if isinstance(k, str) and isinstance(v, (int, float))
                                       and k.lower() in ["joy", "happiness", "anger", "sadness", "fear", 
                                                       "surprise", "disgust", "trust", "anticipation"]}
                
                # Aspect-based sentiment analysis
                aspect_result = process_with_asi(self.api, {
                    "task": "aspect_sentiment",
                    "text": text
                })
                
                if isinstance(aspect_result, dict) and aspect_result.get("success", False) and "result" in aspect_result:
                    aspects_data = aspect_result["result"]
                    if isinstance(aspects_data, dict):
                        # Check for different possible formats of aspect data
                        if "aspects" in aspects_data:
                            aspects = aspects_data["aspects"]
                        elif any(isinstance(v, dict) and "sentiment" in v for v in aspects_data.values()):
                            aspects = aspects_data
            except Exception as e:
                self.logger.error(f"Error in advanced sentiment analysis with ASI: {str(e)}")
                
        # Fallback emotion detection and aspect analysis if needed
        if not emotions:
            emotions = self._detect_emotions_fallback(text)
        if not aspects:
            aspects = self._analyze_aspects_fallback(text)
        
        # Create the result
        result = {
            "text": text,
            "source": source,
            "timestamp": timestamp,
            "sentiment": {
                "score": sentiment_score,  # Range: -1.0 to 1.0
                "label": sentiment_label,  # positive, negative, or neutral
                "confidence": confidence   # Range: 0.0 to 1.0
            },
            "emotions": emotions,
            "aspects": aspects
        }
        
        # Store in history
        self.history.append(result)
        
        # Store in memory
        memory_key = f"sentiment_{len(self.history)}"
        self.api.store_data(memory_key, result)
        
        return result
    
    def analyze_batch(self, texts: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of dictionaries containing text, source, and optional timestamp
            
        Returns:
            List of sentiment analysis results
        """
        self.logger.info(f"Analyzing batch of {len(texts)} texts")
        
        results = []
        for item in texts:
            text = item["text"]
            source = item.get("source", "unknown")
            timestamp = item.get("timestamp", None)
            
            result = self.analyze_sentiment(text, source, timestamp)
            results.append(result)
        
        return results
    
    def get_sentiment_trends(self, time_period: str = "all") -> Dict[str, Any]:
        """
        Get sentiment trends over time.
        
        Args:
            time_period: Time period for trend analysis (all, day, week, month)
            
        Returns:
            Dictionary containing trend analysis
        """
        self.logger.info(f"Generating sentiment trends for period: {time_period}")
        
        if not self.history:
            return {
                "error": "No sentiment data available for trend analysis"
            }
        
        # Filter history based on time period
        filtered_history = self._filter_by_time_period(self.history, time_period)
        
        if not filtered_history:
            return {
                "error": f"No sentiment data available for period: {time_period}"
            }
        
        # Calculate sentiment distribution
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for item in filtered_history:
            label = item["sentiment"]["label"]
            sentiment_counts[label] += 1
        
        total = len(filtered_history)
        sentiment_distribution = {
            "positive": sentiment_counts["positive"] / total,
            "negative": sentiment_counts["negative"] / total,
            "neutral": sentiment_counts["neutral"] / total
        }
        
        # Calculate average sentiment score
        avg_sentiment = sum(item["sentiment"]["score"] for item in filtered_history) / total
        
        # Get top emotions
        all_emotions = {}
        for item in filtered_history:
            for emotion, score in item["emotions"].items():
                if emotion in all_emotions:
                    all_emotions[emotion] += score
                else:
                    all_emotions[emotion] = score
        
        # Normalize emotions
        if all_emotions:
            for emotion in all_emotions:
                all_emotions[emotion] /= total
            
            # Get top 3 emotions
            top_emotions = sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
            top_emotions = {emotion: score for emotion, score in top_emotions}
        else:
            top_emotions = {}
        
        # Get top aspects
        all_aspects = {}
        for item in filtered_history:
            for aspect, data in item["aspects"].items():
                if aspect not in all_aspects:
                    all_aspects[aspect] = {"positive": 0, "negative": 0, "neutral": 0}
                
                sentiment = data.get("sentiment", "neutral")
                all_aspects[aspect][sentiment] += 1
        
        # Calculate dominant sentiment for each aspect
        aspect_sentiments = {}
        for aspect, counts in all_aspects.items():
            total_mentions = sum(counts.values())
            if counts["positive"] > counts["negative"] and counts["positive"] > counts["neutral"]:
                dominant = "positive"
            elif counts["negative"] > counts["positive"] and counts["negative"] > counts["neutral"]:
                dominant = "negative"
            else:
                dominant = "neutral"
            
            aspect_sentiments[aspect] = {
                "dominant_sentiment": dominant,
                "mentions": total_mentions,
                "distribution": {
                    "positive": counts["positive"] / total_mentions,
                    "negative": counts["negative"] / total_mentions,
                    "neutral": counts["neutral"] / total_mentions
                }
            }
        
        # Sort aspects by mentions
        top_aspects = dict(sorted(aspect_sentiments.items(), 
                          key=lambda x: x[1]["mentions"], reverse=True)[:5])
        
        # Create the trend result
        trend_result = {
            "period": time_period,
            "data_points": len(filtered_history),
            "average_sentiment_score": avg_sentiment,
            "sentiment_distribution": sentiment_distribution,
            "top_emotions": top_emotions,
            "top_aspects": top_aspects
        }
        
        return trend_result
    
    def _parse_sentiment_label(self, response: str) -> str:
        """Parse the sentiment label from the MOCK-LLM response."""
        response = response.lower().strip()
        
        if "positive" in response:
            return "positive"
        elif "negative" in response:
            return "negative"
        else:
            return "neutral"
    
    def _analyze_sentiment_fallback(self, text: str) -> Tuple[float, str, float]:
        """Fallback sentiment analysis method."""
        self.logger.info("Using fallback sentiment analysis")
        
        # Simple keyword-based sentiment analysis
        positive_words = [
            "good", "great", "excellent", "amazing", "awesome", "fantastic",
            "wonderful", "happy", "love", "like", "best", "perfect", "pleased",
            "recommend", "positive", "nice", "definitely", "satisfied", "joy"
        ]
        
        negative_words = [
            "bad", "terrible", "awful", "horrible", "worst", "hate", "dislike",
            "poor", "disappointed", "disappointing", "negative", "unfortunately",
            "not good", "problem", "issue", "broken", "useless", "waste", "sad"
        ]
        
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        # Calculate sentiment score
        total_count = len(words)
        if total_count == 0:
            return 0.0, "neutral", 0.5
        
        positive_ratio = positive_count / total_count
        negative_ratio = negative_count / total_count
        
        score = positive_ratio - negative_ratio
        
        # Determine label
        if score > 0.05:
            label = "positive"
            confidence = 0.5 + min(positive_ratio * 0.5, 0.3)
        elif score < -0.05:
            label = "negative"
            confidence = 0.5 + min(negative_ratio * 0.5, 0.3)
        else:
            label = "neutral"
            confidence = 0.5
        
        return score, label, confidence
    
    def _detect_emotions_fallback(self, text: str) -> Dict[str, float]:
        """Fallback emotion detection method."""
        # Simple keyword-based emotion detection
        emotion_keywords = {
            "joy": ["happy", "joy", "delighted", "pleased", "excited", "glad"],
            "sadness": ["sad", "unhappy", "depressed", "miserable", "heartbroken"],
            "anger": ["angry", "mad", "furious", "outraged", "annoyed", "frustrated"],
            "fear": ["afraid", "scared", "fearful", "terrified", "worried", "anxious"],
            "surprise": ["surprised", "amazed", "astonished", "shocked", "stunned"],
            "disgust": ["disgusted", "revolted", "repulsed", "appalled"]
        }
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Count emotion keywords
        emotions = {}
        for emotion, keywords in emotion_keywords.items():
            count = sum(1 for word in words if word in keywords)
            if count > 0:
                emotions[emotion] = min(count / len(words) * 5, 1.0)  # Normalize to 0-1
        
        # If no emotions detected, add neutral
        if not emotions:
            emotions["neutral"] = 0.7
        
        return emotions
    
    def _analyze_aspects_fallback(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Fallback aspect-based sentiment analysis."""
        # Common aspects to check for
        common_aspects = [
            "price", "quality", "service", "design", "performance", 
            "delivery", "customer service", "shipping", "value", "support",
            "features", "reliability", "durability", "usability", "speed"
        ]
        
        text_lower = text.lower()
        
        # Find mentioned aspects
        aspects = {}
        for aspect in common_aspects:
            if aspect in text_lower:
                # Find the sentence containing the aspect
                sentences = text_lower.split('.')
                for sentence in sentences:
                    if aspect in sentence:
                        # Determine sentiment for this aspect
                        score, sentiment, _ = self._analyze_sentiment_fallback(sentence)
                        aspects[aspect] = {
                            "sentiment": sentiment,
                            "score": score,
                            "context": sentence.strip()
                        }
                        break
        
        return aspects
    
    def _filter_by_time_period(self, history: List[Dict[str, Any]], period: str) -> List[Dict[str, Any]]:
        """Filter history by time period."""
        if period == "all" or not history:
            return history
        
        now = datetime.datetime.now()
        filtered = []
        
        for item in history:
            try:
                timestamp = datetime.datetime.strptime(item["timestamp"], "%Y-%m-%d %H:%M:%S")
                
                if period == "day" and (now - timestamp).days < 1:
                    filtered.append(item)
                elif period == "week" and (now - timestamp).days < 7:
                    filtered.append(item)
                elif period == "month" and (now - timestamp).days < 30:
                    filtered.append(item)
            except:
                # If timestamp parsing fails, include it
                filtered.append(item)
        
        return filtered
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get the sentiment analysis history."""
        return self.history
    
    def save_to_file(self, filepath: str) -> bool:
        """Save sentiment analysis history to a file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.history, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Error saving to file: {str(e)}")
            return False
    
    def load_from_file(self, filepath: str) -> bool:
        """Load sentiment analysis history from a file."""
        try:
            with open(filepath, 'r') as f:
                self.history = json.load(f)
            return True
        except Exception as e:
            self.logger.error(f"Error loading from file: {str(e)}")
            return False


def display_sentiment_result(result: Dict[str, Any]):
    """Display the sentiment analysis result in a user-friendly format."""
    print("\n" + "="*80)
    print("SENTIMENT ANALYSIS RESULT".center(80))
    print("="*80 + "\n")
    
    print(f"Text: \"{result['text'][:60]}...\"")
    print(f"Source: {result['source']} | Timestamp: {result['timestamp']}")
    
    sentiment = result["sentiment"]
    print(f"\nSENTIMENT: {sentiment['label'].upper()}")
    print(f"Score: {sentiment['score']:.2f} | Confidence: {sentiment['confidence']:.2f}")
    
    if result["emotions"]:
        print("\nEMOTIONS:")
        for emotion, score in result["emotions"].items():
            print(f"  {emotion.capitalize()}: {score:.2f}")
    
    if result["aspects"]:
        print("\nASPECTS:")
        for aspect, data in result["aspects"].items():
            sentiment_str = data["sentiment"].upper()
            print(f"  {aspect.capitalize()}: {sentiment_str} ({data['score']:.2f})")
            if "context" in data:
                print(f"    Context: \"{data['context']}\"")
    
    print("\n" + "="*80)


def display_trend_results(trend_result: Dict[str, Any]):
    """Display trend analysis results in a user-friendly format."""
    print("\n" + "="*80)
    print("SENTIMENT TREND ANALYSIS".center(80))
    print("="*80 + "\n")
    
    print(f"Period: {trend_result['period'].upper()} | Data Points: {trend_result['data_points']}")
    print(f"Average Sentiment Score: {trend_result['average_sentiment_score']:.2f}")
    
    # Sentiment distribution
    print("\nSENTIMENT DISTRIBUTION:")
    dist = trend_result["sentiment_distribution"]
    print(f"  Positive: {dist['positive']*100:.1f}%")
    print(f"  Neutral:  {dist['neutral']*100:.1f}%")
    print(f"  Negative: {dist['negative']*100:.1f}%")
    
    # Top emotions
    if trend_result["top_emotions"]:
        print("\nTOP EMOTIONS:")
        for emotion, score in trend_result["top_emotions"].items():
            print(f"  {emotion.capitalize()}: {score:.2f}")
    
    # Top aspects
    if trend_result["top_aspects"]:
        print("\nTOP ASPECTS:")
        for aspect, data in trend_result["top_aspects"].items():
            dominant = data["dominant_sentiment"].upper()
            mentions = data["mentions"]
            print(f"  {aspect.capitalize()}: {dominant} ({mentions} mentions)")
            dist = data["distribution"]
            print(f"    Positive: {dist['positive']*100:.1f}% | Neutral: {dist['neutral']*100:.1f}% | Negative: {dist['negative']*100:.1f}%")
    
    print("\n" + "="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Sentiment Analysis Dashboard")
    parser.add_argument("--text", type=str, help="Single text to analyze")
    parser.add_argument("--file", type=str, help="JSON file with texts to analyze in batch")
    parser.add_argument("--demo", action="store_true", help="Run with demo data")
    parser.add_argument("--trends", action="store_true", help="Show sentiment trends")
    parser.add_argument("--period", type=str, choices=["all", "day", "week", "month"], 
                        default="all", help="Time period for trend analysis")
    parser.add_argument("--load", type=str, help="Load sentiment history from file")
    parser.add_argument("--save", type=str, help="Save sentiment history to file")
    parser.add_argument("--source", type=str, default="user_input", help="Source of the text")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Load history if specified
    if args.load:
        if analyzer.load_from_file(args.load):
            print(f"Loaded sentiment history from {args.load}")
    
    # Analyze text if specified
    if args.text:
        result = analyzer.analyze_sentiment(args.text, args.source)
        display_sentiment_result(result)
    
    # Show trends if specified
    if args.trends:
        trend_result = analyzer.get_sentiment_trends(args.period)
        if "error" in trend_result:
            print(f"Error: {trend_result['error']}")
        else:
            display_trend_results(trend_result)
    
    # Save history if specified
    if args.save:
        if analyzer.save_to_file(args.save):
            print(f"Saved sentiment history to {args.save}")
    
    # If no specific action, analyze sample texts
    if not args.text and not args.trends and not args.load and not args.save:
        sample_texts = [
            {
                "text": "I absolutely love this product! It works great and the customer service was excellent.",
                "source": "product_review",
                "timestamp": "2025-04-26 10:30:00"
            },
            {
                "text": "This service was terrible. I waited for hours and no one ever responded to my request.",
                "source": "customer_feedback",
                "timestamp": "2025-04-26 11:15:00"
            },
            {
                "text": "The software is okay but could use some improvements in the user interface.",
                "source": "app_review",
                "timestamp": "2025-04-26 11:45:00"
            },
            {
                "text": "The conference was informative but the venue was too small for all attendees.",
                "source": "event_feedback",
                "timestamp": "2025-04-25 16:20:00"
            },
            {
                "text": "Excited to announce our new product launch! Can't wait to hear your feedback.",
                "source": "social_media",
                "timestamp": "2025-04-25 09:10:00"
            }
        ]
        
        print("Analyzing sample texts...\n")
        
        results = analyzer.analyze_batch(sample_texts)
        for result in results:
            display_sentiment_result(result)
        
        trend_result = analyzer.get_sentiment_trends()
        display_trend_results(trend_result)


if __name__ == "__main__":
    main()
