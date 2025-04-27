#!/usr/bin/env python3
"""
ASI Engine Helper Module
-----------------------

This module provides helper functions for real-world applications to properly
use the real ASI Engine components. This centralizes the integration logic
so all applications can benefit from improvements.
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional

# Add parent directory to import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create logger
logger = logging.getLogger("ASI-Helper")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def initialize_asi_components():
    """
    Initialize ASI Engine components for all applications.
    This should be called before using any ASI functionality.
    
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    try:
        # Try to import and run the fix_component_loading script
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Import the component loader
        try:
            from fix_component_loading import initialize_real_components
            success = initialize_real_components()
            if success:
                logger.info("Successfully initialized real ASI components")
            else:
                logger.warning("Failed to initialize real ASI components")
            
            # Set environment variable to ensure ASI interface uses real components
            os.environ['USE_REAL_ASI'] = 'true'
            
            return success
        except ImportError:
            logger.error("Could not import fix_component_loading.py")
            return False
    except Exception as e:
        logger.error(f"Error initializing ASI components: {str(e)}")
        return False

def process_with_asi(api, input_data, task_type=None):
    """
    Process data with ASI, handling various task types appropriately.
    
    Args:
        api: The AGIAPI instance to use
        input_data: The input data to process
        task_type: Optional task type for special handling (e.g., 'summarize', 'analyze')
        
    Returns:
        dict: Processed result
    """
    # Ensure proper task formatting if task_type is provided
    if task_type and isinstance(input_data, dict) and 'task' not in input_data:
        if task_type == 'summarize':
            input_data['task'] = 'generate_summary'
        elif task_type == 'analyze':
            input_data['task'] = 'analyze_content'
        elif task_type == 'key_points':
            input_data['task'] = 'extract_key_points'
        # Add more task types as needed
    
    # Log the request
    logger.info(f"Processing with ASI: {str(input_data)[:100]}...")
    
    # Process with ASI
    try:
        result = api.process_with_asi(input_data)
        
        # Log success
        if result.get('success'):
            logger.info("ASI processing successful")
        else:
            logger.warning(f"ASI processing unsuccessful: {result.get('error', 'Unknown error')}")
        
        return result
    except Exception as e:
        logger.error(f"Error in ASI processing: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def extract_key_points(api, text, max_points=5):
    """
    Extract key points from text using ASI.
    
    Args:
        api: The AGIAPI instance to use
        text: The text to extract key points from
        max_points: Maximum number of key points to extract
        
    Returns:
        list: Extracted key points
    """
    # Limit text size
    text_sample = text[:5000]
    
    # Process with ASI
    result = process_with_asi(api, {
        'task': 'extract_key_points',
        'content': text_sample,
        'max_points': max_points
    }, task_type='key_points')
    
    # Extract key points
    key_points = []
    if result.get('success', False) and 'result' in result:
        points_data = result['result']
        
        # Different output formats based on which ASI function was used
        if isinstance(points_data, dict):
            if 'points' in points_data:
                key_points = points_data['points']
            elif 'patterns' in points_data:
                # Extract patterns as key points
                patterns = points_data.get('patterns', [])
                for pattern in patterns:
                    if isinstance(pattern, dict):
                        # Try different fields that might contain useful text
                        if 'description' in pattern:
                            key_points.append(pattern['description'])
                        elif 'concept' in pattern:
                            key_points.append(pattern['concept'])
            elif 'text' in points_data:
                # Split insight into sentences
                insight_text = points_data.get('text', '')
                sentences = insight_text.split('. ')
                for sentence in sentences:
                    if sentence and len(sentence) > 10:
                        key_points.append(sentence)
            elif 'insight' in points_data:
                # Split insight into sentences
                insight_text = points_data.get('insight', '')
                sentences = insight_text.split('. ')
                for sentence in sentences:
                    if sentence and len(sentence) > 10:
                        key_points.append(sentence)
        elif isinstance(points_data, list):
            key_points = points_data
            
    # Ensure we don't return more than max_points
    return key_points[:max_points]

def generate_summary(api, text, length='medium'):
    """
    Generate a summary of text using ASI.
    
    Args:
        api: The AGIAPI instance to use
        text: The text to summarize
        length: Summary length - 'short', 'medium', or 'long'
        
    Returns:
        str: Generated summary
    """
    # Limit text size
    text_sample = text[:5000]
    
    # Process with ASI
    result = process_with_asi(api, {
        'task': 'generate_summary',
        'content': text_sample,
        'length': length
    }, task_type='summarize')
    
    # Extract summary
    if result.get('success', False) and 'result' in result:
        points_data = result['result']
        
        # Try to extract a summary from the ASI output
        if isinstance(points_data, dict):
            if 'summary' in points_data:
                return points_data['summary']
            elif 'text' in points_data:
                return points_data['text']
            elif 'insight' in points_data:
                return points_data['insight']
            elif 'points' in points_data:
                # Join points to form a summary
                return '. '.join(points_data['points'])
            else:
                # Try to find any string values in the dictionary
                str_values = [v for v in points_data.values() if isinstance(v, str) and len(v) > 30]
                if str_values:
                    return str_values[0]  # Use the first substantial string
        elif isinstance(points_data, str) and len(points_data) > 30:
            return points_data
            
    # Fallback: extract first few sentences based on desired length
    sentences = text.split('.')
    if length == 'short':
        num_sentences = min(2, len(sentences))
    elif length == 'medium':
        num_sentences = min(4, len(sentences))
    else:  # long
        num_sentences = min(6, len(sentences))
        
    return '. '.join([s.strip() for s in sentences[:num_sentences] if s.strip()]) + '.'

def analyze_sentiment(api, text):
    """
    Analyze sentiment of text using ASI.
    
    Args:
        api: The AGIAPI instance to use
        text: The text to analyze
        
    Returns:
        dict: Sentiment analysis result with score and label
    """
    # Limit text size
    text_sample = text[:1000]
    
    # Process with ASI
    result = process_with_asi(api, {
        'task': 'analyze_sentiment',
        'content': text_sample
    }, task_type='analyze')
    
    # Default sentiment analysis result
    sentiment = {
        'score': 0.0,
        'label': 'neutral'
    }
    
    # Extract sentiment
    if result.get('success', False) and 'result' in result:
        points_data = result['result']
        
        if isinstance(points_data, dict):
            # Look for sentiment-related fields
            if 'sentiment' in points_data:
                sentiment = points_data['sentiment']
            elif 'score' in points_data:
                sentiment['score'] = points_data['score']
                # Determine label based on score
                if points_data['score'] > 0.3:
                    sentiment['label'] = 'positive'
                elif points_data['score'] < -0.3:
                    sentiment['label'] = 'negative'
    
    return sentiment

def translate_text(api, text, source_lang, target_lang):
    """
    Translate text using ASI.
    
    Args:
        api: The AGIAPI instance to use
        text: The text to translate
        source_lang: Source language code
        target_lang: Target language code
        
    Returns:
        str: Translated text
    """
    # Limit text size
    text_sample = text[:1000]
    
    # Process with ASI
    result = process_with_asi(api, {
        'task': 'translate_text',
        'content': text_sample,
        'source_lang': source_lang,
        'target_lang': target_lang
    })
    
    # Extract translation
    if result.get('success', False) and 'result' in result:
        points_data = result['result']
        
        if isinstance(points_data, dict):
            if 'translation' in points_data:
                return points_data['translation']
            elif 'text' in points_data:
                return points_data['text']
        elif isinstance(points_data, str):
            return points_data
            
    # Fallback: return original text with note
    return f"[Translation from {source_lang} to {target_lang}]: {text_sample}"

def analyze_document(api, document_text):
    """
    Analyze a document using ASI.
    
    Args:
        api: The AGIAPI instance to use
        document_text: The document text to analyze
        
    Returns:
        dict: Document analysis result
    """
    # Limit text size
    text_sample = document_text[:5000]
    
    # Process with ASI
    result = process_with_asi(api, {
        'task': 'analyze_document',
        'content': text_sample
    }, task_type='analyze')
    
    # Default document analysis result
    analysis = {
        'summary': generate_summary(api, text_sample, 'short'),
        'key_points': extract_key_points(api, text_sample),
        'topics': [],
        'sentiment': analyze_sentiment(api, text_sample)
    }
    
    # Extract analysis from ASI result if available
    if result.get('success', False) and 'result' in result:
        points_data = result['result']
        
        if isinstance(points_data, dict):
            # Update default analysis with any available fields
            for key in analysis.keys():
                if key in points_data:
                    analysis[key] = points_data[key]
    
    return analysis
