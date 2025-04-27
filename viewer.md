# AGI Toolkit: Building Real-World Applications with ASI Components

## Introduction

The AGI Toolkit provides a powerful framework for building real-world applications using advanced ASI (Artificial Super Intelligence) and MOCK-LLM capabilities. This guide explains how to properly set up, use, and build applications with this toolkit.

## System Requirements

- Python 3.8+
- Required packages (install via `pip install -r requirements.txt`)
- Environment variable: `AGI_TOOLKIT_KEY='AGI-Toolkit-Secure-2025'`

## Component Architecture

The AGI Toolkit consists of the following key components:

1. **ASI Engine**: Provides pattern discovery, insight generation, and timeline prediction
2. **MOCK-LLM**: Offers text generation, embeddings, and NLP capabilities
3. **Unified Memory**: Non-Euclidean memory system for data storage and retrieval
4. **Unified API**: Clean interface for accessing all capabilities

## Getting Started

### 1. Setting Up Your Environment

Before using the AGI Toolkit, you need to set up your environment:

```bash
# Clone the repository
git clone https://github.com/GlobalSushrut/AGI-GitHub-Release.git
cd AGI-GitHub-Release

# Install dependencies
pip install -r requirements.txt

# Set the required environment variable
export AGI_TOOLKIT_KEY='AGI-Toolkit-Secure-2025'

# For licensed features, set license variables
export MRZKELP_LICENSE_KEY="540e4a27d374b9cd58add850949aeed4595ee582570252db538bdb3776d7aa98cd7614c533640914d1df5e03462ff9247b3ff385bff7ebd5b04de66b09c1c231"
export MRZKELP_CLIENT_ID="demo@example.com"
export MRZKELP_SECRET="AGIToolkitMaster"

# To use real ASI components instead of simulation
export USE_REAL_ASI=true
```

### 2. Initialize Core Components

To ensure the real ASI and MOCK-LLM components load properly, always initialize them using the included script:

```bash
# Run the component loader script before your application
python fix_component_loading.py
```

This properly decrypts and initializes the core ASI and MOCK-LLM components needed for real functionality.

## Running Real-World Applications

The AGI-GitHub-Release package includes several ready-to-use real-world applications that leverage ASI capabilities. These applications are located in the `real_world_apps` directory.

### Using the Unified Launcher

The easiest way to run any application with real ASI components is to use the unified launcher script:

```bash
python3 run_real_asi_app.py [app_name] [app_arguments]
```

This launcher automatically:
1. Sets the `USE_REAL_ASI=true` environment variable
2. Initializes ASI components via the `fix_component_loading.py` script
3. Runs the specified application with the given arguments

### Available Applications and Parameters

#### 1. Document Assistant

Processes and analyzes documents to extract insights, entities, and action items.

```bash
python3 run_real_asi_app.py document_assistant --text "Your document content here"
# OR
python3 run_real_asi_app.py document_assistant --file /path/to/document.txt
```

Parameters:
- `--text`: Text content to analyze
- `--file`: Path to a text file to analyze

#### 2. Virtual Assistant

Interactive virtual assistant that can handle conversations, remember information, and manage tasks.

```bash
python3 run_real_asi_app.py virtual_assistant --demo
# OR
python3 run_real_asi_app.py virtual_assistant --interactive --name "Aria"
```

Parameters:
- `--demo`: Run a pre-configured demonstration session
- `--interactive`: Run in interactive mode
- `--name`: Set the assistant's name (default: Aria)

#### 3. Sentiment Dashboard

Analyzes the sentiment, emotions, and aspect-based opinions in text.

```bash
python3 run_real_asi_app.py sentiment_dashboard --text "I really love the new features in the latest update!"
# OR
python3 run_real_asi_app.py sentiment_dashboard --file /path/to/feedback.txt
```

Parameters:
- `--text`: Text content to analyze for sentiment
- `--file`: Path to a text file to analyze
- `--batch`: Path to a file with multiple entries (one per line) for batch analysis

#### 4. Translation Service

Translates text between different languages with domain-specific capabilities.

```bash
python3 run_real_asi_app.py translation_service --text "Hello, how are you?" --target Spanish
```

Parameters:
- `--text`: Text to translate
- `--target`: Target language code or name (required)
- `--source`: Source language code (auto-detect if not provided)
- `--domain`: Domain for specialized translation (default: "general")
- `--list_languages`: List supported languages and domains

#### 5. Content Summarizer

Generates concise summaries and extracts key points from longer content.

```bash
python3 run_real_asi_app.py content_summarizer --text "Your long text here..."
# OR
python3 run_real_asi_app.py content_summarizer --file /path/to/article.txt --length medium
```

Parameters:
- `--text`: Text content to summarize
- `--file`: Path to a text file to summarize
- `--length`: Summary length ("short", "medium", or "long", default: "medium")

#### 6. Learning Platform

Educational platform that generates learning materials, quizzes, and provides explanations.

```bash
python3 run_real_asi_app.py learning_platform --topic "Quantum Physics" --level intermediate
```

Parameters:
- `--topic`: Subject topic to generate learning materials for
- `--level`: Difficulty level ("beginner", "intermediate", or "advanced")
- `--generate_quiz`: Generate a quiz on the topic

#### 7. Recommendation Engine

Generates personalized recommendations based on user preferences and history.

```bash
python3 run_real_asi_app.py recommendation_engine --user_id "user123" --domain "movies"
```

Parameters:
- `--user_id`: User identifier for personalized recommendations
- `--domain`: Domain for recommendations ("movies", "books", "products", etc.)
- `--count`: Number of recommendations to generate (default: 5)

#### 8. Banking Application

Demonstrates financial applications with fraud detection and risk analysis.

```bash
python3 run_real_asi_app.py banking --analyze_transaction "transaction_data.json"
# OR
python3 run_real_asi_app.py banking --risk_profile "customer123"
```

Parameters:
- `--analyze_transaction`: Path to transaction data file for fraud analysis
- `--risk_profile`: Customer ID to generate a risk profile for

### ASI Interface Implementation

All applications use our centralized ASI helper module (`real_world_apps/asi_helper.py`) to interact with the ASI Engine components. This module provides:

1. **Initialization Functions:**
   ```python
   # Initialize real ASI components
   from asi_helper import initialize_asi_components
   initialize_asi_components()
   ```

2. **Processing Functions:**
   ```python
   from asi_helper import process_with_asi
   
   # General ASI processing
   result = process_with_asi({
       "task": "task_name",
       "content": "content to process"
   })
   ```

3. **Specialized Functions:**
   ```python
   from asi_helper import generate_summary, extract_key_points, analyze_sentiment, translate_text
   
   # Generate a summary
   summary = generate_summary("Long text here...", "medium")
   
   # Extract key points
   points = extract_key_points("Content to analyze...")
   
   # Analyze sentiment
   sentiment = analyze_sentiment("Text for sentiment analysis...")
   
   # Translate text
   translated = translate_text("Hello world", "en", "es")
   ```

## Building Your First Application

### Basic Application Structure

A typical AGI Toolkit application follows this structure:

```python
#!/usr/bin/env python3

import os
import logging
from agi_toolkit import AGIAPI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MyApplication")

class MyApplication:
    def __init__(self):
        logger.info("Initializing application")
        
        # Initialize the API
        self.api = AGIAPI()
        
        # Check component availability
        logger.info(f"ASI available: {self.api.has_asi}")
        logger.info(f"MOCK-LLM available: {self.api.has_mock_llm}")
    
    def process_data(self, data):
        """Process data using ASI capabilities"""
        if self.api.has_asi:
            result = self.api.process_with_asi(data)
            return result
        else:
            logger.warning("ASI not available, using fallback")
            # Implement fallback logic
            return {"result": "Fallback processing"}
    
    def generate_content(self, prompt):
        """Generate content using MOCK-LLM"""
        if self.api.has_mock_llm:
            text = self.api.generate_text(prompt)
            return text
        else:
            logger.warning("MOCK-LLM not available, using fallback")
            return "Fallback generated text"
    
    def run(self):
        """Main application logic"""
        # Your application logic here
        pass

if __name__ == "__main__":
    app = MyApplication()
    app.run()
```

## Key API Capabilities

### 1. ASI Capabilities

The ASI component provides three primary capabilities:

#### Pattern Discovery

Discover patterns in complex data domains:

```python
# Format your data as normalized properties (0-1 values)
domain_data = {
    "property1": 0.75,
    "property2": 0.82,
    "property3": 0.45,
}

# Process with ASI
patterns = api.process_with_asi({
    "query": "discover patterns",
    **domain_data
})

# Access discovered patterns
for pattern in patterns["result"]["patterns"]:
    print(f"Pattern: {pattern['concept']}")
    print(f"Description: {pattern['description']}")
    print(f"Significance: {pattern['significance']}")
```

#### Insight Generation

Generate cross-domain insights:

```python
# Provide concepts for insight generation
concepts = ["concept1", "concept2", "concept3"]
insight = api.process_with_asi(" ".join(concepts))
print(f"Insight: {insight['result']['text']}")
print(f"Confidence: {insight['result']['confidence']}")
```

#### Timeline Prediction

Predict potential future timelines:

```python
# Define scenario for timeline prediction
scenario = {
    "query": "predict timeline",
    "domain": "your_domain",
    "name": "scenario_name",
    "complexity": 0.7,
    "factors": {
        "factor1": 0.8,
        "factor2": 0.6
    }
}

timelines = api.process_with_asi(scenario)

# Access timeline predictions
for timeline in timelines["result"]["timelines"]:
    print(f"Timeline: {timeline['type']}")
    print(f"Probability: {timeline['probability']}")
    
    for event in timeline["events"]:
        print(f"  Event: {event['description']}")
```

### 2. MOCK-LLM Capabilities

#### Text Generation

Generate text based on prompts:

```python
text = api.generate_text("Write a poem about artificial intelligence.")
print(text)
```

#### Embeddings

Generate embeddings for text:

```python
embedding = api.get_embedding("This is a sample text")
similar_texts = api.find_similar_texts(embedding, ["text1", "text2", "text3"])
```

### 3. Memory System

Store and retrieve data from the unified memory system:

```python
# Store data
api.store_data("user_preferences", {"theme": "dark", "language": "en"})

# Retrieve data
prefs, metadata = api.retrieve_data("user_preferences")
print(prefs)
```

## Best Practices

1. **Always Check Component Availability**: Before using ASI or MOCK-LLM features, check if they're available.

2. **Implement Fallbacks**: Provide fallback mechanisms for when components aren't available.

3. **Normalize Data**: When working with ASI pattern discovery, normalize numeric values to 0-1 range.

4. **Error Handling**: Implement proper error handling for all API calls.

5. **Memory Management**: Clean up memory when no longer needed to avoid accumulation.

## Common Patterns

### Processing Pipeline

```python
def process_user_input(text):
    # 1. Store input in memory
    api.store_data("last_input", text)
    
    # 2. Generate embedding for search
    embedding = api.get_embedding(text)
    
    # 3. Find similar previous inputs
    similar_items = api.find_similar_texts(embedding, stored_items)
    
    # 4. Process with ASI if available
    if api.has_asi:
        result = api.process_with_asi({"query": text})
    else:
        # Fallback processing
        result = {"result": "Processed without ASI"}
    
    # 5. Generate response with MOCK-LLM
    if api.has_mock_llm:
        response = api.generate_text(f"Input: {text}\nContext: {result}\nResponse:")
    else:
        # Fallback generation
        response = f"Processed result: {result['result']}"
    
    return response
```

### Handling Different Data Types

```python
def normalize_for_asi(data):
    """Convert various data types to ASI-compatible format."""
    properties = {}
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (int, float)):
                # Normalize to 0-1 range
                properties[key] = min(max(float(value) / 100.0, 0.0), 1.0)
            elif isinstance(value, str):
                properties[key] = 0.5  # Default value for strings
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, (int, float)):
                properties[f"item_{i}"] = min(max(float(item) / 100.0, 0.0), 1.0)
    
    return properties
```

## Troubleshooting

If you encounter issues with component availability:

1. **Check Environment Variables**: Ensure `AGI_TOOLKIT_KEY` is set correctly.
2. **Run Component Loader**: Use `python fix_component_loading.py` before your application.
3. **Check License**: For corporate features, ensure license variables are set.
4. **Check Paths**: If you moved the repository, update paths in `core_loader.py`.

## Real-World Application Examples

The repository includes several real-world application examples in the `real_world_apps` directory:

1. **Banking App**: Transaction processing and fraud detection
2. **Military Logistics**: Supply chain optimization with ASI
3. **Content Summarizer**: Text summarization using MOCK-LLM
4. **Document Assistant**: Document analysis and question answering
5. **Learning Platform**: Adaptive learning system
6. **Sentiment Dashboard**: Sentiment analysis and visualization
7. **Translation Service**: Text translation with context awareness
8. **Virtual Assistant**: Task-based virtual assistant with memory

Study these examples to understand effective patterns for building your own applications.

## Advanced Topics

### Custom ASI Configuration

Customize ASI behavior:

```python
# Initialize the ASI engine with custom configuration
from unreal_asi.asi_public_api import initialize_asi, create_asi_instance

initialize_asi()
asi = create_asi_instance(
    name="CustomASI",
    config={
        "response_detail_level": "high",
        "creativity_factor": 0.7,
        "analysis_depth": "deep"
    }
)
```

### Extending the Toolkit

You can extend the toolkit by:

1. Creating domain-specific pre-processors and post-processors
2. Building specialized visualization tools for outputs
3. Implementing caching and optimization layers
4. Creating domain adapters for specific data formats

## Conclusion

The AGI Toolkit provides powerful capabilities for building real-world applications that leverage advanced AI capabilities. By following the best practices and patterns in this guide, you can create sophisticated applications that use pattern discovery, insight generation, timeline prediction, and text generation.

For further assistance, refer to the documentation in the `docs` directory.
