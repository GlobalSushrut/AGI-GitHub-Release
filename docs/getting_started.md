# Getting Started with AGI Toolkit

This guide will help you get started with the AGI Toolkit for integrating ASI and MOCK-LLM capabilities into your applications without modifying the core implementation code.

## Installation

You can install the AGI Toolkit in several ways:

### From PyPI (Recommended)

```bash
pip install agi-toolkit
```

### From Source

```bash
git clone https://github.com/GlobalSushrut/AGI-GitHub-Release.git
cd AGI-GitHub-Release/deployable
pip install -e .
```

## Quick Start

Here's a simple example of using the AGI Toolkit:

```python
from agi_toolkit import AGIAPI

# Initialize the API
api = AGIAPI()

# Check component availability
print(f"ASI available: {api.has_asi}")
print(f"MOCK-LLM available: {api.has_mock_llm}")

# Generate text with MOCK-LLM
if api.has_mock_llm:
    response = api.generate_text("Explain quantum computing in simple terms")
    print(response)

# Process data with ASI
if api.has_asi:
    result = api.process_with_asi({"query": "Analyze market trends for AI"})
    print(result)

# Store and retrieve data from unified memory
api.store_data("user_preferences", {"theme": "dark", "language": "en"})
prefs, metadata = api.retrieve_data("user_preferences")
print(prefs)
```

## Core Components

The AGI Toolkit consists of several core components:

### AGIAPI

The main entry point for all functionality. It provides high-level methods for:

- Text generation
- Data processing
- Memory management
- Model training and evaluation

### ASIInterface

Interface for ASI-specific functionality, including:

- Advanced data processing
- Complex reasoning
- ASI model training

### MOCKLLMInterface

Interface for MOCK-LLM-specific functionality, including:

- Text generation
- Embeddings
- Classification
- Fine-tuning

### UnifiedMemory

Interface to the unified memory system, providing:

- Data storage and retrieval
- Memory statistics
- Key management

## Common Use Cases

### Text Generation

```python
from agi_toolkit import AGIAPI

api = AGIAPI()
response = api.generate_text("Write a short story about a robot learning to paint")
print(response)
```

### Data Analysis

```python
from agi_toolkit import AGIAPI

api = AGIAPI()
data = {"text": "The stock market showed significant volatility last quarter",
        "context": "financial analysis"}
result = api.process_with_asi(data)
print(result)
```

### Memory Management

```python
from agi_toolkit import AGIAPI

api = AGIAPI()

# Store structured data
api.store_data("customer", {"id": 12345, "name": "John Doe", "status": "active"})

# Retrieve data
customer, metadata = api.retrieve_data("customer")
print(f"Customer: {customer['name']}, Status: {customer['status']}")
```

### Model Training

```python
from agi_toolkit import AGIAPI

api = AGIAPI()
result = api.train_model(
    dataset_path="/path/to/dataset.json",
    model_type="mock_llm",
    num_epochs=3,
    learning_rate=5e-5
)
print(result)
```

## Working with Components Separately

If you need more direct control, you can work with individual components:

```python
from agi_toolkit import ASIInterface, MOCKLLMInterface, UnifiedMemory

# Work with ASI directly
asi = ASIInterface()
if asi.is_available:
    result = asi.reason("What are the implications of quantum computing for cryptography?")
    print(result)

# Work with MOCK-LLM directly
mock_llm = MOCKLLMInterface()
if mock_llm.is_available:
    embedding = mock_llm.get_embedding("This is a test sentence")
    print(f"Embedding shape: {embedding.shape}")

# Work with memory directly
memory = UnifiedMemory()
memory.store("config", {"debug": True, "verbose": False})
stats = memory.get_stats()
print(f"Memory stats: {stats}")
```

## Error Handling

The AGI Toolkit is designed to handle component unavailability gracefully:

```python
from agi_toolkit import AGIAPI

api = AGIAPI()

try:
    # This will work even if MOCK-LLM is not available
    response = api.generate_text("Hello world")
    print(response)  # Will return an error message if MOCK-LLM is not available
    
    # Check component availability before operations
    if api.has_asi:
        result = api.process_with_asi({"query": "Complex analysis"})
    else:
        # Fallback to alternative approach
        result = {"result": "Basic analysis only", "reason": "ASI not available"}
        
    print(result)
except Exception as e:
    print(f"Error: {str(e)}")
```

## Next Steps

Check out the example applications in the `examples` directory for more complex use cases:

- Text Analysis Application
- Simple Chatbot
- Data Processing Tool

For more detailed information, see the [API Reference](api_reference.md) or explore the [Example Applications](../examples/README.md).
