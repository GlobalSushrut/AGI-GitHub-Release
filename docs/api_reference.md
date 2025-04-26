# AGI Toolkit API Reference

This document provides detailed information about the AGI Toolkit API, including all public classes, methods, and their parameters.

## AGIAPI

The main entry point for the AGI Toolkit.

### Initialization

```python
from agi_toolkit import AGIAPI

api = AGIAPI(
    memory_dimensions=11,
    log_level="INFO",
    fallback_enabled=True
)
```

#### Parameters:

- `memory_dimensions` (int, optional): Dimensions for the non-euclidean memory. Default: 11
- `log_level` (str, optional): Logging level. Default: "INFO"
- `fallback_enabled` (bool, optional): Whether to enable fallback responses. Default: True

### Properties

- `has_asi` (bool): Whether ASI components are available
- `has_mock_llm` (bool): Whether MOCK-LLM components are available
- `version` (str): Version of the AGI Toolkit

### Methods

#### `generate_text(prompt, max_length=100)`

Generate text using the MOCK-LLM model.

```python
response = api.generate_text("Explain quantum computing")
```

**Parameters:**
- `prompt` (str): Text prompt for generation
- `max_length` (int, optional): Maximum length of generated text. Default: 100

**Returns:**
- `str`: Generated text or error message if MOCK-LLM is not available

#### `process_with_asi(data)`

Process data using the ASI system for advanced reasoning and analysis.

```python
result = api.process_with_asi({"query": "Analyze market trends"})
```

**Parameters:**
- `data` (dict): Data to process

**Returns:**
- `dict`: Processing results or error message if ASI is not available

#### `store_data(key, data, metadata=None)`

Store data in the unified memory system.

```python
api.store_data("user_preferences", {"theme": "dark"}, {"timestamp": "2023-01-01"})
```

**Parameters:**
- `key` (str): Unique identifier for the data
- `data` (any): Data to store
- `metadata` (dict, optional): Additional metadata for the stored data. Default: None

**Returns:**
- `bool`: Success status

#### `retrieve_data(key)`

Retrieve data from the unified memory system.

```python
data, metadata = api.retrieve_data("user_preferences")
```

**Parameters:**
- `key` (str): Unique identifier for the data

**Returns:**
- `tuple`: (data, metadata) or (None, None) if key not found

#### `train_model(dataset_path, model_type="mock_llm", **kwargs)`

Train a model using the specified training dataset.

```python
result = api.train_model(
    dataset_path="/path/to/data.json",
    model_type="mock_llm",
    num_epochs=3,
    learning_rate=5e-5
)
```

**Parameters:**
- `dataset_path` (str): Path to the training dataset
- `model_type` (str, optional): Type of model to train ("mock_llm" or "asi"). Default: "mock_llm"
- `**kwargs`: Additional model-specific training parameters

**Returns:**
- `dict`: Training results

#### `evaluate_model(test_data_path, model_type="mock_llm", **kwargs)`

Evaluate a model using the specified test dataset.

```python
metrics = api.evaluate_model(
    test_data_path="/path/to/test_data.json",
    model_type="mock_llm"
)
```

**Parameters:**
- `test_data_path` (str): Path to the test dataset
- `model_type` (str, optional): Type of model to evaluate ("mock_llm" or "asi"). Default: "mock_llm"
- `**kwargs`: Additional model-specific evaluation parameters

**Returns:**
- `dict`: Evaluation metrics

#### `get_embedding(text)`

Get the embedding for a text string using MOCK-LLM.

```python
embedding = api.get_embedding("Hello world")
```

**Parameters:**
- `text` (str): Input text

**Returns:**
- `torch.Tensor`: Embedding tensor or None if MOCK-LLM is not available

#### `get_status()`

Get the status of all components.

```python
status = api.get_status()
```

**Returns:**
- `dict`: Status information for all components

## ASIInterface

Interface to ASI components.

### Initialization

```python
from agi_toolkit import ASIInterface

asi = ASIInterface()
```

### Properties

- `is_available` (bool): Whether ASI components are available
- `using_real_asi` (bool): Whether real ASI components are being used

### Methods

#### `reason(query)`

Perform reasoning using ASI.

```python
result = asi.reason("What are the implications of quantum computing for cryptography?")
```

**Parameters:**
- `query` (str): The query to reason about

**Returns:**
- `dict`: Reasoning results

#### `process_data(data)`

Process complex data using ASI.

```python
result = asi.process_data({
    "text": "The stock market showed significant volatility",
    "context": "financial analysis"
})
```

**Parameters:**
- `data` (dict): Data to process

**Returns:**
- `dict`: Processing results

#### `train(dataset_path, **kwargs)`

Train the ASI model.

```python
result = asi.train(
    dataset_path="/path/to/dataset.json",
    num_epochs=5,
    batch_size=32
)
```

**Parameters:**
- `dataset_path` (str): Path to the training dataset
- `**kwargs`: Additional training parameters

**Returns:**
- `dict`: Training results

#### `evaluate(test_data_path, **kwargs)`

Evaluate the ASI model.

```python
metrics = asi.evaluate(
    test_data_path="/path/to/test_data.json"
)
```

**Parameters:**
- `test_data_path` (str): Path to the test dataset
- `**kwargs`: Additional evaluation parameters

**Returns:**
- `dict`: Evaluation metrics

#### `get_status()`

Get the status of ASI components.

```python
status = asi.get_status()
```

**Returns:**
- `dict`: Status information for ASI components

## MOCKLLMInterface

Interface to MOCK-LLM components.

### Initialization

```python
from agi_toolkit import MOCKLLMInterface

mock_llm = MOCKLLMInterface()
```

### Properties

- `is_available` (bool): Whether MOCK-LLM components are available
- `using_real_mock_llm` (bool): Whether real MOCK-LLM components are being used

### Methods

#### `generate_text(prompt, max_length=100)`

Generate text using the MOCK-LLM model.

```python
response = mock_llm.generate_text("Explain quantum computing", max_length=200)
```

**Parameters:**
- `prompt` (str): Input text prompt
- `max_length` (int, optional): Maximum length of generated text. Default: 100

**Returns:**
- `str`: Generated text

#### `get_embedding(text)`

Get the embedding for a text string.

```python
embedding = mock_llm.get_embedding("Hello world")
```

**Parameters:**
- `text` (str): Input text

**Returns:**
- `torch.Tensor`: Embedding tensor

#### `train_model(dataset_path, num_epochs=3, learning_rate=5e-5, batch_size=16)`

Train the MOCK-LLM model on a custom dataset.

```python
result = mock_llm.train_model(
    dataset_path="/path/to/dataset.json",
    num_epochs=5,
    learning_rate=1e-5,
    batch_size=32
)
```

**Parameters:**
- `dataset_path` (str): Path to the training dataset
- `num_epochs` (int, optional): Number of training epochs. Default: 3
- `learning_rate` (float, optional): Learning rate. Default: 5e-5
- `batch_size` (int, optional): Batch size. Default: 16

**Returns:**
- `dict`: Dictionary containing training metrics and results

#### `evaluate_model(test_data_path)`

Evaluate the MOCK-LLM model on test data.

```python
metrics = mock_llm.evaluate_model("/path/to/test_data.json")
```

**Parameters:**
- `test_data_path` (str): Path to the test data

**Returns:**
- `dict`: Dictionary containing evaluation metrics

#### `classify_text(text)`

Classify text using the MOCK-LLM model.

```python
classification = mock_llm.classify_text("I love this product!")
```

**Parameters:**
- `text` (str): Input text

**Returns:**
- `dict`: Dictionary mapping class labels to probabilities

#### `get_status()`

Get the current status of the MOCK-LLM system.

```python
status = mock_llm.get_status()
```

**Returns:**
- `dict`: Dictionary containing MOCK-LLM status information

## UnifiedMemory

Interface to the unified memory system.

### Initialization

```python
from agi_toolkit import UnifiedMemory

memory = UnifiedMemory(
    non_euclidean_dimension=11,
    fold_factor=4,
    embedding_size=768
)
```

#### Parameters:

- `non_euclidean_dimension` (int, optional): Dimensions for the non-euclidean memory. Default: 11
- `fold_factor` (int, optional): Fold factor for the memory. Default: 4
- `embedding_size` (int, optional): Embedding size. Default: 768

### Methods

#### `store(key, data, metadata=None)`

Store data in memory.

```python
memory.store("user_123", {"name": "John", "preferences": {"theme": "dark"}})
```

**Parameters:**
- `key` (str): Unique identifier for the data
- `data` (any): Data to store
- `metadata` (dict, optional): Additional metadata. Default: None

**Returns:**
- `bool`: Success status

#### `retrieve(key)`

Retrieve data from memory.

```python
data, metadata = memory.retrieve("user_123")
```

**Parameters:**
- `key` (str): Unique identifier for the data

**Returns:**
- `tuple`: (data, metadata) or (None, None) if key not found

#### `get_stats()`

Get memory statistics.

```python
stats = memory.get_stats()
```

**Returns:**
- `dict`: Dictionary containing memory statistics

#### `list_keys(pattern=None)`

List all keys in memory.

```python
keys = memory.list_keys("user_*")
```

**Parameters:**
- `pattern` (str, optional): Pattern to filter keys. Default: None

**Returns:**
- `list`: List of keys

#### `clear()`

Clear all data from memory.

```python
memory.clear()
```

**Returns:**
- `bool`: Success status

## Exceptions

The AGI Toolkit defines several exceptions:

### `ComponentNotAvailableError`

Raised when a required component is not available.

```python
from agi_toolkit.exceptions import ComponentNotAvailableError

try:
    # Operation that requires a component
except ComponentNotAvailableError as e:
    print(f"Component not available: {str(e)}")
```

### `MemoryAccessError`

Raised when there is an error accessing memory.

```python
from agi_toolkit.exceptions import MemoryAccessError

try:
    # Memory operation
except MemoryAccessError as e:
    print(f"Memory access error: {str(e)}")
```

### `InvalidParameterError`

Raised when an invalid parameter is provided.

```python
from agi_toolkit.exceptions import InvalidParameterError

try:
    # Operation with parameters
except InvalidParameterError as e:
    print(f"Invalid parameter: {str(e)}")
```
