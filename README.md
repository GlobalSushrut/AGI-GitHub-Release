# AGI Toolkit: Unified ASI and MOCK-LLM Integration

## Important: Environment Variable Setup

To use this package, you must set the following environment variable:

```bash
export AGI_TOOLKIT_KEY='AGI-Toolkit-Secure-2025'
```

This environment variable is required for decrypting and loading the core components. Without it, the toolkit will not function properly.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MR--ZKELP-purple)

A powerful toolkit for integrating ASI (Artificial Super Intelligence) and MOCK-LLM systems into real-world applications without modifying the core implementation code.

## Licensing

This software is released under the **Mock Root ZK Entropy Licensing Protocol (MR-ZKELP)**:

- **Free for Development/Open Source**: Build, test, and develop freely
- **Requires License for Corporate/Production**: Enterprise deployments require a license

The software automatically detects usage patterns and enforces the licensing model. For commercial licenses, contact umeshlamton@gmail.com.

## Features

- **Clean, Stable API:** Build applications on top of ASI and MOCK-LLM without touching core code
- **Component Flexibility:** Works with ASI, MOCK-LLM, or both components together
- **Non-Euclidean Memory:** Advanced memory system with high-dimensional storage capabilities
- **Entropic Quantum Compression:** Dramatic memory reduction for i3 compatibility
- **Unified Training:** Train models across ASI and MOCK-LLM systems seamlessly
- **Easy Integration:** Simple high-level functions for common tasks

## Installation

```bash
# Install from PyPI
pip install agi-toolkit

# Install from source
git clone https://github.com/GlobalSushrut/AGI-GitHub-Release.git
cd AGI-GitHub-Release/deployable
pip install -e .
```

## Quick Start

```python
from agi_toolkit import AGIAPI

# Initialize the API
api = AGIAPI()

# Check component availability
print(f"ASI available: {api.has_asi}")
print(f"MOCK-LLM available: {api.has_mock_llm}")

# Generate text with MOCK-LLM
response = api.generate_text("Explain quantum computing in simple terms")
print(response)

# Process data with ASI
result = api.process_with_asi({"query": "Analyze market trends for AI in 2025"})
print(result)

# Store and retrieve data from unified memory
api.store_data("user_preferences", {"theme": "dark", "language": "en"})
prefs, metadata = api.retrieve_data("user_preferences")
print(prefs)
```

## Examples

The `examples` directory contains fully functional applications built with the AGI Toolkit:

- **Text Analysis App:** Analyze text using ASI and MOCK-LLM capabilities
- **Simple Chatbot:** Build conversational AI with memory persistence
- **Data Processor:** Process and analyze structured data with ASI

## System Architecture

The AGI Toolkit provides a simplified API layer on top of the complex ASI and MOCK-LLM systems:

```
┌────────────────────────────────────────────────────────┐
│                     Your Application                    │
└───────────────────────────┬────────────────────────────┘
                           │
┌───────────────────────────┴────────────────────────────┐
│                        AGI Toolkit                      │
├────────────────────────┬──────────────────────────────┬┘
│                        │                               │
┌────────────────────────┴─────┐   ┌──────────────────────┴───┐
│      ASI Integration          │   │   MOCK-LLM Integration   │
└──────────────┬───────────────┘   └─────────────┬────────────┘
              │                                 │
┌─────────────┴─────────────────────────────────┴────────────┐
│              Unified Memory and Training System             │
└────────────────────────────────────────────────────────────┘
```

## Component Status

The toolkit is designed to work with available components:

- **ASI Components:** Advanced reasoning, data processing, and AI capabilities
- **MOCK-LLM Components:** Text generation, embeddings, and language understanding
- **Unified Memory:** Non-Euclidean memory system for both components
- **Training System:** Unified training across both systems

## Documentation

For detailed documentation, see the `docs` directory or visit our [documentation site](https://github.com/GlobalSushrut/AGI-GitHub-Release/docs).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

---

*Note: ASI and MOCK-LLM components availability may vary depending on your system configuration.*
