# AGI Toolkit Example Applications

This directory contains example applications demonstrating how to use the AGI Toolkit in real-world scenarios. These examples show how you can build powerful applications on top of ASI and MOCK-LLM components without modifying the core implementation code.

## Available Examples

### Text Analysis Application

A comprehensive text analysis tool that demonstrates:
- Sentiment analysis with MOCK-LLM
- Advanced text processing with ASI
- Data persistence using the unified memory system
- Command-line interface for seamless interaction

**Usage:**

```bash
cd text_analysis
python app.py --analyze "This is a sample text to analyze"
```

Or use the interactive mode:

```bash
cd text_analysis
python app.py
```

### Simple Chatbot

A conversational AI application that shows:
- How to build a chat interface
- Maintaining conversation context
- Leveraging both ASI and MOCK-LLM capabilities
- Graceful degradation when components are unavailable

**Usage:**

```bash
cd simple_chatbot
python chatbot.py
```

For non-interactive use:

```bash
cd simple_chatbot
python chatbot.py --input "Tell me about quantum computing"
```

## Building Your Own Applications

These examples are meant to serve as starting points for your own custom applications. Here are some tips for creating your own applications:

1. **Start with the simplified API**: The `AGIAPI` class provides a high-level interface that abstracts away the implementation details.

2. **Handle component availability**: Always check for component availability using `api.has_asi` and `api.has_mock_llm` before using specific functionality.

3. **Use error handling**: Implement proper error handling to create resilient applications.

4. **Leverage the unified memory**: Use the memory system for maintaining state across sessions.

## Additional Resources

- [API Reference](../docs/api_reference.md): Detailed documentation of all available classes and methods
- [Getting Started Guide](../docs/getting_started.md): Quick introduction to the AGI Toolkit
- [Main README](../README.md): Overview of the entire project

## Contributing New Examples

We welcome contributions of new example applications! If you have developed an interesting application using the AGI Toolkit, please consider contributing it back to the community.

To contribute:

1. Create a new directory under `examples/`
2. Include a README.md explaining what your example does
3. Ensure your code is well-documented and follows best practices
4. Submit a pull request to the main repository
