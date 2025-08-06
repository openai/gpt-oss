# GPT-OSS Examples

This directory contains practical examples demonstrating how to use gpt-oss models in different scenarios and with various frameworks.

## Available Examples

### [Streamlit Chat Interface](./streamlit/)
A web-based chat application built with Streamlit that provides:
- Interactive chat interface with conversation history
- Model selection and configuration options
- Function calling capabilities
- Real-time streaming responses

**Best for**: Quick prototyping, demos, and interactive testing of gpt-oss models.

### [Agents SDK Python](./agents-sdk-python/)
An advanced example using the OpenAI Agents SDK with:
- Async agent interactions
- Model Context Protocol (MCP) integration
- Custom function tools
- Filesystem operations
- Streaming event processing

**Best for**: Building sophisticated AI agents with tool capabilities and external integrations.

## Getting Started

Each example directory contains its own README with detailed setup and usage instructions. Generally, you'll need:

1. **A local gpt-oss server running** (using Ollama, vLLM, or the gpt-oss responses API server)
2. **Python dependencies** specific to each example
3. **Additional tools** as specified in each example's requirements

## Common Setup

Most examples expect a local gpt-oss server compatible with OpenAI's API format. Here are quick setup options:

### Using Ollama
```bash
ollama pull gpt-oss:20b
ollama run gpt-oss:20b
```

### Using the gpt-oss Responses API Server
```bash
python -m gpt_oss.responses_api.serve --checkpoint /path/to/checkpoint --port 11434
```

### Using vLLM
```bash
python -m vllm.entrypoints.openai.api_server --model openai/gpt-oss-20b --port 11434
```

## Contributing Examples

We welcome contributions of new examples! If you've built something interesting with gpt-oss, consider:

1. Adding it to the [`awesome-gpt-oss.md`](../awesome-gpt-oss.md) file
2. Creating a pull request with a new example directory
3. Including a comprehensive README with setup instructions

## Support

For questions about these examples:
- Check the individual example README files
- Review the main [gpt-oss documentation](../README.md)
- Visit the [OpenAI Cookbook](https://cookbook.openai.com/topic/gpt-oss) for more guides