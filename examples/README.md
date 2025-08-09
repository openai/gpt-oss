# gpt-oss Examples

This directory contains practical examples demonstrating how to use gpt-oss models in different scenarios.

## Available Examples

### 🤖 Agents SDK Examples

- **[JavaScript/TypeScript](./agents-sdk-js/)**: Use gpt-oss with OpenAI Agents SDK in Node.js
- **[Python](./agents-sdk-python/)**: Use gpt-oss with OpenAI Agents SDK in Python

These examples show how to create intelligent agents that can:

- Use custom tools and functions
- Integrate with MCP (Model Context Protocol) servers
- Stream responses in real-time
- Display reasoning and tool calls

### 💬 Streamlit Chat Interface

- **[Streamlit Chat](./streamlit/)**: A web-based chat interface for gpt-oss

This example demonstrates:

- Real-time streaming chat interface
- Configurable model parameters
- Tool integration (functions and browser search)
- Debug mode for API inspection
- Responsive web design

## Quick Start

1. **Choose an example** based on your needs:

   - Use **Agents SDK** for building intelligent applications
   - Use **Streamlit** for quick web interfaces

2. **Set up a gpt-oss server**:

   ```bash
   # With Ollama (recommended for local development)
   ollama pull gpt-oss:20b
   ollama serve

   # Or with vLLM
   vllm serve openai/gpt-oss-20b
   ```

3. **Follow the specific setup instructions** in each example's README

## Prerequisites

- Python 3.12+ (for Python examples)
- Node.js 18+ (for JavaScript examples)
- A running gpt-oss server (Ollama, vLLM, etc.)
- Basic familiarity with the chosen framework

## Getting Help

- Check the individual README files for detailed setup instructions
- Ensure your gpt-oss server is running and accessible
- Use debug modes to inspect API responses
- Refer to the main [gpt-oss documentation](../README.md) for model details

## Contributing

Feel free to contribute new examples or improvements to existing ones! Each example should include:

- Clear setup instructions
- Prerequisites and dependencies
- Usage examples
- Troubleshooting tips
