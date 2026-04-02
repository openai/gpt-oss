# gpt-oss Examples

This directory contains various examples demonstrating how to use gpt-oss in different scenarios and with different frameworks.

## Available Examples

### [Streamlit Chat](./streamlit/)
A simple chat interface built with Streamlit that connects to a local gpt-oss server.

**Features:**
- Interactive web-based chat interface
- Real-time responses
- Easy to customize and extend

### [Agents SDK - Python](./agents-sdk-python/)
Example using the OpenAI Agents SDK with Python to create an intelligent agent that can use tools and MCP servers.

**Features:**
- Tool integration (weather example)
- MCP server connectivity for filesystem operations
- Streaming responses
- Async/await support

### [Agents SDK - JavaScript/TypeScript](./agents-sdk-js/)
TypeScript example using the OpenAI Agents SDK to create an intelligent agent with tool calling capabilities.

**Features:**
- Tool integration
- MCP server connectivity
- TypeScript support
- Modern async/await patterns

### [Gradio Chat](./gradio/)
A simple chat interface using Gradio framework.

**Features:**
- Quick setup with Gradio
- Web-based interface
- Easy deployment

## Getting Started

1. **Start a local gpt-oss server** on `http://localhost:8000`
2. **Choose an example** from the directories above
3. **Follow the README** in each example directory for specific setup instructions

## Prerequisites

- Python 3.12+
- A running gpt-oss server (see main README for setup instructions)
- Framework-specific dependencies (listed in each example's README)

## Common Setup

Most examples assume you have a local gpt-oss server running. You can start one using:

```bash
# Using the responses API server
python -m gpt_oss.responses_api.serve --checkpoint gpt-oss-20b/original/ --port 8000

# Or using vLLM
vllm serve openai/gpt-oss-20b --port 8000

# Or using Ollama
ollama serve
ollama run gpt-oss:20b
```

If you're running the UI locally, it typically serves on `http://localhost:8081`.

## Contributing

When adding new examples:
1. Create a new directory with a descriptive name
2. Include a comprehensive README.md with setup instructions
3. Ensure all dependencies are clearly listed
4. Test the example thoroughly
5. Update this main examples README to include your new example
