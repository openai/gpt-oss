# gpt-oss Streamlit Chat Interface

This example demonstrates how to create a web-based chat interface for gpt-oss models using Streamlit.

## Prerequisites

- Python 3.12+
- A running gpt-oss server (vLLM, Ollama, or other compatible server)
- Streamlit installed

## Setup

1. Install dependencies:

```bash
pip install streamlit requests
```

2. Start your gpt-oss server. For example, with Ollama:

```bash
# Install gpt-oss-20b model
ollama pull gpt-oss:20b

# Start Ollama
ollama serve
```

3. Run the Streamlit app:

```bash
streamlit run streamlit_chat.py
```

## Features

This chat interface includes:

- **Real-time streaming**: See responses as they're generated
- **Reasoning display**: View the model's reasoning process
- **Tool integration**: Use custom functions and browser search
- **Configurable parameters**: Adjust temperature, reasoning effort, and more
- **Debug mode**: View raw API responses for debugging
- **Responsive design**: Clean, modern chat interface

## Configuration

The sidebar allows you to configure:

- **Model selection**: Choose between different model sizes
- **Instructions**: Customize the assistant's behavior
- **Reasoning effort**: Set reasoning effort (low/medium/high)
- **Functions**: Enable and configure custom function calls
- **Browser search**: Enable web search capabilities
- **Temperature**: Control response randomness
- **Max output tokens**: Limit response length
- **Debug mode**: Show raw API responses

## Server Configuration

The app expects a Responses API compatible server running on:

- `http://localhost:8081/v1/responses` (for small model)
- `http://localhost:8000/v1/responses` (for large model)

You can modify these URLs in the code to match your setup.

## Customization

You can customize the example by:

- Adding new function tools
- Modifying the UI layout
- Adding authentication
- Implementing different server backends
- Adding file upload capabilities

## Troubleshooting

- Make sure your gpt-oss server is running and accessible
- Check that the server URLs match your setup
- Verify all dependencies are installed
- Check the debug mode for API response details
