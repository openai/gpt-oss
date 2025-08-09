# Streamlit Chat Example

This example demonstrates how to create a web-based chat interface for gpt-oss models using Streamlit.

## Features

- Interactive chat interface with conversation history
- Model selection (large/small)
- Configurable reasoning effort (low/medium/high)
- Function calling support with custom tools
- Real-time streaming responses

## Prerequisites

Before running this example, you need:

1. **Local gpt-oss server running** - This example expects a local API server compatible with OpenAI's chat completions format
2. **Python packages** - Install the required dependencies

## Installation

1. Install Streamlit and requests:
```bash
pip install streamlit requests
```

2. Make sure you have a local gpt-oss server running (e.g., using Ollama, vLLM, or the gpt-oss responses API server)

## Running the Example

1. Start your local gpt-oss server on `http://localhost:11434` (or modify the base URL in the code)

2. Run the Streamlit app:
```bash
streamlit run streamlit_chat.py
```

3. Open your browser to the URL displayed (typically `http://localhost:8501`)

## Configuration

The app provides several configuration options in the sidebar:

- **Model**: Choose between "large" and "small" models
- **Instructions**: Customize the system prompt for the assistant
- **Reasoning effort**: Control the level of reasoning (low/medium/high)
- **Functions**: Enable/disable function calling with a sample weather function

## Customization

You can customize this example by:

- Modifying the base URL to point to your gpt-oss server
- Adding custom functions in the function properties section
- Changing the default system instructions
- Styling the interface with Streamlit components

## Notes

- This example assumes you're running a local gpt-oss server compatible with OpenAI's API format
- The function calling feature includes a sample weather function for demonstration
- Conversation history is maintained during the session but not persisted between sessions