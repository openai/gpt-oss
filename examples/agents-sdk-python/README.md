# gpt-oss with OpenAI Agents SDK (Python)

This example demonstrates how to use gpt-oss models with the OpenAI Agents SDK in Python.

## Prerequisites

- Python 3.12+
- Ollama installed and running locally
- gpt-oss model downloaded in Ollama
- npx available (for MCP filesystem server)

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure Ollama is running and you have the gpt-oss model:

```bash
# Install gpt-oss-20b model
ollama pull gpt-oss:20b

# Start Ollama (if not already running)
ollama serve
```

3. Run the example:

```bash
python example.py
```

## What this example does

This example creates a simple agent that:

- Uses the gpt-oss-20b model via Ollama
- Has a custom weather tool
- Integrates with an MCP (Model Context Protocol) filesystem server
- Streams responses in real-time
- Shows both reasoning and tool calls

## Key features

- **Real-time streaming**: See the model's reasoning and responses as they're generated
- **Tool integration**: Demonstrates how to create and use custom tools using `@function_tool`
- **MCP integration**: Shows how to connect to external services via MCP
- **Harmony format**: Uses the harmony response format for better reasoning
- **Async support**: Full async/await support for better performance

## Customization

You can modify the example by:

- Changing the model name in the agent configuration
- Adding more tools using the `@function_tool` decorator
- Modifying the agent instructions
- Adding different MCP servers

## Code structure

- `main()`: Main async function that sets up the agent
- `search_tool()`: Example function tool for weather queries
- `prompt_user()`: Helper function for user input
- MCP server setup for filesystem operations

## Troubleshooting

- Make sure Ollama is running on `localhost:11434`
- Ensure you have the correct model name (`gpt-oss:20b`)
- Check that npx is available for the MCP filesystem server
- Verify Python 3.12+ is installed
