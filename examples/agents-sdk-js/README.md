# gpt-oss with OpenAI Agents SDK (JavaScript)

This example demonstrates how to use gpt-oss models with the OpenAI Agents SDK in JavaScript/TypeScript.

## Prerequisites

- Node.js 18+ installed
- Ollama installed and running locally
- gpt-oss model downloaded in Ollama

## Setup

1. Install dependencies:

```bash
npm install
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
npm start
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
- **Tool integration**: Demonstrates how to create and use custom tools
- **MCP integration**: Shows how to connect to external services via MCP
- **Harmony format**: Uses the harmony response format for better reasoning

## Customization

You can modify the example by:

- Changing the model name in the agent configuration
- Adding more tools to the `tools` array
- Modifying the agent instructions
- Adding different MCP servers

## Troubleshooting

- Make sure Ollama is running on `localhost:11434`
- Ensure you have the correct model name (`gpt-oss:20b-test` or `gpt-oss:20b`)
- Check that npx is available for the MCP filesystem server
