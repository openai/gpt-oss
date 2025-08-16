# Agents SDK Python Example

This example demonstrates how to use the OpenAI Agents SDK with Python to create an intelligent agent that can interact with tools and MCP servers.

## Prerequisites

- Python 3.12+
- Node.js and npm (for MCP server)
- A running gpt-oss server

## Installation

1. Install Python dependencies:

```bash
pip install openai agents
```

2. Install Node.js dependencies for MCP server:

```bash
npm install -g npx
```

## Configuration

The example is configured to connect to a local gpt-oss server. Update the configuration in `example.py` if needed:

```python
openai_client = AsyncOpenAI(
    api_key="local",
    base_url="http://localhost:8000/v1",
)
```

## Running the Example

1. Start your local gpt-oss server on `http://localhost:8000`

2. Run the Python example:

```bash
python example.py
```

3. Enter your message when prompted and interact with the agent

## Features

### Tool Integration
The example includes a simple weather tool that demonstrates how to integrate custom functions:

```python
@function_tool
async def get_weather(location: str) -> str:
    return f"The weather in {location} is sunny."
```

### MCP Server Integration
The agent connects to a filesystem MCP server that allows it to:
- Read files
- Write files
- List directories
- Navigate the filesystem

### Streaming Responses
The example demonstrates how to handle streaming responses and different event types:
- Tool calls
- Tool outputs
- Message outputs
- Agent updates

## Customization

### Adding New Tools
You can add new tools by defining functions with the `@function_tool` decorator:

```python
@function_tool
async def my_custom_tool(param: str) -> str:
    # Your tool logic here
    return "Tool result"
```

### Different Models
Change the model by updating the agent configuration:

```python
agent = Agent(
    name="My Agent",
    instructions="You are a helpful assistant.",
    tools=[get_weather],
    model="gpt-oss:120b",  # or other available models
    mcp_servers=[mcp_server],
)
```

### Custom MCP Servers
You can connect to different MCP servers by modifying the server configuration:

```python
mcp_server = MCPServerStdio(
    name="Custom MCP Server",
    params={
        "command": "your-mcp-server-command",
        "args": ["arg1", "arg2"],
    },
)
```

## Troubleshooting

### npx not found
If you get an error about npx not being found:
```bash
npm install -g npx
```

### Connection errors
Ensure your gpt-oss server is running on the correct port (8000) and accessible.

### MCP server issues
The filesystem MCP server requires npx to be installed and accessible in your PATH.
