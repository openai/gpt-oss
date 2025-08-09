# Agents SDK Python Example

This example demonstrates how to use gpt-oss models with the OpenAI Agents SDK for Python, including integration with Model Context Protocol (MCP) servers.

## Features

- Async agent interaction with streaming responses
- Integration with MCP servers for enhanced capabilities
- Custom function tools
- Filesystem operations through MCP
- Real-time event processing

## Prerequisites

Before running this example, you need:

1. **Node.js and npm** - Required for the MCP filesystem server
2. **Local gpt-oss server** - Running on `http://localhost:11434`
3. **Python 3.12+** - As specified in the project configuration

## Installation

1. Install Node.js and npm if not already installed
2. Install the Python dependencies:

```bash
pip install openai-agents>=0.2.4
```

Or if you prefer using the project file:

```bash
pip install -e .
```

## Running the Example

1. **Start your local gpt-oss server** (e.g., using Ollama):
```bash
ollama run gpt-oss:20b
```

2. **Run the example**:
```bash
python example.py
```

3. **Interact with the agent** by typing your message when prompted

## How It Works

The example sets up:

1. **OpenAI Client**: Configured to connect to your local gpt-oss server
2. **MCP Server**: Filesystem operations server via npx
3. **Custom Tools**: A sample weather search function
4. **Streaming Agent**: Processes responses in real-time

## Example Interaction

```
> Can you tell me about the files in the current directory?
Agent updated: My Agent
-- Tool was called
-- Tool output: [filesystem results]
-- Message output: I can see several files in your current directory...
=== Run complete ===
```

## Configuration

You can customize the example by:

- **Model**: Change the model name in the `Agent` configuration (line 70)
- **Instructions**: Modify the agent's system instructions (line 68)
- **Tools**: Add custom function tools using the `@function_tool` decorator
- **Base URL**: Update the OpenAI client base URL for different servers

## MCP Integration

This example uses the Model Context Protocol (MCP) to provide the agent with filesystem capabilities. The MCP server is automatically started and connected, allowing the agent to:

- Read and write files
- List directory contents
- Navigate the filesystem

## Error Handling

The example includes basic error handling:
- Checks for `npx` availability before running
- Graceful connection to MCP servers
- Async/await pattern for proper resource management

## Extending the Example

To add more capabilities:

1. **Add custom tools**:
```python
@function_tool
async def my_custom_tool(param: str) -> str:
    return f"Processed: {param}"
```

2. **Add more MCP servers**:
```python
additional_mcp = MCPServerStdio(name="Another Server", params={...})
agent = Agent(..., mcp_servers=[mcp_server, additional_mcp])
```

3. **Process different event types**:
```python
async for event in result.stream_events():
    if event.type == "your_custom_event_type":
        # Handle custom events
        pass
```