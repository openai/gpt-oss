# Python Agents SDK Example

This example demonstrates how to use GPT-OSS with OpenAI's Agents SDK in Python.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- GPT-OSS model running locally (Ollama, vLLM, etc.)
- Node.js (for MCP server)

### Installation

1. **Install Python dependencies:**
```bash
pip install -e .
```

2. **Install Node.js dependencies (for MCP server):**
```bash
npm install -g npx
```

### Running the Example

1. **Start your GPT-OSS model:**
```bash
# Using Ollama
ollama run gpt-oss:20b

# Using vLLM
vllm serve openai/gpt-oss-20b --port 11434
```

2. **Run the example:**
```bash
python example.py
```

## 🔧 Configuration

### Environment Setup
The example is configured to use a local model server:

```python
openai_client = AsyncOpenAI(
    api_key="local",
    base_url="http://localhost:11434/v1",
)
```

### Model Configuration
```python
agent = Agent(
    name="My Agent",
    instructions="You are a helpful assistant.",
    tools=[search_tool],
    model="gpt-oss:20b-test",  # Model name for local server
    mcp_servers=[mcp_server],
)
```

## 🛠️ Features Demonstrated

### Function Calling
The example includes a weather tool:
```python
@function_tool
async def search_tool(location: str) -> str:
    return f"The weather in {location} is sunny."
```

### MCP (Model Context Protocol) Integration
Filesystem access via MCP server:
```python
mcp_server = MCPServerStdio(
    name="Filesystem MCP Server, via npx",
    params={
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", samples_dir],
    },
)
```

### Streaming Responses
Real-time response streaming:
```python
result = Runner.run_streamed(agent, user_input)
async for event in result.stream_events():
    # Process streaming events
```

## 📝 Code Structure

### Main Components
1. **Client Setup**: OpenAI client configuration for local model
2. **MCP Server**: Filesystem access server
3. **Tool Definition**: Custom function calling tool
4. **Agent Creation**: GPT-OSS agent with tools and MCP
5. **Streaming Execution**: Real-time response processing

### Event Types
- `raw_response_event`: Raw model responses
- `agent_updated_stream_event`: Agent state changes
- `run_item_stream_event`: Tool calls and outputs

## 🐛 Troubleshooting

### Common Issues

1. **"npx is not installed"**
   ```bash
   npm install -g npx
   ```

2. **Connection refused to localhost:11434**
   - Ensure your model server is running
   - Check the port number matches your setup

3. **Model not found**
   - Verify the model name matches your local server
   - Check that the model is properly loaded

### Debug Mode
Enable tracing for detailed logs:
```python
# Remove this line to enable tracing
set_tracing_disabled(True)
```

## 🔗 Related Documentation

- [OpenAI Agents SDK](https://github.com/openai/agents) - Official SDK documentation
- [Model Context Protocol](https://modelcontextprotocol.io/) - MCP specification
- [Main Examples README](../README.md) - Overview of all examples
- [GPT-OSS Tools](../../gpt_oss/tools/) - Available tools for integration

## 🤝 Contributing

Improvements welcome! Please:
- Add more tool examples
- Enhance error handling
- Add configuration options
- Improve documentation
