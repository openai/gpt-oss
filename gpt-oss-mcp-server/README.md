# MCP Servers for gpt-oss reference tools

This directory contains MCP servers for the reference tools in the [gpt-oss](https://github.com/openai/gpt-oss) repository.
You can set up these tools behind MCP servers and use them in your applications.
For inference service that integrates with MCP, you can also use these as reference tools.

In particular, this directory contains a `build-system-prompt.py` script that will generate exactly the same system prompt as `reference-system-prompt.py`.
The build system prompt script show case all the care needed to automatically discover the tools and construct the system prompt before feeding it into Harmony.

## Usage

```bash
# Install the dependencies
uv pip install -r requirements.txt
```

```bash
# Assume we have harmony and gpt-oss installed
uv pip install mcp[cli]
# start the servers
mcp run -t sse browser_server.py:mcp
mcp run -t sse python_server.py:mcp
```

You can now use MCP inspector to play with the tools.
Once opened, set SSE to `http://localhost:8001/sse` and `http://localhost:8000/sse` respectively.

To compare the system prompt and see how to construct it via MCP service discovery, see `build-system-prompt.py`.
This script will generate exactly the same system prompt as `reference-system-prompt.py`.

## Search Backend Configuration

The browser server supports two search backends: **Exa** (default) and **Tavily**. You can configure which backend to use via the `SEARCH_BACKEND` environment variable.

- **Exa**: Uses HTML parsing for content extraction
- **Tavily**: Uses markdown processing for faster and cleaner content extraction

### Using Exa Backend (Default)

The browser server uses Exa by default. To explicitly set it:

```bash
# Default behavior - uses Exa
mcp run -t sse browser_server.py:mcp

# Explicitly set Exa
SEARCH_BACKEND=exa mcp run -t sse browser_server.py:mcp
```

You'll need an Exa API key set in your environment:
```bash
export EXA_API_KEY="your_exa_api_key_here"
```

### Using Tavily Backend

To use Tavily's search and extraction capabilities:

```bash
SEARCH_BACKEND=tavily mcp run -t sse browser_server.py:mcp
```

You'll need a Tavily API key set in your environment:
```bash
export TAVILY_API_KEY="your_tavily_api_key_here"
```
