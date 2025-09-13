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
