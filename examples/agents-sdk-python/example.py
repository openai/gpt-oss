import asyncio
import shutil
from pathlib import Path

from openai import AsyncOpenAI
from agents import (
    Agent,
    ItemHelpers,
    Runner,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
    function_tool,
)
from agents.mcp import MCPServerStdio


async def prompt_user(question: str) -> str:
    """Async input prompt function"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, question)


@function_tool
async def get_weather(location: str) -> str:
    """Dummy weather tool"""
    return f"The weather in {location} is sunny."


async def main():
    # Check if npx is installed
    if not shutil.which("npx"):
        raise RuntimeError(
            "❌ 'npx' is not installed. Please install it with `npm install -g npx`."
        )

    # OpenAI client (e.g., for Ollama or local LLM)
    openai_client = AsyncOpenAI(
        api_key="local",
        base_url="http://localhost:11434/v1",
    )

    # Set up ModelContextProtocol (MCP) server using npx
    samples_dir = str(Path.cwd())
    mcp_server = MCPServerStdio(
        name="Filesystem MCP Server (via npx)",
        params={
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                samples_dir,
            ],
        },
    )

    # Connect to MCP server
    await mcp_server.connect()

    # Configure agents SDK
    set_tracing_disabled(True)
    set_default_openai_client(openai_client)
    set_default_openai_api("chat_completions")

    # Create the agent
    agent = Agent(
        name="My Agent",
        instructions="You are a helpful assistant.",
        tools=[get_weather],
        model="gpt-oss:20b-test",  # Ensure this model is available in your Ollama instance
        mcp_servers=[mcp_server],
    )

    # Get user input
    user_input = await prompt_user("> ")

    # Run agent with streamed output
    result = Runner.run_streamed(agent, user_input)

    # Stream processing
    async for event in result.stream_events():
        if event.type == "raw_response_event":
            continue
        elif event.type == "agent_updated_stream_event":
            print(f"[Agent updated]: {event.new_agent.name}")
        elif event.type == "run_item_stream_event":
            item = event.item
            if item.type == "tool_call_item":
                print("-- Tool was called")
            elif item.type == "tool_call_output_item":
                print(f"-- Tool output: {item.output}")
            elif item.type == "message_output_item":
                print(f"-- Message output:\n{ItemHelpers.text_message_output(item)}")

    print("✅ Run complete.")


if __name__ == "__main__":
    asyncio.run(main())
