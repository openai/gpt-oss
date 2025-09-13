import asyncio
from pathlib import Path
import shutil

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


async def main():
    # Set up OpenAI client for local server (e.g., Ollama)
    openai_client = AsyncOpenAI(
        api_key="local",
        base_url="http://localhost:11434/v1",
    )

    # Get current working directory
    samples_dir = str(Path.cwd())

    # Create MCP server for filesystem operations
    mcp_server = MCPServerStdio(
        name="Filesystem MCP Server, via npx",
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

    # Define weather tool
    @function_tool
    async def get_weather(location: str) -> str:
        return f"The weather in {location} is sunny."

    # Create agent
    agent = Agent(
        name="My Agent",
        instructions="You are a helpful assistant.",
        tools=[get_weather],
        model="gpt-oss:20b-test",
        mcp_servers=[mcp_server],
    )

    # Get user input
    user_input = await prompt_user("> ")

    # Run agent with streaming
    result = Runner.run_streamed(agent, user_input)

    # Process streaming results
    async for event in result.stream_events():
        if event.type == "raw_response_event":
            continue
        elif event.type == "agent_updated_stream_event":
            print(f"Agent updated: {event.new_agent.name}")
        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                print("-- Tool was called")
            elif event.item.type == "tool_call_output_item":
                print(f"-- Tool output: {event.item.output}")
            elif event.item.type == "message_output_item":
                print(
                    f"-- Message output:\n {ItemHelpers.text_message_output(event.item)}"
                )
            else:
                pass

    print("=== Run complete ===")


if __name__ == "__main__":

    if not shutil.which("npx"):
        raise RuntimeError(
            "npx is not installed. Please install it with `npm install -g npx`."
        )
    asyncio.run(main())
