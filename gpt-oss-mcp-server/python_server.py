from mcp.server.fastmcp import FastMCP
from gpt_oss.tools.python_docker.docker_tool import PythonTool
from openai_harmony import Message, TextContent, Author, Role

# Pass lifespan to server
mcp = FastMCP(
    name="python",
    instructions=r"""
Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).
When you send a message containing python code to python, it will be executed in a stateless docker container, and the stdout of that process will be returned to you.
""".strip(),
)


@mcp.tool(
    name="python",
    title="Execute Python code",
    description="""
Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).
When you send a message containing python code to python, it will be executed in a stateless docker container, and the stdout of that process will be returned to you.
    """,
    annotations={
        # Harmony format don't want this schema to be part of it because it's simple text in text out
        "include_in_prompt": False,
    })
async def python(code: str) -> str:
    tool = PythonTool()
    messages = []
    async for message in tool.process(
            Message(author=Author(role=Role.TOOL, name="python"),
                    content=[TextContent(text=code)])):
        messages.append(message)
    return "\n".join([message.content[0].text for message in messages])
