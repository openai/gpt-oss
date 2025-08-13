# Run this before running the tool:
# $ docker image pull python:3.11
import io
import tarfile
from typing import Any, AsyncIterator

import docker
from openai_harmony import (
    Author,
    Content,
    Message,
    Role,
    TextContent,
    ToolNamespaceConfig,
)

from ..tool import Tool

_docker_client = None


def call_python_script(script: str) -> str:
    """
    Call a python script by writing it to a file in the container and executing it.
    """
    global _docker_client
    if _docker_client is None:
        _docker_client = docker.from_env()
        # pull image `python:3.11` if not present
        try:
            _docker_client.images.get("python:3.11")
        except docker.errors.ImageNotFound:
            _docker_client.images.pull("python:3.11")

    # 1. Create a temporary tar archive containing the script
    script_name = "script.py"
    tarstream = io.BytesIO()
    with tarfile.open(fileobj=tarstream, mode="w") as tar:
        script_bytes = script.encode("utf-8")
        tarinfo = tarfile.TarInfo(name=script_name)
        tarinfo.size = len(script_bytes)
        tar.addfile(tarinfo, io.BytesIO(script_bytes))
    tarstream.seek(0)

    # 2. Start the container
    container = _docker_client.containers.create(
        "python:3.11", command="sleep infinity", detach=True
    )
    try:
        container.start()
        # 3. Put the script into the container
        container.put_archive(path="/tmp", data=tarstream.read())
        # 4. Execute the script
        exec_result = container.exec_run(f"python /tmp/{script_name}")
        output = exec_result.output.decode("utf-8")
    finally:
        container.remove(force=True)
    return output


class PythonTool(Tool):
    def __init__(
        self,
        name: str = "python",
    ):
        assert name == "python"

    @classmethod
    def get_tool_name(cls) -> str:
        return "python"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def instruction(self) -> str:
        return """
Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).
When you send a message containing python code to python, it will be executed in a stateless docker container, and the stdout of that process will be returned to you. You have to use print statements to access the output.
        """.strip()

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name=self.get_tool_name(), description=self.instruction, tools=[]
        )

    def _make_response(
        self,
        output: str,
        channel: str | None = None,
    ) -> Message:
        content = TextContent(text=output)
        return self.make_response(content=content, channel=channel)

    def make_response(
        self,
        content: Content,
        *,
        metadata: dict[str, Any] | None = None,
        author: Author | None = None,
        channel: str | None = None,
    ) -> Message:
        tool_name = self.get_tool_name()
        author = Author(role=Role.TOOL, name=f"{tool_name}")

        message = Message(
            author=author,
            content=[content],
        ).with_recipient("assistant")

        if channel:
            message = message.with_channel(channel)

        return message

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        script = message.content[0].text
        channel = message.channel
        output = call_python_script(script)
        yield self._make_response(output, channel=channel)
