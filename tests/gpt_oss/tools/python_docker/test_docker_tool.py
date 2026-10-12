import asyncio
import time

from openai_harmony import Author, Message, Role, TextContent

from gpt_oss.tools.python_docker import docker_tool
from gpt_oss.tools.python_docker.docker_tool import PythonTool

BLOCKING_SECONDS = 0.5
HEARTBEAT_INTERVAL = 0.01


def python_call(script: str) -> Message:
    return (
        Message(
            author=Author(role=Role.ASSISTANT, name=None),
            content=[TextContent(text=script)],
        )
        .with_recipient("python")
        .with_channel("analysis")
    )


def test_python_tool_does_not_block_the_event_loop(monkeypatch):
    """Every python backend blocks, so none of them may run on the event loop.

    `container.exec_run` has no timeout, so a long running (or looping) script
    would otherwise freeze every other request served by the same process --
    including the disconnect checks of in-flight Responses API streams.
    """

    def blocking_backend(script: str) -> str:
        time.sleep(BLOCKING_SECONDS)
        return "done"

    monkeypatch.setattr(docker_tool, "call_python_script", blocking_backend)

    async def scenario() -> tuple[int, list[Message]]:
        ticks = 0
        stop = asyncio.Event()

        async def heartbeat() -> None:
            nonlocal ticks
            while not stop.is_set():
                ticks += 1
                await asyncio.sleep(HEARTBEAT_INTERVAL)

        beat = asyncio.create_task(heartbeat())
        await asyncio.sleep(0)  # let the heartbeat reach its first await

        tool = PythonTool(execution_backend="docker")
        outputs = [message async for message in tool.process(python_call("print('hi')"))]

        stop.set()
        await beat
        return ticks, outputs

    ticks, outputs = asyncio.run(scenario())

    assert [content.text for message in outputs for content in message.content] == ["done"]
    assert ticks >= 5, (
        f"the event loop only advanced {ticks} time(s) while the python tool ran for "
        f"{BLOCKING_SECONDS}s; the blocking backend must be dispatched to a worker thread"
    )
