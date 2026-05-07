# Run this before running the tool:
# $ docker image pull python:3.11
import asyncio
import contextlib
import io
import os
import queue
import subprocess
import tarfile
import tempfile
from pathlib import Path
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

VALID_EXECUTION_BACKENDS = {
    "docker",
    "dangerously_use_uv",
    "dangerously_use_local_jupyter",
}

_default_backend = os.environ.get("PYTHON_EXECUTION_BACKEND", "docker")
if _default_backend not in VALID_EXECUTION_BACKENDS:
    _default_backend = "docker"

PYTHON_EXECUTION_BACKEND = _default_backend


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
        if not output.strip():
            output = "[WARN] No output available. Use print() to output anything to stdout to receive the output"
    finally:
        container.remove(force=True)
    return output


def call_python_script_with_uv(script: str) -> str:
    """
    Call a python script by writing it to a file to a temporary directory
    and executing it with uv.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = os.path.join(temp_dir, "script.py")
        with open(script_path, "w") as f:
            f.write(script)
        exec_result = subprocess.run(
            ["uv", "run", "--no-project", "python", script_path],
            capture_output=True)
        return (
            exec_result.stdout.decode("utf-8")
            if exec_result.returncode == 0
            else exec_result.stderr.decode("utf-8")
        )


class LocalJupyterSession:
    """Stateful helper that proxies execution through a local Jupyter kernel."""

    def __init__(
        self,
        connection_file: str | None = None,
        *,
        timeout: float = 120.0,
    ) -> None:
        try:
            from jupyter_client import BlockingKernelClient, KernelManager
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The dangerously_use_local_jupyter backend requires the jupyter_client package to be installed."
            ) from exc

        self._default_timeout = timeout
        self._owns_kernel = False
        self._client: BlockingKernelClient
        self._km: KernelManager | None = None

        if connection_file:
            connection_path = Path(connection_file).expanduser()
            if not connection_path.exists():
                raise FileNotFoundError(
                    f"Cannot find Jupyter connection file at '{connection_path}'."
                )
            client = BlockingKernelClient()
            client.load_connection_file(str(connection_path))
            client.start_channels()
            # Ensure the connection is ready before executing.
            client.wait_for_ready(timeout=self._default_timeout)
            self._client = client
        else:
            km = KernelManager()
            km.start_kernel()
            client = km.blocking_client()
            client.start_channels()
            client.wait_for_ready(timeout=self._default_timeout)
            self._client = client
            self._km = km
            self._owns_kernel = True

    def execute(self, code: str, *, timeout: float | None = None) -> str:
        """Execute code in the kernel, returning combined stdout/stderr output."""

        client = self._client
        effective_timeout = timeout or self._default_timeout
        msg_id = client.execute(
            code,
            store_history=True,
            allow_stdin=False,
            stop_on_error=False,
        )

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []

        while True:
            try:
                msg = client.get_iopub_msg(timeout=effective_timeout)
            except queue.Empty as exc:
                raise TimeoutError("Timed out waiting for Jupyter kernel output.") from exc

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type")
            content = msg.get("content", {})

            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stdout":
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)
            elif msg_type == "error":
                traceback_data = content.get("traceback")
                if traceback_data:
                    stderr_parts.append("\n".join(traceback_data))
                else:
                    ename = content.get("ename", "")
                    evalue = content.get("evalue", "")
                    stderr_parts.append(f"{ename}: {evalue}".strip())
            elif msg_type in {"execute_result", "display_data"}:
                data = content.get("data", {})
                text = data.get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else f"{text}\n")
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break

        # Drain the shell channel to capture final execution status.
        while True:
            try:
                reply = client.get_shell_msg(timeout=effective_timeout)
            except queue.Empty as exc:
                raise TimeoutError(
                    "Timed out waiting for Jupyter kernel execution reply."
                ) from exc

            if reply.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            reply_content = reply.get("content", {})
            if reply_content.get("status") == "error":
                traceback_data = reply_content.get("traceback")
                if traceback_data:
                    stderr_parts.append("\n".join(traceback_data))
                else:
                    ename = reply_content.get("ename", "")
                    evalue = reply_content.get("evalue", "")
                    stderr_parts.append(f"{ename}: {evalue}".strip())
            break

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)

        if stderr:
            if stdout:
                stdout = f"{stdout.rstrip()}\n{stderr}"
            else:
                stdout = stderr

        if not stdout.strip():
            stdout = (
                "[WARN] No output available. Use print() to output anything to stdout to "
                "receive the output"
            )

        return stdout

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._client.stop_channels()

        if self._owns_kernel and self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        self.close()

class PythonTool(Tool):
    def __init__(
        self,
        name: str = "python",
        *,
        execution_backend: str | None = None,
        local_jupyter_connection_file: str | None = None,
        local_jupyter_timeout: float = 60.0,
    ):
        assert name == "python"

        backend = execution_backend or PYTHON_EXECUTION_BACKEND
        if backend not in VALID_EXECUTION_BACKENDS:
            raise ValueError(
                "execution_backend must be one of: "
                + ", ".join(sorted(VALID_EXECUTION_BACKENDS))
            )

        self._execution_backend = backend
        self._local_jupyter_connection_file = (
            local_jupyter_connection_file
            or os.environ.get("PYTHON_LOCAL_JUPYTER_CONNECTION_FILE")
        )
        self._local_jupyter_timeout = local_jupyter_timeout

        self._jupyter_session: LocalJupyterSession | None = None
        self._execution_lock: asyncio.Lock | None = None

        if self._execution_backend == "dangerously_use_local_jupyter":
            self._execution_lock = asyncio.Lock()
            self._jupyter_session = LocalJupyterSession(
                connection_file=self._local_jupyter_connection_file,
                timeout=self._local_jupyter_timeout,
            )

    @classmethod
    def get_tool_name(cls) -> str:
        return "python"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def instruction(self) -> str:
        if self._execution_backend == "dangerously_use_local_jupyter":
            return """
Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).
When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 120.0 seconds. Internet access for this session is UNKNOWN. Depends on the cluster.
            """.strip()

        return """
Use this tool to execute STATELESS Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).
When you send a message containing python code to python, it will be executed in a stateless docker container, and the stdout of that process will be returned to you. You have to use print statements to access the output.

IMPORTANT: Your python environment is not shared between calls. You will have to pass your entire code each time.
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

        if self._execution_backend == "docker":
            output = call_python_script(script)
        elif self._execution_backend == "dangerously_use_uv":
            output = call_python_script_with_uv(script)
        elif self._execution_backend == "dangerously_use_local_jupyter":
            assert self._jupyter_session is not None
            lock = self._execution_lock
            if lock is not None:
                async with lock:
                    try:
                        output = self._jupyter_session.execute(script)
                    except TimeoutError as exc:
                        output = f"[ERROR] {exc}"
            else:
                try:
                    output = self._jupyter_session.execute(script)
                except TimeoutError as exc:
                    output = f"[ERROR] {exc}"
        else:
            raise ValueError(
                f"Invalid PYTHON_EXECUTION_BACKEND: {self._execution_backend}"
            )
        yield self._make_response(output, channel=channel)

    def close(self) -> None:
        if self._jupyter_session is not None:
            self._jupyter_session.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        self.close()
