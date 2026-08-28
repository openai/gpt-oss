import json
import pytest
from openai_harmony import Author, Message, Role, TextContent

from gpt_oss.tools.simple_browser.backend import YouComBackend
from gpt_oss.tools.simple_browser.page_contents import PageContents
from gpt_oss.tools.simple_browser.simple_browser_tool import (
    SimpleBrowserTool,
    maybe_get_function_args,
)


def _make_tool_message(recipient: str | None, payload: dict | str | None) -> Message:
    if payload is None:
        content = []
    elif isinstance(payload, str):
        content = [TextContent(text=payload)]
    else:
        content = [TextContent(text=json.dumps(payload))]

    msg = Message(
        author=Author.new(Role.ASSISTANT, "assistant"),
        content=content,
    )
    if recipient is not None:
        msg = msg.with_recipient(recipient)
    return msg


def test_maybe_get_function_args_none_recipient():
    msg = _make_tool_message(recipient=None, payload={"query": "test"})
    assert maybe_get_function_args(msg, tool_name="browser") is None


def test_maybe_get_function_args_unrelated_recipient():
    msg = _make_tool_message(recipient="python", payload={"code": "print(1)"})
    assert maybe_get_function_args(msg, tool_name="browser") is None


def test_maybe_get_function_args_valid():
    msg = _make_tool_message(recipient="browser.search", payload={"query": "openai"})
    assert maybe_get_function_args(msg, tool_name="browser") == {"query": "openai"}


def test_maybe_get_function_args_empty():
    msg = _make_tool_message(recipient="browser.search", payload=None)
    assert maybe_get_function_args(msg, tool_name="browser") == {}


def test_maybe_get_function_args_invalid_json():
    msg = _make_tool_message(recipient="browser.search", payload="not-json")
    assert maybe_get_function_args(msg, tool_name="browser") is None


def test_process_arguments_invalid_message():
    tool = SimpleBrowserTool(backend=YouComBackend(source="web"))
    msg = _make_tool_message(recipient=None, payload=None)
    with pytest.raises(ValueError, match="Invalid function arguments"):
        tool.process_arguments(msg)


def test_process_arguments_open_integer_id_default_cursor():
    tool = SimpleBrowserTool(backend=YouComBackend(source="web"))
    search_page = PageContents(
        url="http://example.com/search",
        title="Search",
        text="Results",
        urls={"0": "https://example.com/article1", "1": "https://example.com/article2"},
    )
    tool.tool_state.add_page(search_page)

    msg = _make_tool_message(recipient="browser.open", payload={"id": 0})
    args = tool.process_arguments(msg)

    assert args["id"] == 0
    assert args["url"] == "https://example.com/article1"


def test_process_arguments_open_string_url():
    tool = SimpleBrowserTool(backend=YouComBackend(source="web"))
    page = PageContents(
        url="http://example.com/home",
        title="Home",
        text="Welcome",
        urls={},
    )
    tool.tool_state.add_page(page)

    msg = _make_tool_message(
        recipient="browser.open", payload={"id": "https://docs.example.com"}
    )
    args = tool.process_arguments(msg)

    assert args["id"] == "https://docs.example.com"
    assert args["url"] == "https://docs.example.com"


def test_process_arguments_open_explicit_cursor():
    tool = SimpleBrowserTool(backend=YouComBackend(source="web"))
    page0 = PageContents(
        url="http://example.com/page0",
        title="Page 0",
        text="P0",
        urls={"0": "https://example.com/from-page0"},
    )
    page1 = PageContents(
        url="http://example.com/page1",
        title="Page 1",
        text="P1",
        urls={"0": "https://example.com/from-page1"},
    )
    tool.tool_state.add_page(page0)
    tool.tool_state.add_page(page1)

    msg = _make_tool_message(
        recipient="browser.open", payload={"cursor": 0, "id": 0}
    )
    args = tool.process_arguments(msg)

    assert args["cursor"] == 0
    assert args["id"] == 0
    assert args["url"] == "https://example.com/from-page0"


def test_process_arguments_find_resolves_page_url():
    tool = SimpleBrowserTool(backend=YouComBackend(source="web"))
    page = PageContents(
        url="https://example.com/guide",
        title="Guide",
        text="Detailed documentation",
        urls={},
    )
    tool.tool_state.add_page(page)

    msg = _make_tool_message(
        recipient="browser.find", payload={"pattern": "documentation"}
    )
    args = tool.process_arguments(msg)

    assert args["pattern"] == "documentation"
    assert args["url"] == "https://example.com/guide"
