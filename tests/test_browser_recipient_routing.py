"""Tests for routing model tool calls to the built-in browser tool."""

import json

import chz
import pytest
from fastapi.testclient import TestClient
from openai_harmony import HarmonyEncodingName, load_harmony_encoding

import gpt_oss.responses_api.api_server as api_server
from gpt_oss.responses_api.api_server import create_api_server
from gpt_oss.tools.simple_browser.backend import Backend
from gpt_oss.tools.simple_browser.page_contents import PageContents

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

FINAL_MESSAGE = "<|channel|>final<|message|>done<|return|>"
STUB_URL = "https://example.com/result"


@chz.chz(typecheck=True)
class StubBackend(Backend):
    """Browser backend that never touches the network."""

    source: str = chz.field(doc="Description of the backend source")

    async def search(self, query: str, topn: int, session) -> PageContents:
        return PageContents(url=STUB_URL, text="stub results", title=query, urls={})

    async def fetch(self, url: str, session) -> PageContents:
        return PageContents(url=url, text="stub page", title="stub", urls={})


@pytest.fixture(autouse=True)
def stub_browser_backend(monkeypatch):
    monkeypatch.setattr(api_server, "YouComBackend", StubBackend)


def make_client(*scripts: str) -> TestClient:
    queue: list[int] = []
    for script in scripts:
        queue.extend(encoding.encode(script, allowed_special="all"))

    def infer_next_token(
        tokens: list[int], temperature: float = 0.0, new_request: bool = False
    ) -> int:
        assert queue, "stub inference ran out of tokens"
        return queue.pop(0)

    return TestClient(create_api_server(infer_next_token=infer_next_token, encoding=encoding))


def output_types(response) -> list[str]:
    return [item.get("type") for item in response.json()["output"]]


SEARCH_VIA_FUNCTIONS = (
    '<|channel|>commentary to=functions.search <|constrain|>json'
    '<|message|>{"query":"weather"}<|call|>'
)


def test_functions_search_falls_back_to_browser_tool():
    """`functions.search` should be routed to the browser tool.

    The reserved-name set held prefixed names (`browser.search`) but was
    compared against the bare name (`search`), so this fallback never fired and
    the call was reported as a plain function_call instead.
    """
    client = make_client(SEARCH_VIA_FUNCTIONS, FINAL_MESSAGE)
    response = client.post(
        "/v1/responses",
        json={"model": "gpt-oss-120b", "input": "hi", "tools": [{"type": "web_search"}]},
    )
    assert response.status_code == 200
    assert "web_search_call" in output_types(response)


def test_user_defined_search_function_is_not_hijacked():
    """A user function named `search` must win over the browser fallback."""
    client = make_client(SEARCH_VIA_FUNCTIONS, FINAL_MESSAGE)
    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": "hi",
            "tools": [
                {"type": "web_search"},
                {
                    "type": "function",
                    "name": "search",
                    "description": "my own search",
                    "parameters": {"type": "object", "properties": {}},
                },
            ],
        },
    )
    assert response.status_code == 200
    types = output_types(response)
    assert "function_call" in types
    assert "web_search_call" not in types


def test_browser_search_recipient_still_routes():
    """The non-fallback path must keep working."""
    client = make_client(
        '<|channel|>commentary to=browser.search <|constrain|>json'
        '<|message|>{"query":"weather"}<|call|>',
        FINAL_MESSAGE,
    )
    response = client.post(
        "/v1/responses",
        json={"model": "gpt-oss-120b", "input": "hi", "tools": [{"type": "web_search"}]},
    )
    assert response.status_code == 200
    assert "web_search_call" in output_types(response)


def test_annotation_index_starts_at_zero():
    """annotation_index was read after the append, so it started at 1."""
    client = make_client(
        '<|channel|>commentary to=browser.search <|constrain|>json'
        '<|message|>{"query":"weather"}<|call|>',
        "<|channel|>final<|message|>Rain is likely 【0†L1】<|return|>",
    )
    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": "hi",
            "stream": True,
            "tools": [{"type": "web_search"}],
        },
    ) as response:
        assert response.status_code == 200
        indices = []
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            event = json.loads(line[len("data:") :])
            if event.get("type") == "response.output_text.annotation.added":
                indices.append(event["annotation_index"])

    assert indices, "expected at least one annotation event"
    assert indices == list(range(len(indices))), (
        f"annotation_index should be 0-based and contiguous, got {indices}"
    )
