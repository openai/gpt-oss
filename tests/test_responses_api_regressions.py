"""Regression tests for crashes in the Responses API server.

Each test here corresponds to a request that previously returned a 500 (or raised
out of the endpoint entirely) rather than a well-formed response.
"""

import chz
import pytest
from fastapi.testclient import TestClient
from openai_harmony import HarmonyEncodingName, load_harmony_encoding

import gpt_oss.responses_api.api_server as api_server
from gpt_oss.responses_api.api_server import create_api_server
from gpt_oss.tools.simple_browser.backend import Backend
from gpt_oss.tools.simple_browser.page_contents import PageContents

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

# A well-formed final message, used to let the model "finish" after a tool call.
FINAL_MESSAGE = "<|channel|>final<|message|>done<|return|>"


@chz.chz(typecheck=True)
class StubBackend(Backend):
    """Browser backend that never touches the network."""

    source: str = chz.field(doc="Description of the backend source")

    async def search(self, query: str, topn: int, session) -> PageContents:
        return PageContents(url="", text="stub results", title=query, urls={})

    async def fetch(self, url: str, session) -> PageContents:
        return PageContents(url=url, text="stub page", title="stub", urls={})


@pytest.fixture(autouse=True)
def stub_browser_backend(monkeypatch):
    monkeypatch.setattr(api_server, "YouComBackend", StubBackend)


def make_client(*scripts: str) -> TestClient:
    """Build a client whose model emits `scripts` back to back, token by token."""
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


def test_unknown_browser_function_does_not_crash():
    """An unrecognized `browser.*` recipient used to raise UnboundLocalError.

    `web_search_call_id` was only assigned when the recipient mapped to a known
    action, but the in_progress event referencing it was emitted unconditionally.
    """
    client = make_client(
        '<|channel|>commentary to=browser.foo <|constrain|>json<|message|>{"x":1}<|call|>',
        FINAL_MESSAGE,
    )
    response = client.post(
        "/v1/responses",
        json={"model": "gpt-oss-120b", "input": "hi", "tools": [{"type": "web_search"}]},
    )
    assert response.status_code == 200
    # No web_search_call item, because there is no action to describe.
    assert "web_search_call" not in output_types(response)


def test_malformed_browser_arguments_do_not_crash():
    """Non-JSON tool arguments used to raise ValueError out of the stream.

    `process_arguments` is guarded in `generate_response` but was unguarded on
    the streaming path.
    """
    client = make_client(
        "<|channel|>commentary to=browser.search <|constrain|>json<|message|>not json<|call|>",
        FINAL_MESSAGE,
    )
    response = client.post(
        "/v1/responses",
        json={"model": "gpt-oss-120b", "input": "hi", "tools": [{"type": "web_search"}]},
    )
    assert response.status_code == 200


def test_well_formed_browser_search_still_emits_web_search_call():
    """Guard against over-correcting the two tests above."""
    client = make_client(
        '<|channel|>commentary to=browser.search <|constrain|>json<|message|>{"query":"weather"}<|call|>',
        FINAL_MESSAGE,
    )
    response = client.post(
        "/v1/responses",
        json={"model": "gpt-oss-120b", "input": "hi", "tools": [{"type": "web_search"}]},
    )
    assert response.status_code == 200
    assert "web_search_call" in output_types(response)


def test_null_tools_with_instructions():
    """`tools` is Optional, so an explicit null must not be iterated."""
    client = make_client(FINAL_MESSAGE)
    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": "hi",
            "instructions": "be nice",
            "tools": None,
        },
    )
    assert response.status_code == 200


def test_null_max_output_tokens():
    """`max_output_tokens` is Optional and must not reach the `>=` comparison."""
    client = make_client(FINAL_MESSAGE)
    response = client.post(
        "/v1/responses",
        json={"model": "gpt-oss-120b", "input": "hi", "max_output_tokens": None},
    )
    assert response.status_code == 200
