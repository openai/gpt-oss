"""Regression tests for `output_index` bookkeeping on the streaming path.

The Responses API contract is that `output_index` is the position of the item
in `response.output`, and that it is identical across every event belonging to
the same item (`added`, the deltas, and `done`).
"""

import json

import chz
from fastapi.testclient import TestClient
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
)

from gpt_oss.responses_api import api_server
from gpt_oss.responses_api.api_server import create_api_server
from gpt_oss.tools.simple_browser.backend import Backend
from gpt_oss.tools.simple_browser.page_contents import PageContents, process_html

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def make_client(completion: str) -> TestClient:
    """A client whose model replays `completion` token by token."""
    token_queue = encoding.encode(completion, allowed_special="all").copy()

    def stub_infer_next_token(
        tokens: list[int], temperature: float = 0.0, new_request: bool = False
    ) -> int:
        return token_queue.pop(0)

    return TestClient(
        create_api_server(infer_next_token=stub_infer_next_token, encoding=encoding)
    )


def stream_events(client: TestClient, body: dict) -> list[dict]:
    body = {**body, "stream": True}
    with client.stream("POST", "/v1/responses", json=body) as response:
        assert response.status_code == 200
        return [
            json.loads(line[len("data: ") :])
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]


def assert_output_indices_consistent(
    events: list[dict], expected_item_types: list[str]
) -> None:
    final_output = None
    added = {}
    done = {}
    for event in events:
        if event["type"] == "response.completed":
            final_output = event["response"]["output"]
            continue
        output_index = event.get("output_index")
        if output_index is None:
            continue
        assert output_index >= 0, f"negative output_index on {event['type']}"
        if event["type"] == "response.output_item.added":
            added[event["item"]["id"]] = output_index
        elif event["type"] == "response.output_item.done":
            done[event["item"]["id"]] = output_index

    assert final_output is not None, "no response.completed event"
    assert [item["type"] for item in final_output] == expected_item_types
    positions = {item["id"]: i for i, item in enumerate(final_output)}

    for item_id, index in done.items():
        if item_id in added:
            assert added[item_id] == index, (
                f"output_item.added and output_item.done disagree for {item_id}"
            )
        assert positions[item_id] == index, (
            f"output_index {index} for {item_id} is not its position "
            f"{positions[item_id]} in response.output"
        )


def test_output_index_matches_response_output_for_reasoning_and_message():
    client = make_client(
        "<|channel|>analysis<|message|>Thinking<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Hey there<|return|>"
    )
    events = stream_events(client, {"model": "gpt-oss-120b", "input": "Hello!"})
    assert_output_indices_consistent(events, ["reasoning", "message"])


def test_output_index_matches_response_output_for_function_call():
    client = make_client(
        "<|channel|>analysis<|message|>Thinking<|end|>"
        "<|start|>assistant<|channel|>commentary to=functions.get_weather"
        '<|constrain|>json<|message|>{"city":"SF"}<|call|>'
    )
    events = stream_events(
        client,
        {
            "model": "gpt-oss-120b",
            "input": "What is the weather?",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
        },
    )
    assert_output_indices_consistent(events, ["reasoning", "function_call"])


@chz.chz(typecheck=True)
class StubBackend(Backend):
    source: str = chz.field(doc="Description of the backend source", default="web")

    async def search(self, query: str, topn: int, session) -> PageContents:
        return process_html(
            html="<html><body>result</body></html>",
            url="https://example.com",
            title="Example",
            display_urls=False,
        )

    async def fetch(self, url: str, session) -> PageContents:
        return await self.search(query="", topn=1, session=session)


def test_output_index_matches_response_output_for_web_search_call(monkeypatch):
    monkeypatch.setenv("BROWSER_BACKEND", "youcom")
    monkeypatch.setattr(api_server, "YouComBackend", StubBackend)
    client = make_client(
        "<|channel|>analysis<|message|>Thinking<|end|>"
        "<|start|>assistant<|channel|>analysis to=browser.search"
        '<|constrain|>json<|message|>{"query":"weather"}<|call|>'
        "<|channel|>final<|message|>Hey there<|return|>"
    )
    events = stream_events(
        client,
        {
            "model": "gpt-oss-120b",
            "input": "What is the weather?",
            "tools": [{"type": "browser_search"}],
        },
    )
    assert_output_indices_consistent(
        events, ["reasoning", "web_search_call", "message"]
    )
