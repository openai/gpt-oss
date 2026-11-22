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

# search -> open a result by link id -> find on the current page -> answer.
# open and find omit `cursor`, so their parsed arguments carry no "url".
MODEL_TURNS = [
    '<|channel|>analysis to=browser.search code<|message|>{"query": "paris weather"}<|call|>',
    '<|channel|>analysis to=browser.open code<|message|>{"id": 0}<|call|>',
    '<|channel|>analysis to=browser.find code<|message|>{"pattern": "sunny"}<|call|>',
    '<|channel|>analysis to=browser.search code<|message|>{"query": "lyon weather"}<|call|>',
    "<|channel|>final<|message|>It is sunny.<|return|>",
]


@chz.chz
class FakeBackend(Backend):
    async def search(self, query, topn, session):
        return PageContents(
            url="",
            title=query,
            text="result 【0†Example】",
            urls={"0": "https://example.com/a"},
        )

    async def fetch(self, url, session):
        return PageContents(url=url, title="Example", text="Paris is sunny.", urls={})


def make_infer():
    turns = list(MODEL_TURNS)
    queue: list[int] = []

    def infer(tokens, temperature=0.0, new_request=False):
        if new_request or not queue:
            queue[:] = encoding.encode(turns.pop(0), allowed_special="all")
        return queue.pop(0)

    return infer


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_server, "YouComBackend", FakeBackend)
    app = create_api_server(infer_next_token=make_infer(), encoding=encoding)
    return TestClient(app)


BODY = {"model": "gpt-oss-120b", "input": "weather?", "tools": [{"type": "browser_search"}]}


def web_search_calls(output):
    return [item for item in output if item["type"] == "web_search_call"]


def test_non_streaming_response_keeps_open_and_find_calls(client):
    output = client.post("/v1/responses", json=BODY).json()["output"]

    assert [c["action"] for c in web_search_calls(output)] == [
        {"type": "search", "query": "paris weather"},
        {"type": "open_page", "url": None},
        {"type": "find", "pattern": "sunny", "url": None},
        {"type": "search", "query": "lyon weather"},
    ]


def test_completed_event_matches_streamed_web_search_calls(client):
    response = client.post("/v1/responses", json=BODY | {"stream": True})
    events = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    streamed = [
        e["item"]
        for e in events
        if e["type"] == "response.output_item.done"
        and e["item"]["type"] == "web_search_call"
    ]
    completed = web_search_calls(events[-1]["response"]["output"])

    assert len(streamed) == 4
    assert [(c["id"], c["action"]) for c in completed] == [
        (c["id"], c["action"]) for c in streamed
    ]
