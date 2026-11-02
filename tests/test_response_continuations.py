import json

import pytest
from fastapi.testclient import TestClient

from gpt_oss.responses_api.api_server import create_api_server


@pytest.fixture
def continuation_client(harmony_encoding):
    output_tokens = harmony_encoding.encode(
        "<|channel|>final<|message|>Saved answer<|return|>",
        allowed_special="all",
    )
    prompts = []
    remaining = []

    def infer(tokens, temperature=0.0, new_request=False):
        if new_request:
            prompts.append(harmony_encoding.decode_utf8(tokens))
            remaining[:] = output_tokens
        return remaining.pop(0)

    with TestClient(create_api_server(infer, harmony_encoding)) as client:
        yield client, prompts


def completed_response(response, stream):
    assert response.status_code == 200
    if not stream:
        return response.json()
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[-1]["type"] == "response.completed"
    return events[-1]["response"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("previous_id", ["resp_missing", ""])
def test_unknown_previous_response_rejected_before_inference(
    continuation_client, stream, previous_id
):
    client, prompts = continuation_client
    response = client.post(
        "/v1/responses",
        json={
            "input": "Continue the earlier task",
            "previous_response_id": previous_id,
            "stream": stream,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == f"Previous response {previous_id} not found"
    assert prompts == []


@pytest.mark.parametrize("stream", [False, True])
def test_unstored_response_cannot_be_continued(continuation_client, stream):
    client, prompts = continuation_client
    first = completed_response(
        client.post("/v1/responses", json={"input": "Remember this", "store": False}),
        False,
    )
    prompts.clear()

    response = client.post(
        "/v1/responses",
        json={
            "input": "What did I ask you to remember?",
            "previous_response_id": first["id"],
            "stream": stream,
        },
    )

    assert response.status_code == 400
    assert first["id"] in response.json()["detail"]
    assert prompts == []


def test_unknown_previous_response_rejected_before_tool_setup(
    continuation_client, monkeypatch
):
    client, prompts = continuation_client

    def unexpected_tool_setup(*args, **kwargs):
        raise AssertionError("Tool setup should not run for an unavailable continuation")

    monkeypatch.setattr(
        "gpt_oss.responses_api.api_server.YouComBackend", unexpected_tool_setup
    )
    monkeypatch.setattr(
        "gpt_oss.responses_api.api_server.PythonTool", unexpected_tool_setup
    )
    monkeypatch.setenv("BROWSER_BACKEND", "youcom")
    response = client.post(
        "/v1/responses",
        json={
            "input": "Continue using the tools",
            "previous_response_id": "resp_missing",
            "tools": [{"type": "web_search"}, {"type": "code_interpreter"}],
        },
    )

    assert response.status_code == 400
    assert prompts == []


@pytest.mark.parametrize("stream", [False, True])
def test_stored_response_chain_preserves_history(continuation_client, stream):
    client, prompts = continuation_client
    previous_id = None
    for text in ("First question", "Second question", "Third question"):
        response = completed_response(
            client.post(
                "/v1/responses",
                json={
                    "input": text,
                    "previous_response_id": previous_id,
                    "store": True,
                    "stream": stream,
                },
            ),
            stream,
        )
        assert response["previous_response_id"] == previous_id
        previous_id = response["id"]

    assert len(prompts) == 3
    assert prompts[-1].count("First question") == 1
    assert prompts[-1].count("Second question") == 1
    assert prompts[-1].count("Third question") == 1
    assert prompts[-1].count("Saved answer") == 2
    assert prompts[-1].index("First question") < prompts[-1].index("Second question")
    assert prompts[-1].index("Second question") < prompts[-1].index("Third question")
