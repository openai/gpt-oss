import json

import pytest
from fastapi.testclient import TestClient
from openai_harmony import HarmonyEncodingName, load_harmony_encoding

from gpt_oss.responses_api.api_server import create_api_server

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def _make_client(script: str) -> tuple[TestClient, list[int]]:
    tokens = encoding.encode(script, allowed_special="all")
    queue = tokens.copy()
    calls = [0]

    def infer_next_token(
        input_tokens: list[int], temperature: float = 0.0, new_request: bool = False
    ) -> int:
        calls[0] += 1
        return queue.pop(0)

    client = TestClient(
        create_api_server(infer_next_token=infer_next_token, encoding=encoding),
        raise_server_exceptions=False,
    )
    return client, calls


def test_max_output_tokens_stops_at_exact_budget_and_marks_incomplete():
    client, calls = _make_client(
        "<|channel|>final<|message|>One two three four five<|return|>"
    )
    response = client.post(
        "/v1/responses",
        json={"model": "gpt-oss-120b", "input": "Count", "max_output_tokens": 5},
    )

    assert response.status_code == 200
    assert calls[0] == 5
    data = response.json()
    assert data["status"] == "incomplete"
    assert data["incomplete_details"] == {"reason": "max_output_tokens"}
    assert data["usage"]["output_tokens"] == 5
    assert len(data["output"]) == 1
    assert data["output"][0]["status"] == "incomplete"
    assert data["output"][0]["content"][0]["text"] == "One two"


def test_max_output_tokens_streaming_deltas_match_final_output():
    client, calls = _make_client(
        "<|channel|>final<|message|>One two three four five<|return|>"
    )
    deltas: list[str] = []
    completed_response = None

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": "Count",
            "max_output_tokens": 5,
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            event = json.loads(line[len("data: ") :])
            if event["type"] == "response.output_text.delta":
                deltas.append(event["delta"])
            elif event["type"] == "response.completed":
                completed_response = event["response"]

    assert calls[0] == 5
    assert completed_response is not None
    assert completed_response["status"] == "incomplete"
    assert completed_response["incomplete_details"] == {"reason": "max_output_tokens"}
    assert "".join(deltas) == completed_response["output"][0]["content"][0]["text"]


@pytest.mark.parametrize("max_tokens", [1, 2, 9, 10, 11, 12])
def test_max_output_tokens_truncating_inside_header_does_not_500(max_tokens: int):
    script = (
        "<|channel|>analysis<|message|>Let me think.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Answer here.<|return|>"
    )
    client, calls = _make_client(script)
    response = client.post(
        "/v1/responses",
        json={"model": "gpt-oss-120b", "input": "Hi", "max_output_tokens": max_tokens},
    )

    assert response.status_code == 200
    assert calls[0] == max_tokens
    data = response.json()
    assert data["status"] == "incomplete"
    assert data["incomplete_details"] == {"reason": "max_output_tokens"}
    if max_tokens <= 2:
        assert data["output"] == []
    else:
        assert len(data["output"]) == 1
        assert data["output"][0]["type"] == "reasoning"
        assert data["output"][0]["content"][0]["text"] == "Let me think."


def test_natural_stop_at_budget_boundary_stays_completed():
    client, calls = _make_client("<|channel|>final<|message|>Hello<|return|>")
    response = client.post(
        "/v1/responses",
        json={"model": "gpt-oss-120b", "input": "Hi", "max_output_tokens": 5},
    )

    assert response.status_code == 200
    assert calls[0] == 5
    data = response.json()
    assert data["status"] == "completed"
    assert data["incomplete_details"] is None
    assert data["output"][0]["content"][0]["text"] == "Hello"


def test_zero_max_output_tokens_does_not_infer():
    client, calls = _make_client("<|channel|>final<|message|>Hello<|return|>")
    response = client.post(
        "/v1/responses",
        json={"model": "gpt-oss-120b", "input": "Hi", "max_output_tokens": 0},
    )

    assert response.status_code == 200
    assert calls[0] == 0
    data = response.json()
    assert data["status"] == "incomplete"
    assert data["incomplete_details"] == {"reason": "max_output_tokens"}
    assert data["output"] == []
    assert data["usage"]["output_tokens"] == 0


def test_max_output_tokens_flushes_buffered_partial_citation_in_stream():
    client, calls = _make_client(
        "<|channel|>final<|message|>See \u30100\u2020L1\u3011<|return|>"
    )
    deltas: list[str] = []
    completed_response = None

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": "Cite",
            "tools": [{"type": "web_search"}],
            "max_output_tokens": 6,
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            event = json.loads(line[len("data: ") :])
            if event["type"] == "response.output_text.delta":
                deltas.append(event["delta"])
            elif event["type"] == "response.completed":
                completed_response = event["response"]

    assert calls[0] == 6
    assert completed_response is not None
    assert completed_response["status"] == "incomplete"
    assert "".join(deltas) == completed_response["output"][0]["content"][0]["text"]


@pytest.mark.parametrize("max_tokens", [6, 7, 8])
def test_max_output_tokens_after_final_partial_citation_flushes_before_item_done(
    max_tokens: int,
):
    script = (
        "<|channel|>final<|message|>See \u3010<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Later<|return|>"
    )
    client, calls = _make_client(script)
    added_ids: list[str] = []
    done_ids: set[str] = set()
    deltas_by_item: dict[str, list[str]] = {}
    completed_response = None

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": "Cite",
            "tools": [{"type": "web_search"}],
            "max_output_tokens": max_tokens,
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            event = json.loads(line[len("data: ") :])
            event_type = event["type"]
            if event_type == "response.output_item.added":
                added_ids.append(event["item"]["id"])
            elif event_type == "response.output_text.delta":
                item_id = event["item_id"]
                assert item_id in added_ids
                assert item_id not in done_ids
                deltas_by_item.setdefault(item_id, []).append(event["delta"])
            elif event_type == "response.output_item.done":
                done_ids.add(event["item"]["id"])
            elif event_type == "response.completed":
                completed_response = event["response"]

    assert calls[0] == max_tokens
    assert completed_response is not None
    assert completed_response["status"] == "incomplete"
    assert len(completed_response["output"]) == 1
    msg_item = completed_response["output"][0]
    assert added_ids == [msg_item["id"]]
    assert "".join(deltas_by_item[msg_item["id"]]) == msg_item["content"][0]["text"]
