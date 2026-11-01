import time

import pytest
from fastapi.testclient import TestClient
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
)

from gpt_oss.responses_api.api_server import create_api_server

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

fake_tokens = encoding.encode(
    "<|channel|>final<|message|>Hey there<|return|>", allowed_special="all"
)

token_queue = fake_tokens.copy()


def stub_infer_next_token(
    tokens: list[int], temperature: float = 0.0, new_request: bool = False
) -> int:
    global token_queue
    next_tok = token_queue.pop(0)
    if len(token_queue) == 0:
        token_queue = fake_tokens.copy()
    time.sleep(0.1)
    return next_tok


@pytest.fixture
def test_client():
    return TestClient(
        create_api_server(infer_next_token=stub_infer_next_token, encoding=encoding)
    )


def test_health_check(test_client):
    response = test_client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": "Hello, world!",
        },
    )
    print(response.json())
    assert response.status_code == 200


def _function_call(call_id, name="get_weather"):
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": "{}",
    }


def _function_call_output(call_id, output="21C sunny"):
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def test_duplicate_function_call_id_rejected(test_client):
    response = test_client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": [
                _function_call("call_0", "get_weather"),
                _function_call("call_0", "delete_file"),
                _function_call_output("call_0"),
            ],
        },
    )
    assert response.status_code == 422
    assert "Duplicate function call id call_0" in response.text


def test_distinct_function_call_ids_accepted(test_client):
    response = test_client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": [
                _function_call("call_0", "get_weather"),
                _function_call("call_1", "delete_file"),
                _function_call_output("call_0"),
                _function_call_output("call_1", "done"),
            ],
        },
    )
    assert response.status_code == 200


def test_unknown_function_call_output_rejected_as_422(test_client):
    response = test_client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": [_function_call_output("call_missing")],
        },
    )
    assert response.status_code == 422
    assert "not found" in response.text
