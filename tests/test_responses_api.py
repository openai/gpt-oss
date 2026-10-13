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


def _function_call(call_id=None, name="get_weather"):
    item = {
        "type": "function_call",
        "name": name,
        "arguments": "{}",
    }
    if call_id is not None:
        item["call_id"] = call_id
    return item


def _function_call_output(call_id=None, output="21C sunny"):
    item = {
        "type": "function_call_output",
        "output": output,
    }
    if call_id is not None:
        item["call_id"] = call_id
    return item


def _assert_missing_call_id(response):
    assert response.status_code == 422
    assert any(
        error["type"] == "missing" and error["loc"][-1] == "call_id"
        for error in response.json()["detail"]
    )


def test_function_call_requires_explicit_call_id(test_client):
    response = test_client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": [_function_call()],
        },
    )

    _assert_missing_call_id(response)


def test_function_call_output_requires_explicit_call_id(test_client):
    response = test_client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": [
                _function_call("call_explicit"),
                _function_call_output(),
            ],
        },
    )

    _assert_missing_call_id(response)


def test_missing_call_ids_do_not_fabricate_binding(test_client):
    response = test_client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": [
                _function_call(name="get_weather"),
                _function_call(name="delete_file"),
                _function_call_output(output="21C sunny"),
            ],
        },
    )

    _assert_missing_call_id(response)


def _assert_invalid_call_id(response):
    assert response.status_code == 422
    assert any(
        error["loc"][-1] == "call_id"
        and error["type"] in {"string_too_short", "string_pattern_mismatch"}
        for error in response.json()["detail"]
    )


@pytest.mark.parametrize("call_id", ["", "   ", "\t\n"])
def test_function_call_rejects_blank_call_id(test_client, call_id):
    response = test_client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": [_function_call(call_id)],
        },
    )

    _assert_invalid_call_id(response)


@pytest.mark.parametrize("call_id", ["", "   ", "\t\n"])
def test_function_call_output_rejects_blank_call_id(test_client, call_id):
    response = test_client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": [
                _function_call("call_explicit"),
                _function_call_output(call_id),
            ],
        },
    )

    _assert_invalid_call_id(response)


def test_explicit_function_call_id_is_accepted(test_client):
    response = test_client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": [
                _function_call("call_explicit"),
                _function_call_output("call_explicit"),
            ],
        },
    )

    assert response.status_code == 200
