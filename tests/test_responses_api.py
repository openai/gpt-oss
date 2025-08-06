import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from openai_harmony import (
    DeveloperContent,
    HarmonyEncoding,
    HarmonyEncodingName,
    Role,
    load_harmony_encoding,
)

from gpt_oss.responses_api.api_server import create_api_server
from gpt_oss.responses_api.types import ResponsesRequest, Item, TextContentItem

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

# Create tokens that will generate a proper assistant response
fake_tokens = encoding.encode(
    "<|channel|>final<|message|>Hello! I'm here to help you.<|return|>", allowed_special="all"
)

token_queue = fake_tokens.copy()


def stub_infer_next_token(
    tokens: list[int], temperature: float = 0.0, new_request: bool = False
) -> int:
    global token_queue
    if len(token_queue) == 0:
        # Return a proper stop token instead of -1
        return encoding.stop_tokens_for_assistant_actions()[0]
    next_tok = token_queue.pop(0)
    return next_tok


def reset_token_queue():
    """Reset the token queue for each test"""
    global token_queue
    token_queue = fake_tokens.copy()


@pytest.fixture
def test_client():
    reset_token_queue()
    app = create_api_server(infer_next_token=stub_infer_next_token, encoding=encoding)
    return TestClient(app)


def test_health_check(test_client):
    reset_token_queue()
    response = test_client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": "Hello, world!",
        },
    )
    print(response.json())
    assert response.status_code == 200


def test_system_message_processing():
    """Test that system messages are correctly processed and combined with developer instructions."""
    reset_token_queue()
    app = create_api_server(infer_next_token=stub_infer_next_token, encoding=encoding)
    client = TestClient(app)
    
    # Test with system message
    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": [
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "Always provide accurate and up-to-date information."}]
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello, world!"}]
                }
            ],
            "instructions": "You are a helpful assistant."
        },
    )
    
    assert response.status_code == 200
    result = response.json()
    assert "output" in result
    assert result["status"] == "completed"


def test_system_message_without_instructions():
    """Test system message processing when no initial instructions are provided."""
    reset_token_queue()
    app = create_api_server(infer_next_token=stub_infer_next_token, encoding=encoding)
    client = TestClient(app)
    
    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": [
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "Always respond in a professional and courteous manner."}]
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello!"}]
                }
            ]
        },
    )
    
    assert response.status_code == 200
    result = response.json()
    assert "output" in result


def test_multiple_system_messages():
    """Test that multiple system messages are properly combined."""
    reset_token_queue()
    app = create_api_server(infer_next_token=stub_infer_next_token, encoding=encoding)
    client = TestClient(app)
    
    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": [
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "Provide concise and clear explanations."}]
                },
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "Use examples when helpful to illustrate concepts."}]
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Tell me about AI."}]
                }
            ],
            "instructions": "You are a helpful assistant."
        },
    )
    
    assert response.status_code == 200
    result = response.json()
    assert "output" in result


def test_system_message_with_string_content():
    """Test system message processing when content is a string instead of array."""
    reset_token_queue()
    app = create_api_server(infer_next_token=stub_infer_next_token, encoding=encoding)
    client = TestClient(app)
    
    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": [
                {
                    "type": "message",
                    "role": "system",
                    "content": "Always prioritize user safety and provide helpful guidance."
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello!"}]
                }
            ]
        },
    )
    
    assert response.status_code == 200
    result = response.json()
    assert "output" in result


def test_mixed_message_types():
    """Test processing of mixed message types including system, user, and assistant messages."""
    reset_token_queue()
    app = create_api_server(infer_next_token=stub_infer_next_token, encoding=encoding)
    client = TestClient(app)
    
    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": [
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "Be empathetic and understanding in your responses."}]
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hi there!"}]
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello! How can I help you?"}]
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Tell me a joke."}]
                }
            ]
        },
    )
    
    assert response.status_code == 200
    result = response.json()
    assert "output" in result


def test_no_system_messages():
    """Test that the API works correctly when no system messages are provided."""
    reset_token_queue()
    app = create_api_server(infer_next_token=stub_infer_next_token, encoding=encoding)
    client = TestClient(app)
    
    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello, world!"}]
                }
            ],
            "instructions": "You are a helpful assistant."
        },
    )
    
    assert response.status_code == 200
    result = response.json()
    assert "output" in result
