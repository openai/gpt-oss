from gpt_oss.responses_api.types import DEFAULT_MAX_OUTPUT_TOKENS, ResponsesRequest


def test_explicit_null_server_fields_use_concrete_defaults() -> None:
    request = ResponsesRequest(
        input="hello",
        metadata=None,
        tools=None,
        max_output_tokens=None,
    )

    assert request.metadata == {}
    assert request.tools == []
    assert request.max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS


def test_non_null_server_fields_are_preserved() -> None:
    request = ResponsesRequest(
        input="hello",
        metadata={"key": "value"},
        tools=[],
        max_output_tokens=32,
    )

    assert request.metadata == {"key": "value"}
    assert request.tools == []
    assert request.max_output_tokens == 32
