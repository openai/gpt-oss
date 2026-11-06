from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import openai
import pytest

import gpt_oss.evals.chat_completions_sampler as chat_module
import gpt_oss.evals.responses_sampler as responses_module
from gpt_oss.evals.chat_completions_sampler import ChatCompletionsSampler
from gpt_oss.evals.responses_sampler import ResponsesSampler


class PermanentAPIStatusError(Exception):
    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


def _responses_sampler_raising(error: Exception) -> ResponsesSampler:
    sampler = object.__new__(ResponsesSampler)
    sampler.client = SimpleNamespace(
        responses=SimpleNamespace(create=Mock(side_effect=error))
    )
    sampler.model = "test-model"
    sampler.developer_message = None
    sampler.temperature = 1.0
    sampler.max_tokens = 16
    sampler.reasoning_model = False
    sampler.reasoning_effort = None
    return sampler


def _chat_sampler_raising(error: Exception) -> ChatCompletionsSampler:
    sampler = object.__new__(ChatCompletionsSampler)
    sampler.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=Mock(side_effect=error))
        )
    )
    sampler.model = "test-model"
    sampler.system_message = None
    sampler.temperature = 1.0
    sampler.max_tokens = 16
    sampler.reasoning_model = False
    sampler.reasoning_effort = None
    return sampler


def test_responses_sampler_surfaces_permanent_api_status_without_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = PermanentAPIStatusError(401)
    sampler = _responses_sampler_raising(error)
    sleep = Mock()
    monkeypatch.setattr(
        responses_module.openai, "APIStatusError", PermanentAPIStatusError
    )
    monkeypatch.setattr(responses_module.time, "sleep", sleep)

    with pytest.raises(PermanentAPIStatusError) as raised:
        sampler([{"role": "user", "content": "hello"}])

    assert raised.value is error
    sleep.assert_not_called()


def test_chat_sampler_surfaces_permanent_api_status_without_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = PermanentAPIStatusError(403)
    sampler = _chat_sampler_raising(error)
    sleep = Mock()
    monkeypatch.setattr(chat_module.openai, "APIStatusError", PermanentAPIStatusError)
    monkeypatch.setattr(chat_module.time, "sleep", sleep)

    with pytest.raises(PermanentAPIStatusError) as raised:
        sampler([{"role": "user", "content": "hello"}])

    assert raised.value is error
    sleep.assert_not_called()


@pytest.fixture(params=[ChatCompletionsSampler, ResponsesSampler])
def sdk_sampler(request, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-offline")
    sampler = request.param(model="test-model")
    if isinstance(sampler, ChatCompletionsSampler):
        resource = sampler.client.chat.completions
        success = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="ok", reasoning=None))
            ],
            usage=None,
        )
    else:
        resource = sampler.client.responses
        success = SimpleNamespace(output=[], output_text="ok", usage=None)
    create = Mock()
    monkeypatch.setattr(resource, "create", create)
    sleep = Mock()
    monkeypatch.setattr(chat_module.time, "sleep", sleep)
    try:
        yield sampler, create, sleep, success
    finally:
        sampler.client.close()


@pytest.mark.parametrize("status", [401, 403, 404, 422])
def test_real_sdk_permanent_errors_do_not_retry(sdk_sampler, status):
    sampler, create, sleep, success = sdk_sampler
    response = httpx.Response(
        status, request=httpx.Request("POST", "https://example.invalid")
    )
    error = openai.APIStatusError("permanent", response=response, body={})
    create.side_effect = [error, success]
    with pytest.raises(openai.APIStatusError) as raised:
        sampler([{"role": "user", "content": "test"}])
    assert raised.value is error
    assert create.call_count == 1
    sleep.assert_not_called()


@pytest.mark.parametrize("status", [408, 409, 429, 500, 503])
def test_real_sdk_transient_errors_retry_then_succeed(sdk_sampler, status):
    sampler, create, sleep, success = sdk_sampler
    response = httpx.Response(
        status, request=httpx.Request("POST", "https://example.invalid")
    )
    create.side_effect = [
        openai.APIStatusError("temporary", response=response, body={}),
        success,
    ]
    assert sampler([{"role": "user", "content": "test"}]).response_text == "ok"
    assert create.call_count == 2
    sleep.assert_called_once_with(1)


def test_real_sdk_connection_error_retries_then_succeeds(sdk_sampler):
    sampler, create, sleep, success = sdk_sampler
    error = openai.APIConnectionError(
        request=httpx.Request("POST", "https://example.invalid")
    )
    create.side_effect = [error, success]
    assert sampler([{"role": "user", "content": "test"}]).response_text == "ok"
    assert create.call_count == 2
    sleep.assert_called_once_with(1)
