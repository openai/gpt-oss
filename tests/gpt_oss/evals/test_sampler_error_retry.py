from types import SimpleNamespace
from unittest.mock import Mock

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
    monkeypatch.setattr(responses_module.openai, "APIStatusError", PermanentAPIStatusError)
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
