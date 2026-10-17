from types import SimpleNamespace
from unittest.mock import Mock

from gpt_oss.evals.chat_completions_sampler import ChatCompletionsSampler


def _sampler_with_response(response) -> ChatCompletionsSampler:
    sampler = object.__new__(ChatCompletionsSampler)
    sampler.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=Mock(return_value=response))
        )
    )
    sampler.model = "test-model"
    sampler.system_message = None
    sampler.temperature = 1.0
    sampler.max_tokens = 16
    sampler.reasoning_model = False
    sampler.reasoning_effort = None
    return sampler


def test_reasoning_is_not_added_to_queried_prompt_history() -> None:
    usage = SimpleNamespace(total_tokens=10)
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="final answer",
                    reasoning="private reasoning",
                )
            )
        ],
        usage=usage,
    )
    sampler = _sampler_with_response(response)
    messages = [{"role": "user", "content": "question"}]

    result = sampler(messages.copy())

    assert result.response_text == "final answer"
    assert result.actual_queried_message_list == messages
    assert result.response_metadata == {
        "usage": usage,
        "reasoning": "private reasoning",
    }


def test_missing_reasoning_does_not_add_metadata_field() -> None:
    usage = SimpleNamespace(total_tokens=10)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="final answer"))],
        usage=usage,
    )
    sampler = _sampler_with_response(response)

    result = sampler([{"role": "user", "content": "question"}])

    assert result.response_metadata == {"usage": usage}
