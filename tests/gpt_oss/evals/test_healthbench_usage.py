from types import SimpleNamespace

from gpt_oss.evals.healthbench_eval import get_usage_dict


def test_get_usage_dict_supports_responses_usage() -> None:
    usage = SimpleNamespace(input_tokens=120, output_tokens=30, total_tokens=150)

    assert get_usage_dict(usage) == {
        "input_tokens": 120,
        "input_cached_tokens": None,
        "output_tokens": 30,
        "output_reasoning_tokens": None,
        "total_tokens": 150,
    }


def test_get_usage_dict_supports_chat_completions_usage() -> None:
    usage = SimpleNamespace(prompt_tokens=80, completion_tokens=20, total_tokens=100)

    assert get_usage_dict(usage) == {
        "input_tokens": 80,
        "input_cached_tokens": None,
        "output_tokens": 20,
        "output_reasoning_tokens": None,
        "total_tokens": 100,
    }


def test_get_usage_dict_handles_missing_usage() -> None:
    assert get_usage_dict(None) == {
        "input_tokens": None,
        "input_cached_tokens": None,
        "output_tokens": None,
        "output_reasoning_tokens": None,
        "total_tokens": None,
    }
