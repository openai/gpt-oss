import importlib
import sys
from types import ModuleType

import pytest


def test_tp_environment_is_parsed_as_integer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TP", "4")
    captured = {}
    fake_engine = object()

    class FakeLLM:
        def __new__(cls, **kwargs):
            captured.update(kwargs)
            return fake_engine

    class FakeSamplingParams:
        pass

    class FakeTokensPrompt:
        pass

    fake_vllm = ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams
    fake_inputs = ModuleType("vllm.inputs")
    fake_inputs.TokensPrompt = FakeTokensPrompt
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.inputs", fake_inputs)
    monkeypatch.delitem(
        sys.modules, "gpt_oss.responses_api.inference.vllm", raising=False
    )

    backend = importlib.import_module("gpt_oss.responses_api.inference.vllm")
    result = backend.load_model("checkpoint")

    assert result is fake_engine
    assert backend.TP == 4
    assert captured["model"] == "checkpoint"
    assert captured["tensor_parallel_size"] == 4
    assert isinstance(captured["tensor_parallel_size"], int)
