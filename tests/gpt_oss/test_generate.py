import sys
from types import ModuleType, SimpleNamespace

import gpt_oss.generate as generate_module


class StubTokenizer:
    eot_token = 999

    def encode(self, prompt: str) -> list[int]:
        return [1]

    def decode(self, tokens: list[int]) -> str:
        return "x"


def test_generate_passes_zero_limit_to_backend(monkeypatch) -> None:
    captured = {}

    class StubGenerator:
        def __init__(self, checkpoint, device):
            captured["checkpoint"] = checkpoint
            captured["device"] = device

        def generate(
            self,
            prompt_tokens,
            stop_tokens,
            temperature,
            max_tokens,
            return_logprobs,
        ):
            captured["max_tokens"] = max_tokens
            yield 7, -0.1

    fake_utils = ModuleType("gpt_oss.torch.utils")
    fake_utils.init_distributed = lambda: "device"
    fake_model = ModuleType("gpt_oss.torch.model")
    fake_model.TokenGenerator = StubGenerator
    monkeypatch.setitem(sys.modules, "gpt_oss.torch.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "gpt_oss.torch.model", fake_model)
    monkeypatch.setattr(generate_module, "get_tokenizer", lambda: StubTokenizer())

    args = SimpleNamespace(
        backend="torch",
        checkpoint="checkpoint",
        context_length=4096,
        tensor_parallel_size=1,
        prompt="hello",
        temperature=0.0,
        limit=0,
    )

    generate_module.main(args)

    assert captured["max_tokens"] == 0
