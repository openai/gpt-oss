"""Tests for the `max_tokens` contract shared by the reference token generators.

`0` means "no limit". `gpt_oss.generate` used to translate its `--limit 0`
default into `None`, which the torch and triton loops compared against with
`<`, raising TypeError before generating a single token.
"""

import pytest

torch = pytest.importorskip("torch")

from gpt_oss.torch.model import TokenGenerator

VOCAB_SIZE = 16
FORCED_TOKEN = 7
NEVER_EMITTED = 3


class StubModel:
    """Model whose argmax is always `FORCED_TOKEN`."""

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(tokens.shape[0], VOCAB_SIZE)
        logits[:, FORCED_TOKEN] = 10.0
        return logits


def make_generator() -> TokenGenerator:
    """Build a TokenGenerator without loading a checkpoint."""
    generator = object.__new__(TokenGenerator)
    generator.device = torch.device("cpu")
    generator.model = StubModel()
    return generator


def test_max_tokens_none_means_no_limit():
    """`None` must not reach the `<` comparison in the loop bound."""
    tokens = list(
        make_generator().generate(
            [1, 2],
            stop_tokens=[FORCED_TOKEN],
            temperature=0.0,
            max_tokens=None,
        )
    )
    assert tokens == [FORCED_TOKEN]


def test_max_tokens_zero_means_no_limit():
    tokens = list(
        make_generator().generate(
            [1, 2],
            stop_tokens=[FORCED_TOKEN],
            temperature=0.0,
            max_tokens=0,
        )
    )
    assert tokens == [FORCED_TOKEN]


def test_positive_max_tokens_is_respected():
    """A real budget still caps generation when no stop token is hit."""
    tokens = list(
        make_generator().generate(
            [1, 2],
            stop_tokens=[NEVER_EMITTED],
            temperature=0.0,
            max_tokens=3,
        )
    )
    assert tokens == [FORCED_TOKEN] * 3


def test_generate_script_passes_the_limit_through_unchanged():
    """`gpt_oss.generate` must hand `--limit` to the backend as-is."""
    import inspect

    import gpt_oss.generate as generate_module

    source = inspect.getsource(generate_module.main)
    assert "max_tokens=args.limit" in source, (
        "generate.py should pass --limit straight through; translating 0 to None "
        "breaks the torch and triton loop bounds"
    )
