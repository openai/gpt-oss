"""Metal backend for :mod:`gpt_oss.responses_api`."""

from typing import Callable

from gpt_oss.metal import Context, Model


def setup_model(checkpoint: str) -> Callable[[list[int], float], int]:
    """Load the Metal model and return an inference function."""

    model = Model(checkpoint)
    context = Context(model)

    def infer_next_token(
        tokens: list[int], temperature: float = 0.0, new_request: bool = False
    ) -> int:
        """Infer next token using incremental LCP caching when possible."""

        # Context handles LCP caching internally; if `tokens` matches the
        # tokens in the KV cache, the KV cache is reused after reset+append.
        context.reset()
        for t in tokens:
            context.append(t)
        return int(context.sample(temperature=temperature))

    return infer_next_token
