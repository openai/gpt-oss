"""Metal backend for :mod:`gpt_oss.responses_api`."""

from typing import Callable

from gpt_oss.metal import Context, Model


# Tunables
MAX_OUTPUT_TOKENS = 100


def setup_model(checkpoint: str) -> Callable[[list[int], float], int]:
    """Load the Metal model and return an inference function."""

    model = Model(checkpoint)
    context = Context(model)

    seed = 0
    output_tokens = []

    def infer_next_token(
        tokens: list[int], temperature: float = 0.0, new_request: bool = False
    ) -> int:
        """Infer next token using incremental LCP caching when possible."""
        nonlocal output_tokens

        if new_request:
            output_tokens = []

        if len(output_tokens) == 0:
            # Context handles LCP caching internally; if `tokens` matches the
            # tokens in the KV cache, the KV cache is reused after reset+append.
            context.reset()
            for t in tokens:
                context.append(t)

            output_tokens = context.sample(max_output_tokens=MAX_OUTPUT_TOKENS,
                                           temperature=temperature,
                                           seed=seed)

        return int(output_tokens.pop(0))

    return infer_next_token
