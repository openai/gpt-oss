"""Metal backend for :mod:`gpt_oss.responses_api`."""

from typing import Callable

from gpt_oss.metal import Context, Model


def setup_model(checkpoint: str) -> Callable[[list[int], float], int]:
    """Load the Metal model and return an inference function."""

    model = Model(checkpoint)
    context = Context(model)

    def lcp(cache: list[int], inp: list[int]) -> list[int]:
        i = 0
        max_len = min(len(cache), len(inp))
        while i < max_len and cache[i] == inp[i]:
            i += 1
        return cache[:i]

    tokens_so_far = []

    def infer_next_token(tokens: list[int], temperature: float = 0.0, new_request: bool = False) -> int:
        """Infer next token using incremental LCP caching when possible."""
        nonlocal tokens_so_far

        # Fast path: first call or explicitly new request.
        if new_request or not tokens_so_far:
            context.reset()
            for t in tokens:
                context.append(t)
            tokens_so_far = tokens.copy()
            context.process()
            return int(context.sample(temperature=temperature))

        # Longest common prefix length
        overlap = lcp(tokens_so_far, tokens)
        ol = len(overlap)
        prev_len = len(tokens_so_far)
        cur_len = len(tokens)

        diverged_midstream = (ol < prev_len) and (ol < cur_len)  # mismatch not at the end

        if diverged_midstream:
            # safest: rebuild
            context.reset()
            for t in tokens:
                context.append(t)
            tokens_so_far = tokens.copy()
            context.process()
            return int(context.sample(temperature=temperature))

        if cur_len > prev_len:
            # pure extension (good for KV reuse)
            extension = tokens[prev_len:]
            for t in extension:
                context.append(t)
            tokens_so_far = tokens.copy()
            context.process()
            return int(context.sample(temperature=temperature))

        if cur_len < prev_len:
            # truncation/backspace; easiest correct behavior is rebuild
            context.reset()
            for t in tokens:
                context.append(t)
            tokens_so_far = tokens.copy()
            context.process()
            return int(context.sample(temperature=temperature))

        # cur_len == prev_len and everything matches => no new tokens; just sample.
        return int(context.sample(temperature=temperature))

    return infer_next_token
