import datetime
import os
from typing import Callable

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.distributed as dist

from gpt_oss.triton.model import Cache, ModelConfig, Transformer

DEFAULT_TEMPERATURE = 0.0
CONTEXT = 16_384
CONCURRENT_SESSIONS = 1

rank = int(
    os.environ.get("RANK", 0)
)  # set this env var to another value to run on other GPUs


def load_model(checkpoint: str):
    print(f"[{rank}] loading model...")

    torch.cuda.set_device(rank)
    torch.set_grad_enabled(False)
    device = torch.device(f"cuda:{rank}")

    # Load model
    model = Transformer.from_checkpoint(checkpoint, device=device)

    print(f"[{rank}] loaded")
    return model, device


def get_infer_next_token(model, device):
    caches = [
        Cache(CONCURRENT_SESSIONS, CONTEXT, model.config.num_key_value_heads)
        for _ in range(len(model.block))
    ]
    # offsets = torch.zeros(CONCURRENT_SESSIONS, dtype=torch.int32, device=device) # TBD
    input_token = torch.zeros(
        1, dtype=torch.int32, device=device
    )  # add concurrent sessions support
    tokens_so_far = []

    model.prefill(torch.zeros(1, 4, dtype=torch.int32, device=device), caches)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits = model(input_token[None, :], caches=caches)[0]

    def lcp(cache: list[int], inp: list[int]) -> list[int]:
        i = 0
        max_len = min(len(cache), len(inp))
        while i < max_len and cache[i] == inp[i]:
            i += 1
        return cache[:i]

    def sample_next_token(
        logits: torch.Tensor, temperature: float = DEFAULT_TEMPERATURE
    ) -> int:
        """Executed only on rank 0."""
        if temperature == 0.0:
            return torch.argmax(logits[-1, :], dim=-1).item()
        probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
        return torch.multinomial(probs[-1, :], num_samples=1).item()

    @torch.inference_mode()
    def infer_next_token(
        tokens: list[int],
        temperature: float = DEFAULT_TEMPERATURE,
        new_request: bool = False,
    ) -> int:
        nonlocal tokens_so_far
        tokens_so_far = lcp(tokens_so_far, tokens)
        for cache in caches:
            cache.truncate(len(tokens_so_far))
        all_tokens = tokens  # for pdb
        tokens = tokens[len(tokens_so_far) :]

        if len(tokens) > 1:
            model.prefill(
                torch.as_tensor(tokens[:-1], dtype=torch.int32, device=device)[None, :],
                caches,
            )

        if len(tokens) == 0:
            breakpoint()

        input_token[-1] = tokens[-1]
        graph.replay()

        # decide next token on rank‑0
        next_tok = sample_next_token(logits, temperature=temperature)

        return next_tok

    return infer_next_token


def setup_model(checkpoint: str) -> Callable[[list[int], float], int]:
    model, device = load_model(checkpoint)
    infer_next_token = get_infer_next_token(model, device)
    return infer_next_token
