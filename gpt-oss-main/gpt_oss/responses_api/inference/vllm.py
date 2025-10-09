"""
NOTE: this is not the most efficient way to use vLLM. It's a simple implementation that infers 
one token at a time to mimic the behavior of the Triton implementation. 
"""

import os
from typing import Callable, List, Optional

# vLLM imports
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

DEFAULT_TEMPERATURE = 0.0
TP = os.environ.get("TP", 2)

def load_model(checkpoint: str):
    """
    Create the vLLM engine. We enable prefix caching so repeated prefixes
    across calls can reuse KV cache for faster prefill.
    """

    llm = LLM(
        model=checkpoint,
        tensor_parallel_size=TP,          # set >1 if you want TP across GPUs
        enable_prefix_caching=True,      # reuse KV for shared prefixes
        disable_log_stats=True,        # uncomment to quiet logs
    )

    return llm


def get_infer_next_token(llm: LLM):
    """
    Return a callable with the same shape as your original:
      infer_next_token(tokens: List[int], temperature: float, new_request: bool) -> int

    Implementation detail:
      - We issue a single-token generation with TokensPrompt(prompt_token_ids=tokens).
      - vLLM handles sampling (temperature=0 => greedy).
      - With enable_prefix_caching=True, the shared prefix prefill can be reused
        across calls that share the same prefix.
    """

    # Maintain compatibility with your previous closure signature.
    def infer_next_token(
        tokens: List[int],
        temperature: float = DEFAULT_TEMPERATURE,
        new_request: bool = False,  # kept for interface compatibility; unused here
    ) -> int:
        if not tokens:
            raise ValueError("tokens must contain at least one input token id")

        sampling = SamplingParams(
            temperature=float(temperature),
            max_tokens=1,            # we only want the next token
            n=1,                     # single continuation
            # You can expose/enable more controls here (top_p, top_k, etc.)
        )

        # Provide token IDs directly (no re-tokenization).
        outputs = llm.generate(
            TokensPrompt(prompt_token_ids=tokens),
            sampling_params=sampling,
        )

        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM returned empty outputs")

        gen = outputs[0].outputs[0]
        if not gen.token_ids:
            # If the model immediately finished (e.g., EOS), decide how you'd like
            # to signal that. Here we raise; you could also return an EOS id.
            raise RuntimeError("No next token was generated (possibly EOS).")

        next_tok = int(gen.token_ids[0])
        return next_tok

    return infer_next_token


def setup_model(checkpoint: str) -> Callable[[List[int], float, bool], int]:
    llm = load_model(checkpoint)
    infer_next_token = get_infer_next_token(llm)
    return infer_next_token
