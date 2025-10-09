"""
NOTE: this is not the most efficient way to use transformers. It's a simple implementation that infers
one token at a time to mimic the behavior of the Triton implementation.
"""

import os
from typing import Callable, List

# Transformers imports
from transformers import AutoModelForCausalLM, PreTrainedModel
import torch


DEFAULT_TEMPERATURE = 0.0
TP = os.environ.get("TP", 2)

def load_model(checkpoint: str):
    """
    Serve the model directly with the Auto API.
    """

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    return model


def get_infer_next_token(model: PreTrainedModel):
    """
    Return a callable with the same shape as the original triton implementation:
      infer_next_token(tokens: List[int], temperature: float, new_request: bool) -> int

    Implementation detail:
      - We issue a single-token generation with using model.generate
      - generate handles sampling (temperature=0 => greedy, otherwise, sampling).
    """

    def infer_next_token(
        tokens: List[int],
        temperature: float = DEFAULT_TEMPERATURE,
        new_request: bool = False, # kept for interface compatibility; unused here
    ) -> int:
        tokens = torch.tensor([tokens], dtype=torch.int64, device=model.device)
        output = model.generate(tokens, max_new_tokens=1, do_sample=temperature != 0, temperature=temperature)
        return output[0, -1].tolist()

    return infer_next_token


def setup_model(checkpoint: str) -> Callable[[List[int], float, bool], int]:
    model = load_model(checkpoint)
    infer_next_token = get_infer_next_token(model)
    return infer_next_token
