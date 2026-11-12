"""Curated registry of upstream vintage models and research assets."""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ModelProfile:
    key: str
    provider: str
    model_id: str
    task: Literal["generation", "retrieval", "catalogue"]
    period_start: int | None
    period_end: int | None
    instruction_tuned: bool
    notes: str
    upstream: str


MODEL_REGISTRY: dict[str, ModelProfile] = {
    "talkie-1930-it": ModelProfile(
        "talkie-1930-it", "talkie", "talkie-1930-13b-it", "generation",
        None, 1930, True,
        "Instruction-tuned 13B model trained on pre-1931 English text; not a strict pre-1900 model.",
        "https://github.com/talkie-lm/talkie",
    ),
    "talkie-1930-base": ModelProfile(
        "talkie-1930-base", "talkie", "talkie-1930-13b-base", "generation",
        None, 1930, False, "Pre-1931 base model; use completion rather than chat.",
        "https://github.com/talkie-lm/talkie",
    ),
    "vintage-llm-340m": ModelProfile(
        "vintage-llm-340m", "transformers", "croqaz/vintage-LLM-340m-v1-base",
        "generation", None, 1899, False,
        "Small pre-1900 base-model experiment; not an instruction-tuned assistant.",
        "https://github.com/croqaz/vintage-LLM",
    ),
    "histlm-bert-1760-1900": ModelProfile(
        "histlm-bert-1760-1900", "histlm", "Livingwithmachines/bert_1760_1900",
        "retrieval", 1760, 1900, False,
        "Historical BERT encoder for representation and retrieval; cannot generate chat responses.",
        "https://github.com/Living-with-machines/histLM",
    ),
    "histlm-bert-1890-1900": ModelProfile(
        "histlm-bert-1890-1900", "histlm", "Livingwithmachines/bert_1890_1900",
        "retrieval", 1890, 1900, False,
        "Time-sliced historical BERT encoder for retrieval and semantic comparison.",
        "https://github.com/Living-with-machines/histLM",
    ),
}


def get_profile(key: str) -> ModelProfile:
    try:
        return MODEL_REGISTRY[key]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown vintage model {key!r}. Available: {choices}") from exc
