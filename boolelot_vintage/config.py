"""Configuration and model profiles for historical language models."""
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class VintageConfig:
    model_id: str = "talkie-lm/talkie-1930-13b-it"
    backend: str = "talkie"
    period_start: int = 1800
    period_end: int = 1899
    max_new_tokens: int = 384
    temperature: float = 0.75
    top_p: float = 0.9
    repetition_penalty: float = 1.08
    device_map: str = "auto"
    torch_dtype: str = "auto"

    @classmethod
    def from_env(cls):
        return cls(
            model_id=os.getenv("BOOLELOT_MODEL", cls.model_id),
            backend=os.getenv("BOOLELOT_BACKEND", cls.backend),
            period_start=int(os.getenv("BOOLELOT_PERIOD_START", "1800")),
            period_end=int(os.getenv("BOOLELOT_PERIOD_END", "1899")),
            max_new_tokens=int(os.getenv("BOOLELOT_MAX_NEW_TOKENS", "384")),
        )


MODEL_PROFILES = {
    "victorian-small": {
        "model_id": "croqaz/vintage-LLM-340m-v1-base",
        "backend": "transformers",
        "note": "Small pre-1900 base model; limited instruction-following ability.",
    },
    "victorian-research": {
        "model_id": "talkie-lm/talkie-1930-13b-it",
        "backend": "talkie",
        "note": "Instruction-tuned pre-1931 model; high hardware requirements.",
    },
    "historical-retrieval": {
        "model_id": "Livingwithmachines/bert_1760_1900",
        "backend": "retrieval",
        "note": "Historical encoder for retrieval/classification, not generation.",
    },
}
