"""Lazy inference adapters for Talkie and Transformers vintage generators."""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VintageResponse:
    text: str
    model_id: str
    provider: str
    finish_reason: str | None = None


class VintageEngine:
    """Load one selected generator lazily and expose a common generation API."""

    def __init__(
        self, model_key: str = "talkie-1930-it", *, device: str | None = None,
        cache_dir: str | None = None, max_new_tokens: int = 384,
        temperature: float = 0.7, top_p: float = 0.9,
    ):
        from .registry import get_profile
        self.profile = get_profile(model_key)
        if self.profile.task != "generation":
            raise ValueError(
                f"{model_key!r} is a retrieval model, not a text generator. "
                "Use HistLMRetriever for historical semantic retrieval."
            )
        self.device, self.cache_dir = device, cache_dir
        self.max_new_tokens, self.temperature, self.top_p = max_new_tokens, temperature, top_p
        self._model: Any = None
        self._tokenizer: Any = None

    def _load_talkie(self):
        try:
            from talkie import Talkie
        except ImportError as exc:
            raise RuntimeError(
                "Talkie is optional. Clone https://github.com/talkie-lm/talkie "
                "and run uv sync in that repository."
            ) from exc
        kwargs = {}
        if self.device:
            kwargs["device"] = self.device
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        self._model = Talkie(self.profile.model_id, **kwargs)

    def _load_transformers(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install optional dependencies with pip install -r requirements-vintage.txt."
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.profile.model_id, cache_dir=self.cache_dir
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.profile.model_id, device_map="auto", torch_dtype="auto",
            cache_dir=self.cache_dir,
        )
        self._model.eval()

    def generate(self, prompt: str, *, system: str | None = None) -> VintageResponse:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if self.profile.provider == "talkie":
            if self._model is None:
                self._load_talkie()
            if self.profile.instruction_tuned:
                from talkie import Message
                messages = []
                if system:
                    messages.append(Message(role="system", content=system))
                messages.append(Message(role="user", content=prompt.strip()))
                result = self._model.chat(
                    messages, temperature=self.temperature,
                    max_tokens=self.max_new_tokens, top_p=self.top_p,
                )
            else:
                prefix = f"{system}\n\n" if system else ""
                result = self._model.generate(
                    prefix + prompt.strip(), temperature=self.temperature,
                    max_tokens=self.max_new_tokens, top_p=self.top_p,
                )
            return VintageResponse(
                result.text.strip(), self.profile.model_id, "talkie",
                getattr(result, "finish_reason", None),
            )

        if self._model is None:
            self._load_transformers()
        rendered = f"{system}\n\n{prompt.strip()}" if system else prompt.strip()
        if getattr(self._tokenizer, "chat_template", None):
            rendered = self._tokenizer.apply_chat_template(
                [{"role": "system", "content": system or ""},
                 {"role": "user", "content": prompt.strip()}],
                tokenize=False, add_generation_prompt=True,
            )
        inputs = self._tokenizer(rendered, return_tensors="pt")
        device = getattr(self._model, "device", None)
        if device is not None:
            inputs = {key: value.to(device) for key, value in inputs.items()}
        import torch
        with torch.inference_mode():
            output = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5), top_p=self.top_p,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[-1]:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        return VintageResponse(text, self.profile.model_id, "transformers")
