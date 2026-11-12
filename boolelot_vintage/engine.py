"""Inference adapters for vintage language models."""
from dataclasses import dataclass
from typing import Optional

from .config import VintageConfig
from .prompts import build_system_prompt


@dataclass
class Generation:
    text: str
    model_id: str
    backend: str


class VintageEngine:
    """Lazy-loading runtime; model weights download only on first use."""

    def __init__(self, config: Optional[VintageConfig] = None):
        self.config = config or VintageConfig.from_env()
        self._model = None
        self._tokenizer = None

    def _load_transformers(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install inference dependencies with: "
                "pip install -r requirements-vintage.txt"
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        dtype = self.config.torch_dtype
        if dtype != "auto" and hasattr(torch, dtype):
            dtype = getattr(torch, dtype)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
            device_map=self.config.device_map,
        )
        self._model.eval()

    def _generate_transformers(self, prompt: str) -> str:
        import torch
        if self._model is None:
            self._load_transformers()

        messages = [
            {"role": "system", "content": build_system_prompt(self.config)},
            {"role": "user", "content": prompt},
        ]
        if getattr(self._tokenizer, "chat_template", None):
            rendered = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            rendered = (
                f"System: {messages[0]['content']}\n\n"
                f"User: {prompt}\n\nAssistant:"
            )

        inputs = self._tokenizer(rendered, return_tensors="pt")
        device = getattr(self._model, "device", None)
        if device is not None:
            inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.temperature > 0,
                temperature=max(self.config.temperature, 1e-5),
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _generate_talkie(self, prompt: str) -> str:
        try:
            from talkie import Talkie, Message
        except ImportError as exc:
            raise RuntimeError(
                "Talkie backend selected. Install it with: "
                "pip install 'git+https://github.com/talkie-lm/talkie.git'"
            ) from exc

        if self._model is None:
            self._model = Talkie(self.config.model_id)
        messages = [
            Message(role="system", content=build_system_prompt(self.config)),
            Message(role="user", content=prompt),
        ]
        result = self._model.chat(
            messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_new_tokens,
        )
        return result.text.strip()

    def generate(self, prompt: str) -> Generation:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if self.config.period_start > self.config.period_end:
            raise ValueError("period_start must not exceed period_end")

        if self.config.backend == "talkie":
            text = self._generate_talkie(prompt.strip())
        elif self.config.backend == "transformers":
            text = self._generate_transformers(prompt.strip())
        else:
            raise ValueError(
                f"Unsupported generation backend: {self.config.backend!r}. "
                "Use 'transformers' or 'talkie'."
            )
        return Generation(text=text, model_id=self.config.model_id,
                          backend=self.config.backend)
