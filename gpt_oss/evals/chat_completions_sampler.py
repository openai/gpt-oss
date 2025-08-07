import time
from typing import Any

import openai
import structlog
from openai import OpenAI

from .types import MessageList, SamplerBase, SamplerResponse

OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."
OPENAI_SYSTEM_MESSAGE_CHATGPT = (
    "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
    + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
)


class ChatCompletionsSampler(SamplerBase):
    """Sample from a Chat Completions compatible API."""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        reasoning_model: bool = False,
        reasoning_effort: str | None = None,
        base_url: str = "http://localhost:8000/v1",
    ):
        self.client = OpenAI(base_url=base_url, timeout=24 * 60 * 60)
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_model = reasoning_model
        self.reasoning_effort = reasoning_effort
        self.image_format = "url"

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        trial = 0
        while True:
            try:
                if self.reasoning_model:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=message_list,
                        reasoning_effort=self.reasoning_effort,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=message_list,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )

                choice = response.choices[0]
                content = choice.message.content
                if getattr(choice.message, "reasoning", None):
                    message_list.append(
                        self._pack_message("assistant", choice.message.reasoning)
                    )

                if not content:
                    raise ValueError("OpenAI API returned empty response; retrying")
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                )
            except openai.BadRequestError as e:
                logger = structlog.get_logger()
                logger.error("Bad request error from OpenAI API", error=str(e))
                return SamplerResponse(
                    response_text="No response (bad request).",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
            ) as e:
                exception_backoff = min(
                    2**trial, 60
                )  # exponential backoff with max 60s
                logger = structlog.get_logger()
                logger.warning(
                    "API error, retrying",
                    trial=trial,
                    backoff_seconds=exception_backoff,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                time.sleep(exception_backoff)
                trial += 1
            except Exception as e:
                logger = structlog.get_logger()
                logger.error(
                    "Unexpected error during API call", error=str(e), trial=trial
                )
                raise  # Re-raise unexpected errors
            # unknown error shall throw exception
