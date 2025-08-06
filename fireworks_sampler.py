#!/usr/bin/env python3
"""
Fireworks AI sampler for gpt-oss evaluations.
This creates a sampler that works with Fireworks AI API for running evaluations.
"""
import os
import time
from typing import Any

import openai
from openai import OpenAI

from gpt_oss.evals.types import MessageList, SamplerBase, SamplerResponse


class FireworksSampler(SamplerBase):
    """
    Sample from Fireworks AI API
    """

    def __init__(
        self,
        model: str = "accounts/fireworks/models/gpt-oss-120b",
        system_message: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ):
        # Use Fireworks API configuration
        api_key = os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError("FIREWORKS_API_KEY environment variable is required")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.fireworks.ai/inference/v1"
        )
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        
        trial = 0
        while True:
            try:
                # Prepare API call parameters
                params = {
                    "model": self.model,
                    "messages": message_list,
                    "temperature": self.temperature,
                }
                
                # Add max_tokens if specified
                if self.max_tokens is not None:
                    params["max_tokens"] = self.max_tokens
                
                # Add reasoning effort if specified (for gpt-oss models)
                if self.reasoning_effort is not None:
                    params["reasoning_effort"] = self.reasoning_effort
                
                response = self.client.chat.completions.create(**params)
                choice = response.choices[0]
                message = choice.message
                
                # For gpt-oss models, the actual response might be in reasoning_content
                content = message.content
                if content is None and hasattr(message, 'reasoning_content'):
                    content = message.reasoning_content
                
                if content is None:
                    print("Warning: Fireworks API returned empty content")
                    raise ValueError("Fireworks API returned empty response; retrying")
                
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                )
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    response_text="No response (bad request).",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = 2**trial  # exponential back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
                if trial > 5:  # Max 5 retries
                    raise e


if __name__ == "__main__":
    # Test the sampler
    sampler = FireworksSampler()
    test_messages = [{"role": "user", "content": "What is 2+2?"}]
    response = sampler(test_messages)
    print("Response:", response.response_text)