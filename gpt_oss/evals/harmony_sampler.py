"""
Harmony Sampler - converts chat messages to Harmony tokens and sends to SGLang /generate endpoint.
"""
import json
import os
import threading
import time
from typing import Any

import requests
from transformers import AutoTokenizer
from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Role,
    Message,
    Conversation,
    SystemContent,
    DeveloperContent,
    ReasoningEffort,
)

from .types import MessageList, SamplerBase, SamplerResponse


# Map string reasoning effort to enum
REASONING_EFFORT_MAP = {
    "low": ReasoningEffort.LOW,
    "medium": ReasoningEffort.MEDIUM,
    "high": ReasoningEffort.HIGH,
}


class HarmonySampler(SamplerBase):
    """
    Sample from SGLang's /generate endpoint using Harmony tokenization.
    
    Converts chat messages to Harmony format, tokenizes them, and sends
    raw tokens to the /generate endpoint.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        max_tokens: int = 32768,
        reasoning_model: bool = False,
        reasoning_effort: str | None = None,
        base_url: str = "http://localhost:8080",
        top_p: float | None = None,
        top_k: int | None = None,
        dump_inputs_dir: str | None = None,
        decode_output_tokens: bool = False,
        timeout: int = 1800,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_model = reasoning_model
        self.reasoning_effort = reasoning_effort or "high"
        self.base_url = base_url.rstrip("/")
        self.top_p = top_p
        self.top_k = top_k
        self.image_format = "url"
        self.dump_inputs_file = dump_inputs_dir  # renamed but keeping param name for compatibility
        self.decode_output_tokens = decode_output_tokens
        self.timeout = timeout
        self._dump_lock = threading.Lock()
        
        # Load tokenizer for decoding tokens to text (always needed for HTML reports)
        print(f"Loading tokenizer for model: {model}")
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        print("Tokenizer loaded successfully")

        # Initialize dump file if specified
        if self.dump_inputs_file:
            # Create parent directory if needed
            dump_dir = os.path.dirname(self.dump_inputs_file)
            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)
            # Clear/create the file
            with open(self.dump_inputs_file, "w") as f:
                pass  # Create empty file
        
        # Load the Harmony encoding for gpt-oss models
        self.enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": role, "content": content}

    def _convert_to_harmony_messages(self, message_list: MessageList) -> list[Message]:
        """
        Convert chat messages (role/content dicts) to Harmony Message objects.
        """
        harmony_messages = []
        reasoning_effort_enum = REASONING_EFFORT_MAP.get(
            self.reasoning_effort.lower(), ReasoningEffort.HIGH
        )
        
        # Check if there's a system message, if not create a default one
        has_system = any(msg.get("role") == "system" for msg in message_list)
        assert not has_system, "System message not supported"
        
        if not has_system:
            # Create default system message with reasoning effort
            system_content = (
                SystemContent.new()
                .with_reasoning_effort(reasoning_effort_enum)
                .with_conversation_start_date("2025-09-30")
                .with_required_channels(["analysis", "commentary", "final"])
            )
            harmony_messages.append(
                Message.from_role_and_content(Role.SYSTEM, system_content)
            )
        
        for msg in message_list:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "developer":
                developer_content = DeveloperContent.new().with_instructions(content)
                harmony_messages.append(
                    Message.from_role_and_content(Role.DEVELOPER, developer_content)
                )
            elif role == "user":
                harmony_messages.append(
                    Message.from_role_and_content(Role.USER, content)
                )
            elif role == "assistant":
                harmony_messages.append(
                    Message.from_role_and_content(Role.ASSISTANT, content)
                )
            else:
                # Default to user role for unknown roles
                harmony_messages.append(
                    Message.from_role_and_content(Role.USER, content)
                )
        
        return harmony_messages

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        trial = 0
        while True:
            try:
                # Convert chat messages to Harmony format
                harmony_messages = self._convert_to_harmony_messages(message_list)
                
                # Create conversation
                convo = Conversation.from_messages(harmony_messages)
                
                # Tokenize for completion
                tokens = self.enc.render_conversation_for_completion(convo, Role.ASSISTANT)
                tokens_list = tokens.tolist() if hasattr(tokens, 'tolist') else list(tokens)
                
                # Decode tokens to text for HTML reports
                text_input = self.tokenizer.decode(tokens_list, skip_special_tokens=False)
                
                # Dump inputs if file is specified
                if self.dump_inputs_file:
                    dump_data = {
                        "input_tokens": tokens_list,
                        "num_tokens": len(tokens_list),
                        "text_input": text_input,
                        "original_messages": message_list,
                        "sampling_params": {
                            "temperature": self.temperature,
                            "max_new_tokens": self.max_tokens,
                            "top_p": self.top_p,
                            "top_k": self.top_k,
                        },
                    }
                    # Thread-safe append to JSONL file
                    with self._dump_lock:
                        with open(self.dump_inputs_file, "a") as f:
                            f.write(json.dumps(dump_data) + "\n")
                
                # Create de-tokenized message list for HTML reports
                detokenized_message_list = [
                    {"role": "user", "content": text_input}
                ]
                
                # Build sampling params
                sampling_params = {
                    "temperature": self.temperature,
                    "max_new_tokens": self.max_tokens,
                }
                if self.top_p is not None:
                    sampling_params["top_p"] = self.top_p
                if self.top_k is not None:
                    sampling_params["top_k"] = self.top_k
                
                # Send to SGLang /generate endpoint
                response = requests.post(
                    f"{self.base_url}/generate",
                    json={
                        "model": self.model,
                        "input_ids": tokens_list,
                        "sampling_params": sampling_params,
                    },
                    timeout=self.timeout,
                )
                
                if response.status_code != 200:
                    raise ValueError(f"Generate endpoint returned {response.status_code}: {response.text}")
                
                result = response.json()
                
                # Extract response text - optionally decode output tokens ourselves
                if self.decode_output_tokens and "output_ids" in result:
                    output_ids = result["output_ids"]
                    response_text = self.tokenizer.decode(output_ids, skip_special_tokens=False)
                else:
                    response_text = result.get("text", "")
                
                if not response_text:
                    raise ValueError("Generate endpoint returned empty response; retrying")
                
                return SamplerResponse(
                    response_text=response_text,
                    response_metadata={
                        "input_tokens": len(tokens_list),
                        "output_tokens": result.get("meta_info", {}).get("completion_tokens"),
                    },
                    actual_queried_message_list=detokenized_message_list,
                )
                
            except requests.exceptions.RequestException as e:
                exception_backoff = 2 ** trial
                print(
                    f"Request exception, wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
                if trial > 10:
                    return SamplerResponse(
                        response_text="No response (request failed).",
                        response_metadata={"error": str(e)},
                        actual_queried_message_list=message_list,
                    )
            except Exception as e:
                exception_backoff = 2 ** trial
                print(
                    f"Exception, wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
                if trial > 10:
                    return SamplerResponse(
                        response_text="No response (error).",
                        response_metadata={"error": str(e)},
                        actual_queried_message_list=message_list,
                    )
