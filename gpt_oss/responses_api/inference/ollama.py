"""
NOTE: this is a stitched together implementation that uses Ollama for inference. It's primarily used
for testing and development. It does not leverage any prompt caching or other optimizations and
can therefore be slow between turns.
"""

import json
import threading
import time
from typing import Callable, Optional

import requests
from openai_harmony import HarmonyEncodingName, load_harmony_encoding

EOS_TOKEN = 200002

# Tunables
NO_TOKEN_TIMEOUT_S = 15.0  # overall inactivity timeout before emitting EOS
FIRST_BYTE_TIMEOUT_S = 30.0  # time to wait for first token before EOS

# Shared state
_token_buffer: list[int] = []
_buffer_lock = threading.Lock()
_stream_thread: Optional[threading.Thread] = None
_stream_done = threading.Event()
_stream_has_output = threading.Event()
_stream_error: Optional[Exception] = None
_stream_started_ts: float = 0.0
_last_progress_ts: float = 0.0  # updated whenever we enqueue or dequeue tokens
_previous_request_tokens: list[int] = []


def lcp(cache: list[int], inp: list[int]) -> list[int]:
    i = 0
    max_len = min(len(cache), len(inp))
    while i < max_len and cache[i] == inp[i]:
        i += 1
    return cache[:i]


def _now():
    return time.monotonic()


def _touch_progress():
    global _last_progress_ts
    _last_progress_ts = _now()


def _reset_stream_state():
    global _token_buffer, _stream_thread, _stream_error
    global _stream_started_ts, _last_progress_ts

    with _buffer_lock:
        _token_buffer = []
    _stream_done.clear()
    _stream_has_output.clear()
    _stream_thread = None
    _stream_error = None
    now = _now()
    _stream_started_ts = now
    _last_progress_ts = now


def setup_model(
    checkpoint: str,
) -> Callable[[list[int], float, bool], Optional[int]]:
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    model_name = checkpoint

    def _start_stream(token_ids: list[int], temperature: float):
        prompt_text = encoding.decode(token_ids)

        def run():
            nonlocal prompt_text, temperature
            global _stream_error
            global _previous_request_tokens

            accum_text = ""
            last_len = 0  # number of tokens already emitted

            try:
                url = "http://localhost:11434/api/generate"

                payload = {
                    "model": model_name,
                    "prompt": prompt_text,
                    "stream": True,
                    "options": {"temperature": temperature},
                    "raw": True,
                }

                with requests.post(url, json=payload, stream=True, timeout=60) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        obj = json.loads(line)

                        if isinstance(obj.get("response"), str):
                            accum_text += obj["response"]
                            toks = encoding.encode(accum_text, allowed_special="all")
                            if len(toks) > last_len:
                                new_toks = toks[last_len:]
                                with _buffer_lock:
                                    _token_buffer.extend(new_toks)
                                last_len = len(toks)
                                _stream_has_output.set()
                                _touch_progress()

                        if obj.get("done", False):
                            with _buffer_lock:
                                _token_buffer.append(EOS_TOKEN)
                            _touch_progress()
                            break

                _stream_done.set()

            except Exception as e:
                _stream_error = e
                _stream_done.set()

        t = threading.Thread(target=run, name="ollama-stream", daemon=True)
        t.start()
        return t

    def infer_next_token(
        tokens: list[int], temperature: float = 0.0, new_request: bool = False
    ) -> Optional[int]:
        """
        - Starts a new Ollama stream on new_request.
        - Forwards only tokens produced by that stream.
        - Returns None while the stream is still active but no token is buffered.
        - Emits EOS_TOKEN when the stream completes or a timeout expires.
        """
        global _stream_thread

        if new_request:
            _reset_stream_state()
            _stream_thread = _start_stream(token_ids=tokens, temperature=temperature)

        if _stream_error is not None:
            raise RuntimeError(f"Ollama stream error: {_stream_error!r}")

        with _buffer_lock:
            if _token_buffer:
                tok = _token_buffer.pop(0)
                _touch_progress()
                return tok

        if _stream_done.is_set():
            return EOS_TOKEN

        now = _now()
        if not _stream_has_output.is_set():
            if now - _stream_started_ts > FIRST_BYTE_TIMEOUT_S:
                return EOS_TOKEN
        elif now - _last_progress_ts > NO_TOKEN_TIMEOUT_S:
            return EOS_TOKEN

        return None

    return infer_next_token
