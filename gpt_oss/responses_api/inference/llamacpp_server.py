"""
Inference with llama.cpp server /completion endpoint with raw token ids.
"""

import json
import threading
import time
from typing import Callable, Optional

import requests

EOS_TOKEN = 200002  # only used on hard timeout

# Tunables
POLL_INTERVAL_S = 0.01  # 10ms between buffer checks
CALL_MAX_WAIT_S = 0.250  # max time to block inside a single infer call
NO_TOKEN_TIMEOUT_S = 15.0  # overall inactivity timeout before emitting EOS
FIRST_BYTE_TIMEOUT_S = 30.0  # time to wait for first token before EOS

# Shared state
_token_buffer: list[int] = []
_buffer_lock = threading.Lock()
_stream_thread: Optional[threading.Thread] = None
_stream_done = threading.Event()
_stream_error: Optional[Exception] = None
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
    with _buffer_lock:
        _token_buffer = []
    _stream_done.clear()
    _stream_thread = None
    _stream_error = None
    _touch_progress()


def setup_model(checkpoint: str) -> Callable[[list[int], float, bool], int]:
    # For llama-server, checkpoint is the base URL (e.g., "http://localhost:8080")
    server_url = checkpoint if checkpoint.startswith("http") else f"http://{checkpoint}"

    def _start_stream(token_ids: list[int], temperature: float):
        def run():
            nonlocal temperature
            global _stream_error
            global _previous_request_tokens

            toks = []
            last_len = 0  # number of tokens already emitted

            try:
                url = f"{server_url}/completion"

                payload = {
                    "prompt": token_ids,
                    "stream": True,
                    "temperature": temperature,
                    "return_tokens": True,
                    "cache_prompt": True,  # Re-use KV cache for better performance
                    "n_predict": -1,  # Generate until EOS or stop condition
                }

                with requests.post(url, json=payload, stream=True, timeout=60) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line:
                            continue

                        # llama-server uses Server-sent events format
                        if line.startswith("data: "):
                            line = line[6:]  # Remove "data: " prefix

                        obj = json.loads(line)
                        chunk_tokens = obj.get('tokens')

                        if chunk_tokens is not None:
                            toks += chunk_tokens
                            if len(toks) > last_len:
                                new_toks = toks[last_len:]
                                with _buffer_lock:
                                    _token_buffer.extend(new_toks)
                                last_len = len(toks)
                                _touch_progress()

                        # Check if generation is complete
                        if obj.get("stop", False):
                            _token_buffer.append(EOS_TOKEN)
                            _touch_progress()
                            break

                _stream_done.set()

            except Exception as e:
                _stream_error = e
                _stream_done.set()

        t = threading.Thread(target=run, name="llama-server-stream", daemon=True)
        t.start()
        return t

    def infer_next_token(
        tokens: list[int], temperature: float = 0.0, new_request: bool = False
    ) -> int:
        """
        - Starts a new llama-server stream on new_request.
        - Forwards tokens as they arrive.
        - Only emits EOS_TOKEN if we exceed an inactivity timeout.
        """
        global _stream_thread

        if new_request:
            _reset_stream_state()
            _stream_thread = _start_stream(token_ids=tokens, temperature=temperature)
            # Wait for first byte within FIRST_BYTE_TIMEOUT_S (without emitting EOS early)
            start = _now()
            while _now() - start < FIRST_BYTE_TIMEOUT_S:
                with _buffer_lock:
                    if _token_buffer:
                        tok = _token_buffer.pop(0)
                        _touch_progress()
                        return tok
                if _stream_error is not None:
                    raise RuntimeError(f"llama-server stream error: {_stream_error!r}")
                # If llama-server finished instantly with no output, continue loop until timeout
                time.sleep(POLL_INTERVAL_S)
            # Hard first-byte timeout -> emit EOS so the server can stop this request
            return EOS_TOKEN

        if _stream_error is not None:
            raise RuntimeError(f"llama-server stream error: {_stream_error!r}")

        # Normal path: wait up to CALL_MAX_WAIT_S for a token to arrive
        wait_start = _now()
        while _now() - wait_start < CALL_MAX_WAIT_S:
            with _buffer_lock:
                if _token_buffer:
                    tok = _token_buffer.pop(0)
                    _touch_progress()
                    return tok
            # No token yet; if we've been idle too long overall, end with EOS
            if _now() - _last_progress_ts > NO_TOKEN_TIMEOUT_S:
                return EOS_TOKEN
            time.sleep(POLL_INTERVAL_S)

        # Still no token in this call slice. Do NOT send EOS unless we've timed out.
        if _now() - _last_progress_ts > NO_TOKEN_TIMEOUT_S:
            return EOS_TOKEN

        # Tell caller to call us again; block minimally by returning *nothing new*.
        # We must return an int; safest is to wait a tiny bit longer for a token.
        # If still none, keep returning only after short waits. Avoid EOS here.
        # One more short wait to reduce hot-looping:
        time.sleep(POLL_INTERVAL_S)
        with _buffer_lock:
            if _token_buffer:
                tok = _token_buffer.pop(0)
                _touch_progress()
                return tok

        # As a last resort for this call slice, return EOS only on true inactivity timeout.
        if _now() - _last_progress_ts > NO_TOKEN_TIMEOUT_S:
            return EOS_TOKEN

        # If we reach here, we still haven't got a token—ask the caller to call again soon.
        # Return a harmless token that the server will replace/ignore if your interface supports it.
        # If your interface does NOT allow a sentinel, keep the short-blocking behavior above.
        return (
            EOS_TOKEN if False else 0
        )  # replace `0` with a PAD/NOOP token your server ignores

    return infer_next_token

