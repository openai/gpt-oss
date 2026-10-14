import gpt_oss.responses_api.inference.ollama as ollama_backend


class FakeEncoding:
    def decode(self, tokens):
        return "prompt"


def _make_infer(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr(
        ollama_backend,
        "load_harmony_encoding",
        lambda name: FakeEncoding(),
    )
    monkeypatch.setattr(ollama_backend, "_now", lambda: clock["now"])
    infer = ollama_backend.setup_model("model")
    ollama_backend._reset_stream_state()
    return infer, clock


def test_normal_polling_returns_none_until_real_token_arrives(monkeypatch) -> None:
    infer, _clock = _make_infer(monkeypatch)

    def unexpected_sleep(_delay):
        raise AssertionError("infer_next_token must not block while waiting")

    monkeypatch.setattr(ollama_backend.time, "sleep", unexpected_sleep)

    assert infer([1, 2, 3], new_request=False) is None

    with ollama_backend._buffer_lock:
        ollama_backend._token_buffer.append(42)
    ollama_backend._stream_has_output.set()
    ollama_backend._touch_progress()

    token = infer([1, 2, 3], new_request=False)

    assert token == 42
    assert token != 0


def test_completed_stream_without_buffer_returns_eos(monkeypatch) -> None:
    infer, _clock = _make_infer(monkeypatch)
    ollama_backend._stream_done.set()

    token = infer([1, 2, 3], new_request=False)

    assert token == ollama_backend.EOS_TOKEN


def test_first_byte_timeout_returns_eos(monkeypatch) -> None:
    infer, clock = _make_infer(monkeypatch)
    clock["now"] = ollama_backend.FIRST_BYTE_TIMEOUT_S + 0.001

    token = infer([1, 2, 3], new_request=False)

    assert token == ollama_backend.EOS_TOKEN


def test_inactivity_timeout_after_output_returns_eos(monkeypatch) -> None:
    infer, clock = _make_infer(monkeypatch)
    ollama_backend._stream_has_output.set()
    ollama_backend._last_progress_ts = 0.0
    clock["now"] = ollama_backend.NO_TOKEN_TIMEOUT_S + 0.001

    token = infer([1, 2, 3], new_request=False)

    assert token == ollama_backend.EOS_TOKEN
