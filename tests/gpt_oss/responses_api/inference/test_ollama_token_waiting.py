import json

import pytest

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


@pytest.mark.parametrize("old_event", ["tokens", "done", "error"])
def test_superseded_stream_cannot_change_current_request(monkeypatch, old_event):
    infer, _clock = _make_infer(monkeypatch)
    pending = []

    class DeferredThread:
        def __init__(self, *, target, **kwargs):
            self.target = target

        def start(self):
            pending.append(self.target)

    class Response:
        def __init__(self, old):
            self.old = old
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def raise_for_status(self):
            pass

        def iter_lines(self, **kwargs):
            if self.old:
                # The old HTTP worker is already running when a replacement starts.
                assert infer([2], new_request=True) is None
                if old_event == "error":
                    raise RuntimeError("obsolete stream failure")
                text = "old" if old_event == "tokens" else ""
            else:
                text = "new"
            yield json.dumps({"response": text, "done": True})

    responses = [Response(old=True), Response(old=False)]
    post_results = iter(responses)
    monkeypatch.setattr(ollama_backend.threading, "Thread", DeferredThread)
    monkeypatch.setattr(
        ollama_backend.requests, "post", lambda *a, **kw: next(post_results)
    )
    monkeypatch.setattr(
        FakeEncoding,
        "encode",
        lambda self, text, **kw: list(text.encode()),
        raising=False,
    )

    assert infer([1], new_request=True) is None
    pending[0]()
    assert responses[0].closed
    assert infer([2]) is None
    pending[1]()
    assert [infer([2]) for _ in range(3)] == list(b"new")
    assert infer([2]) == ollama_backend.EOS_TOKEN
    assert responses[1].closed


@pytest.mark.parametrize("stream", [False, True])
def test_api_waits_without_inventing_tokens(harmony_encoding, stream):
    from fastapi.testclient import TestClient

    from gpt_oss.responses_api.api_server import create_api_server

    produced = harmony_encoding.encode(
        "<|channel|>final<|message|>Only real output<|return|>", allowed_special="all"
    )
    pending = iter([None, None, *produced])
    request_flags = []

    def infer(tokens, temperature=0.0, new_request=False):
        request_flags.append(new_request)
        return next(pending)

    with TestClient(create_api_server(infer, harmony_encoding)) as client:
        response = client.post(
            "/v1/responses",
            json={"input": "test", "stream": stream, "max_output_tokens": 64},
        )
    assert response.status_code == 200
    if stream:
        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: {")
        ]
        completed = [event for event in events if event["type"] == "response.completed"]
        assert len(completed) == 1
        result = completed[0]["response"]
    else:
        result = response.json()
    texts = [
        part["text"]
        for item in result["output"]
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    ]
    assert texts == ["Only real output"]
    assert request_flags == [True] + [False] * (len(produced) + 1)
