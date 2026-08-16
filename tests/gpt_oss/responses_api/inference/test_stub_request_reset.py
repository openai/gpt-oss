import gpt_oss.responses_api.inference.stub as stub_backend


def test_new_request_restarts_fake_token_sequence(monkeypatch) -> None:
    monkeypatch.setattr(stub_backend.time, "sleep", lambda delay: None)
    stub_backend.token_queue = stub_backend.fake_tokens.copy()

    first = stub_backend.stub_infer_next_token([], new_request=False)
    second = stub_backend.stub_infer_next_token([], new_request=False)
    restarted = stub_backend.stub_infer_next_token([], new_request=True)

    assert first == stub_backend.fake_tokens[0]
    assert second == stub_backend.fake_tokens[1]
    assert restarted == stub_backend.fake_tokens[0]
