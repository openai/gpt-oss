import requests
from requests import Response, RequestException

from gpt_oss.utils.http import request_with_retry


def test_request_with_retry_success(monkeypatch):
    calls = {"count": 0}
    sleeps: list[float] = []

    def fake_request(method, url, **kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            raise RequestException("boom")
        resp = Response()
        resp.status_code = 200
        return resp

    def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("gpt_oss.utils.http.requests.request", fake_request)
    monkeypatch.setattr("gpt_oss.utils.http.time.sleep", fake_sleep)

    resp = request_with_retry("GET", "http://example.com", max_retries=3, backoff_factor=1)
    assert resp.status_code == 200
    assert calls["count"] == 3
    assert sleeps == [1, 2]


def test_request_with_retry_exhausts(monkeypatch):
    calls = {"count": 0}

    def always_fail(method, url, **kwargs):
        calls["count"] += 1
        raise RequestException("boom")

    monkeypatch.setattr("gpt_oss.utils.http.requests.request", always_fail)
    monkeypatch.setattr("gpt_oss.utils.http.time.sleep", lambda s: None)

    try:
        request_with_retry("GET", "http://example.com", max_retries=2, backoff_factor=0)
    except RequestException:
        pass
    else:
        assert False, "RequestException not raised"
    assert calls["count"] == 2
