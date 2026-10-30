"""Regression test: streaming url_citation annotations must use a 0-based index."""

import json
from unittest import mock

from fastapi.testclient import TestClient
from openai_harmony import HarmonyEncodingName, load_harmony_encoding

from gpt_oss.responses_api.api_server import create_api_server
from gpt_oss.tools.simple_browser.simple_browser_tool import SimpleBrowserTool

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

_final_tokens = encoding.encode(
    "<|channel|>final<|message|>See cats and also dogs done<|return|>",
    allowed_special="all",
)

# Two citations at distinct positions so both survive the start_index dedup filter.
_ANNOTATIONS = [
    {"type": "url_citation", "start_index": 4, "end_index": 8,
     "url": "https://a.example", "title": "Foo"},
    {"type": "url_citation", "start_index": 18, "end_index": 22,
     "url": "https://b.example", "title": "Bar"},
]


def _fake_normalize_citations(self, text):
    return (text, [dict(a) for a in _ANNOTATIONS], False)


def test_streaming_annotation_index_is_zero_based():
    queue = _final_tokens.copy()

    def stub_infer_next_token(tokens, temperature=0.0, new_request=False):
        nonlocal queue
        if not queue:
            queue = _final_tokens.copy()
        return queue.pop(0)

    client = TestClient(
        create_api_server(infer_next_token=stub_infer_next_token, encoding=encoding)
    )

    with mock.patch.object(
        SimpleBrowserTool, "normalize_citations", _fake_normalize_citations
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "gpt-oss-120b",
                "input": "hi",
                "stream": True,
                "tools": [{"type": "web_search"}],
            },
        )

    indices = []
    for line in response.text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload:
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "response.output_text.annotation.added":
            indices.append(event.get("annotation_index"))

    # The first annotation must report index 0 (not 1); indices are 0-based,
    # matching output_index/content_index and the UrlCitation event default.
    assert indices == [0, 1], indices
