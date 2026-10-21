"""Regression test: a model tool-call to ``functions.{search,open,find}`` must be
routed to the built-in browser when the browser tool is enabled and the user has
not defined a function of that name."""

import json
from unittest import mock

from fastapi.testclient import TestClient
from openai_harmony import HarmonyEncodingName, load_harmony_encoding

from gpt_oss.responses_api.api_server import create_api_server
from gpt_oss.tools.simple_browser.simple_browser_tool import SimpleBrowserTool

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

# Phase 1: a tool call addressed via the functions.* namespace to the builtin browser.
# Phase 2: a terminating final message so the stream ends.
_TOOLCALL = (
    '<|channel|>commentary to=functions.search'
    '<|constrain|>json<|message|>{"query":"cats"}<|call|>'
)
_FINAL = "<|channel|>final<|message|>done<|return|>"
_SEQUENCE = (
    encoding.encode(_TOOLCALL, allowed_special="all")
    + encoding.encode(_FINAL, allowed_special="all")
)


async def _empty_process(self, message):
    # Avoid any network; the browser call events are emitted before this runs.
    if False:
        yield
    return


def test_functions_namespaced_browser_call_is_routed_to_browser():
    cursor = {"i": 0}

    def stub_infer_next_token(tokens, temperature=0.0, new_request=False):
        i = cursor["i"]
        if i < len(_SEQUENCE):
            cursor["i"] += 1
            return _SEQUENCE[i]
        return _SEQUENCE[-1]  # keep returning <|return|> to guarantee termination

    client = TestClient(
        create_api_server(infer_next_token=stub_infer_next_token, encoding=encoding)
    )

    with mock.patch.object(SimpleBrowserTool, "process", _empty_process):
        response = client.post(
            "/v1/responses",
            json={
                "model": "gpt-oss-120b",
                "input": "find cats",
                "stream": True,
                "tools": [{"type": "web_search"}],  # no user function named "search"
            },
        )

    web_search_events = []
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
        if "web_search_call" in event.get("type", ""):
            web_search_events.append(event["type"])

    # The browser must actually run: without the fix this list is empty because the
    # functions.* -> browser fallback is unreachable dead code.
    assert "response.web_search_call.in_progress" in web_search_events, web_search_events
