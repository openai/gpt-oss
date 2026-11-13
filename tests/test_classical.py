"""驗年界、書證及問答之銜接。"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from openai_harmony import HarmonyError, HarmonyEncodingName, load_harmony_encoding

from gpt_oss.classical import ClassicalCanon
from gpt_oss.classical_library import ClassicalLibrary
from gpt_oss.prepare_classical import compile_siku
from gpt_oss.responses_api.api_server import create_api_server


CATALOGUE = Path(__file__).parents[1] / "docs" / "classical_sources.json"


@pytest.fixture(scope="session")
def harmony_encoding():
    try:
        return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    except HarmonyError as exc:
        pytest.skip(f"真詞彙檔未備，無法驗完整串流：{exc}")


@pytest.fixture
def siku(tmp_path):
    destination = tmp_path / "siku.sqlite"
    assert compile_siku(CATALOGUE, destination) == 8
    return ClassicalLibrary(destination)


def test_cutoff_and_transcription_dates(siku):
    assert "天工開物" in siku.consult("水車", 1799)
    assert "布爾" not in siku.consult("布爾 logic", 1799)
    assert "布爾" in siku.consult("布爾 logic", 1899)
    assert "授時曆故" not in siku.consult("授時曆故", 1899)
    assert "General History" not in siku.consult("General History", 1899)
    assert siku.consult("", 1899) == ""
    assert siku.consult('" OR * --', 1899) == ""


@pytest.mark.parametrize("year", [0, 1900, 2026])
def test_invalid_cutoff(year):
    with pytest.raises(ValueError):
        ClassicalCanon(year)


def test_invalid_note_not_imported(tmp_path):
    data = {"sources": [{"status": "reviewed_notes", "title": "書",
                         "notes": [{"knowledge_year": 1987}]}]}
    catalogue = tmp_path / "bad.json"
    catalogue.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        compile_siku(catalogue, tmp_path / "bad.sqlite")


@pytest.mark.parametrize("user_input", [
    "請以 Python 模擬水車。",
    [{"type": "message", "role": "user", "content": "請以 Python 模擬水車。"}],
])
def test_prompt_assembly_without_vocab(siku, user_input):
    class ScrollReady(Exception):
        pass

    class RecordingEncoding:
        def render_conversation_for_completion(self, conversation, role):
            self.scroll = conversation.to_dict()
            raise ScrollReady

    encoding = RecordingEncoding()
    with TestClient(create_api_server(lambda *a: 0, encoding, ClassicalCanon(1799), siku)) as client:
        with pytest.raises(ScrollReady):
            client.post("/v1/responses", json={
                "input": user_input, "instructions": "簡述即可。",
            })
    messages = encoding.scroll["messages"]
    assert messages[0]["content"][0]["knowledge_cutoff"] == "1799-12"
    assert "西元1799年" in messages[0]["content"][0]["model_identity"]
    assert "簡述即可" in messages[1]["content"][0]["instructions"]
    assert "卷一乃粒" in messages[-1]["content"][0]["text"]
    assert "1987" not in messages[-1]["content"][0]["text"]


@pytest.mark.parametrize("stream", [False, True])
def test_policy_and_notes_reach_inference(harmony_encoding, mock_infer_token, siku, stream):
    prompts = []

    def infer(tokens, *args, **kwargs):
        prompts.append(harmony_encoding.decode(tokens))
        return mock_infer_token(tokens, *args, **kwargs)

    with TestClient(create_api_server(infer, harmony_encoding, ClassicalCanon(1799), siku)) as client:
        response = client.post("/v1/responses", json={
            "input": "請以 Python 模擬水車。", "instructions": "簡述即可。", "stream": stream,
        })
        assert response.status_code == 200
        assert "西元1799年" in prompts[0]
        assert "簡述即可" in prompts[0]
        assert "卷一乃粒" in prompts[0]
        assert "1987" not in prompts[0]


def test_continuation_preserves_policy(harmony_encoding, mock_infer_token, siku):
    prompts = []

    def infer(tokens, *args, **kwargs):
        prompts.append(harmony_encoding.decode(tokens))
        return mock_infer_token(tokens, *args, **kwargs)

    with TestClient(create_api_server(infer, harmony_encoding, library=siku)) as client:
        first = client.post("/v1/responses", json={"input": "談水車。"}).json()
        result = client.post("/v1/responses", json={
            "previous_response_id": first["id"],
            "input": [{"type": "message", "role": "user", "content": "再談布爾 logic。"}],
        })
        assert result.status_code == 200
        assert "西元1899年" in prompts[-1]
        assert "印頁 22–23" in prompts[-1]
