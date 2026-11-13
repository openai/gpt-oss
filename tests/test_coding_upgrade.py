"""驗現代程式特許與一八九九年界之分，不以提示自稱保證。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from openai_harmony import HarmonyEncodingName, HarmonyError, load_harmony_encoding

from gpt_oss.chronology import ChronologyGuard
from gpt_oss.classical import ClassicalCanon
from gpt_oss.coding_library import CodingLibrary
from gpt_oss.evals.coding_scope_eval import CASES
from gpt_oss.responses_api.api_server import create_api_server
from training.prepare_corpus import (CLASSICAL_CATALOGUE, CODING_CATALOGUE,
                                     build_corpus)
from training.verify_coding_examples import verify_all


@pytest.fixture(scope="module")
def harmony_encoding():
    try:
        return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    except HarmonyError as exc:
        pytest.skip(f"真詞彙檔未備，無法驗完整串流：{exc}")


def final_channel_inference(encoding, text: str):
    tokens = list(encoding.encode(
        f"<|channel|>final<|message|>{text}<|return|>", allowed_special="all"))
    queue: list[int] = []

    def infer(*_args, **kwargs) -> int:
        nonlocal queue
        if kwargs.get("new_request", False) or not queue:
            queue = tokens.copy()
        return queue.pop(0) if queue else tokens[-1]

    return infer


def sse_payloads(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        raw = line[6:]
        if raw == "[DONE]":
            continue
        events.append(json.loads(raw))
    return events


def final_text(response: dict) -> str:
    pieces = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") in ("output_text", "text"):
                pieces.append(part.get("text", ""))
    return "\n".join(pieces)


def test_guard_separates_code_literals_from_history():
    guard = ChronologyGuard(1899)
    strict = ChronologyGuard(1799)
    assert strict.classify_request("布爾在 1854 年提出的學說為何？").blocked
    assert not strict.classify_request("請談 1799 年以前之算學。").blocked
    assert guard.classify_request("請說明 1969 年人類登月的經過。").blocked
    assert guard.classify_request("第二次世界大戰的結果為何？").blocked
    assert not guard.classify_request("用 Python 解析 year = 1969 的資料。").blocked
    assert guard.classify_request("請說明二十世紀的戰爭，並用 Python 寫統計程式。").blocked
    assert guard.sanitize("某語言於 1991 年創立。", programming=True).blocked
    assert not guard.sanitize("```python\nyear = 1969\n```", programming=True).blocked
    assert "程式語法" in guard.refusal


def test_coding_reference_reaches_prompt_without_world_history():
    library = CodingLibrary()
    assert {"Python", "C++", "TypeScript", "SQL", "Rust"} <= set(library.languages)
    reference = library.consult("請用 Python 寫 dataclass 與型別檢查。")
    assert "符術準繩" in reference
    assert "dataclass" in reference
    assert not ChronologyGuard().reasons(reference, programming=True)

    class ScrollReady(Exception):
        pass

    class RecordingEncoding:
        def render_conversation_for_completion(self, conversation, role):
            self.scroll = conversation.to_dict()
            raise ScrollReady

    encoding = RecordingEncoding()
    app = create_api_server(lambda *a: 0, encoding, ClassicalCanon(),
                            coding_library=library, guard=ChronologyGuard())
    with TestClient(app) as client, pytest.raises(ScrollReady):
        client.post("/v1/responses", json={"input": "請用 Python 寫 dataclass 與型別檢查。"})
    messages = encoding.scroll["messages"]
    assert any("符術準繩" in entry.get("text", "")
               for message in messages for entry in message.get("content", []))
    assert all("1969" not in entry.get("text", "")
               for message in messages for entry in message.get("content", []))


def test_post_cutoff_request_never_calls_model(harmony_encoding, capsys):
    def forbidden_infer(*_args, **_kwargs):
        raise AssertionError("明確越界之問不得呼叫底座模型")

    guard = ChronologyGuard()
    app = create_api_server(forbidden_infer, harmony_encoding, ClassicalCanon(),
                            coding_library=None, guard=guard)
    with TestClient(app) as client:
        blocked = client.post("/v1/responses", json={
            "input": "請說明 1969 年人類登月的經過。", "stream": False,
            "max_output_tokens": 128,
        })
        assert blocked.status_code == 200
        assert final_text(blocked.json()) == guard.refusal
        assert blocked.json()["metadata"]["chronology_scope"]["blocked"] is True
        assert blocked.json()["metadata"]["chronology_guard"]["request_refused"] is True

        streamed = client.post("/v1/responses", json={
            "input": "請說明 1914 年爆發的戰爭與結果。", "stream": True,
            "max_output_tokens": 128,
        })
        assert streamed.status_code == 200
        events = sse_payloads(streamed.text)
        visible = []
        for event in events:
            if event.get("type") in ("response.output_text.delta", "response.output_text.done"):
                visible.append(event.get("delta") or event.get("text") or "")
            if event.get("type") == "response.completed":
                visible.append(final_text(event["response"]))
        assert visible
        assert all("登月" not in text and "1969" not in text for text in visible)
        assert any(guard.refusal in text for text in visible)
    capsys.readouterr()


def test_generated_post_cutoff_answer_is_screened_in_both_modes(harmony_encoding, capsys):
    bad_answer = "一九六九年，人乘火箭登月，此乃後世確鑿之史事。"
    guard = ChronologyGuard()
    app = create_api_server(final_channel_inference(harmony_encoding, bad_answer),
                            harmony_encoding, ClassicalCanon(),
                            coding_library=None, guard=guard)
    with TestClient(app) as client:
        plain = client.post("/v1/responses", json={
            "input": "請談史事。", "stream": False, "max_output_tokens": 192,
        })
        assert plain.status_code == 200
        assert final_text(plain.json()) == guard.refusal
        assert plain.json()["metadata"]["chronology_guard"]["blocked"] is True

        streamed = client.post("/v1/responses", json={
            "input": "請再談史事。", "stream": True, "max_output_tokens": 192,
        })
        assert streamed.status_code == 200
        events = sse_payloads(streamed.text)
        visible = []
        for event in events:
            if event.get("type") in ("response.output_text.delta", "response.output_text.done"):
                visible.append(event.get("delta") or event.get("text") or "")
            if event.get("type") == "response.completed":
                visible.append(final_text(event["response"]))
        assert visible
        assert all("登月" not in text and "一九六九" not in text for text in visible)
        assert any(guard.refusal in text for text in visible)
    capsys.readouterr()


def test_programming_answer_with_date_literal_is_not_erased(harmony_encoding, capsys):
    answer = ("茲以 Python 之型別與邊界驗法示之；年份只為輸入之數，非史事。\n\n"
              "```python\nyear = 1969\nassert isinstance(year, int)\n```")
    guard = ChronologyGuard()
    app = create_api_server(final_channel_inference(harmony_encoding, answer),
                            harmony_encoding, ClassicalCanon(),
                            coding_library=None, guard=guard)
    with TestClient(app) as client:
        response = client.post("/v1/responses", json={
            "input": "請用 Python 寫型別檢查，年份作輸入資料。",
            "stream": False, "max_output_tokens": 256,
        })
        assert response.status_code == 200
        text = final_text(response.json())
        assert "```python" in text and "year = 1969" in text
        assert guard.refusal not in text
        assert response.json()["metadata"]["chronology_guard"]["programming"] is True
    capsys.readouterr()


def test_curriculum_is_split_verified_and_rejects_history(tmp_path):
    manifest = build_corpus(CLASSICAL_CATALOGUE, CODING_CATALOGUE, tmp_path)
    assert manifest["train_count"] == 38
    assert manifest["eval_count"] == 10
    assert manifest["domains"]["train"] == {"historical": 6, "programming": 32}
    assert manifest["domains"]["eval"] == {"historical": 2, "programming": 8}
    assert not set(manifest["train_sources"]) & set(manifest["eval_sources"])
    train = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert any(row["knowledge_domain"] == "programming" and row["knowledge_year"] is None
               for row in train)

    payload = json.loads(CODING_CATALOGUE.read_text(encoding="utf-8"))
    payload["lessons"][0]["prompt"] = "請說明 1969 年人類登月，並順便寫 Python。"
    poisoned = tmp_path / "poisoned.json"
    poisoned.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="年界後"):
        build_corpus(CLASSICAL_CATALOGUE, poisoned, tmp_path / "poisoned-output")


def test_curated_examples_have_no_failed_checks():
    report = verify_all()
    assert report["examples"] == 24
    assert report["counts"]["failed"] == 0
    assert report["cards"] == 12


def test_live_evaluation_cases_respect_the_domain_split():
    guard = ChronologyGuard()
    coding = [case for case in CASES if case["category"] in ("coding", "mixed")]
    post_cutoff = [case for case in CASES if case["category"] == "post_cutoff"]
    assert len(coding) >= 8 and len(post_cutoff) >= 4
    for case in coding:
        # Coding prompts may name modern languages, but must not seed history.
        assert not guard.reasons(case["prompt"], programming=True)
    for case in post_cutoff:
        assert guard.classify_request(case["prompt"]).blocked
        assert case["forbidden"]


def test_system_context_does_not_reveal_a_post_cutoff_conversation_date():
    class ScrollReady(Exception):
        pass

    class RecordingEncoding:
        def render_conversation_for_completion(self, conversation, role):
            self.scroll = conversation.to_dict()
            raise ScrollReady

    encoding = RecordingEncoding()
    app = create_api_server(lambda *_args, **_kwargs: 0, encoding,
                            ClassicalCanon(1899), coding_library=None,
                            guard=ChronologyGuard(1899))
    with TestClient(app) as client, pytest.raises(ScrollReady):
        client.post("/v1/responses", json={"input": "請說明古代水車的原理。"})
    system = encoding.scroll["messages"][0]["content"][0]
    assert system["conversation_start_date"] == "1899-12-31"
    assert "今世編程" not in system["model_identity"]
    assert "不得自稱知道" in system["model_identity"]


def test_strict_historical_profile_rejects_live_web_search():
    class RecordingEncoding:
        def render_conversation_for_completion(self, conversation, role):
            raise AssertionError("browser requests must be rejected before prompt assembly")

    called = False

    def forbidden_infer(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("live research must not reach inference")

    app = create_api_server(forbidden_infer, RecordingEncoding(), ClassicalCanon(),
                            coding_library=None, guard=ChronologyGuard())
    with TestClient(app) as client:
        response = client.post("/v1/responses", json={
            "input": "請查詢最新資料。",
            "tools": [{"type": "web_search"}],
        })
    assert response.status_code == 400
    assert "Live web search is disabled" in response.json()["detail"]
    assert not called


def test_current_demo_corpus_is_not_eligible_for_candidate_training(tmp_path):
    from training.train_classical import (
        MIN_HISTORICAL_EVAL, MIN_HISTORICAL_TRAIN,
        MIN_PROGRAMMING_EVAL, MIN_PROGRAMMING_TRAIN,
    )

    manifest = build_corpus(CLASSICAL_CATALOGUE, CODING_CATALOGUE, tmp_path)
    assert manifest["domains"]["train"]["historical"] < MIN_HISTORICAL_TRAIN
    assert manifest["domains"]["train"]["programming"] < MIN_PROGRAMMING_TRAIN
    assert manifest["domains"]["eval"]["historical"] < MIN_HISTORICAL_EVAL
    assert manifest["domains"]["eval"]["programming"] < MIN_PROGRAMMING_EVAL
