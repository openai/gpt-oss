import pandas as pd
import pytest

from gpt_oss.evals import aime_eval, gpqa_eval


def test_aime_zero_examples_produces_empty_subset(monkeypatch) -> None:
    frame = pd.DataFrame([{"question": "q", "answer": "1"}])
    monkeypatch.setattr(aime_eval.pandas, "read_json", lambda *args, **kwargs: frame)

    evaluation = aime_eval.AIME25Eval(num_examples=0)

    assert evaluation.examples == []


def test_aime_rejects_negative_example_count(monkeypatch) -> None:
    frame = pd.DataFrame([{"question": "q", "answer": "1"}])
    monkeypatch.setattr(aime_eval.pandas, "read_json", lambda *args, **kwargs: frame)

    with pytest.raises(ValueError, match="non-negative"):
        aime_eval.AIME25Eval(num_examples=-1)


def test_gpqa_zero_examples_produces_empty_subset(monkeypatch) -> None:
    frame = pd.DataFrame(
        [
            {
                "Question": "q",
                "Correct Answer": "a",
                "Incorrect Answer 1": "b",
                "Incorrect Answer 2": "c",
                "Incorrect Answer 3": "d",
            }
        ]
    )
    monkeypatch.setattr(gpqa_eval.pandas, "read_csv", lambda *args, **kwargs: frame)

    evaluation = gpqa_eval.GPQAEval(num_examples=0)

    assert evaluation.examples == []


def test_gpqa_rejects_negative_example_count(monkeypatch) -> None:
    frame = pd.DataFrame(
        [
            {
                "Question": "q",
                "Correct Answer": "a",
                "Incorrect Answer 1": "b",
                "Incorrect Answer 2": "c",
                "Incorrect Answer 3": "d",
            }
        ]
    )
    monkeypatch.setattr(gpqa_eval.pandas, "read_csv", lambda *args, **kwargs: frame)

    with pytest.raises(ValueError, match="non-negative"):
        gpqa_eval.GPQAEval(num_examples=-1)
