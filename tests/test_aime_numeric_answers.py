"""Offline regressions for numeric values accepted by the AIME evaluator."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpt_oss.evals.aime_eval import AIME25Eval, extract_boxed_text
from gpt_oss.evals.types import SamplerResponse


INVALID_ANSWERS = [
    ("The answer is -42.", 42),
    ("The answer is −42.", 42),
    ("The answer is 4.2.", 2),
    ("The answer is .42.", 42),
    ("The answer is 1e3.", 3),
    ("The answer is 2e-3.", 3),
    ("The answer is 1,234.", 234),
    ("The answer is 1/42.", 42),
    (r"The answer is \boxed{1,234}.", 234),
    (r"The answer is \boxed{12,42}.", 42),
    (r"The answer is \framebox{12,42}.", 42),
]


class SyntheticSampler:
    def __init__(self, text):
        self.text = text

    def _pack_message(self, **message):
        return message

    def __call__(self, messages):
        return SamplerResponse(self.text, messages, {})


def evaluate(monkeypatch, text, correct):
    # Both dataset reads are intercepted before the evaluator is constructed.
    monkeypatch.setattr(
        pd,
        "read_json",
        lambda *args, **kwargs: pd.DataFrame(
            [{"question": "Synthetic arithmetic question.", "answer": correct}]
        ),
    )
    evaluation = AIME25Eval(n_repeats=1, num_examples=1, n_threads=1)
    return evaluation(SyntheticSampler(text))


@pytest.mark.parametrize("answer,correct", INVALID_ANSWERS)
def test_numeric_fragments_do_not_receive_credit(monkeypatch, answer, correct):
    result = evaluate(monkeypatch, answer, correct)
    assert result.score == 0
    assert len(result.htmls) == 1
    assert result.convos[0][-1]["content"] == answer


@pytest.mark.parametrize(
    "answer,correct",
    [
        ("The answer is 42.", 42),
        ("Earlier I found 12; the final answer is +42.", 42),
        ("Answer: 000", 0),
        ("Answer: 999", 999),
        (r"Earlier \boxed{12}; finally \boxed{42}.", 42),
        (r"Answer: \framebox{ 42 }", 42),
        ("No numerical answer is available.", 42),
    ],
)
def test_integer_and_no_answer_controls(monkeypatch, answer, correct):
    result = evaluate(monkeypatch, answer, correct)
    assert result.score == (0 if answer.startswith("No numerical") else 1)


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("Answer: -42", "-42"),
        ("Answer: 4.2", "4.2"),
        ("Answer: 2e-3", "2e-3"),
        ("Answer: 1,234", "1,234"),
        ("Answer: 1 / 42", "1 / 42"),
        (r"Answer: \boxed{12,42}", "12,42"),
    ],
)
def test_extraction_preserves_the_entire_numeric_candidate(answer, expected):
    assert extract_boxed_text(answer) == expected
