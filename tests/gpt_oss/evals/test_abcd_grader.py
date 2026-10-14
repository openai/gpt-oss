import pytest

from gpt_oss.evals.abcd_grader import extract_abcd


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Answer: C", "C"),
        ("ANSWER: C", "C"),
        ("**Answer:** B", "B"),
        ("The answer is (D).", "D"),
        (r"\boxed{A}", "A"),
        ("D) The fourth option.", "D"),
        ("**D**", "D"),
        ("Answer: C\nAdditional reasoning follows.", "C"),
    ],
)
def test_valid_answer_declarations_are_still_extracted(text, expected):
    assert extract_abcd(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "As an AI I cannot help with that.",
        "Because the data is insufficient, I cannot decide.",
        "Clearly the answer is B.",
        "Data not available.",
        "I cannot answer a question about this.",
        "Answer: 42",
        "Answer: Both are correct",
        "Answer: After checking, C",
        "Answer: c",
        "Answer: Caution",
        "After checking, C",
    ],
)
def test_prose_and_incomplete_declarations_are_not_answers(text):
    assert extract_abcd(text) is None


def test_explicit_answer_later_in_prose_is_preserved():
    assert extract_abcd("Clearly, the answer is (B).") == "B"
    assert extract_abcd("The final choice is C.") is None
