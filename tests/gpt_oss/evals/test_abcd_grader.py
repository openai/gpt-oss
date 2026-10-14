import pytest

from gpt_oss.evals.abcd_grader import extract_abcd


@pytest.mark.parametrize(
    "text",
    [
        "As an AI I cannot help with that.",
        "Because the data is insufficient, I cannot decide.",
        "Considering all factors, the value is 3.2 m/s.",
        "Data not available.",
        "I cannot answer a question about this.",
        "Sorry, I can't answer a medical question.",
        "Answer: 42",
        "Answer: Approximately 3 km",
        "Answer: Both are correct",
        "Clearly the answer is B.",
    ],
)
def test_rejects_prose_and_non_answers(text: str) -> None:
    assert extract_abcd(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Answer: C", "C"),
        ("**Answer:** B", "B"),
        ("The answer is (D).", "D"),
        (r"\boxed{A}", "A"),
        ("D) The fourth option.", "D"),
    ],
)
def test_preserves_explicit_answer_declarations(text: str, expected: str) -> None:
    assert extract_abcd(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("answer: B", "B"),
        ("ANSWER: (C)", "C"),
        ("answers - D", "D"),
        ("option A", "A"),
        ("Choice: B", "B"),
        ("Answer: correct option is (B)", "B"),
        ("Explanation\nB", "B"),
    ],
)
def test_labels_are_case_insensitive_but_choices_remain_uppercase(
    text: str, expected: str
) -> None:
    assert extract_abcd(text) == expected


@pytest.mark.parametrize(
    "text",
    ["Answer: c", "answer: (d)", "Option b", "choice: a"],
)
def test_lowercase_choices_are_rejected(text: str) -> None:
    assert extract_abcd(text) is None
