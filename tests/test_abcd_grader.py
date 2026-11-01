from gpt_oss.evals.abcd_grader import extract_abcd


def test_extracts_explicit_answer_declarations() -> None:
    cases = {
        "Answer: A": "A",
        "Answer: (B)": "B",
        "Option: C": "C",
        "**Answer:** D": "D",
        "*Answer*: A.": "A",
        "**Answer**: B.": "B",
        "__Answer__: C!": "C",
        "A!": "A",
        "B?": "B",
        "C)": "C",
        "**D**": "D",
    }

    for text, expected in cases.items():
        assert extract_abcd(text) == expected


def test_does_not_classify_letter_prefixes_as_answers() -> None:
    cases = [
        "As an AI assistant, I cannot help with that.",
        "Answer: 42",
        "Answer: Both options are correct.",
        "**Answer**: B. This is why.",
        "Answer: B. This is why.",
        "Answer: B because this explanation follows.",
        "Caution: the premise is false.",
        "Difficult to determine from the supplied data.",
        "A useful way to approach this is...",
    ]

    for text in cases:
        assert extract_abcd(text) is None
