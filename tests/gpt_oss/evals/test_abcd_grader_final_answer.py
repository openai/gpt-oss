from gpt_oss.evals.abcd_grader import extract_abcd


def test_latest_explicit_answer_declaration_wins() -> None:
    text = "Answer: A\nAfter checking the reasoning, I need to correct this.\nAnswer: B"

    assert extract_abcd(text) == "B"


def test_latest_markdown_answer_declaration_wins() -> None:
    text = "**Answer:** A\nAfter checking the reasoning, I need to correct this.\n**Answer:** C"

    assert extract_abcd(text) == "C"


def test_latest_declaration_wins_across_answer_and_choice_formats() -> None:
    text = "Answer: A\nI reconsidered.\nChoice: B"

    assert extract_abcd(text) == "B"


def test_latest_declaration_wins_across_plain_and_parenthesized_formats() -> None:
    text = "Answer: A\nI reconsidered.\nAnswer: (B)"

    assert extract_abcd(text) == "B"


def test_explicit_answer_priority_is_preserved() -> None:
    text = "Answer: A\nA later aside mentions (B), without declaring it as the answer."

    assert extract_abcd(text) == "A"


def test_fallback_answer_is_not_overridden_by_later_prose() -> None:
    text = "D) The fourth option.\nBecause it follows."

    assert extract_abcd(text) == "D"


def test_single_answer_declaration_is_unchanged() -> None:
    assert extract_abcd("Reasoning complete.\nAnswer: D") == "D"
