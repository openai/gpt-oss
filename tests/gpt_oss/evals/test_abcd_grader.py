from gpt_oss.evals.abcd_grader import extract_abcd


def test_extracts_explicit_answer():
    assert extract_abcd("Answer: B") == "B"


def test_extracts_markdown_wrapped_answer():
    assert extract_abcd("**Answer:** A") == "A"


def test_bare_letter_still_extracted():
    assert extract_abcd("C") == "C"


def test_returns_none_for_unparseable_text():
    # Regression: the fallback used to return the first character of the text
    # (e.g. "T"), violating the documented `-> str | None` choice contract.
    assert extract_abcd("The answer is unclear.") is None
    assert extract_abcd("") is None
