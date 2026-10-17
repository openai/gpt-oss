from gpt_oss.evals.abcd_grader import extract_abcd


def test_extracts_explicit_answer():
    assert extract_abcd("Answer: B") == "B"


def test_extracts_markdown_wrapped_answer():
    assert extract_abcd("**Answer:** A") == "A"


def test_bare_letter_still_extracted():
    assert extract_abcd("C") == "C"


def test_bare_bold_letter_extracted():
    assert extract_abcd("**A**") == "A"
    assert extract_abcd("**b**") == "B"


def test_bare_letter_with_surrounding_whitespace():
    assert extract_abcd("  **B**  ") == "B"
    assert extract_abcd("\nA") == "A"


def test_bare_letter_with_trailing_punctuation():
    assert extract_abcd("A.") == "A"
    assert extract_abcd("A)") == "A"


def test_returns_none_for_unparseable_text():
    # Regression: the fallback used to return the first character of the text
    # (e.g. "T"), violating the documented `-> str | None` choice contract.
    assert extract_abcd("The answer is unclear.") is None
    assert extract_abcd("") is None


def test_does_not_leak_leading_lowercase_letter():
    # Regression: a non-answer that merely *starts* with a/b/c/d must not be
    # scored as that choice (the fallback only accepts a whole bare letter).
    assert extract_abcd("because the answer is unclear") is None
    assert extract_abcd("dunno") is None
