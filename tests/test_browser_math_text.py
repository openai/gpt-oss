"""Offline checks for prose around removed MathML elements."""

import unittest
import lxml.html
from gpt_oss.tools.simple_browser.page_contents import _remove_math, process_html


class BrowserMathTextTests(unittest.TestCase):
    def test_surrounding_text_is_preserved(self):
        cases = [
            ("<p>Before <math>x</math> after.</p>", "Before  after."),
            (
                "<p><math>x</math>Beginning of the explanation.</p>",
                "Beginning of the explanation.",
            ),
            ("<p><b>Bold</b> before <math>x</math> after.</p>", "Bold before  after."),
            ("<p>one<math>x</math>two<math>y</math>three</p>", "onetwothree"),
            ("<div><p>A<math>x</math>B</p><p>C</p></div>", "ABC"),
            (
                "<p>Before <math><mrow><mi>x</mi></mrow></math> text.</p>",
                "Before  text.",
            ),
            (
                "<p>Before <math>outer<math>inner</math>discard</math> after.</p>",
                "Before  after.",
            ),
            (
                "<p><math>x</math><b>Middle</b><math>y</math> after.</p>",
                "Middle after.",
            ),
        ]
        for source, expected in cases:
            with self.subTest(source=source):
                root = lxml.html.fromstring(source)
                _remove_math(root)
                self.assertEqual(root.text_content(), expected)
                self.assertEqual(root.findall(".//math"), [])

    def test_public_html_conversion_keeps_explanation_and_links(self):
        result = process_html(
            "<html><head><title>Math fixture</title></head><body>"
            "<p>Before <math><mi>discarded_formula</mi></math>important explanation "
            "<a href='/reference'>reference</a> remains.</p></body></html>",
            "https://example.invalid/article",
            None,
        )
        self.assertIn("Before", result.text)
        self.assertIn("important explanation", result.text)
        self.assertIn("remains.", result.text)
        self.assertNotIn("discarded_formula", result.text)
        self.assertEqual(result.urls, {"0": "https://example.invalid/reference"})
        self.assertEqual(result.title, "Math fixture")

    def test_formula_only_paragraph_has_no_remaining_text(self):
        root = lxml.html.fromstring("<p><math><mi>x</mi></math></p>")
        _remove_math(root)
        self.assertEqual(root.text_content(), "")

    def test_document_without_math_is_unchanged(self):
        root = lxml.html.fromstring("<p>Before <strong>emphasis</strong> after.</p>")
        original = lxml.html.tostring(root)
        _remove_math(root)
        self.assertEqual(lxml.html.tostring(root), original)


if __name__ == "__main__":
    unittest.main()
