import pytest

from gpt_oss.tools.simple_browser import page_contents


def test_html_to_text_restores_escape_helpers_after_failure(monkeypatch) -> None:
    original_escape_md = page_contents.html2text.utils.escape_md
    original_escape_md_section = page_contents.html2text.utils.escape_md_section

    def fail_handle(self, html):
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(page_contents.html2text.HTML2Text, "handle", fail_handle)

    with pytest.raises(RuntimeError, match="conversion failed"):
        page_contents.html_to_text("<p>test</p>")

    assert page_contents.html2text.utils.escape_md is original_escape_md
    assert page_contents.html2text.utils.escape_md_section is original_escape_md_section
