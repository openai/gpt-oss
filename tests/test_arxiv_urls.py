"""Offline checks for arXiv link conversion in the simple browser."""
import unittest

import lxml.html

from gpt_oss.tools.simple_browser.page_contents import _clean_links, arxiv_to_ar5iv


class ArxivURLTests(unittest.TestCase):
    def test_canonical_arxiv_urls_convert(self):
        for scheme in ("http", "https"):
            with self.subTest(scheme=scheme):
                self.assertEqual(arxiv_to_ar5iv(f"{scheme}://arxiv.org/abs/1234.56789"),
                                 f"{scheme}://ar5iv.org/abs/1234.56789")

    def test_only_hostname_changes(self):
        source = "https://arxiv.org/abs/arxiv.org?next=https://arxiv.org/abs/1#arxiv.org"
        expected = "https://ar5iv.org/abs/arxiv.org?next=https://arxiv.org/abs/1#arxiv.org"
        self.assertEqual(arxiv_to_ar5iv(source), expected)

    def test_unrelated_url_components_are_not_rewritten(self):
        for source in (
            "https://example.com/arxiv.org/paper",
            "https://example.com/?url=https://arxiv.org/abs/1234.56789",
            "https://example.com/#arxiv.org",
            "https://arxiv.org@example.com/paper",
        ):
            with self.subTest(source=source):
                self.assertEqual(arxiv_to_ar5iv(source), source)

    def test_lookalike_and_subdomain_hosts_are_not_rewritten(self):
        for hostname in ("arxivXorg", "notarxiv.org", "arxiv.org.example.com", "export.arxiv.org"):
            source = f"https://{hostname}/abs/1234.56789"
            with self.subTest(hostname=hostname):
                self.assertEqual(arxiv_to_ar5iv(source), source)

    def test_case_insensitive_hostname_and_port(self):
        self.assertEqual(arxiv_to_ar5iv("https://ARXIV.ORG:443/abs/1"), "https://ar5iv.org:443/abs/1")

    def test_user_info_and_port_are_preserved(self):
        source = "https://reader:arxiv.org@arxiv.org:443/abs/1"
        self.assertEqual(arxiv_to_ar5iv(source), "https://reader:arxiv.org@ar5iv.org:443/abs/1")

    def test_protocol_relative_and_unparseable_inputs(self):
        self.assertEqual(arxiv_to_ar5iv("//arxiv.org/abs/1"), "//ar5iv.org/abs/1")
        for source in ("", "/arxiv.org/paper", "https://[invalid/arxiv.org"):
            with self.subTest(source=source):
                self.assertEqual(arxiv_to_ar5iv(source), source)

    def test_link_cleaning_retains_original_external_destinations(self):
        root = lxml.html.fromstring('''<div>
          <a href="https://example.com/?source=arxiv.org">External</a>
          <a href="https://arxiv.org/abs/1">Paper</a>
          <a href="https://example.com/?source=arxiv.org">Again</a>
        </div>''')
        urls = _clean_links(root, "https://example.com/index")
        self.assertEqual(urls, {"0": "https://example.com/?source=arxiv.org", "1": "https://ar5iv.org/abs/1"})
        self.assertEqual(len(root.findall(".//a")), 0)
        self.assertIn("【0†External】", root.text_content())
        self.assertIn("【0†Again】", root.text_content())


if __name__ == "__main__":
    unittest.main()
