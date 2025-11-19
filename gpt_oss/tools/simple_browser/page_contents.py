"""
Page contents for the simple browser tool.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import re
from urllib.parse import urljoin, urlparse

import aiohttp
import html2text
import lxml
import lxml.etree
import lxml.html
import pydantic

import tiktoken

logger = logging.getLogger(__name__)


HTML_SUP_RE = re.compile(r"<sup( [^>]*)?>([\w\-]+)</sup>")
HTML_SUB_RE = re.compile(r"<sub( [^>]*)?>([\w\-]+)</sub>")
HTML_TAGS_SEQ_RE = re.compile(r"(?<=\w)((<[^>]*>)+)(?=\w)")
WHITESPACE_ANCHOR_RE = re.compile(r"(【\@[^】]+】)(\s+)")
EMPTY_LINE_RE = re.compile(r"^\s+$", flags=re.MULTILINE)
EXTRA_NEWLINE_RE = re.compile(r"\n(\s*\n)+")


class Extract(pydantic.BaseModel):  # A search result snippet or a quotable extract
    url: str
    text: str
    title: str
    line_idx: int | None = None


class FetchResult(pydantic.BaseModel):
    url: str
    success: bool
    title: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    html: str | None = None
    raw_content: bytes | None = None
    plaintext: str | None = None


class PageContents(pydantic.BaseModel):
    url: str
    text: str
    title: str
    urls: dict[str, str]
    snippets: dict[str, Extract] | None = None
    error_message: str | None = None


@dataclasses.dataclass(frozen=True)
class Tokens:
    tokens: list[int]
    tok2idx: list[int]  # Offsets = running sum of lengths.


def get_domain(url: str) -> str:
    """Extracts the domain from a URL."""
    if "http" not in url:
        # If `get_domain` is called on a domain, add a scheme so that the
        # original domain is returned instead of the empty string.
        url = "http://" + url
    return urlparse(url).netloc


def multiple_replace(text: str, replacements: dict[str, str]) -> str:
    """Performs multiple string replacements using regex pass."""
    regex = re.compile("(%s)" % "|".join(map(re.escape, replacements.keys())))
    return regex.sub(lambda mo: replacements[mo.group(1)], text)


@functools.lru_cache(maxsize=1024)
def mark_lines(text: str) -> str:
    """Adds line numbers (ex: 'L0:') to the beginning of each line in a string."""
    # Split the string by newline characters
    lines = text.split("\n")

    # Add lines numbers to each line and join into a single string
    numbered_text = "\n".join([f"L{i}: {line}" for i, line in enumerate(lines)])
    return numbered_text


@functools.cache
def _tiktoken_vocabulary_lengths(enc_name: str) -> list[int]:
    """Gets the character lengths of all tokens in the specified TikToken vocabulary."""
    encoding = tiktoken.get_encoding(enc_name)
    return [len(encoding.decode([i])) for i in range(encoding.n_vocab)]


def warmup_caches(enc_names: list[str]) -> None:
    """Warm up the cache by computing token length lists for the given TikToken encodings."""
    for _ in map(_tiktoken_vocabulary_lengths, enc_names):
        pass


def _replace_special_chars(text: str) -> str:
    """Replaces specific special characters with visually similar alternatives."""
    replacements = {
        "【": "〖",
        "】": "〗",
        "◼": "◾",
        # "━": "─",
        "\u200b": "",  # zero width space
        # Note: not replacing †
    }
    return multiple_replace(text, replacements)


def merge_whitespace(text: str) -> str:
    """Replace newlines with spaces and merge consecutive whitespace into a single space."""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def arxiv_to_ar5iv(url: str) -> str:
    """Converts an arxiv.org URL to its ar5iv.org equivalent."""
    return re.sub(r"arxiv.org", r"ar5iv.org", url)


def _clean_links(root: lxml.html.HtmlElement, cur_url: str) -> dict[str, str]:
    """Processes all anchor tags in the HTML, replaces them with a custom format and returns an ID-to-URL mapping."""
    cur_domain = get_domain(cur_url)
    urls: dict[str, str] = {}
    urls_rev: dict[str, str] = {}
    for a in root.findall(".//a[@href]"):
        assert a.getparent() is not None
        link = a.attrib["href"]
        if link.startswith(("mailto:", "javascript:")):
            continue
        text = _get_text(a).replace("†", "‡")
        if not re.sub(r"【\@([^】]+)】", "", text):  # Probably an image
            continue
        if link.startswith("#"):
            replace_node_with_text(a, text)
            continue
        try:
            link = urljoin(cur_url, link)  # works with both absolute and relative links
            domain = get_domain(link)
        except Exception:
            domain = ""
        if not domain:
            logger.debug("SKIPPING LINK WITH URL %s", link)
            continue
        link = arxiv_to_ar5iv(link)
        if (link_id := urls_rev.get(link)) is None:
            link_id = f"{len(urls)}"
            urls[link_id] = link
            urls_rev[link] = link_id
        if domain == cur_domain:
            replacement = f"【{link_id}†{text}】"
        else:
            replacement = f"【{link_id}†{text}†{domain}】"
        replace_node_with_text(a, replacement)
    return urls


def _get_text(node: lxml.html.HtmlElement) -> str:
    """Extracts all text from an HTML element and merges it into a whitespace-normalized string."""
    return merge_whitespace(" ".join(node.itertext()))


def _remove_node(node: lxml.html.HtmlElement) -> None:
    """Removes a node from its parent in the lxml tree."""
    node.getparent().remove(node)


def _escape_md(text: str) -> str:
    return text


def _escape_md_section(text: str, snob: bool = False) -> str:
    return text


def html_to_text(html: str) -> str:
    """Converts an HTML string to clean plaintext."""
    html = re.sub(HTML_SUP_RE, r"^{\2}", html)
    html = re.sub(HTML_SUB_RE, r"_{\2}", html)
    # add spaces between tags such as table cells
    html = re.sub(HTML_TAGS_SEQ_RE, r" \1", html)
    # we don't need to escape markdown, so monkey-patch the logic
    orig_escape_md = html2text.utils.escape_md
    orig_escape_md_section = html2text.utils.escape_md_section
    html2text.utils.escape_md = _escape_md
    html2text.utils.escape_md_section = _escape_md_section
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.body_width = 0  # no wrapping
    h.ignore_tables = True
    h.unicode_snob = True
    h.ignore_emphasis = True
    result = h.handle(html).strip()
    html2text.utils.escape_md = orig_escape_md
    html2text.utils.escape_md_section = orig_escape_md_section
    return result


def _remove_math(root: lxml.html.HtmlElement) -> None:
    """Removes all <math> elements from the lxml tree."""
    for node in root.findall(".//math"):
        _remove_node(node)


def remove_unicode_smp(text: str) -> str:
    """Removes Unicode characters in the Supplemental Multilingual Plane (SMP) from `text`.

    SMP characters are not supported by lxml.html processing.
    """
    smp_pattern = re.compile(r"[\U00010000-\U0001FFFF]", re.UNICODE)
    return smp_pattern.sub("", text)


def replace_node_with_text(node: lxml.html.HtmlElement, text: str) -> None:
    """Replaces an lxml node with a text string while preserving surrounding text."""
    previous = node.getprevious()
    parent = node.getparent()
    tail = node.tail or ""
    if previous is None:
        parent.text = (parent.text or "") + text + tail
    else:
        previous.tail = (previous.tail or "") + text + tail
    parent.remove(node)


def replace_images(
    root: lxml.html.HtmlElement,
    base_url: str,
    session: aiohttp.ClientSession | None,
) -> None:
    """Finds all image tags and replaces them with numbered placeholders (includes alt/title if available)."""
    cnt = 0
    for img_tag in root.findall(".//img"):
        image_name = img_tag.get("alt", img_tag.get("title"))
        if image_name:
            replacement = f"[Image {cnt}: {image_name}]"
        else:
            replacement = f"[Image {cnt}]"
        replace_node_with_text(img_tag, replacement)
        cnt += 1


def process_html(
    html: str,
    url: str,
    title: str | None,
    session: aiohttp.ClientSession | None = None,
    display_urls: bool = False,
) -> PageContents:
    """Convert HTML into model-readable version."""
    html = remove_unicode_smp(html)
    html = _replace_special_chars(html)
    root = lxml.html.fromstring(html)

    # Parse the title.
    title_element = root.find(".//title")
    if title:
        final_title = title
    elif title_element is not None:
        final_title = title_element.text or ""
    elif url and (domain := get_domain(url)):
        final_title = domain
    else:
        final_title = ""

    urls = _clean_links(root, url)
    replace_images(
        root=root,
        base_url=url,
        session=session,
    )
    _remove_math(root)
    clean_html = lxml.etree.tostring(root, encoding="UTF-8").decode()
    text = html_to_text(clean_html)
    text = re.sub(WHITESPACE_ANCHOR_RE, lambda m: m.group(2) + m.group(1), text)
    # ^^^ move anchors to the right thru whitespace
    # This way anchors don't create extra whitespace
    text = re.sub(EMPTY_LINE_RE, "", text)
    # ^^^ Get rid of empty lines
    text = re.sub(EXTRA_NEWLINE_RE, "\n\n", text)
    # ^^^ Get rid of extra newlines

    top_parts = []
    if display_urls:
        top_parts.append(f"\nURL: {url}\n")
    # NOTE: Publication date is currently not extracted due
    # to performance costs.

    return PageContents(
        url=url,
        text="".join(top_parts) + text,
        urls=urls,
        title=final_title,
    )
