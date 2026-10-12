import chz
import pytest
from aiohttp import ClientSession

from gpt_oss.tools.simple_browser.backend import Backend
from gpt_oss.tools.simple_browser.page_contents import PageContents, process_html
from gpt_oss.tools.simple_browser.simple_browser_tool import SimpleBrowserTool


@chz.chz(typecheck=True)
class FakeBackend(Backend):
    """Backend that serves deterministic, query-derived pages."""

    source: str = chz.field(doc="Description of the backend source", default="web")

    async def search(
        self, query: str, topn: int, session: ClientSession
    ) -> PageContents:
        html_page = f"""
<html><body>
<h1>Search Results</h1>
<ul>
<li><a href='https://{query}.example/1'>{query} result</a> summary</li>
</ul>
</body></html>
"""
        # Like the real backends, search result pages have no url.
        return process_html(html=html_page, url="", title=query, display_urls=True)

    async def fetch(self, url: str, session: ClientSession) -> PageContents:
        return process_html(
            html=f"<html><body><p>contents of {url}</p></body></html>",
            url=url,
            title=url,
        )


async def _drain(messages):
    last = None
    async for message in messages:
        last = message
    assert last is not None
    return last.content[0].text


@pytest.mark.asyncio
async def test_consecutive_searches_keep_distinct_cursors():
    tool = SimpleBrowserTool(backend=FakeBackend())
    await _drain(tool.search(query="alpha"))
    await _drain(tool.search(query="beta"))

    assert tool.tool_state.current_cursor == 1
    assert tool.tool_state.get_page(0).title == "alpha"
    assert tool.tool_state.get_page(1).title == "beta"


@pytest.mark.asyncio
async def test_open_link_from_earlier_search():
    tool = SimpleBrowserTool(backend=FakeBackend())
    await _drain(tool.search(query="alpha"))
    await _drain(tool.search(query="beta"))

    # Click the first result of the *first* search.
    text = await _drain(tool.open(id=0, cursor=0))
    assert "https://alpha.example/1" in text
    assert "https://beta.example/1" not in text


@pytest.mark.asyncio
async def test_search_cursors_are_not_cited_as_urls():
    tool = SimpleBrowserTool(backend=FakeBackend())
    await _drain(tool.search(query="alpha"))
    await _drain(tool.search(query="beta"))

    content, annotations, _ = tool.normalize_citations("see 【0†L1】 and 【1†L2】")
    # Search result pages have no url, so they produce no url_citation.
    assert content == "see 【0†L1】 and 【1†L2】"
    assert annotations == []


@pytest.mark.asyncio
async def test_opened_pages_are_still_cited_as_urls():
    tool = SimpleBrowserTool(backend=FakeBackend())
    await _drain(tool.search(query="alpha"))
    await _drain(tool.open(id=0, cursor=0))

    content, annotations, _ = tool.normalize_citations("see 【1†L1】")
    assert "https://alpha.example/1" in content
    assert [annotation["url"] for annotation in annotations] == [
        "https://alpha.example/1"
    ]
