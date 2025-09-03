"""
Tavily backend for the simple browser tool.
"""

import os

import chz
from aiohttp import ClientSession

from .base_backend import Backend, BackendError
from ..page_contents import PageContents, process_html
from ..markdown_processor import process_markdown


VIEW_SOURCE_PREFIX = "view-source:"


@chz.chz(typecheck=True)
class TavilyBackend(Backend):
    """Backend that uses the Tavily Search API."""

    source: str = chz.field(doc="Description of the backend source")
    api_key: str | None = chz.field(
        doc="Tavily API key. Uses TAVILY_API_KEY environment variable if not provided.",
        default=None,
    )
    use_markdown: bool = chz.field(
        doc="Use markdown processing instead of HTML processing for better performance",
        default=True,
    )

    BASE_URL: str = "https://api.tavily.com"

    def _get_api_key(self) -> str:
        key = self.api_key or os.environ.get("TAVILY_API_KEY")
        if not key:
            raise BackendError("Tavily API key not provided")
        return key

    async def _post(self, session: ClientSession, endpoint: str, payload: dict) -> dict:
        headers = {
            "Authorization": f"Bearer {self._get_api_key()}",
            "Content-Type": "application/json"
        }
        async with session.post(f"{self.BASE_URL}{endpoint}", json=payload, headers=headers) as resp:
            if resp.status != 200:
                raise BackendError(
                    f"Tavily API error {resp.status}: {await resp.text()}"
                )
            return await resp.json()

    async def search(
            self, query: str, topn: int, session: ClientSession
    ) -> PageContents:
        data = await self._post(
            session,
            "/search",
            {"query": query, "max_results": topn},
        )
        # make a simple HTML page to work with browser format
        titles_and_urls = [
            (result["title"], result["url"], result["content"])
            for result in data["results"]
        ]
        html_page = f"""
<html><body>
<h1>Search Results</h1>
<ul>
{"".join([f"<li><a href='{url}'>{title}</a> {content}</li>" for title, url, content in titles_and_urls])}
</ul>
</body></html>
"""

        return process_html(
            html=html_page,
            url="",
            title=query,
            display_urls=True,
            session=session,
        )

    async def fetch(self, url: str, session: ClientSession) -> PageContents:
        is_view_source = url.startswith(VIEW_SOURCE_PREFIX)
        if is_view_source:
            url = url[len(VIEW_SOURCE_PREFIX) :]

        format_type = "markdown" if self.use_markdown else "html_tags"
        
        data = await self._post(
            session,
            "/extract",
            {"urls": [url], "format": format_type},
        )
        results = data.get("results", [])
        if not results:
            raise BackendError(f"No contents returned for {url}")

        result = results[0]
        if self.use_markdown:
            return process_markdown(
                content=result.get("raw_content", ""),
                url=url,
                title=result.get("title", ""),
                display_urls=True,
            )
        else:
            return process_html(
                html=result.get("raw_content", ""),
                url=url,
                title=result.get("title", ""),
                display_urls=True,
                session=session,
            )
