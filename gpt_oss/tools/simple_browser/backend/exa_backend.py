"""
Exa backend for the simple browser tool.
"""

import os

import chz
from aiohttp import ClientSession

from .base_backend import Backend, BackendError
from ..page_contents import PageContents, process_html


VIEW_SOURCE_PREFIX = "view-source:"


@chz.chz(typecheck=True)
class ExaBackend(Backend):
    """Backend that uses the Exa Search API."""

    source: str = chz.field(doc="Description of the backend source")
    api_key: str | None = chz.field(
        doc="Exa API key. Uses EXA_API_KEY environment variable if not provided.",
        default=None,
    )

    BASE_URL: str = "https://api.exa.ai"

    def _get_api_key(self) -> str:
        key = self.api_key or os.environ.get("EXA_API_KEY")
        if not key:
            raise BackendError("Exa API key not provided")
        return key

    async def _post(self, session: ClientSession, endpoint: str, payload: dict) -> dict:
        headers = {"x-api-key": self._get_api_key()}
        async with session.post(f"{self.BASE_URL}{endpoint}", json=payload, headers=headers) as resp:
            if resp.status != 200:
                raise BackendError(
                    f"Exa API error {resp.status}: {await resp.text()}"
                )
            return await resp.json()

    async def search(
        self, query: str, topn: int, session: ClientSession
    ) -> PageContents:
        data = await self._post(
            session,
            "/search",
            {"query": query, "numResults": topn, "contents": {"text": True, "summary": True}},
        )
        # make a simple HTML page to work with browser format
        titles_and_urls = [
            (result["title"], result["url"], result["summary"])
            for result in data["results"]
        ]
        html_page = f"""
<html><body>
<h1>Search Results</h1>
<ul>
{"".join([f"<li><a href='{url}'>{title}</a> {summary}</li>" for title, url, summary in titles_and_urls])}
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
        data = await self._post(
            session,
            "/contents",
            {"urls": [url], "text": { "includeHtmlTags": True }},
        )
        results = data.get("results", [])
        if not results:
            raise BackendError(f"No contents returned for {url}")
        return process_html(
            html=results[0].get("text", ""),
            url=url,
            title=results[0].get("title", ""),
            display_urls=True,
            session=session,
        )
