from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Union, Optional

from mcp.server.fastmcp import Context, FastMCP
from gpt_oss.tools.simple_browser import SimpleBrowserTool
from gpt_oss.tools.simple_browser.backend import ExaBackend


@dataclass
class AppContext:
    browsers: dict[str, SimpleBrowserTool] = field(default_factory=dict)

    def create_or_get_browser(self, session_id: str) -> SimpleBrowserTool:
        if session_id not in self.browsers:
            backend = ExaBackend(source="web")
            self.browsers[session_id] = SimpleBrowserTool(backend=backend)
        return self.browsers[session_id]

    def remove_browser(self, session_id: str) -> None:
        self.browsers.pop(session_id, None)


@asynccontextmanager
async def app_lifespan(_server: FastMCP) -> AsyncIterator[AppContext]:
    yield AppContext()


# Pass lifespan to server
mcp = FastMCP(
    name="browser",
    instructions=r"""
Tool for browsing.
The `cursor` appears in brackets before each browsing display: `[{cursor}]`.
Cite information from the tool using the following format:
`【{cursor}†L{line_start}(-L{line_end})?】`, for example: `【6†L9-L11】` or `【8†L3】`. 
Do not quote more than 10 words directly from the tool output.
sources=web
""".strip(),
    lifespan=app_lifespan,
    port=8001,
)


@mcp.tool(
    name="search",
    title="Search for information",
    description=
    "Searches for information related to `query` and displays `topn` results.",
)
async def search(ctx: Context,
                 query: str,
                 topn: int = 10,
                 source: Optional[str] = None) -> str:
    """Search for information related to a query"""
    browser = ctx.request_context.lifespan_context.create_or_get_browser(
        ctx.client_id)
    messages = []
    async for message in browser.search(query=query, topn=topn, source=source):
        if message.content and hasattr(message.content[0], 'text'):
            messages.append(message.content[0].text)
    return "\n".join(messages)


@mcp.tool(
    name="open",
    title="Open a link or page",
    description="""
Opens the link `id` from the page indicated by `cursor` starting at line number `loc`, showing `num_lines` lines.
Valid link ids are displayed with the formatting: `【{id}†.*】`.
If `cursor` is not provided, the most recent page is implied.
If `id` is a string, it is treated as a fully qualified URL associated with `source`.
If `loc` is not provided, the viewport will be positioned at the beginning of the document or centered on the most relevant passage, if available.
Use this function without `id` to scroll to a new location of an opened page.
""".strip(),
)
async def open_link(ctx: Context,
                    id: Union[int, str] = -1,
                    cursor: int = -1,
                    loc: int = -1,
                    num_lines: int = -1,
                    view_source: bool = False,
                    source: Optional[str] = None) -> str:
    """Open a link or navigate to a page location"""
    browser = ctx.request_context.lifespan_context.create_or_get_browser(
        ctx.client_id)
    messages = []
    async for message in browser.open(id=id,
                                      cursor=cursor,
                                      loc=loc,
                                      num_lines=num_lines,
                                      view_source=view_source,
                                      source=source):
        if message.content and hasattr(message.content[0], 'text'):
            messages.append(message.content[0].text)
    return "\n".join(messages)


@mcp.tool(
    name="find",
    title="Find pattern in page",
    description=
    "Finds exact matches of `pattern` in the current page, or the page given by `cursor`.",
)
async def find_pattern(ctx: Context, pattern: str, cursor: int = -1) -> str:
    """Find exact matches of a pattern in the current page"""
    browser = ctx.request_context.lifespan_context.create_or_get_browser(
        ctx.client_id)
    messages = []
    async for message in browser.find(pattern=pattern, cursor=cursor):
        if message.content and hasattr(message.content[0], 'text'):
            messages.append(message.content[0].text)
    return "\n".join(messages)
