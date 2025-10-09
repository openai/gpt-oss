import contextvars
import dataclasses
import functools
import itertools
import json
import re
import textwrap
from typing import Any, AsyncIterator, Callable, ParamSpec, Sequence
from urllib.parse import quote, unquote

import pydantic
import structlog
import tiktoken
from aiohttp import ClientSession
from openai_harmony import (
    Author,
    Content,
    Message,
    Role,
    TextContent,
    ToolNamespaceConfig
)

from ..tool import Tool

# from functions import Function, from_python
from .backend import (
    VIEW_SOURCE_PREFIX,
    Backend,
    BackendError,
    maybe_truncate,
)
from .page_contents import Extract, PageContents

logger = structlog.stdlib.get_logger(component=__name__)


# TODO(zhuohan): Use the correct encoding at release
ENC_NAME = "o200k_base"
FIND_PAGE_LINK_FORMAT = "# 【{idx}†{title}】"
PARTIAL_INITIAL_LINK_PATTERN = re.compile(r"^[^【】]*】")
PARTIAL_FINAL_LINK_PATTERN = re.compile(
    r"【\d*(?:†(?P<content>[^†】]*)(?:†[^†】]*)?)?$"
)
LINK_PATTERN = re.compile(r"【\d+†(?P<content>[^†】]+)(?:†[^†】]+)?】")

CITATION_OUTPUT_PATTERN = re.compile(r"【(?P<cursor>\d+)†(?P<content>[^†】]+)(?:†[^†】]+)?】")

CallParams = ParamSpec("CallParams")


_P = ParamSpec("_P")
_live_function_name = contextvars.ContextVar[str]("_live_function_name")


class ToolUsageError(Exception):
    pass


def function_the_model_can_call(
    fn: Callable[_P, AsyncIterator[Message]],
) -> Callable[_P, AsyncIterator[Message]]:
    fn.__fn_calling_tool_fn_type__ = "function_the_model_can_call"  # type: ignore

    @functools.wraps(fn)
    async def inner(*args: _P.args, **kwargs: _P.kwargs) -> AsyncIterator[Message]:
        token = _live_function_name.set(fn.__name__)
        try:
            async for m in fn(*args, **kwargs):
                yield m
        finally:
            _live_function_name.reset(token)

    return inner


@functools.cache
def _tiktoken_vocabulary_lengths(enc_name: str) -> list[int]:
    encoding = tiktoken.get_encoding(enc_name)
    results = []
    for i in range(encoding.n_vocab):
        try:
            results.append(len(encoding.decode([i])))
        except Exception as e:
            results.append(1)
    return results


@dataclasses.dataclass(frozen=True)
class Tokens:
    tokens: list[int]
    tok2idx: list[int]  # Offsets = running sum of lengths.


@functools.cache
def max_chars_per_token(enc_name: str) -> int:
    """Typical value is 128, but let's be safe."""
    tok_lens = _tiktoken_vocabulary_lengths(enc_name)
    return max(tok_lens)


def get_tokens(text: str, enc_name: str) -> Tokens:
    encoding = tiktoken.get_encoding(enc_name)
    tokens = encoding.encode(text, disallowed_special=())
    _vocabulary_lengths = _tiktoken_vocabulary_lengths(enc_name)
    tok2idx = [0] + list(itertools.accumulate(_vocabulary_lengths[i] for i in tokens))[
        :-1
    ]
    result = Tokens(tokens=tokens, tok2idx=tok2idx)
    return result


def get_end_loc(
    loc: int,
    num_lines: int,
    total_lines: int,
    lines: list[str],
    view_tokens: int,
    encoding_name: str,
) -> int:
    if num_lines <= 0:
        # COMPUTE NUMBER OF LINES TO SHOW
        txt = join_lines(lines[loc:], add_line_numbers=True, offset=loc)
        # if the text is very short, no need to truncate at all
        # at least one char per token
        if len(txt) > view_tokens:
            # limit the amount of text we tokenize here
            upper_bound = max_chars_per_token(encoding_name)
            tok2idx = get_tokens(
                txt[: (view_tokens + 1) * upper_bound], encoding_name
            ).tok2idx
            if len(tok2idx) > view_tokens:
                end_idx = tok2idx[view_tokens]
                num_lines = txt[:end_idx].count("\n") + 1  # round up
            else:
                num_lines = total_lines
        else:
            num_lines = total_lines

    return min(loc + num_lines, total_lines)


def get_page_metadata(
    curr_page: PageContents,
) -> dict[str, str | None | dict[str, str] | list[str]]:
    """Some attributes of the current page."""
    page_metadata: dict[str, str | None | dict[str, str] | list[str]] = {
        "url": curr_page.url,
        "title": curr_page.title,
    }
    return page_metadata


def join_lines(
    lines: list[str], add_line_numbers: bool = False, offset: int = 0
) -> str:
    if add_line_numbers:
        return "\n".join([f"L{i + offset}: {line}" for i, line in enumerate(lines)])
    else:
        return "\n".join(lines)


def wrap_lines(text: str, width: int = 80) -> list[str]:
    lines = text.split("\n")
    wrapped = itertools.chain.from_iterable(
        (
            textwrap.wrap(
                line, width=width, replace_whitespace=False, drop_whitespace=False
            )
            if line
            else [""]
        )  # preserve empty lines
        for line in lines
    )
    return list(wrapped)


def strip_links(text: str) -> str:
    text = re.sub(PARTIAL_INITIAL_LINK_PATTERN, "", text)
    text = re.sub(PARTIAL_FINAL_LINK_PATTERN, lambda mo: mo.group("content"), text)
    text = re.sub(LINK_PATTERN, lambda mo: mo.group("content"), text)
    return text


def maybe_get_function_args(
    message: Message, tool_name: str = "browser"
) -> dict[str, Any] | None:
    if not message.recipient.startswith(f"{tool_name}."):
        return None

    contents = ""
    if len(message.content) == 1 and isinstance(message.content[0], TextContent):
        contents = message.content[0].text

    if not contents:
        return {}

    try:
        parsed_contents = json.loads(contents)
        if isinstance(parsed_contents, dict):
            return parsed_contents
    except json.JSONDecodeError:
        pass

    return None


async def run_find_in_page(
    pattern: str,
    page: PageContents,
    max_results: int = 50,
    num_show_lines: int = 4,
) -> PageContents:
    lines = wrap_lines(text=page.text)
    txt = join_lines(lines, add_line_numbers=False)
    without_links = strip_links(txt)
    lines = without_links.split("\n")

    result_chunks, snippets = [], []
    line_idx, match_idx = 0, 0
    while line_idx < len(lines):
        line = lines[line_idx]
        if pattern not in line.lower():
            line_idx += 1
            continue
        snippet = "\n".join(lines[line_idx : line_idx + num_show_lines])
        link_title = FIND_PAGE_LINK_FORMAT.format(
            idx=f"{match_idx}", title=f"match at L{line_idx}"
        )
        result_chunks.append(f"{link_title}\n{snippet}")
        snippets.append(
            Extract(
                url=page.url, text=snippet, title=f"#{match_idx}", line_idx=line_idx
            )
        )
        if len(result_chunks) == max_results:
            break
        match_idx += 1
        line_idx += num_show_lines

    urls = [page.url for _ in result_chunks]

    if result_chunks:
        display_text = "\n\n".join(result_chunks)
    else:
        display_text = f"No `find` results for pattern: `{pattern}`"

    result_page = PageContents(
        url=f"{page.url}/find?pattern={quote(pattern)}",
        title=f"Find results for text: `{pattern}` in `{page.title}`",
        text=display_text,
        urls={str(i): url for i, url in enumerate(urls)},
        snippets={str(i): snip for i, snip in enumerate(snippets)},
    )
    return result_page


def handle_errors(
    func: Callable[CallParams, AsyncIterator["Message"]],
) -> Callable[CallParams, AsyncIterator["Message"]]:
    @functools.wraps(func)
    async def inner(
        *args: CallParams.args, **kwargs: CallParams.kwargs
    ) -> AsyncIterator[Message]:
        tool = args[0]
        # Could be cool to type this explicitly, but mypy makes it hard
        assert isinstance(tool, SimpleBrowserTool)
        try:
            async for msg in func(*args, **kwargs):
                yield msg
        except (ToolUsageError, BackendError) as e:
            yield tool.make_error_message(e)

    return inner


class SimpleBrowserState(pydantic.BaseModel):
    # maps page url to page contents
    pages: dict[str, PageContents] = pydantic.Field(default_factory=dict)
    # a sequential list of page urls
    page_stack: list[str] = pydantic.Field(default_factory=list)

    @property
    def current_cursor(self) -> int:
        return len(self.page_stack) - 1

    def add_page(self, page: PageContents) -> None:
        self.pages[page.url] = page
        self.page_stack.append(page.url)

    def get_page(self, cursor: int = -1) -> PageContents:
        if self.current_cursor < 0:
            raise ToolUsageError("No pages to access!")
        if cursor == -1 or cursor == self.current_cursor:
            return self.pages[self.page_stack[-1]]
        try:
            page_url = self.page_stack[cursor]
        except TypeError as e:
            raise ToolUsageError(
                f"`cursor` should be an integer, not `{type(cursor).__name__}`"
            ) from e
        except IndexError as e:
            raise ToolUsageError(
                f"Cursor `{cursor}` is out of range. "
                f"Available cursor indices: [0 - {self.current_cursor}]."
            ) from e
        return self.pages[page_url]

    def get_page_by_url(self, url: str) -> PageContents | None:
        if url in self.pages:
            return self.pages[url]
        return None

    def pop_page_stack(self) -> None:
        assert self.current_cursor >= 0, "No page to pop!"
        self.page_stack.pop()


class SimpleBrowserTool(Tool):
    def __init__(
        self,
        backend: Backend,
        encoding_name: str = ENC_NAME,
        max_search_results: int = 20,
        tool_state: dict[str, Any] | None = None,
        view_tokens: int = 1024,
        name: str = "browser",
    ):
        assert name == "browser"
        self.backend = backend
        if tool_state is None:
            self.tool_state = SimpleBrowserState()
        else:
            self.tool_state = SimpleBrowserState.model_validate(tool_state)

        self.encoding_name = encoding_name
        self.max_search_results = max_search_results
        self.view_tokens = view_tokens

    def get_tool_state(self) -> dict[str, Any]:
        return {"tool_state": self.tool_state.model_dump()}

    @classmethod
    def get_tool_name(cls) -> str:
        return "browser"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        config = ToolNamespaceConfig.browser()
        config.name = self.name
        config.description = """Tool for browsing.
The `cursor` appears in brackets before each browsing display: `[{cursor}]`.
Cite information from the tool using the following format:
`【{cursor}†L{line_start}(-L{line_end})?】`, for example: `【6†L9-L11】` or `【8†L3】`.
Do not quote more than 10 words directly from the tool output.
sources=""" + self.backend.source
        return config

    @property
    def instruction(self) -> str:
        return self.tool_config.description

    def _render_browsing_display(
        self,
        tether_id: int,
        result: str,
        summary: str | None = None,
    ):
        to_return = ""
        # Always show summaries.
        if summary:
            to_return += summary
        to_return += result
        to_return = f"[{tether_id}] {to_return}"
        return to_return

    def _make_response(
        self,
        page: PageContents,
        cursor: int,
        body: str,
        scrollbar: str,
    ) -> Message:
        domain = maybe_truncate(unquote(page.url))
        header = f"{page.title}"
        if domain:
            header += f" ({domain})"
        header += f"\n**{scrollbar}**\n\n"

        content = TextContent(text=self._render_browsing_display(cursor, body, header))
        return self.make_response(
            content=content, metadata=get_page_metadata(self.tool_state.get_page())
        )

    async def show_page(self, loc: int = 0, num_lines: int = -1) -> Message:
        page = self.tool_state.get_page()
        cursor = self.tool_state.current_cursor
        lines = wrap_lines(text=page.text)
        total_lines = len(lines)

        if loc >= total_lines:
            err_msg = (
                f"Invalid location parameter: `{loc}`. "
                f"Cannot exceed page maximum of {total_lines - 1}."
            )
            raise ToolUsageError(err_msg)

        end_loc = get_end_loc(
            loc, num_lines, total_lines, lines, self.view_tokens, self.encoding_name
        )

        lines_to_show = lines[loc:end_loc]
        body = join_lines(lines_to_show, add_line_numbers=True, offset=loc)

        scrollbar = f"viewing lines [{loc} - {end_loc - 1}] of {total_lines - 1}"
        return self._make_response(page, cursor, body, scrollbar)

    async def show_page_safely(self, loc: int = 0, num_lines: int = -1) -> Message:
        try:
            return await self.show_page(loc=loc, num_lines=num_lines)
        except ToolUsageError as e:
            self.tool_state.pop_page_stack()
            raise e

    async def _open_url(self, url: str, direct_url_open: bool) -> PageContents:
        """Use the cache, if available."""
        backend = self.backend
        # direct_url_open should be regarded as a refresh
        if not direct_url_open and (page := self.tool_state.get_page_by_url(url)):
            assert page.url == url
            return page

        try:
            async with ClientSession() as session:
                page = await backend.fetch(url, session=session)
            return page
        except Exception as e:
            msg = maybe_truncate(str(e))
            logger.warning("Error fetching URL in lean browser tool", exc_info=e)
            raise BackendError(
                f"Error fetching URL `{maybe_truncate(url)}`: {msg}"
            ) from e

    def make_error_message(self, error: Exception) -> Message:
        """Uses the message creation codepath from the base class."""
        error_name = error.__class__.__name__
        content = TextContent(text=str(error))
        return self.make_response(content=content)

    @function_the_model_can_call
    @handle_errors
    async def search(
        self,
        query: str,
        topn: int = 10,
        top_n: int = 10,
        source: str | None = None,
    ) -> AsyncIterator[Message]:
        del topn
        del top_n
        try:
            async with ClientSession() as session:
                search_page = await self.backend.search(
                    query=query,
                    topn=self.max_search_results,
                    session=session,
                )
        except Exception as e:
            msg = maybe_truncate(str(e))
            raise BackendError(f"Error during search for `{query}`: {msg}") from e

        self.tool_state.add_page(search_page)
        yield await self.show_page_safely(loc=0)

    @function_the_model_can_call
    @handle_errors
    async def open(
        self,
        id: int | str = -1,
        cursor: int = -1,
        loc: int = -1,
        num_lines: int = -1,
        view_source: bool = False,
        source: str | None = None,
    ) -> AsyncIterator[Message]:
        curr_page: PageContents | None = None
        stay_on_current_page = False
        direct_url_open = False
        if isinstance(id, str):
            snippet = None
            url = id
            direct_url_open = True
        else:  # Operate on a previously opened page
            curr_page = self.tool_state.get_page(cursor)

            if id >= 0:  # click a link
                try:
                    url = curr_page.urls[str(id)]
                except KeyError as e:
                    raise ToolUsageError(f"Invalid link id `{id}`.") from e
                snippet = (curr_page.snippets or {}).get(str(id))
                if snippet and curr_page.url == "":
                    # current page is a search result page
                    assert isinstance(snippet, Extract)
            else:  # navigate to new position on the current page
                if not view_source:
                    stay_on_current_page = True
                url = curr_page.url
                snippet = None

        new_page: PageContents
        if view_source:
            url = f"{VIEW_SOURCE_PREFIX}{url}"
            snippet = None
        if stay_on_current_page:
            assert curr_page is not None
            new_page = curr_page
        else:
            new_page = await self._open_url(url, direct_url_open)

        self.tool_state.add_page(new_page)

        if loc < 0:  # unset
            if snippet is not None and snippet.line_idx is not None:
                loc = snippet.line_idx
                if loc > 4:
                    loc -= 4
            else:
                loc = 0
        yield await self.show_page_safely(loc=loc, num_lines=num_lines)

    @function_the_model_can_call
    @handle_errors
    async def find(self, pattern: str, cursor: int = -1) -> AsyncIterator[Message]:
        page = self.tool_state.get_page(cursor)
        if page.snippets is not None:
            raise ToolUsageError(
                "Cannot run `find` on search results page or find results page"
            )

        pc = await run_find_in_page(
            pattern=str(pattern).lower(),
            page=page,
        )
        self.tool_state.add_page(pc)
        yield await self.show_page_safely(loc=0)

    def make_response(
        self,
        content: Content,
        *,
        metadata: dict[str, Any] | None = None,
        author: Author | None = None,
    ) -> Message:
        """
        Make a response message.

        Should be used from `@function_the_model_can_call` if author is not provided.
        """
        if author is None:
            tool_name = self.get_tool_name()
            function_name = _live_function_name.get()
            assert function_name is not None
            author = Author(role=Role.TOOL, name=f"{tool_name}.{function_name}")

        return Message(
            author=author,
            content=[content],
        ).with_recipient("assistant")

    def process_arguments(self, message: Message) -> dict[str, Any]:
        function_args = maybe_get_function_args(message, tool_name=self.name)
        if function_args is None:
            raise ValueError("Invalid function arguments")

        if "cursor" in function_args and function_args["cursor"] >= 0:
            page = self.tool_state.get_page(cursor=function_args["cursor"])
            if "id" in function_args:
                function_args["url"] = page.urls[str(function_args["id"])]
            else:
                function_args["url"] = page.url
        elif "id" in function_args and isinstance(function_args["id"], str):
            function_args["url"] = function_args["id"]
        return function_args

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        def make_error_message(error: str) -> Message:
            return self.make_response(
                content=TextContent(text=json.dumps({"error": error})),
                author=Author(role=Role.TOOL, name=message.recipient),
            )

        function_args = maybe_get_function_args(message, tool_name=self.name)
        if function_args is None:
            yield make_error_message("Invalid function arguments")
            return

        _, function_name = message.recipient.split(".")
        if function_name not in ["search", "open", "find"]:
            yield make_error_message(f"Unknown function: {function_name}")
            return

        if function_name == "search":
            async for msg in self.search(**function_args):
                yield msg
        elif function_name == "open":
            async for msg in self.open(**function_args):
                yield msg
        elif function_name == "find":
            async for msg in self.find(**function_args):
                yield msg
        else:
            raise ValueError("should not be here")


    def normalize_citations(self, old_content: str, hide_partial_citations: bool = False) -> tuple[str, list[dict[str, Any]], bool]:
        """
        Returns a tuple of (new_message, annotations, has_partial_citations)
        - new_message: Message with citations replaced by ([domain](url))
        - annotations: list of dicts with start_index, end_index, and title (url)
        - has_partial_citations: whether the text includes an unfinished citation
        """

        has_partial_citations = PARTIAL_FINAL_LINK_PATTERN.search(old_content) is not None
        if hide_partial_citations and has_partial_citations:
            old_content = PARTIAL_FINAL_LINK_PATTERN.sub("", old_content)

        matches = []
        for match in CITATION_OUTPUT_PATTERN.finditer(old_content):
            cursor = match.group("cursor")
            content = match.group("content")
            start_idx = match.start()
            end_idx = match.end()
            matches.append({
                "cursor": cursor,
                "content": content,
                "start": start_idx,
                "end": end_idx
            })

        # Build a mapping from cursor to url
        cursor_to_url = {}
        for idx, url in enumerate(self.tool_state.page_stack):
            cursor_to_url[str(idx)] = url

        def extract_domain(url):
            try:
                return unquote(url).split("/")[2]
            except Exception:
                return url

        new_content = ""
        last_idx = 0
        annotations = []
        running_offset = 0  # Offset due to length changes in replacements

        for m in matches:
            cursor = m["cursor"]
            url = cursor_to_url.get(cursor, None)
            orig_start = m["start"]
            orig_end = m["end"]

            # Add text before the citation
            new_content += old_content[last_idx:orig_start]

            if url:
                domain = extract_domain(url)
                replacement = f" ([{domain}]({url})) "
                # The start and end indices in the new content
                start_index = len(new_content)
                end_index = start_index + len(replacement)
                annotations.append({
                    "start_index": start_index,
                    "end_index": end_index,
                    "title": domain,
                    "url": url,
                    "type": "url_citation",
                })
                new_content += replacement
            else:
                # Keep the original citation format if cursor is missing
                replacement = old_content[orig_start:orig_end]
                start_index = len(new_content)
                end_index = start_index + len(replacement)
                # No annotation for missing url, but could add if desired
                new_content += replacement

            last_idx = orig_end

        new_content += old_content[last_idx:]
        return new_content, annotations, has_partial_citations

