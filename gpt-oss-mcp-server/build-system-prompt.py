import asyncio
import copy
import datetime as dt
import logging
from typing import Any, Dict, List, Optional, Tuple

from gpt_oss.tokenizer import tokenizer
from openai_harmony import (
    Conversation, DeveloperContent, HarmonyEncodingName, Message,
    ReasoningEffort, Role, SystemContent, ToolNamespaceConfig, ToolDescription,
    load_harmony_encoding,
)
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import ListToolsResult

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TOOL_SERVER_URLS = [
    "http://localhost:8001/sse",  # browser
    "http://localhost:8000/sse",  # python
]

def _strip_none_default(d: Dict[str, Any]) -> None:
    if "default" in d and d["default"] is None:
        d.pop("default", None)

def _flatten_type_list(types: List[str]) -> List[str]:
    # remove duplicates and "null" (Harmony ignores it)
    return sorted({t for t in types if t != "null"})

def _normalize_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep-copied Harmony-friendly variant of a JSON Schema."""
    s = copy.deepcopy(schema)
    s.pop("title", None)
    _strip_none_default(s)

    # Handle nullable (OpenAPI) → type list
    if s.get("nullable") is True:
        t = s.get("type")
        if isinstance(t, str):
            s["type"] = _flatten_type_list([t, "null"])
        elif isinstance(t, list):
            s["type"] = _flatten_type_list(t + ["null"])
        s.pop("nullable", None)

    # anyOf/oneOf → type union when they’re simple type unions
    for key in ("anyOf", "oneOf"):
        if key in s:
            variants = s[key]
            if all(isinstance(v, dict) and "type" in v for v in variants):
                s["type"] = _flatten_type_list([v["type"] for v in variants])
                s.pop(key, None)

    # allOf – naive merge for common simple cases
    if "allOf" in s:
        merged: Dict[str, Any] = {}
        for part in s.pop("allOf"):
            merged.update(part)
        # Recurse on merged piece (avoid infinite loop)
        s = _normalize_schema({**s, **merged})

    # Recurse into properties/items
    if "properties" in s and isinstance(s["properties"], dict):
        s["properties"] = {k: _normalize_schema(v) for k, v in s["properties"].items()}

    if "items" in s and isinstance(s["items"], dict):
        s["items"] = _normalize_schema(s["items"])

    # Keep description/enum/const/format if present; Harmony tolerates these
    return s

def _filter_tools(list_tools: ListToolsResult) -> ListToolsResult:
    # Guard annotations (MCP servers differ)
    kept = []
    for t in list_tools.tools:
        include = True
        ann = getattr(t, "annotations", None)
        if ann is not None:
            include = getattr(ann, "include_in_prompt", True)
        if include:
            kept.append(t)
        else:
            log.info("Excluding tool from prompt: %s", getattr(t, "name", "<unnamed>"))
    list_tools.tools = kept
    return list_tools

async def fetch_server(tools_url: str) -> Optional[Tuple[Any, ListToolsResult]]:
    try:
        async with sse_client(url=tools_url, timeout=10) as streams, ClientSession(*streams) as session:
            init = await session.initialize()
            tools: ListToolsResult = await session.list_tools()
            return init, tools
    except Exception as e:
        log.warning("Failed to fetch tools from %s: %s", tools_url, e)
        return None

async def gather_servers(urls: List[str]) -> List[Tuple[Any, ListToolsResult]]:
    results = await asyncio.gather(*(fetch_server(u) for u in urls))
    return [r for r in results if r is not None]

def build_harmony_system_content(
    server_results: List[Tuple[Any, ListToolsResult]],
    conversation_start_date: str,
    reasoning_effort: ReasoningEffort = ReasoningEffort.LOW,
) -> SystemContent:
    sc = SystemContent.new().with_reasoning_effort(reasoning_effort).with_conversation_start_date(conversation_start_date)
    for init, tools_result in server_results:
        tools_result = _filter_tools(tools_result)
        namespace = ToolNamespaceConfig(
            name=init.serverInfo.name,
            description=init.instructions,
            tools=[
                ToolDescription.new(
                    name=t.name,
                    description=t.description,
                    parameters=_normalize_schema(t.inputSchema),
                )
                for t in tools_result.tools
            ],
        )
        sc = sc.with_tools(namespace)
    return sc

def main(tool_urls: List[str]) -> str:
    # Fetch tools concurrently
    server_results = asyncio.run(gather_servers(tool_urls))
    if not server_results:
        raise RuntimeError("No tool servers available; cannot build system message.")

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    start_date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")

    system_content = build_harmony_system_content(server_results, start_date)
    system_msg = Message.from_role_and_content(Role.SYSTEM, system_content)

    dev_msg = Message.from_role_and_content(Role.DEVELOPER, DeveloperContent.new().with_instructions(""))

    convo = Conversation.from_messages([system_msg, dev_msg])
    token_ids = encoding.render_conversation(convo)

    rendered_system_text = tokenizer.decode(token_ids)
    return rendered_system_text

if __name__ == "__main__":
    print(main(TOOL_SERVER_URLS))
