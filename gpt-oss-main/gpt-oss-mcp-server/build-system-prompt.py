import datetime
import asyncio

from gpt_oss.tokenizer import get_tokenizer

from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    ToolNamespaceConfig,
    ToolDescription,
    load_harmony_encoding,
)

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import ListToolsResult


async def list_server_and_tools(server_url: str):
    async with sse_client(url=server_url) as streams, ClientSession(
            *streams) as session:
        initialize_response = await session.initialize()
        list_tools_response = await session.list_tools()
        return initialize_response, list_tools_response


def trim_schema(schema: dict) -> dict:
    # Turn JSON Schema from MCP generated into Harmony's variant.
    if "title" in schema:
        del schema["title"]
    if "default" in schema and schema["default"] is None:
        del schema["default"]
    if "anyOf" in schema:
        # Turn "anyOf": [{"type": "type-1"}, {"type": "type-2"}] into "type": ["type-1", "type-2"]
        # if there's more than 1 types, also remove "null" type as Harmony will just ignore it
        types = [
            type_dict["type"] for type_dict in schema["anyOf"]
            if type_dict["type"] != 'null'
        ]
        schema["type"] = types
        del schema["anyOf"]
    if "properties" in schema:
        schema["properties"] = {
            k: trim_schema(v)
            for k, v in schema["properties"].items()
        }
    return schema


def post_process_tools_description(
        list_tools_result: ListToolsResult) -> ListToolsResult:
    # Adapt the MCP tool result for Harmony
    for tool in list_tools_result.tools:
        tool.inputSchema = trim_schema(tool.inputSchema)

    # Some tools schema don't need to be part of the prompt (e.g. simple text in text out for Python)
    list_tools_result.tools = [
        tool for tool in list_tools_result.tools
        if getattr(tool.annotations, "include_in_prompt", True)
    ]

    return list_tools_result

tokenizer = get_tokenizer()

tools_urls = [
    "http://localhost:8001/sse",  # browser
    "http://localhost:8000/sse",  # python
]
harmony_tool_descriptions = []
for tools_url in tools_urls:

    initialize_response, list_tools_response = asyncio.run(
        list_server_and_tools(tools_url))

    list_tools_response = post_process_tools_description(list_tools_response)

    tool_from_mcp = ToolNamespaceConfig(
        name=initialize_response.serverInfo.name,
        description=initialize_response.instructions,
        tools=[
            ToolDescription.new(name=tool.name,
                                description=tool.description,
                                parameters=tool.inputSchema)
            for tool in list_tools_response.tools
        ])
    harmony_tool_descriptions.append(tool_from_mcp)

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

system_message_content = (SystemContent.new().with_reasoning_effort(
    ReasoningEffort.LOW).with_conversation_start_date(
        datetime.datetime.now().strftime("%Y-%m-%d")))

for tool_description in harmony_tool_descriptions:
    system_message_content = system_message_content.with_tools(
        tool_description)

system_message = Message.from_role_and_content(Role.SYSTEM,
                                               system_message_content)

developer_message_content = DeveloperContent.new().with_instructions("")
developer_message = Message.from_role_and_content(Role.DEVELOPER,
                                                  developer_message_content)

messages = [system_message, developer_message]

conversation = Conversation.from_messages(messages)
tokens = encoding.render_conversation(conversation)
system_message = tokenizer.decode(tokens)
print(system_message)
