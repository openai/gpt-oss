import datetime

from gpt_oss.tools.simple_browser import SimpleBrowserTool
from gpt_oss.tools.simple_browser.backend import ExaBackend
from gpt_oss.tools.python_docker.docker_tool import PythonTool
from gpt_oss.tokenizer import tokenizer

from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    load_harmony_encoding,
)

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

system_message_content = (SystemContent.new().with_reasoning_effort(
    ReasoningEffort.LOW).with_conversation_start_date(
        datetime.datetime.now().strftime("%Y-%m-%d")))

backend = ExaBackend(source="web", )
browser_tool = SimpleBrowserTool(backend=backend)
system_message_content = system_message_content.with_tools(
    browser_tool.tool_config)

python_tool = PythonTool()
system_message_content = system_message_content.with_tools(
    python_tool.tool_config)

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
