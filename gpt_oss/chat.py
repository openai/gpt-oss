"""
Harmony chat with tools
"""

import atexit
import argparse
import asyncio
import datetime
import os
from pathlib import Path

try:
    import gnureadline as readline
except ImportError:
    import readline

import torch
import termcolor

from gpt_oss.tools import apply_patch
from gpt_oss.classical import ClassicalCanon
from gpt_oss.classical_library import ClassicalLibrary
from gpt_oss.coding_library import DEFAULT_CATALOGUE, CodingLibrary
from gpt_oss.chronology import ChronologyGuard
from gpt_oss.tools.simple_browser import SimpleBrowserTool
from gpt_oss.tools.simple_browser.backend import ExaBackend, YouComBackend
from gpt_oss.tools.python_docker.docker_tool import PythonTool

from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    StreamState,
    SystemContent,
    TextContent,
    ToolDescription,
    load_harmony_encoding,
)


REASONING_EFFORT = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}


def get_user_input():
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        user_input = input()
    else:
        user_input = ""
    user_input_list = [user_input]
    if torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(user_input_list, 0)
    return user_input_list[0]


def main(args):
    canon = ClassicalCanon(args.knowledge_cutoff_year)
    library = ClassicalLibrary(args.classical_library) if args.classical_library else None
    coding_library = CodingLibrary(args.coding_library)
    guard = ChronologyGuard(canon.cutoff_year)
    match args.backend:
        case "triton":
            from gpt_oss.triton.model import TokenGenerator as TritonGenerator
            from gpt_oss.torch.utils import init_distributed
            device = init_distributed()
            generator = TritonGenerator(args.checkpoint, args.context, device)
        case "torch":
            from gpt_oss.torch.model import TokenGenerator as TorchGenerator
            from gpt_oss.torch.utils import init_distributed
            device = init_distributed()
            generator = TorchGenerator(args.checkpoint, device)
        case "vllm":
            from gpt_oss.vllm.token_generator import TokenGenerator as VLLMGenerator
            generator = VLLMGenerator(args.checkpoint, tensor_parallel_size=2)
        case _:
            raise ValueError(f"Invalid backend: {args.backend}")

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    system_message_content = (
        SystemContent.new()
        .with_model_identity(canon.instructions)
        .with_knowledge_cutoff(f"{canon.cutoff_year}-12")
        .with_reasoning_effort(REASONING_EFFORT[args.reasoning_effort])
        .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
    )

    if args.browser:
        tool_backend = os.getenv("BROWSER_BACKEND", "youcom")
        if tool_backend == "youcom":
            backend = YouComBackend(source="web")
        elif tool_backend == "exa":
            backend = ExaBackend(source="web")
        else:
            raise ValueError(f"Invalid tool backend: {tool_backend}")
        browser_tool = SimpleBrowserTool(backend=backend)
        system_message_content = system_message_content.with_tools(browser_tool.tool_config)

    if args.python:
        python_tool = PythonTool()
        system_message_content = system_message_content.with_tools(python_tool.tool_config)

    system_message = Message.from_role_and_content(Role.SYSTEM, system_message_content)
    messages = [system_message]

    if args.apply_patch:
        apply_patch_instructions = Path(apply_patch.__file__).parent / "apply_patch.md"
        developer_message = ""
        if args.developer_message:
            developer_message = args.developer_message + "\n"
        developer_message += apply_patch_instructions.read_text()
        developer_message_content = (
            DeveloperContent.new()
            .with_instructions(developer_message)
            .with_function_tools([
                ToolDescription.new(
                    "apply_patch",
                    "Patch a file",
                    parameters={
                        "type": "string",
                        "description": "Formatted patch code",
                        "default": "*** Begin Patch\n*** End Patch\n",
                    }
                ),
            ])
        )
        messages.append(Message.from_role_and_content(Role.DEVELOPER, developer_message_content))
    elif args.developer_message:
        developer_message_content = DeveloperContent.new().with_instructions(args.developer_message)
        messages.append(Message.from_role_and_content(Role.DEVELOPER, developer_message_content))
    else:
        developer_message_content = None

    if args.raw:
        conversation = Conversation.from_messages(messages)
        tokens = encoding.render_conversation(conversation)
        system_message = encoding.decode(tokens)
        print(system_message, flush=True, end="")
        empty_user_message_tokens = encoding.render(Message.from_role_and_content(Role.USER, ""))
        user_message_start = encoding.decode(empty_user_message_tokens[:-1])
        user_message_end = encoding.decode(empty_user_message_tokens[-1:])
    else:
        # System message
        print(termcolor.colored("System Message:", "cyan"), flush=True)
        print(termcolor.colored("Model Identity:", "cyan"), system_message_content.model_identity, flush=True)
        print(termcolor.colored("Reasoning Effort:", "cyan"), system_message_content.reasoning_effort, flush=True)
        print(termcolor.colored("Conversation Start Date:", "cyan"), system_message_content.conversation_start_date, flush=True)
        print(termcolor.colored("Knowledge Cutoff:", "cyan"), system_message_content.knowledge_cutoff, flush=True)
        print(termcolor.colored("Browser Tool:", "cyan"), "Enabled" if args.browser else "Disabled", flush=True)
        print(termcolor.colored("Python Tool:", "cyan"), "Enabled" if args.python else "Disabled", flush=True)
        print(termcolor.colored("Apply Patch Function:", "cyan"), "Enabled" if args.apply_patch else "Disabled", flush=True)
        print(termcolor.colored("Chronology Guard:", "cyan"),
              "Raw diagnostics bypass output screening" if args.raw else "Enabled", flush=True)
        print(termcolor.colored("Coding Library:", "cyan"),
              "、".join(coding_library.languages), flush=True)
        if developer_message_content:
            print(termcolor.colored("Developer Message:", "yellow"), flush=True)
            print(developer_message_content.instructions, flush=True)

    # Print the system message and the user message start
    MESSAGE_PADDING = 12
    programming_context = False

    def screened(text: str, channel: str, programming: bool) -> str:
        """Screen one completed channel; tool arguments are treated as code."""
        if channel == "tool":
            verdict = guard.sanitize(f"```text\n{text}\n```", programming=True)
            return guard.refusal if verdict.blocked else text
        return guard.sanitize(text, programming=programming).text

    while True:
        last_message = messages[-1]
        if last_message.recipient is None:
            if args.raw:
                print(user_message_start, end="", flush=True)
                user_message = get_user_input()
                print(user_message_end, flush=True, end="")
            else:
                print(termcolor.colored("User:".ljust(MESSAGE_PADDING), "red"), flush=True)
                user_message = get_user_input()
            user_message = Message.from_role_and_content(Role.USER, user_message)
            messages.append(user_message)
            question = user_message.content[0].text
            scope = guard.classify_request(question)
            programming_context = scope.programming
            if scope.blocked:
                print(termcolor.colored("Assistant:".ljust(MESSAGE_PADDING), "green"), flush=True)
                print(guard.refusal, flush=True)
                messages.append(Message.from_role_and_content(
                    Role.ASSISTANT, guard.refusal).with_channel("final"))
                continue
            code_reference = coding_library.consult(question)
            if code_reference:
                messages.append(Message.from_role_and_content(Role.USER, code_reference))
            if library is not None:
                book_evidence = library.consult(question, canon.cutoff_year)
                if book_evidence:
                    messages.append(Message.from_role_and_content(Role.USER, book_evidence))
        else:
            # Tool or function call
            if last_message.recipient.startswith("browser."):
                assert args.browser, "Browser tool is not enabled"
                tool_name = "Search"
                async def run_tool():
                    results = []
                    async for msg in browser_tool.process(last_message):
                        results.append(msg)
                    return results

                result = asyncio.run(run_tool())
                messages += result
            elif last_message.recipient.startswith("python"):
                assert args.python, "Python tool is not enabled"
                tool_name = "Python"
                async def run_tool():
                    results = []
                    async for msg in python_tool.process(last_message):
                        results.append(msg)
                    return results

                result = asyncio.run(run_tool())
                messages += result
            elif last_message.recipient == "functions.apply_patch":
                assert args.apply_patch, "Apply patch tool is not enabled"
                tool_name = "Apply Patch"
                text = last_message.content[0].text
                tool_output = None

                if text.startswith("{"):
                    # this is json, try to extract the patch from it
                    import json
                    try:
                        some_dict = json.loads(text)
                        _, text = some_dict.popitem()
                    except Exception as e:
                        tool_output = f"Error parsing JSON: {e}"

                if tool_output is None:
                    try:
                        tool_output = apply_patch.apply_patch(text)
                    except Exception as e:
                        tool_output = f"Error applying patch: {e}"

                message = (
                    Message(
                        author=Author.new(Role.TOOL, last_message.recipient),
                        content=[TextContent(text=tool_output)]
                    )
                    .with_recipient("assistant")
                )
                if last_message.channel:
                    message = message.with_channel(last_message.channel)

                result = [message]
                messages += result
            else:
                raise ValueError(f"Unknown tool or function call: {last_message.recipient}")
            # Print the tool or function call result
            if args.raw:
                rendered_result = encoding.render_conversation(Conversation.from_messages(result))
                print(encoding.decode(rendered_result), flush=True, end="")
            else:
                print(termcolor.colored(f"{tool_name} output:".ljust(MESSAGE_PADDING), "magenta"), flush=True)
                if tool_name == "Search" and not args.show_browser_results:
                    print("[Search results fed to the model]")
                else:
                    print(screened(result[0].content[0].text, "tool",
                                   programming_context))

        conversation = Conversation.from_messages(messages)
        tokens = encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT
        )

        if args.raw:
            # Print the last two tokens, which are the start of the assistant message
            print(encoding.decode(tokens[-2:]), flush=True, end="")

        parser = StreamableParser(encoding, role=Role.ASSISTANT)
        # The chronology guard buffers each channel so no partial post-cutoff
        # claim can be printed before the completed message is screened.
        channel_buffers: dict[str, str] = {"final": "", "analysis": "", "tool": ""}
        channel_headers: dict[str, str] = {}

        def flush_channels():
            for channel, header in channel_headers.items():
                text = channel_buffers.get(channel, "")
                if not text:
                    continue
                if args.browser and channel == "final":
                    text, _annotations, _partial = browser_tool.normalize_citations(text)
                print(header, flush=True)
                print(screened(text, channel, programming_context), end="", flush=True)
                channel_buffers[channel] = ""
            channel_headers.clear()

        for predicted_token in generator.generate(tokens, encoding.stop_tokens_for_assistant_actions()):
            parser.process(predicted_token)
            if args.raw:
                # Raw diagnostics intentionally expose the token stream; the
                # request-level refusal still applies. Do not use raw mode as
                # a user-facing guarded chat.
                print(encoding.decode([predicted_token]), end="", flush=True)
                continue

            if parser.state == StreamState.EXPECT_START and any(channel_buffers.values()):
                flush_channels()
                print("")  # new line

            if not parser.last_content_delta:
                continue

            if parser.current_channel == "final":
                channel = "final"
                header = termcolor.colored("Assistant:".ljust(MESSAGE_PADDING), "green")
            elif parser.current_recipient is not None:
                channel = "tool"
                header = termcolor.colored(
                    f"Tool call to {parser.current_recipient}:".ljust(MESSAGE_PADDING), "cyan")
            else:
                channel = "analysis"
                header = termcolor.colored("CoT:".ljust(MESSAGE_PADDING), "yellow")
            channel_headers.setdefault(channel, header)
            channel_buffers[channel] += parser.last_content_delta

        if not args.raw:
            flush_channels()
        messages += parser.messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chat example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint",
        metavar="FILE",
        type=str,
        help="Path to the SafeTensors checkpoint",
    )
    parser.add_argument(
        "-r",
        "--reasoning-effort",
        metavar="REASONING_EFFORT",
        type=str,
        default="low",
        choices=["high", "medium", "low"],
        help="Reasoning effort",
    )
    parser.add_argument(
        "-a",
        "--apply-patch",
        action="store_true",
        help="Make apply_patch function available to the model",
    )
    parser.add_argument(
        "-b",
        "--browser",
        default=False,
        action="store_true",
        help="Use browser tool",
    )
    parser.add_argument(
        "--show-browser-results",
        default=False,
        action="store_true",
        help="Show browser results",
    )
    parser.add_argument(
        "-p",
        "--python",
        default=False,
        action="store_true",
        help="Use python tool",
    )
    parser.add_argument(
        "--developer-message",
        default="",
        help="Developer message",
    )
    parser.add_argument(
        "-c",
        "--context",
        metavar="CONTEXT",
        type=int,
        default=8192,
        help="Max context length",
    )
    parser.add_argument(
        "--raw",
        default=False,
        action="store_true",
        help="Raw mode (does not render Harmony encoding)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="triton",
        choices=["triton", "torch", "vllm"],
        help="Inference backend",
    )
    parser.add_argument("--knowledge-cutoff-year", type=int, choices=range(1, 1900), default=1899,
                        metavar="YEAR", help="古典學問之年界")
    parser.add_argument("--classical-library", help="已校書證之索引路徑")
    parser.add_argument("--coding-library", type=Path, default=DEFAULT_CATALOGUE,
                        help="已校現代程式準繩；預設隨套件提供")
    args = parser.parse_args()

    if int(os.environ.get("WORLD_SIZE", 1)) == 1:
        histfile = os.path.join(os.path.expanduser("~"), ".chat")
        try:
            readline.read_history_file(histfile)
            readline.set_history_length(10000)
        except FileNotFoundError:
            pass

        atexit.register(readline.write_history_file, histfile)

    main(args)
