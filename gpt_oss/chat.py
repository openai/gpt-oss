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
from gpt_oss.tools.simple_browser import SimpleBrowserTool
from gpt_oss.tools.simple_browser.backend import ExaBackend
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
        .with_reasoning_effort(REASONING_EFFORT[args.reasoning_effort])
        .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
    )

    if args.browser:
        backend = ExaBackend(
            source="web",
        )
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
        if developer_message_content:
            print(termcolor.colored("Developer Message:", "yellow"), flush=True)
            print(developer_message_content.instructions, flush=True)

    # Print the system message and the user message start
    MESSAGE_PADDING = 12
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
                    print(result[0].content[0].text)

        conversation = Conversation.from_messages(messages)
        tokens = encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT
        )

        if args.raw:
            # Print the last two tokens, which are the start of the assistant message
            print(encoding.decode(tokens[-2:]), flush=True, end="")

        parser = StreamableParser(encoding, role=Role.ASSISTANT)
        field_created = False
        current_output_text = ""
        output_text_delta_buffer = ""
        for predicted_token in generator.generate(tokens, encoding.stop_tokens_for_assistant_actions()):
            parser.process(predicted_token)
            if args.raw:
                print(encoding.decode([predicted_token]), end="", flush=True)
                continue

            if parser.state == StreamState.EXPECT_START:
                print("")  # new line
                field_created = False

            if not parser.last_content_delta:
                continue

            if not field_created:
                field_created = True
                if parser.current_channel == "final":
                    print(termcolor.colored("Assistant:", "green"), flush=True)
                elif parser.current_recipient is not None:
                    print(termcolor.colored(f"Tool call to {parser.current_recipient}:", "cyan"), flush=True)
                else:
                    print(termcolor.colored("CoT:", "yellow"), flush=True)

            should_send_output_text_delta = True
            output_text_delta_buffer += parser.last_content_delta
            if args.browser:
                updated_output_text, _annotations, has_partial_citations = browser_tool.normalize_citations(current_output_text + output_text_delta_buffer)
                output_text_delta_buffer = updated_output_text[len(current_output_text):]
                if has_partial_citations:
                    should_send_output_text_delta = False
            if should_send_output_text_delta:
                print(output_text_delta_buffer, end="", flush=True)
                current_output_text += output_text_delta_buffer
                output_text_delta_buffer = ""

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
