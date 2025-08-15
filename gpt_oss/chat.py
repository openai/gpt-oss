"""
Harmony chat with tools

This module provides an interactive chat interface that supports multiple inference backends
(Triton, Torch, VLLM) and various tools including browser search, Python execution, and 
patch application functionality.

BUG FIXES AND IMPROVEMENTS MADE:
- Added comprehensive error handling for tool execution failures
- Fixed missing file handling for apply_patch.md instructions
- Added safety checks for message content access before processing
- Fixed hardcoded tensor_parallel_size=2 in VLLM backend (marked as FIXME)
- Added proper initialization checks for browser_tool and python_tool
- Improved error handling for readline history operations
- Added comprehensive docstrings and GitHub-style comments
- Enhanced argument descriptions for better CLI usability
- Added graceful handling of token generation errors
- Fixed browser tool citation normalization safety checks
"""

import atexit
import argparse
import asyncio
import datetime
import os
from pathlib import Path
import sys

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


# Mapping of string reasoning effort levels to enum values
# This allows CLI users to specify reasoning effort in a human-readable way
REASONING_EFFORT = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}


def get_user_input():
    """
    Get user input in a distributed setting.
    
    In distributed training/inference, only rank 0 should read from stdin
    to avoid multiple processes trying to read input simultaneously.
    The input is then broadcast to all other ranks.
    
    Returns:
        str: User input string
    """
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
    """
    Main chat loop with support for multiple backends and tools.
    
    Args:
        args: Parsed command line arguments containing backend choice,
              tool configurations, and other settings.
    """
    # Initialize the appropriate token generator based on backend choice
    # TODO: Consider adding error handling for missing dependencies per backend
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
            # Use configurable tensor parallel size
            generator = VLLMGenerator(args.checkpoint, tensor_parallel_size=args.tensor_parallel_size)
        case _:
            raise ValueError(f"Invalid backend: {args.backend}")

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # Configure system message with reasoning effort and current date
    system_message_content = (
        SystemContent.new()
        .with_reasoning_effort(REASONING_EFFORT[args.reasoning_effort])
        .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
    )

    # Initialize browser tool if requested
    browser_tool = None
    if args.browser:
        backend = ExaBackend(
            source="web",
        )
        browser_tool = SimpleBrowserTool(backend=backend)
        system_message_content = system_message_content.with_tools(browser_tool.tool_config)

    # Initialize Python execution tool if requested  
    python_tool = None
    if args.python:
        python_tool = PythonTool()
        system_message_content = system_message_content.with_tools(python_tool.tool_config)

    system_message = Message.from_role_and_content(Role.SYSTEM, system_message_content)
    messages = [system_message]

    # Configure apply_patch functionality if requested
    if args.apply_patch:
        apply_patch_instructions = Path(apply_patch.__file__).parent / "apply_patch.md"
        developer_message = ""
        if args.developer_message:
            developer_message = args.developer_message + "\n"
        # BUG FIX: Add error handling for missing apply_patch.md file
        try:
            developer_message += apply_patch_instructions.read_text()
        except FileNotFoundError:
            print(f"Warning: apply_patch.md not found at {apply_patch_instructions}")
            developer_message += "Apply patch functionality enabled but instructions file not found."
        
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

    # Handle raw mode for debugging/development - outputs raw tokens
    if args.raw:
        conversation = Conversation.from_messages(messages)
        tokens = encoding.render_conversation(conversation)
        system_message = encoding.decode(tokens)
        print(system_message, flush=True, end="")
        empty_user_message_tokens = encoding.render(Message.from_role_and_content(Role.USER, ""))
        user_message_start = encoding.decode(empty_user_message_tokens[:-1])
        user_message_end = encoding.decode(empty_user_message_tokens[-1:])
    else:
        # Display system configuration in human-readable format
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

    # Main chat loop
    MESSAGE_PADDING = 12
    while True:
        last_message = messages[-1]
        
        # Handle user input or tool/function call responses
        if last_message.recipient is None:
            # Get user input
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
            # Process tool or function calls
            # BUG FIX: Add proper error handling for tool processing
            try:
                if last_message.recipient.startswith("browser."):
                    assert args.browser, "Browser tool is not enabled"
                    assert browser_tool is not None, "Browser tool not initialized"
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
                    assert python_tool is not None, "Python tool not initialized"
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
                    # BUG FIX: Add safety check for message content
                    if not last_message.content or len(last_message.content) == 0:
                        tool_output = "Error: No content provided for patch application"
                    else:
                        text = last_message.content[0].text
                        tool_output = None

                        # Handle JSON-wrapped patch content
                        if text.startswith("{"):
                            # this is json, try to extract the patch from it
                            import json
                            try:
                                some_dict = json.loads(text)
                                _, text = some_dict.popitem()
                            except Exception as e:
                                tool_output = f"Error parsing JSON: {e}"

                        # Apply the patch
                        if tool_output is None:
                            try:
                                tool_output = apply_patch.apply_patch(text)
                            except Exception as e:
                                tool_output = f"Error applying patch: {e}"

                    # Create tool response message
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
            except Exception as e:
                # BUG FIX: Handle tool execution errors gracefully
                error_message = f"Error executing tool {last_message.recipient}: {e}"
                print(termcolor.colored(f"Error: {error_message}", "red"), flush=True)
                
                # Create error response message
                error_response = Message(
                    author=Author.new(Role.TOOL, last_message.recipient),
                    content=[TextContent(text=error_message)]
                ).with_recipient("assistant")
                if last_message.channel:
                    error_response = error_response.with_channel(last_message.channel)
                messages.append(error_response)
                continue
                
            # Display tool execution results
            if args.raw:
                rendered_result = encoding.render_conversation(Conversation.from_messages(result))
                print(encoding.decode(rendered_result), flush=True, end="")
            else:
                print(termcolor.colored(f"{tool_name} output:".ljust(MESSAGE_PADDING), "magenta"), flush=True)
                if tool_name == "Search" and not args.show_browser_results:
                    print("[Search results fed to the model]")
                else:
                    # BUG FIX: Add safety check for result content access
                    if result and len(result) > 0 and result[0].content and len(result[0].content) > 0:
                        print(result[0].content[0].text)
                    else:
                        print("[No output returned from tool]")

        # Generate assistant response using the selected backend
        conversation = Conversation.from_messages(messages)
        tokens = encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT
        )

        if args.raw:
            # Print the last two tokens, which are the start of the assistant message
            print(encoding.decode(tokens[-2:]), flush=True, end="")

        # Stream the model's response token by token
        parser = StreamableParser(encoding, role=Role.ASSISTANT)
        field_created = False
        current_output_text = ""
        output_text_delta_buffer = ""
        
        # BUG FIX: Add error handling for token generation
        try:
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

                # Create field headers for different types of content
                if not field_created:
                    field_created = True
                    if parser.current_channel == "final":
                        print(termcolor.colored("Assistant:", "green"), flush=True)
                    elif parser.current_recipient is not None:
                        print(termcolor.colored(f"Tool call to {parser.current_recipient}:", "cyan"), flush=True)
                    else:
                        print(termcolor.colored("CoT:", "yellow"), flush=True)

                # Handle citation normalization for browser tool if enabled
                should_send_output_text_delta = True
                output_text_delta_buffer += parser.last_content_delta
                if args.browser and browser_tool is not None:
                    # BUG FIX: Ensure browser_tool exists before using it
                    updated_output_text, _annotations, has_partial_citations = browser_tool.normalize_citations(current_output_text + output_text_delta_buffer)
                    output_text_delta_buffer = updated_output_text[len(current_output_text):]
                    if has_partial_citations:
                        should_send_output_text_delta = False
                
                # Print the content delta
                if should_send_output_text_delta:
                    print(output_text_delta_buffer, end="", flush=True)
                    current_output_text += output_text_delta_buffer
                    output_text_delta_buffer = ""
        except Exception as e:
            # BUG FIX: Handle token generation errors
            print(termcolor.colored(f"\nError during token generation: {e}", "red"), flush=True)
            print(termcolor.colored("Continuing with chat...", "yellow"), flush=True)
            
        # Add the parser's messages to the conversation
        messages += parser.messages


if __name__ == "__main__":
    # Configure command line argument parser with comprehensive options
    parser = argparse.ArgumentParser(
        description="Interactive chat interface with support for multiple backends and tools",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint",
        metavar="FILE",
        type=str,
        help="Path to the SafeTensors checkpoint file for the model",
    )
    parser.add_argument(
        "-r",
        "--reasoning-effort",
        metavar="REASONING_EFFORT",
        type=str,
        default="low",
        choices=["high", "medium", "low"],
        help="Set the reasoning effort level for the model",
    )
    parser.add_argument(
        "-a",
        "--apply-patch",
        action="store_true",
        help="Enable apply_patch function for code modification capabilities",
    )
    parser.add_argument(
        "-b",
        "--browser",
        default=False,
        action="store_true",
        help="Enable browser tool for web search capabilities",
    )
    parser.add_argument(
        "--show-browser-results",
        default=False,
        action="store_true",
        help="Display browser search results in the chat output",
    )
    parser.add_argument(
        "-p",
        "--python",
        default=False,
        action="store_true",
        help="Enable Python execution tool for code running capabilities",
    )
    parser.add_argument(
        "--developer-message",
        default="",
        help="Custom developer message to include in the system prompt",
    )
    parser.add_argument(
        "-c",
        "--context",
        metavar="CONTEXT",
        type=int,
        default=8192,
        help="Maximum context length for the model (tokens)",
    )
    parser.add_argument(
        "--raw",
        default=False,
        action="store_true",
        help="Enable raw mode (outputs raw tokens without Harmony encoding rendering)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="triton",
        choices=["triton", "torch", "vllm"],
        help="Choose the inference backend for token generation",
    )
    # Add tensor parallel size (CLI overrides env TP)
    parser.add_argument(
        "--tensor-parallel-size", "--tp",
        dest="tensor_parallel_size",
        type=int,
        default=int(os.environ.get("TP", "2")),
        help="Tensor parallel size (overrides env TP; default from TP or 2)",
    )
    args = parser.parse_args()

    # Validate TP value
    if args.tensor_parallel_size < 1:
        print("Error: --tensor-parallel-size must be >= 1", file=sys.stderr)
        sys.exit(2)

    # Set up readline history for better user experience
    # Only do this for single-process execution (not distributed)
    if int(os.environ.get("WORLD_SIZE", 1)) == 1:
        histfile = os.path.join(os.path.expanduser("~"), ".chat")
        try:
            readline.read_history_file(histfile)
            readline.set_history_length(10000)
        except FileNotFoundError:
            # BUG FIX: Handle missing history file gracefully
            pass
        except Exception as e:
            # BUG FIX: Handle other potential readline errors
            print(f"Warning: Could not set up readline history: {e}")

        # Ensure history is saved on exit
        atexit.register(readline.write_history_file, histfile)

    main(args)
