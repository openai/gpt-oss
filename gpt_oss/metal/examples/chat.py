#!/usr/bin/env python

import argparse
import sys
from datetime import date
from gpt_oss.metal import Context, Model

# Constants
DEFAULT_PROMPT = f"""You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: {date.today().isoformat()}

reasoning effort high

# Valid channels: analysis, final. Channel must be included for every message."""

# ANSI escape codes for terminal colors and styles
GREY = "\33[90m"
BOLD = "\33[1m"
RESET = "\33[0m"
RED = "\33[91m"

# Argument Parsing
parser = argparse.ArgumentParser(
    description="Chat with gpt-oss",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("model", metavar="PATH", type=str, help="Path to gpt-oss model in Metal inference format")
parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="System prompt")
parser.add_argument("--context-length", type=int, default=0, help="The maximum context length")
parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
parser.add_argument("--seed", type=int, default=0, help="Sampling seed")


def main(args):
    """
    Main function to run the interactive chat session.
    """
    options = parser.parse_args(args)

    # Graceful Model Loading
    try:
        model = Model(options.model)
        tokenizer = model.tokenizer
    except Exception as e:
        print(f"{RED}{BOLD}Error:{RESET} Failed to load the model from '{options.model}'.")
        print(f"Please ensure the path is correct and the model files are not corrupted.")
        print(f"Details: {e}")
        sys.exit(1) # Exit with a non-zero status code to indicate an error

    # Token Initialization
    start_token = tokenizer.encode_special_token("<|start|>")
    message_token = tokenizer.encode_special_token("<|message|>")
    end_token = tokenizer.encode_special_token("<|end|>")
    return_token = tokenizer.encode_special_token("<|return|>")
    channel_token = tokenizer.encode_special_token("<|channel|>")

    # Context Initialization
    context = Context(model, context_length=options.context_length)
    context.append(start_token)
    context.append("system")
    context.append(message_token)
    context.append(options.prompt)
    context.append(end_token)

    # Main Chat Loop
    try:
        while True:
            context.append(start_token)
            context.append("user")
            context.append(message_token)
            
            # Get user input
            message = input(f"{BOLD}User:{RESET} ").rstrip()
            context.append(message)
            context.append(end_token)
            
            print(f"{BOLD}Assistant:{RESET} {GREY}", end="", flush=True)
            context.append(start_token)
            context.append("assistant")
            context.append(channel_token)

            # Response Generation Loop
            inside_start_block = True
            inside_channel_block = True
            role = "assistant"
            channel = ""
            while True:
                token = context.sample(
                    temperature=options.temperature,
                    seed=options.seed,
                )
                context.append(token)
                
                if token == return_token:
                    print(flush=True)
                    break
                elif token == start_token:
                    inside_start_block = True
                    role = ""
                    channel = ""
                elif token == message_token:
                    inside_start_block = False
                    inside_channel_block = False
                    if channel == "analysis":
                        print(f"{GREY}", end="", flush=True)
                elif token == end_token:
                    print(f"{RESET}", flush=True)
                elif token == channel_token:
                    inside_channel_block = True
                elif token < tokenizer.num_text_tokens:
                    # Decode and process the token
                    decoded_token = tokenizer.decode(token)
                    if inside_channel_block:
                        channel += str(decoded_token, encoding="utf-8")
                    elif inside_start_block:
                        role += str(decoded_token, encoding="utf-8")
                    else:
                        sys.stdout.buffer.write(decoded_token)
                        sys.stdout.buffer.flush()

    except KeyboardInterrupt:
        # Handle Ctrl+C for a clean exit
        print(f"\n{BOLD}Exiting chat.{RESET} Goodbye!")
        sys.exit(0)
    except EOFError:
        # Handle end-of-file (Ctrl+D) for a clean exit
        print(f"\n{BOLD}Exiting chat.{RESET} Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
