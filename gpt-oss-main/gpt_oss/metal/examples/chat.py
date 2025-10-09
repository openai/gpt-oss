#!/usr/bin/env python

import argparse
import sys

from datetime import date
from gpt_oss.metal import Context, Model


DEFAULT_PROMPT = f"""You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: {date.today().isoformat()}

reasoning effort high

# Valid channels: analysis, final. Channel must be included for every message."""


parser = argparse.ArgumentParser(description="Chat with gpt-oss", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("model", metavar="PATH", type=str, help="Path to gpt-oss model in Metal inference format")
parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="System prompt")
parser.add_argument(
    "--context-length", type=int, default=0, help="The maximum context length"
)
parser.add_argument(
    "--temperature", type=float, default=1.0, help="Sampling temperature"
)
parser.add_argument(
    "--seed", type=int, default=0, help="Sampling seed"
)


GREY = "\33[90m"
BOLD = "\33[1m"
RESET = "\33[0m"


def main(args):
    options = parser.parse_args(args)
    model = Model(options.model)
    tokenizer = model.tokenizer
    start_token = tokenizer.encode_special_token("<|start|>")
    message_token = tokenizer.encode_special_token("<|message|>")
    end_token = tokenizer.encode_special_token("<|end|>")
    return_token = tokenizer.encode_special_token("<|return|>")
    channel_token = tokenizer.encode_special_token("<|channel|>")

    context = Context(model, context_length=options.context_length)
    context.append(start_token)
    context.append("system")
    context.append(message_token)
    context.append(options.prompt)
    context.append(end_token)

    while True:
        context.append(start_token)
        context.append("user")
        context.append(message_token)
        message = input(f"{BOLD}User:{RESET} ").rstrip()
        context.append(message)
        context.append(end_token)
        print(f"{BOLD}Assistant:{RESET} {GREY}", end="", flush=True)
        context.append(start_token)
        context.append("assistant")
        context.append(channel_token)

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
                if inside_channel_block:
                    channel += str(tokenizer.decode(token), encoding="utf-8")
                elif inside_start_block:
                    role += str(tokenizer.decode(token), encoding="utf-8")
                else:
                    sys.stdout.buffer.write(tokenizer.decode(token))
                    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main(sys.argv[1:])
