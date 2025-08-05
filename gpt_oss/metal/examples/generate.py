#!/usr/bin/env python

import argparse
import sys

from gpt_oss.metal import Context, Model


parser = argparse.ArgumentParser(description='Chat with gpt-oss', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('model', metavar='PATH', type=str, help='Path to gpt-oss checkpoint')
parser.add_argument('-p', '--prompt', type=str, required=True, help='Prompt')
parser.add_argument('-l', '--limit', type=int, default=100, help='Number of tokens to generate')
parser.add_argument('--context-length', type=int, default=0, help='The maximum context length')


def main(args):
    options = parser.parse_args(args)
    model = Model(options.model)

    context = Context(model, context_length=options.context_length)
    context.append(options.prompt)
    print(context.tokens)
    prompt_tokens = context.num_tokens

    tokenizer = model.tokenizer

    while context.num_tokens - prompt_tokens < options.limit:
        token = context.sample()
        context.append(token)
        print(str(tokenizer.decode(token), encoding="utf-8"), end='', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
