<img alt="gpt-oss-120" src="./docs/gpt-oss.svg">
<p align="center">
  <a href="https://gpt-oss.com"><strong>Try gpt-oss</strong></a> ·
  <a href="https://cookbook.openai.com/topic/gpt-oss"><strong>Guides</strong></a> ·
  <a href="https://arxiv.org/abs/2508.10925"><strong>Model card</strong></a> ·
  <a href="https://openai.com/index/introducing-gpt-oss/"><strong>OpenAI blog</strong></a>
</p>
<p align="center">
  <strong>Download <a href="https://huggingface.co/openai/gpt-oss-120b">gpt-oss-120b</a> and <a href="https://huggingface.co/openai/gpt-oss-20b">gpt-oss-20b</a> on Hugging Face</strong>
</p>

<br>

Welcome to the gpt-oss series, [OpenAI's open-weight models](https://openai.com/open-models/) designed for powerful reasoning, agentic tasks, and versatile developer use cases.

We're releasing two flavors of these open models:

- `gpt-oss-120b` — for production, general purpose, high reasoning use cases that fit into a single 80GB GPU (like NVIDIA H100 or AMD MI300X) (117B parameters with 5.1B active parameters)
- `gpt-oss-20b` — for lower latency, and local or specialized use cases (21B parameters with 3.6B active parameters)

Both models were trained using our [harmony response format][harmony] and should only be used with this format; otherwise, they will not work correctly.

## Table of Contents
- [Highlights](#highlights)
- [Inference examples](#inference-examples)
- [About this repository](#about-this-repository)
- [Setup](#setup)
- [Download the model](#download-the-model)
- [Reference PyTorch implementation](#reference-pytorch-implementation)
- [Reference Triton implementation (single GPU)](#reference-triton-implementation-single-gpu)
- [Reference Metal implementation](#reference-metal-implementation)
- [Harmony format & tools](#harmony-format--tools)
- [Clients](#clients)
- [Tools](#tools)
- [Other details](#other-details)
- [Contributing](#contributing)

### Highlights

- **Permissive Apache 2.0 license:** Build freely without copyleft restrictions or patent risk—ideal for experimentation, customization, and commercial deployment.
- **Configurable reasoning effort:** Easily adjust the reasoning effort (low, medium, high) based on your specific use case and latency needs.
- **Full chain-of-thought:** Provides complete access to the model's reasoning process, facilitating easier debugging and greater trust in outputs. This information is not intended to be shown to end users.
- **Fine-tunable:** Fully customize models to your specific use case through parameter fine-tuning.
- **Agentic capabilities:** Use the models' native capabilities for function calling, [web browsing](#browser), [Python code execution](#python), and Structured Outputs.
- **MXFP4 quantization:** The models were post-trained with MXFP4 quantization of the MoE weights, making `gpt-oss-120b` run on a single 80GB GPU (like NVIDIA H100 or AMD MI300X) and the `gpt-oss-20b` model run within 16GB of memory. All evals were performed with the same MXFP4 quantization.

### Inference examples

#### Transformers

You can use `gpt-oss-120b` and `gpt-oss-20b` with the Transformers library. If you use Transformers' chat template, it will automatically apply the [harmony response format][harmony]. If you use `model.generate` directly, you need to apply the harmony format manually using the chat template or use our [`openai-harmony`][harmony] package.

```python
from transformers import pipeline
import torch

model_id = "openai/gpt-oss-120b"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
```

[Learn more about how to use gpt-oss with Transformers.](https://cookbook.openai.com/articles/gpt-oss/run-transformers)

#### vLLM

vLLM recommends using [`uv`](https://docs.astral.sh/uv/) for Python dependency management. You can use vLLM to spin up an OpenAI-compatible web server. The following command will automatically download the model and start the server.

```bash
uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

vllm serve openai/gpt-oss-20b
```

[Learn more about how to use gpt-oss with vLLM.](https://cookbook.openai.com/articles/gpt-oss/run-vllm)

Offline Serve Code:
- run this code after installing proper libraries as described, while additionally installing this:
- `uv pip install openai-harmony`
```python
# source .oss/bin/activate

import os
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

import json
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
)
 
from vllm import LLM, SamplingParams
import os

# --- 1) Render the prefill with Harmony ---
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
 
convo = Conversation.from_messages(
    [
        Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
        Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent.new().with_instructions("Always respond in riddles"),
        ),
        Message.from_role_and_content(Role.USER, "What is the weather like in SF?"),
    ]
)
 
prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
 
# Harmony stop tokens (pass to sampler so they won't be included in output)
stop_token_ids = encoding.stop_tokens_for_assistant_actions()
 
# --- 2) Run vLLM with prefill ---
llm = LLM(
    model="openai/gpt-oss-20b",
    trust_remote_code=True,
    gpu_memory_utilization = 0.95,
    max_num_batched_tokens=4096,
    max_model_len=5000,
    tensor_parallel_size=1
)
 
sampling = SamplingParams(
    max_tokens=128,
    temperature=1,
    stop_token_ids=stop_token_ids,
)
 
outputs = llm.generate(
    prompt_token_ids=[prefill_ids],   # batch of size 1
    sampling_params=sampling,
)
 
# vLLM gives you both text and token IDs
gen = outputs[0].outputs[0]
text = gen.text
output_tokens = gen.token_ids  # <-- these are the completion token IDs (no prefill)
 
# --- 3) Parse the completion token IDs back into structured Harmony messages ---
entries = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
 
# 'entries' is a sequence of structured conversation entries (assistant messages, tool calls, etc.).
for message in entries:
    print(f"{json.dumps(message.to_dict())}")
```

#### PyTorch / Triton / Metal

These implementations are largely reference implementations for educational purposes and are not expected to be run in production.

[Learn more below.](#reference-pytorch-implementation)

#### Ollama

If you are trying to run `gpt-oss` on consumer hardware, you can use Ollama by running the following commands after [installing Ollama](https://ollama.com/download).

```bash
# gpt-oss-20b
ollama pull gpt-oss:20b
ollama run gpt-oss:20b

# gpt-oss-120b
ollama pull gpt-oss:120b
ollama run gpt-oss:120b
```

[Learn more about how to use gpt-oss with Ollama.](https://cookbook.openai.com/articles/gpt-oss/run-locally-ollama)

#### LM Studio

If you are using [LM Studio](https://lmstudio.ai/) you can use the following commands to download.

```bash
# gpt-oss-20b
lms get openai/gpt-oss-20b
# gpt-oss-120b
lms get openai/gpt-oss-120b
```

Check out our [awesome list](./awesome-gpt-oss.md) for a broader collection of gpt-oss resources and inference partners.

## About this repository

This repository provides a collection of reference implementations:

- **Inference:**
  - [`torch`](#reference-pytorch-implementation) — a non-optimized [PyTorch](https://pytorch.org/) implementation for educational purposes only. Requires at least 4× H100 GPUs due to lack of optimization.
  - [`triton`](#reference-triton-implementation-single-gpu) — a more optimized implementation using [PyTorch](https://pytorch.org/) & [Triton](https://github.com/triton-lang/triton) incl. using CUDA graphs and basic caching
  - [`metal`](#reference-metal-implementation) — a Metal-specific implementation for running the models on Apple Silicon hardware
- **Tools:**
  - [`browser`](#browser) — a reference implementation of the browser tool the models got trained on
  - [`python`](#python) — a stateless reference implementation of the python tool the model got trained on
- **Client examples:**
  - [`chat`](#terminal-chat) — a basic terminal chat application that uses the PyTorch or Triton implementations for inference along with the python and browser tools
  - [`responses_api`](#responses-api) — an example Responses API compatible server that implements the browser tool along with other Responses-compatible functionality

## Setup

### Requirements

- Python 3.12
- On macOS: Install the Xcode CLI tools --> `xcode-select --install`
- On Linux: These reference implementations require CUDA
- On Windows: These reference implementations have not been tested on Windows. Try using solutions like Ollama if you are trying to run the model locally.

### Installation

If you want to try any of the code you can install it directly from [PyPI](https://pypi.org/project/gpt-oss/)

```shell
# if you just need the tools
pip install gpt-oss
# if you want to try the torch implementation
pip install gpt-oss[torch]
# if you want to try the triton implementation
pip install gpt-oss[triton]
```

If you want to modify the code or try the metal implementation set the project up locally:

```shell
git clone https://github.com/openai/gpt-oss.git
GPTOSS_BUILD_METAL=1 pip install -e ".[metal]"
```

## Download the model

You can download the model weights from the [Hugging Face Hub](https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4) directly from Hugging Face CLI:

```shell
# gpt-oss-120b
hf download openai/gpt-oss-120b --include "original/*" --local-dir gpt-oss-120b/

# gpt-oss-20b
hf download openai/gpt-oss-20b --include "original/*" --local-dir gpt-oss-20b/
```

## Reference PyTorch implementation

We include an inefficient reference PyTorch implementation in [gpt_oss/torch/model.py](gpt_oss/torch/model.py). This code uses basic PyTorch operators to show the exact model architecture, with a small addition of supporting tensor parallelism in MoE so that the larger model can run with this code (e.g., on 4xH100 or 2xH200). In this implementation, we upcast all weights to BF16 and run the model in BF16.

To run the reference implementation, install the dependencies:

```shell
pip install -e ".[torch]"
```

And then run:

```shell
# On 4xH100:
torchrun --nproc-per-node=4 -m gpt_oss.generate gpt-oss-120b/original/
```

## Reference Triton implementation (single GPU)

We also include an optimized reference implementation that uses [an optimized triton MoE kernel](https://github.com/triton-lang/triton/tree/main/python/triton_kernels/triton_kernels) that supports MXFP4. It also has some optimization on the attention code to reduce the memory cost. To run this implementation, the nightly version of triton and torch will be installed. This version can be run on a single 80GB GPU for `gpt-oss-120b`.

To install the reference Triton implementation run

```shell
# You need to install triton from source to use the triton implementation
git clone https://github.com/triton-lang/triton
cd triton/
pip install -r python/requirements.txt
pip install -e . --verbose --no-build-isolation
pip install -e python/triton_kernels

# Install the gpt-oss triton implementation
pip install -e ".[triton]"
```

And then run:

```shell
# On 1xH100
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m gpt_oss.generate --backend triton gpt-oss-120b/original/
```

If you encounter `torch.OutOfMemoryError`, make sure to turn on the expandable allocator to avoid crashes when loading weights from the checkpoint.

## Reference Metal implementation

Additionally we are providing a reference implementation for Metal to run on Apple Silicon. This implementation is not production-ready but is accurate to the PyTorch implementation.

The implementation will get automatically compiled when running the `.[metal]` installation on an Apple Silicon device:

```shell
GPTOSS_BUILD_METAL=1 pip install -e ".[metal]"
```

To perform inference you'll need to first convert the SafeTensor weights from Hugging Face into the right format using:

```shell
python gpt_oss/metal/scripts/create-local-model.py -s <model_dir> -d <output_file>
```

Or download the pre-converted weights:

```shell
hf download openai/gpt-oss-120b --include "metal/*" --local-dir gpt-oss-120b/metal/
hf download openai/gpt-oss-20b --include "metal/*" --local-dir gpt-oss-20b/metal/
```

To test it you can run:

```shell
python gpt_oss/metal/examples/generate.py gpt-oss-20b/metal/model.bin -p "why did the chicken cross the road?"
```

## Harmony format & tools

Along with the model, we are also releasing a new chat format library `harmony` to interact with the model. Check [this guide](https://cookbook.openai.com/articles/openai-harmony) for more info about harmony.

We also include two system tools for the model: browsing and python container. Check [gpt_oss/tools](gpt_oss/tools) for the tool implementation.

## Clients

### Terminal Chat

The terminal chat application is a basic example of how to use the harmony format together with the PyTorch, Triton, and vLLM implementations. It also exposes both the python and browser tool as optional tools that can be used.

```bash
usage: python -m gpt_oss.chat [-h] [-r REASONING_EFFORT] [-a] [-b] [--show-browser-results] [-p] [--developer-message DEVELOPER_MESSAGE] [-c CONTEXT] [--raw] [--backend {triton,torch,vllm}] FILE

Chat example

positional arguments:
  FILE                  Path to the SafeTensors checkpoint

options:
  -h, --help            show this help message and exit
  -r REASONING_EFFORT, --reasoning-effort REASONING_EFFORT
                        Reasoning effort (default: low)
  -a, --apply-patch     Make apply_patch tool available to the model (default: False)
  -b, --browser         Use browser tool (default: False)
  --show-browser-results
                        Show browser results (default: False)
  -p, --python          Use python tool (default: False)
  --developer-message DEVELOPER_MESSAGE
                        Developer message (default: )
  -c CONTEXT, --context CONTEXT
                        Max context length (default: 8192)
  --raw                 Raw mode (does not render Harmony encoding) (default: False)
  --backend {triton,torch,vllm}
                        Inference backend (default: triton)
```

> [!NOTE]
> The torch and triton implementations require original checkpoint under `gpt-oss-120b/original/` and `gpt-oss-20b/original/` respectively. While vLLM uses the Hugging Face converted checkpoint under `gpt-oss-120b/` and `gpt-oss-20b/` root directory respectively.

### Responses API

We also include an example Responses API server. This server does not implement every feature and event of the Responses API but should be compatible with most of the basic use cases and serve as inspiration for anyone building their own server. Some of our inference partners are also offering their own Responses API.

You can start this server with the following inference backends:

- `triton` — uses the triton implementation
- `metal` — uses the metal implementation on Apple Silicon only
- `ollama` — uses the Ollama /api/generate API as an inference solution
- `vllm` — uses your installed vllm version to perform inference
- `transformers` — uses your installed transformers version to perform local inference

```bash
usage: python -m gpt_oss.responses_api.serve [-h] [--checkpoint FILE] [--port PORT] [--inference-backend BACKEND]

Responses API server

options:
  -h, --help                    show this help message and exit
  --checkpoint FILE             Path to the SafeTensors checkpoint
  --port PORT                   Port to run the server on
  --inference-backend BACKEND   Inference backend to use
```

### Codex

We support [codex](https://github.com/openai/codex) as a client for gpt-oss. To run the 20b version, set this to `~/.codex/config.toml`:

```
disable_response_storage = true
show_reasoning_content = true

[model_providers.local]
name = "local"
base_url = "http://localhost:11434/v1"

[profiles.oss]
model = "gpt-oss:20b"
model_provider = "local"
```

This will work with any chat completions-API compatible server listening on port 11434, like ollama. Start the server and point codex to the oss model:

```
ollama run gpt-oss:20b
codex -p oss
```

## Tools

### Browser

> [!WARNING]
> This implementation is purely for educational purposes and should not be used in production. You should implement your own equivalent of the [`YouComBackend`](gpt_oss/tools/simple_browser/backend.py) class with your own browsing environment. Currently we have available `YouComBackend` and `ExaBackend`. 

Both gpt-oss models were trained with the capability to browse using the `browser` tool that exposes the following three methods:

- `search` to search for key phrases
- `open` to open a particular page
- `find` to look for contents on a page

#### Usage

To enable the browser tool, you'll have to place the definition into the `system` message of your harmony formatted prompt. You can either use the `with_browser_tool()` method if your tool implements the full interface or modify the definition using `with_tools()`. For example:

```python
import datetime
from gpt_oss.tools.simple_browser import SimpleBrowserTool
from gpt_oss.tools.simple_browser.backend import YouComBackend
from openai_harmony import SystemContent, Message, Conversation, Role, load_harmony_encoding, HarmonyEncodingName

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

# Depending on the choice of the browser backend you need corresponding env variables setup
# In case you use You.com backend requires you to have set the YDC_API_KEY environment variable,
# while for Exa you might need EXA_API_KEY environment variable set
backend = YouComBackend(
    source="web",
)
# backend = ExaBackend(
#  source="web",
# )
browser_tool = SimpleBrowserTool(backend=backend)

# create a basic system prompt
system_message_content = SystemContent.new().with_conversation_start_date(
    datetime.datetime.now().strftime("%Y-%m-%d")
)

# if you want to use the browser tool
if use_browser_tool:
    # enables the tool
    system_message_content = system_message_content.with_tools(browser_tool.tool_config)
    # alternatively you could use the following if your tool is not stateless
    system_message_content = system_message_content.with_browser_tool()

# construct the system message
system_message = Message.from_role_and_content(Role.SYSTEM, system_message_content)

# create the overall prompt
messages = [system_message, Message.from_role_and_content(Role.USER, "What's the weather in SF?")]
conversation = Conversation.from_messages(messages)

# convert to tokens
token_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

# perform inference
# ...

# parse the output
messages = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
last_message = messages[-1]
if last_message.recipient.startswith("browser"):
  # perform browser call
  response_messages = await browser_tool.process(last_message)

  # extend the current messages and run inference again
  messages.extend(response_messages)
```

#### Details

To control the context window size this tool uses a scrollable window of text that the model can interact with. So it might fetch the first 50 lines of a page and then scroll to the next 20 lines after that. The model has also been trained to then use citations from this tool in its answers.

To improve performance the tool caches requests so that the model can revisit a different part of a page without having to reload the page. For that reason you should create a new browser instance for every request.

### Python

The model was trained to use a python tool to perform calculations and other actions as part of its chain-of-thought. During the training the model used a stateful tool which makes running tools between CoT loops easier. This reference implementation, however, uses a stateless mode. As a result the PythonTool defines its own tool description to override the definition in [`openai-harmony`][harmony].

> [!WARNING]
> This implementation runs in a permissive Docker container which could be problematic in cases like prompt injections. It's serving as an example and you should consider implementing your own container restrictions in production.

#### Usage

To enable the python tool, you'll have to place the definition into the `system` message of your harmony formatted prompt. You can either use the `with_python()` method if your tool implements the full interface or modify the definition using `with_tools()`. For example:

```python
import datetime
from gpt_oss.tools.python_docker.docker_tool import PythonTool
from openai_harmony import SystemContent, Message, Conversation, Role, load_harmony_encoding, HarmonyEncodingName

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

python_tool = PythonTool()

# create a basic system prompt
system_message_content = SystemContent.new().with_conversation_start_date(
    datetime.datetime.now().strftime("%Y-%m-%d")
)

# if you want to use the python tool
if use_python_tool:
    # enables the tool making sure that the prompt gets set with the stateless tool description
    system_message_content = system_message_content.with_tools(python_tool.tool_config)
    # alternatively you could use the following if your tool is not stateless
    system_message_content = system_message_content.with_python()

# construct the system message
system_message = Message.from_role_and_content(Role.SYSTEM, system_message_content)

# create the overall prompt
messages = [system_message, Message.from_role_and_content(Role.USER, "What's the square root of 9001?")]
conversation = Conversation.from_messages(messages)

# convert to tokens
token_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

# perform inference
# ...

# parse the output
messages = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
last_message = messages[-1]
if last_message.recipient == "python":
  # perform python call
  response_messages = await python_tool.process(last_message)

  # extend the current messages and run inference again
  messages.extend(response_messages)
```

### Apply Patch

`apply_patch` can be used to create, update or delete files locally.

## Other details

### Precision format

We released the models with native quantization support. Specifically, we use [MXFP4](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) for the linear projection weights in the MoE layer. We store the MoE tensor in two parts:

- `tensor.blocks` stores the actual fp4 values. We pack every two values in one `uint8` value.
- `tensor.scales` stores the block scale. The block scaling is done among the last dimension for all MXFP4 tensors.

All other tensors will be in BF16. We also recommend using BF16 as the activation precision for the model.

### Recommended Sampling Parameters

We recommend sampling with `temperature=1.0` and `top_p=1.0`.

## Contributing

The reference implementations in this repository are meant as a starting point and inspiration. Outside of bug fixes we do not intend to accept new feature contributions. If you build implementations based on this code such as new tool implementations you are welcome to contribute them to the [`awesome-gpt-oss.md`](./awesome-gpt-oss.md) file.

[harmony]: https://github.com/openai/harmony

## Citation

```bibtex
@misc{openai2025gptoss120bgptoss20bmodel,
      title={gpt-oss-120b & gpt-oss-20b Model Card}, 
      author={OpenAI},
      year={2025},
      eprint={2508.10925},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.10925}, 
}
```
