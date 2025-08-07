<img alt="gpt-oss-120" src="./docs/gpt-oss.svg">

<p align="center">
  <a href="https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue" alt="Hugging Face Models"></a>
  <a href="https://github.com/openai/gpt-oss/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License"></a>
  <a href="https://cookbook.openai.com/topic/gpt-oss"><img src="https://img.shields.io/badge/OpenAI-Cookbook_Guides-orange" alt="OpenAI Cookbook"></a>
  <a href="https://openai.com/index/introducing-gpt-oss/"><img src="https://img.shields.io/badge/Blog-Introducing_gpt--oss-black" alt="OpenAI Blog"></a>
</p>

<p align="center">
  <a href="https://gpt-oss.com"><strong>Try gpt-oss</strong></a> ·
  <a href="https://cookbook.openai.com/topic/gpt-oss"><strong>Guides</strong></a> ·
  <a href="https://openai.com/index/gpt-oss-model-card"><strong>Model Card</strong></a>
</p>
<p align="center">
  <strong>Download <a href="https://huggingface.co/openai/gpt-oss-120b">gpt-oss-120b</a> and <a href="https://huggingface.co/openai/gpt-oss-20b">gpt-oss-20b</a> on Hugging Face</strong>
</p>

---

Welcome to the `gpt-oss` series, [OpenAI's family of open-weight models](https://openai.com/open-models/) designed for powerful reasoning, agentic tasks, and versatile developer use cases.

We are releasing two powerful open models:

-   `gpt-oss-120b`: Our premier model for production-grade, general-purpose, and high-reasoning tasks. It features 117B parameters (5.1B active) and is optimized to run on a single NVIDIA H100 GPU.
-   `gpt-oss-20b`: A compact and efficient model for lower-latency needs, local deployment, or specialized applications. It has 21B parameters (3.6B active) and can run within 16GB of memory.

> **Note:** Both models were trained using our custom `harmony` response format. They must be used with this format to function correctly; otherwise, outputs will be incoherent. Inference solutions like Transformers, vLLM, and Ollama handle this formatting automatically.

## Key Features

-   **Permissive Apache 2.0 License**: Build, experiment, customize, and deploy commercially without copyleft restrictions or patent risks.
-   **Configurable Reasoning Effort**: Easily adjust the model's reasoning effort (low, medium, high) to balance performance and latency for your specific needs.
-   **Full Chain-of-Thought Access**: Gain complete visibility into the model's reasoning process for easier debugging, enhanced trust, and deeper integration. This data is not intended for end-users.
-   **Fully Fine-Tunable**: Adapt the models to your specific domain or task through full parameter fine-tuning.
-   **Agentic Capabilities**: Natively leverage function calling, [web browsing](#browser), [Python code execution](#python), and structured outputs for complex, agent-like tasks.
-   **Native MXFP4 Quantization**: Models are trained with native MXFP4 precision for the Mixture-of-Experts (MoE) layer, a key innovation that enables `gpt-oss-120b` to run on a single H100 GPU and `gpt-oss-20b` to fit within 16GB of memory.

---

## Table of Contents

-   [Quickstart: Inference](#quickstart-inference)
    -   [Hugging Face Transformers](#transformers)
    -   [vLLM](#vllm)
    -   [Ollama (Local)](#ollama)
    -   [LM Studio (Local)](#lm-studio)
-   [About This Repository](#about-this-repository)
-   [Setup and Installation](#setup-and-installation)
-   [Model Downloads](#download-the-model)
-   [Reference Implementations](#reference-implementations)
    -   [PyTorch](#reference-pytorch-implementation)
    -   [Triton (Single GPU)](#reference-triton-implementation-single-gpu)
    -   [Metal (Apple Silicon)](#reference-metal-implementation)
-   [Core Concepts](#core-concepts)
    -   [The Harmony Format](#harmony-format--tools)
    -   [MXFP4 Precision](#precision-format)
-   [Clients & Integrations](#clients)
    -   [Terminal Chat](#terminal-chat)
    -   [Responses API Server](#responses-api)
    -   [Codex](#codex)
-   [Built-in Tools](#tools)
    -   [Browser](#browser)
    -   [Python](#python)
    -   [Apply Patch](#apply-patch)
-   [Recommended Sampling Parameters](#recommended-sampling-parameters)
-   [Contributing](#contributing)

---

## Quickstart: Inference

### Transformers

Use the `transformers` library for easy integration. The library's chat templates automatically apply the required `harmony` response format.

```python
from transformers import pipeline
import torch

# Choose your model
model_id = "openai/gpt-oss-120b"

# Set up the pipeline
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

# Prepare your messages
messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

# Get the response
outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
```

[**Learn more about using gpt-oss with Transformers.**](https://cookbook.openai.com/articles/gpt-oss/run-transformers)

### vLLM

For high-throughput serving, `vLLM` provides an OpenAI-compatible web server. The following command downloads the model and starts the server.

```bash
# Install vLLM with gpt-oss support (uv is recommended)
uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

# Serve the model
vllm serve openai/gpt-oss-20b
```

[**Learn more about using gpt-oss with vLLM.**](https://cookbook.openai.com/articles/gpt-oss/run-vllm)

### Ollama

To run `gpt-oss` on local consumer hardware, use [Ollama](https://ollama.com/).

```bash
# Pull and run the 20B model
ollama pull gpt-oss:20b
ollama run gpt-oss:20b

# Or, pull and run the 120B model
ollama pull gpt-oss:120b
ollama run gpt-oss:120b```

[**Learn more about using gpt-oss with Ollama.**](https://cookbook.openai.com/articles/gpt-oss/run-locally-ollama)

### LM Studio

You can also run the models locally using [LM Studio](https://lmstudio.ai/).

```bash
# Get the 20B model
lms get openai/gpt-oss-20b

# Get the 120B model
lms get openai/gpt-oss-120b
```

---

## About This Repository

This repository provides official reference implementations and tools for `gpt-oss`:

-   **Inference Implementations**:
    -   `torch`: A non-optimized PyTorch implementation for educational purposes.
    -   `triton`: An optimized implementation using Triton for high-performance inference on NVIDIA GPUs.
    -   `metal`: A reference implementation for running on Apple Silicon hardware.
-   **Tools**:
    -   `browser`: A reference implementation of the web browsing tool the models were trained on.
    -   `python`: A stateless reference implementation of the Python execution tool.
-   **Client Examples**:
    -   `chat`: A basic terminal chat application demonstrating tools and various backends.
    -   `responses_api`: An example server compatible with the Responses API.

Check out our [awesome-gpt-oss list](./awesome-gpt-oss.md) for a broader collection of community resources and inference partners.

---

## Setup and Installation

### Requirements

-   Python 3.12+
-   **macOS**: Xcode CLI tools (`xcode-select --install`).
-   **Linux**: CUDA is required for the PyTorch and Triton reference implementations.
-   **Windows**: The reference implementations are not tested on Windows. Please use solutions like Ollama or LM Studio for local execution.

### Installation

Install the library directly from PyPI:

```shell
# Install only the tools (browser, python)
pip install gpt-oss

# To use the reference torch implementation
pip install gpt-oss[torch]

# To use the reference triton implementation
pip install gpt-oss[triton]
```

To contribute or modify the code, clone the repository and install it in editable mode:

```shell
git clone https://github.com/openai/gpt-oss.git
cd gpt-oss

# To include the Metal implementation for Apple Silicon
GPTOSS_BUILD_METAL=1 pip install -e ".[metal]"
```

---

## Download the Model

Download the model weights from the Hugging Face Hub using the `huggingface-cli`:

```shell
# Download gpt-oss-120b (original weights for reference implementations)
huggingface-cli download openai/gpt-oss-120b --include "original/*" --local-dir gpt-oss-120b/

# Download gpt-oss-20b (original weights)
huggingface-cli download openai/gpt-oss-20b --include "original/*" --local-dir gpt-oss-20b/
```

---

## Reference Implementations

These are provided for educational and research purposes and are not optimized for production use.

### Reference PyTorch Implementation

This simple implementation in `gpt_oss/torch/model.py` demonstrates the model architecture using basic PyTorch operators.

```shell
# Install dependencies
pip install -e .[torch]

# Run on 4x H100 GPUs
torchrun --nproc-per-node=4 -m gpt_oss.generate gpt-oss-120b/original/
```

### Reference Triton Implementation (Single GPU)

This version uses an optimized Triton kernel for MoE that supports MXFP4, allowing `gpt-oss-120b` to run on a single 80GB GPU.

```shell
# You must install Triton from source
git clone https://github.com/triton-lang/triton
cd triton/
pip install -e . --verbose --no-build-isolation

# Install the gpt-oss Triton dependencies
pip install -e .[triton]

# Run on 1x H100 GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m gpt_oss.generate --backend triton gpt-oss-120b/original/
```

> **Note:** If you encounter a `torch.OutOfMemoryError`, ensure the expandable allocator is enabled.

### Reference Metal Implementation

This implementation allows the models to run on Apple Silicon.

```shell
# Install with Metal support
pip install -e .[metal]

# Convert Hugging Face weights to Metal format
python gpt_oss/metal/scripts/create-local-model.py -s <model_dir> -d <output_file>

# Or download pre-converted weights
huggingface-cli download openai/gpt-oss-20b --include "metal/*" --local-dir gpt-oss-20b/metal/

# Run inference
python gpt_oss/metal/examples/generate.py gpt-oss-20b/metal/model.bin -p "Why did the chicken cross the road?"
```

---

## Core Concepts

### The Harmony Format & Tools

The `gpt-oss` models are trained on `harmony`, a specialized chat format designed to handle complex conversational structures, chain-of-thought reasoning, and tool use. It separates model outputs into different channels, such as `analysis` for internal thoughts and `final` for the user-facing response. Our `openai-harmony` library handles this encoding and decoding for you.

[**Learn more about the Harmony format in this guide.**](https://cookbook.openai.com/articles/openai-harmony)

### Precision Format

We use **MXFP4 (Microscaling FP4)**, a 4-bit floating-point format, for the linear projection weights in the MoE layers. This fine-grained quantization significantly reduces the memory footprint and is the key to running our largest models on a single GPU with minimal accuracy loss. All other tensors are in BF16, which is also the recommended activation precision.

---

## Clients & Integrations

### Terminal Chat

A basic terminal chat client is included to demonstrate how to use the `harmony` format with different backends (PyTorch, Triton, vLLM) and tools.

```bash
# See all available options
python -m gpt_oss.chat --help

# Example: Run with the browser and python tools enabled
python -m gpt_oss.chat gpt-oss-20b/ --backend vllm -b -p
```

### Responses API

We provide an example server that is compatible with the OpenAI Responses API. It supports various backends and serves as a starting point for building your own integrations.

```bash
# See all available options
python -m gpt_oss.responses_api.serve --help

# Example: Start the server with the Ollama backend
python -m gpt_oss.responses_api.serve --inference-backend ollama
```

### Codex

You can use `gpt-oss` with [Codex](https://github.com/openai/codex). Point it to any OpenAI-compatible server (like the one from Ollama or vLLM). Edit `~/.codex/config.toml`:

```toml
disable_response_storage = true
show_reasoning_content = true

[model_providers.local]
name = "local"
base_url = "http://localhost:11434/v1"

[profiles.oss]
model = "gpt-oss:20b"
model_provider = "local"
```

Then, run the Ollama server and start Codex:

```bash
ollama run gpt-oss:20b
codex -p oss
```

---

## Built-in Tools

### Browser

> **Warning:** This browser tool is an educational reference and not for production use. Implement your own secure browsing environment based on the `ExaBackend` class.

The `browser` tool enables models to `search` the web, `open` pages, and `find` content, allowing them to access information beyond their training data.

### Python

> **Warning:** This reference Python tool runs code in a permissive Docker container. For production, you must implement your own sandboxed environment with appropriate restrictions to mitigate risks like prompt injection.

The `python` tool allows the model to execute Python code to perform calculations, data manipulation, and more as part of its reasoning process.

### Apply Patch

The `apply_patch` tool can be used by the model to create, update, or delete files on the local file system.

---

## Recommended Sampling Parameters

For best results, we recommend sampling with the following parameters:
-   `temperature=1.0`
-   `top_p=1.0`

---

## Contributing

The reference implementations in this repository are intended as a starting point and for educational purposes. While we welcome bug fixes, we do not plan to accept major feature contributions to this codebase.

If you build something new using `gpt-oss`, such as a novel tool implementation or a new inference backend, we encourage you to share it with the community! Please open a pull request to add your project to our [**awesome-gpt-oss.md**](./awesome-gpt-oss.md) list.

[harmony]: https://github.com/openai/harmony
