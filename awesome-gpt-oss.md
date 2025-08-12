![gpt-oss](./docs/gpt-oss.svg)

# Awesome gpt-oss

This is a list of guides and resources to help you get started with the gpt-oss models.

- [Inference](#inference)
  - [Local](#local)
  - [Server](#server)
  - [Cloud](#cloud)
- [Examples / Tutorials](#examples--tutorials)
- [Tools](#tools)

## Inference

### Local

- Ollama
  - [How to run gpt-oss locally with Ollama](https://cookbook.openai.com/articles/gpt-oss/run-locally-ollama)
  - [Ollama & gpt-oss launch blog](https://ollama.com/blog/gpt-oss)
  - [Check out the models Ollama](https://ollama.com/library/gpt-oss)
- LM Studio
  - [LM Studio & gpt-oss launch blog](https://lmstudio.ai/blog/gpt-oss)
  - [Use gpt-oss-20b with LM Studio](https://lmstudio.ai/models/openai/gpt-oss-20b)
  - [Use gpt-oss-120b with LM Studio](https://lmstudio.ai/models/openai/gpt-oss-120b)
- Hugging Face & Transformers
  - [How to run gpt-oss with Transformers](https://cookbook.openai.com/articles/gpt-oss/run-transformers)
  - [Hugging Face & gpt-oss launch blog](https://huggingface.co/blog/welcome-openai-gpt-oss)
  - [Collection of Hugging Face examples](https://github.com/huggingface/gpt-oss-recipes)
- NVIDIA
  - [gpt-oss on RTX](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss)
- AMD
  - [Running gpt-oss models on AMD Ryzen AI Processors and Radeon Graphics Cards](https://www.amd.com/en/blogs/2025/how-to-run-openai-gpt-oss-20b-120b-models-on-amd-ryzen-ai-radeon.html)

### Server

- vLLM
  - [How to run gpt-oss with vLLM](https://cookbook.openai.com/articles/gpt-oss/run-vllm)
  - [vLLM & gpt-oss recipies](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html)
- NVIDIA
  - [Optimizing gpt-oss with NVIDIA TensorRT-LLM](https://cookbook.openai.com/articles/run-nvidia)
  - [Deploying gpt-oss on TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog9_Deploying_GPT_OSS_on_TRTLLM.md)
- AMD
  - [Running the Latest Open Models from OpenAI on AMD AI Hardware](https://rocm.blogs.amd.com/ecosystems-and-partners/openai-day-0/README.html)

### Cloud

- Groq
  - [Groq & gpt-oss launch blog](https://groq.com/blog/day-zero-support-for-openai-open-models)
  - [gpt-oss-120b model on the GroqCloud Playground](https://console.groq.com/playground?model=openai/gpt-oss-120b)
  - [gpt-oss-20b model on the GroqCloud Playground](https://console.groq.com/playground?model=openai/gpt-oss-20b)
  - [gpt-oss with built-in web search on GroqCloud](https://console.groq.com/docs/browser-search)
  - [gpt-oss with built-in code execution on GroqCloud](https://console.groq.com/docs/code-execution)
  - [Responses API on Groq](https://console.groq.com/docs/responses-api)
- NVIDIA
  - [NVIDIA launch blog post](https://blogs.nvidia.com/blog/openai-gpt-oss/)
  - [NVIDIA & gpt-oss developer launch blog post](https://developer.nvidia.com/blog/delivering-1-5-m-tps-inference-on-nvidia-gb200-nvl72-nvidia-accelerates-openai-gpt-oss-models-from-cloud-to-edge/)
  - Use [gpt-oss-120b](https://build.nvidia.com/openai/gpt-oss-120b) and [gpt-oss-20b](https://build.nvidia.com/openai/gpt-oss-20b) on NVIDIA's Cloud
- Cloudflare
  - [Cloudflare & gpt-oss launch blog post](https://blog.cloudflare.com/openai-gpt-oss-on-workers-ai)
  - [gpt-oss-120b on Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/models/gpt-oss-120b)
  - [gpt-oss-20b on Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/models/gpt-oss-20b)
- AMD
  - [gpt-oss-120B on AMD MI300X](https://huggingface.co/spaces/amd/gpt-oss-120b-chatbot)

## Examples & Tutorials

- [OpenAI harmony response format](https://cookbook.openai.com/articles/openai-harmony)

## Tools

- [Example `python` tool for gpt-oss](./gpt_oss/tools/python_docker/)
- [Example `browser` tool for gpt-oss](./gpt_oss/tools/simple_browser/)

## Contributing

Feel free to open a PR to add your own guides and resources on how to run gpt-oss. We will try to review it and add it here.
