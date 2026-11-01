# GPT-OSS Examples

This directory contains practical examples demonstrating how to use GPT-OSS models with different frameworks and tools.

## 📁 Examples Overview

### 🤖 **Agents SDK Examples**
- **Python**: `agents-sdk-python/` - Example using OpenAI's Agents SDK with Python
- **JavaScript**: `agents-sdk-js/` - Example using OpenAI's Agents SDK with TypeScript/JavaScript

### 🎨 **Streamlit Chat Interface**
- **Streamlit**: `streamlit/` - Interactive web-based chat interface using Streamlit

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Node.js 18+ (for JavaScript examples)
- GPT-OSS model running locally (via Ollama, vLLM, or other inference backend)

### Running the Examples

#### 1. Agents SDK (Python)
```bash
cd examples/agents-sdk-python
pip install -r requirements.txt  # if requirements.txt exists
python example.py
```

#### 2. Agents SDK (JavaScript)
```bash
cd examples/agents-sdk-js
npm install
npm start
```

#### 3. Streamlit Chat Interface
```bash
cd examples/streamlit
pip install streamlit requests
streamlit run streamlit_chat.py
```

## 🔧 Configuration

### Local Model Setup
Most examples expect a GPT-OSS model running locally. You can use:

- **Ollama**: `ollama run gpt-oss:20b`
- **vLLM**: `vllm serve openai/gpt-oss-20b`
- **Local Responses API**: Run the included responses API server

### Environment Variables
Some examples may require environment variables:
```bash
export OPENAI_API_KEY="local"  # for local models
export OPENAI_BASE_URL="http://localhost:11434/v1"  # Ollama default
```

## 📚 Example Details

### Agents SDK Examples
These examples demonstrate:
- Setting up GPT-OSS with OpenAI's Agents SDK
- Using function calling and tools
- MCP (Model Context Protocol) integration
- Streaming responses

### Streamlit Chat Interface
Features:
- Interactive web-based chat
- Model selection (large/small)
- Reasoning effort control
- Function calling support
- Browser search integration
- Debug mode for development

## 🛠️ Troubleshooting

### Common Issues

1. **Connection Refused**: Make sure your local model server is running
2. **Model Not Found**: Verify the model name matches your local setup
3. **Port Conflicts**: Check that ports 11434 (Ollama) or 8000 (vLLM) are available

### Getting Help
- Check the main [README.md](../README.md) for setup instructions
- Review the [awesome-gpt-oss.md](../awesome-gpt-oss.md) for additional resources
- Open an issue on GitHub for bugs or questions

## 🤝 Contributing

We welcome improvements to these examples! Please:
- Add clear comments and documentation
- Include setup instructions
- Test with different model backends
- Follow the project's coding standards

## 📖 Related Documentation

- [Main README](../README.md) - Project overview and setup
- [Tools Documentation](../gpt_oss/tools/) - Available tools and their usage
- [Responses API](../gpt_oss/responses_api/) - API server implementation
