# GPT-OSS Mac Setup Guide

This guide provides step-by-step instructions for setting up and running GPT-OSS on macOS, including the Responses API server.

## Prerequisites

### System Requirements
- **macOS**: 10.15 (Catalina) or later
- **Architecture**: Apple Silicon (M1/M2/M3) recommended for Metal backend
- **Memory**: 16GB+ RAM recommended for gpt-oss-20b, 32GB+ for gpt-oss-120b
- **Storage**: 50GB+ free space for model downloads

### Required Tools

#### 1. Xcode Command Line Tools
```bash
xcode-select --install
```

#### 2. Python 3.12
Install via Homebrew (recommended):
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.12
brew install python@3.12
```

Or download from [python.org](https://www.python.org/downloads/macos/)

#### 3. Hugging Face CLI (for model downloads)
```bash
pip3 install huggingface_hub
```

## Installation Options

### Option 1: Metal Backend (Recommended for Apple Silicon)

**Best performance on M1/M2/M3 Macs**

1. **Clone and install with Metal support**:
   ```bash
   git clone https://github.com/openai/gpt-oss.git
   cd gpt-oss
   GPTOSS_BUILD_METAL=1 pip3 install -e ".[metal]"
   ```

2. **Download model weights**:
   ```bash
   # Download pre-converted Metal weights (recommended)
   hf download openai/gpt-oss-20b --include "metal/*" --local-dir gpt-oss-20b/
   
   # Or for 120b model (requires more RAM)
   hf download openai/gpt-oss-120b --include "metal/*" --local-dir gpt-oss-120b/
   ```

3. **Start server with Metal backend**:
   ```bash
   python3 -m gpt_oss.responses_api.serve --inference-backend metal --checkpoint gpt-oss-20b/metal/model.bin --port 8080
   ```

### Option 2: Quick Start with Ollama (Alternative)

**Easier setup, but less optimized for Apple Silicon**

1. **Install Ollama**:
   ```bash
   brew install ollama
   ```

2. **Install GPT-OSS**:
   ```bash
   pip3 install gpt-oss
   ```

3. **Download and run model**:
   ```bash
   # Download gpt-oss-20b (smaller, faster)
   ollama pull gpt-oss:20b
   
   # Or download gpt-oss-120b (larger, more capable)
   ollama pull gpt-oss:120b
   ```

4. **Start Responses API server**:
   ```bash
   python3 -m gpt_oss.responses_api.serve --inference-backend ollama --checkpoint "gpt-oss:20b" --port 8080
   ```

### Option 3: Development Setup

1. **Clone repository**:
   ```bash
   git clone https://github.com/openai/gpt-oss.git
   cd gpt-oss
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install in development mode**:
   ```bash
   pip install -e ".[torch,triton,metal]"
   ```

## Usage Examples

### Basic Chat Interface
```bash
# Terminal chat with Metal backend (recommended)
python3 -m gpt_oss.chat gpt-oss-20b/metal/model.bin --backend metal

# With browser tool enabled
python3 -m gpt_oss.chat gpt-oss-20b/metal/model.bin --backend metal --browser

# With Python code execution
python3 -m gpt_oss.chat gpt-oss-20b/metal/model.bin --backend metal --python

# Alternative: Ollama backend
python3 -m gpt_oss.chat gpt-oss:20b --backend vllm --browser
```

### Responses API Server

Start the server (Metal backend recommended):
```bash
# Metal backend (best performance)
python3 -m gpt_oss.responses_api.serve --inference-backend metal --checkpoint gpt-oss-20b/metal/model.bin --port 8080

# Alternative: Ollama backend (easier setup)
python3 -m gpt_oss.responses_api.serve --inference-backend ollama --checkpoint "gpt-oss:20b" --port 8080
```

Test with curl:
```bash
curl -X POST http://127.0.0.1:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Write a haiku about coding",
    "reasoning": {"effort": "low"},
    "stream": false
  }'
```

### Python Integration
```python
import requests

response = requests.post("http://127.0.0.1:8080/v1/responses", json={
    "input": [
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "text", "text": "You are a helpful coding assistant."}]
        },
        {
            "type": "message", 
            "role": "user",
            "content": [{"type": "text", "text": "Explain async/await in Python"}]
        }
    ],
    "reasoning": {"effort": "medium"},
    "tools": [{"type": "web_search_preview"}],
    "stream": False
})

print(response.json())
```

## Testing

### Run Tests
```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests
pytest

# Run specific test categories
pytest tests/test_api_endpoints.py -v
pytest tests/test_responses_api.py -v

# Run with coverage
pytest --cov=gpt_oss tests/
```

### Test the API Server
```bash
# Start server in background (Metal backend)
python3 -m gpt_oss.responses_api.serve --inference-backend metal --checkpoint gpt-oss-20b/metal/model.bin --port 8080 &

# Or with Ollama backend
python3 -m gpt_oss.responses_api.serve --inference-backend ollama --checkpoint "gpt-oss:20b" --port 8080 &

# Run API tests
pytest tests/test_api_endpoints.py::TestCompatibilityFeatures -v

# Stop background server
pkill -f "gpt_oss.responses_api.serve"
```

## Troubleshooting

### Common Issues

#### 1. Metal Compilation Errors
If you see C++ compilation errors with Metal (like `std::format` not found):
```bash
# Fallback: Use Ollama backend instead
brew install ollama
ollama pull gpt-oss:20b
pip3 install gpt-oss  # Basic installation without Metal

# Then use Ollama backend
python3 -m gpt_oss.responses_api.serve --inference-backend ollama --checkpoint "gpt-oss:20b" --port 8080
```

#### 2. Memory Issues
For large models on limited RAM:
```bash
# Use smaller model
ollama pull gpt-oss:20b  # Instead of 120b

# Or reduce context length
python3 -m gpt_oss.chat gpt-oss:20b --context 4096
```

#### 3. Port Already in Use
```bash
# Check what's using the port
lsof -i :8080

# Use different port
python3 -m gpt_oss.responses_api.serve --port 8081
```

#### 4. Model Download Issues
```bash
# Set HF token if needed
export HF_TOKEN="your_token_here"

# Or download manually
hf download openai/gpt-oss-20b --local-dir ./gpt-oss-20b/
```

### Performance Tips

#### 1. Optimize for Apple Silicon
- Use Metal backend when possible
- Enable unified memory: `export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`

#### 2. Memory Management
```bash
# For Metal backend
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Monitor memory usage
top -pid $(pgrep -f gpt_oss)
```

#### 3. Speed Optimization
- Use gpt-oss-20b for faster responses
- Set reasoning effort to "low" for speed
- Reduce max_output_tokens for shorter responses

## Environment Variables

```bash
# Optional: Set browser tool backend
export BROWSER_BACKEND=exa  # or youcom
export EXA_API_KEY="your_exa_key"
export YDC_API_KEY="your_youcom_key"

# Optional: Hugging Face token for private models
export HF_TOKEN="your_hf_token"

# Optional: Metal optimization
export GPTOSS_BUILD_METAL=1
```

## Integration Examples

### With OpenAI Python SDK
```python
from openai import OpenAI

# Point to local server
client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed"  # Local server doesn't require auth
)

# Use Responses API
response = client.responses.create(
    model="gpt-oss-20b",
    input="What is quantum computing?",
    reasoning={"effort": "low"}
)

print(response.output)
```

### With Cursor/VS Code
Configure in your IDE settings:
```json
{
  "ai.baseUrl": "http://127.0.0.1:8080/v1",
  "ai.model": "gpt-oss-20b"
}
```

## Next Steps

- Explore the [main README](README.md) for more advanced usage
- Check out [example implementations](examples/)
- Read about [harmony format](https://github.com/openai/harmony)
- Join the community discussions

## Support

- **Issues**: [GitHub Issues](https://github.com/openai/gpt-oss/issues)
- **Discussions**: [GitHub Discussions](https://github.com/openai/gpt-oss/discussions)
- **Documentation**: [OpenAI Cookbook](https://cookbook.openai.com/topic/gpt-oss)
