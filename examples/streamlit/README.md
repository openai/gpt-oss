# Streamlit Chat Interface

This example provides an interactive web-based chat interface for GPT-OSS using Streamlit.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- GPT-OSS model running locally (Ollama, vLLM, etc.)
- Streamlit

### Installation

1. **Install dependencies:**
```bash
pip install streamlit requests
```

2. **Install GPT-OSS (if not already installed):**
```bash
pip install -e ../../  # From the root directory
```

### Running the Example

1. **Start your GPT-OSS model server:**
```bash
# Using vLLM (recommended)
vllm serve openai/gpt-oss-20b --port 8000

# Using Ollama with Responses API
ollama run gpt-oss:20b --port 8000

# Using local Responses API server
python -m gpt_oss.responses_api.serve --port 8000
```

2. **Run the Streamlit app:**
```bash
streamlit run streamlit_chat.py
```

3. **Open your browser** to the URL shown in the terminal (usually `http://localhost:8501`)

## 🎨 Features

### Model Selection
- **Large Model**: Uses `localhost:8000` (gpt-oss-120b)
- **Small Model**: Uses `localhost:8081` (gpt-oss-20b)

### Chat Interface
- **Interactive Chat**: Real-time conversation with the model
- **Message History**: View and continue previous conversations
- **Streaming Responses**: See responses as they're generated

### Configuration Options

#### Reasoning Effort
- **Low**: Fast responses, minimal reasoning
- **Medium**: Balanced speed and reasoning
- **High**: Maximum reasoning effort

#### Tools and Functions
- **Browser Search**: Enable web search capabilities
- **Function Calling**: Use custom functions
- **Apply Patch**: File manipulation capabilities

#### Generation Parameters
- **Temperature**: Control response randomness (0.0-1.0)
- **Max Output Tokens**: Limit response length (1000-20000)

### Debug Features
- **Debug Mode**: View raw conversation data
- **JSON Output**: See the full conversation structure
- **Tool Interactions**: Monitor function calls and responses

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set default model server URLs
export GPTOSS_LARGE_MODEL_URL="http://localhost:8000/v1"
export GPTOSS_SMALL_MODEL_URL="http://localhost:8081/v1"
```

### Custom Functions
You can define custom functions in the sidebar:

```json
{
  "type": "object",
  "properties": {
    "location": {
      "type": "string",
      "description": "The city and state, e.g. San Francisco, CA"
    }
  },
  "required": ["location"]
}
```

## 🛠️ Advanced Usage

### Custom Model Endpoints
Modify the URL configuration in the code:
```python
URL = (
    "http://your-custom-endpoint:8000/v1"  # Large model
    if selection == options[1]
    else "http://your-custom-endpoint:8081/v1"  # Small model
)
```

### Adding New Tools
Extend the tool configuration in the sidebar:
```python
# Add new tool toggles
use_custom_tool = st.sidebar.toggle("Use Custom Tool", value=False)
```

### Custom Styling
Modify the Streamlit theme in `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

## 🐛 Troubleshooting

### Common Issues

1. **"Connection refused"**
   - Ensure your model server is running
   - Check the port numbers (8000 for large, 8081 for small)
   - Verify the server supports the Responses API

2. **"Model not found"**
   - Check that your model server has the correct model loaded
   - Verify the model name in your server configuration

3. **Streamlit not starting**
   ```bash
   # Check Streamlit installation
   streamlit --version
   
   # Reinstall if needed
   pip install --upgrade streamlit
   ```

4. **Browser search not working**
   - Ensure you have an Exa API key set
   - Check that the browser tool is properly configured

### Debug Mode
Enable debug mode in the sidebar to see:
- Raw conversation data
- Tool interaction logs
- Model configuration details

## 📊 Performance Tips

### For Better Performance
- Use the smaller model for faster responses
- Set reasoning effort to "low" for quick interactions
- Limit max output tokens for shorter responses
- Use local model servers for lower latency

### For Development
- Enable debug mode to monitor interactions
- Use the JSON output to understand the conversation flow
- Test with different model configurations

## 🔗 Related Documentation

- [Streamlit Documentation](https://docs.streamlit.io/) - Streamlit framework guide
- [GPT-OSS Main README](../../README.md) - Project overview
- [Responses API](../gpt_oss/responses_api/) - API server documentation
- [Tools Documentation](../gpt_oss/tools/) - Available tools

## 🤝 Contributing

Improvements welcome! Please:
- Add new UI features
- Enhance error handling
- Improve accessibility
- Add more configuration options
- Create custom themes
