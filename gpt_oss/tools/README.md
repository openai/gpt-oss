# GPT-OSS Tools

This directory contains the tools that GPT-OSS models can use during inference. These tools enable the models to perform actions like web browsing, code execution, and file manipulation.

## 🛠️ Available Tools

### 🌐 **Browser Tool** (`simple_browser/`)
A web browsing tool that allows the model to search and read web pages.

**Features:**
- Web search functionality
- Page content extraction
- Scrolling through long pages
- Citation support for answers

**Usage:**
```python
from gpt_oss.tools.simple_browser import SimpleBrowserTool
from gpt_oss.tools.simple_browser.backend import ExaBackend

backend = ExaBackend(source="web")
browser_tool = SimpleBrowserTool(backend=backend)
```

**⚠️ Note:** This is for educational purposes. Implement your own browsing environment for production use.

### 🐍 **Python Tool** (`python_docker/`)
A Python code execution tool that runs code in a Docker container.

**Features:**
- Safe code execution in isolated environment
- Stateless execution model
- Support for calculations and data processing
- Chain-of-thought reasoning integration

**Usage:**
```python
from gpt_oss.tools.python_docker.docker_tool import PythonTool

python_tool = PythonTool()
```

**⚠️ Note:** Runs in a permissive Docker container. Implement proper security restrictions for production.

### 📝 **Apply Patch Tool** (`apply_patch.py`)
A tool for creating, updating, or deleting files locally.

**Features:**
- File creation and modification
- Patch application
- Safe file operations

**Usage:**
```python
from gpt_oss.tools.apply_patch import apply_patch_tool
```

## 🔧 Tool Integration

### Using Tools with Harmony Format
Tools are integrated using the Harmony response format:

```python
from openai_harmony import SystemContent, Message, Conversation, Role

# Create system message with tools
system_content = SystemContent.new().with_tools([
    browser_tool.tool_config,
    python_tool.tool_config
])

system_message = Message.from_role_and_content(Role.SYSTEM, system_content)
```

### Tool Processing
```python
# Parse model output
messages = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
last_message = messages[-1]

# Route to appropriate tool
if last_message.recipient.startswith("browser"):
    response_messages = await browser_tool.process(last_message)
elif last_message.recipient == "python":
    response_messages = await python_tool.process(last_message)
```

## 🚀 Getting Started

### Prerequisites
- Docker (for Python tool)
- Exa API key (for browser tool)
- GPT-OSS model with Harmony format support

### Environment Setup
```bash
# Set up environment variables
export EXA_API_KEY="your_exa_api_key"  # for browser tool
export DOCKER_HOST="unix:///var/run/docker.sock"  # for Python tool
```

### Basic Example
```python
import asyncio
from gpt_oss.tools.simple_browser import SimpleBrowserTool
from gpt_oss.tools.python_docker.docker_tool import PythonTool
from openai_harmony import SystemContent, Message, Conversation, Role

async def main():
    # Initialize tools
    browser_tool = SimpleBrowserTool()
    python_tool = PythonTool()
    
    # Create conversation with tools
    system_content = SystemContent.new().with_tools([
        browser_tool.tool_config,
        python_tool.tool_config
    ])
    
    conversation = Conversation.from_messages([
        Message.from_role_and_content(Role.SYSTEM, system_content),
        Message.from_role_and_content(Role.USER, "What's the weather in San Francisco?")
    ])
    
    # Process with your model...
```

## 🔒 Security Considerations

### Browser Tool
- Implement your own browsing environment
- Add rate limiting and access controls
- Consider content filtering

### Python Tool
- Use restricted Docker containers
- Implement code execution limits
- Add security sandboxing

### Apply Patch Tool
- Validate file paths and operations
- Implement backup mechanisms
- Add user confirmation for destructive operations

## 📚 Advanced Usage

### Custom Tool Development
You can create custom tools by implementing the `Tool` interface:

```python
from gpt_oss.tools.tool import Tool
from openai_harmony import Message

class CustomTool(Tool):
    @property
    def name(self) -> str:
        return "custom_tool"
    
    async def _process(self, message: Message):
        # Implement your tool logic here
        yield Message(...)
    
    def instruction(self) -> str:
        return "Description of what this tool does"
```

### Tool Configuration
Tools can be configured with different backends and settings:

```python
# Browser tool with custom backend
from gpt_oss.tools.simple_browser.backend import CustomBackend

backend = CustomBackend(
    source="web",
    max_results=10,
    include_domains=["example.com"]
)
browser_tool = SimpleBrowserTool(backend=backend)
```

## 🐛 Troubleshooting

### Common Issues

1. **Docker Connection Error**: Ensure Docker is running and accessible
2. **Exa API Error**: Verify your API key is valid and has sufficient credits
3. **Tool Not Found**: Check that the tool is properly registered in the system message

### Debug Mode
Enable debug mode to see tool interactions:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📖 Related Documentation

- [Main README](../../README.md) - Project overview
- [Harmony Format](https://github.com/openai/harmony) - Response format documentation
- [Tool Interface](tool.py) - Base tool implementation
- [Examples](../../examples/) - Usage examples

## 🤝 Contributing

We welcome tool improvements and new tool implementations! Please:
- Follow the existing tool interface
- Add comprehensive documentation
- Include security considerations
- Provide usage examples
