# Streamlit Chat Example

This example demonstrates how to create a simple chat interface using Streamlit and gpt-oss.

## Prerequisites

- Python 3.12+
- A running gpt-oss server
- Streamlit installed

## Installation

1. Install dependencies:

```bash
pip install streamlit openai
```

2. Ensure you have a local gpt-oss server running

## Running the Example

1. Start your local gpt-oss server on `http://localhost:8000` (or modify the base URL in the code)

2. Run the Streamlit application:

```bash
streamlit run streamlit_chat.py
```

3. Open your browser to the URL shown in the terminal (typically `http://localhost:8501`)

## Configuration

You can modify the base URL and other settings by editing the configuration in `streamlit_chat.py`:

```python
client = OpenAI(
    api_key="local",
    base_url="http://localhost:8000/v1",
)
```

## Features

- Interactive chat interface
- Real-time responses from gpt-oss
- Simple and clean UI using Streamlit

## Customization

Feel free to modify the interface and add additional features such as:
- Chat history persistence
- Different model configurations
- Custom styling
- Tool integration
