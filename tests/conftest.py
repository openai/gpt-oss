import os
import sys
import pytest
from typing import Generator, Any
from unittest.mock import Mock, MagicMock
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
)
from gpt_oss.responses_api.api_server import create_api_server


@pytest.fixture(scope="session")
def harmony_encoding():
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


@pytest.fixture
def mock_infer_token(harmony_encoding):
    fake_tokens = harmony_encoding.encode(
        "<|channel|>final<|message|>Test response<|return|>", 
        allowed_special="all"
    )
    token_queue = fake_tokens.copy()
    
    def _mock_infer(tokens: list[int], temperature: float = 0.0, new_request: bool = False) -> int:
        nonlocal token_queue
        if len(token_queue) == 0:
            token_queue = fake_tokens.copy()
        return token_queue.pop(0)
    return _mock_infer


@pytest.fixture
def api_client(harmony_encoding, mock_infer_token) -> Generator[TestClient, None, None]:
    app = create_api_server(
        infer_next_token=mock_infer_token,
        encoding=harmony_encoding
    )
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_request_data():
    return {
        "model": "gpt-oss-120b",
        "input": "Hello, how can I help you today?",
        "stream": False,
        "reasoning_effort": "low",
        "temperature": 0.7,
        "tools": []
    }


@pytest.fixture
def mock_browser_tool():
    mock = MagicMock()
    mock.search.return_value = ["Result 1", "Result 2"]
    mock.open_page.return_value = "Page content"
    mock.find_on_page.return_value = "Found text"
    return mock


@pytest.fixture
def mock_python_tool():
    mock = MagicMock()
    mock.execute.return_value = {
        "output": "print('Hello')",
        "error": None,
        "exit_code": 0
    }
    return mock


@pytest.fixture(autouse=True)
def reset_test_environment():
    test_env_vars = ['OPENAI_API_KEY', 'GPT_OSS_MODEL_PATH']
    original_values = {}
    
    for var in test_env_vars:
        if var in os.environ:
            original_values[var] = os.environ[var]
            del os.environ[var]
    
    yield
    
    for var, value in original_values.items():
        os.environ[var] = value


@pytest.fixture
def performance_timer():
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.elapsed
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()