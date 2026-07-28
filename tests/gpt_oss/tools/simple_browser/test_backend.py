import os
import pytest
from typing import Generator, Any
from unittest import mock
from aiohttp import ClientSession

from gpt_oss.tools.simple_browser.backend import ExaBackend, YouComBackend

class MockAiohttpResponse:
    """Mocks responses for get/post requests from async libraries."""

    def __init__(self, json: dict, status: int):
        self._json = json
        self.status = status

    async def json(self):
        return self._json

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def __aenter__(self):
        return self

def mock_os_environ_get(name: str, default: Any = "test_api_key"):
    assert name in ["YDC_API_KEY"]
    return default

def test_youcom_backend():
    backend = YouComBackend(source="web")
    assert backend.source == "web"

def test_youcom_backend_api_key_param():
    backend = YouComBackend(source="web", api_key="my_custom_key")
    assert backend.api_key == "my_custom_key"
    assert backend._get_api_key() == "my_custom_key"

@pytest.mark.asyncio
@mock.patch("aiohttp.ClientSession.get")
async def test_youcom_backend_search(mock_session_get):
    backend = YouComBackend(source="web")
    api_response = {
        "results": {
            "web": [
                {"title": "Web Result 1", "url": "https://www.example.com/web1", "snippets": "Web Result 1 snippets"},
                {"title": "Web Result 2", "url": "https://www.example.com/web2", "snippets": "Web Result 2 snippets"},
            ],
            "news": [
                {"title": "News Result 1", "url": "https://www.example.com/news1", "description": "News Result 1 description"},
                {"title": "News Result 2", "url": "https://www.example.com/news2", "description": "News Result 2 description"},
            ],
        }
    }
    with mock.patch("os.environ.get", wraps=mock_os_environ_get):
        mock_session_get.return_value = MockAiohttpResponse(api_response, 200)
        async with ClientSession() as session:
            result = await backend.search(query="test", topn=10, session=session)
        assert result.title == "test"
        assert result.urls == {"0": "https://www.example.com/web1", "1": "https://www.example.com/web2", "2": "https://www.example.com/news1", "3": "https://www.example.com/news2"}

@pytest.mark.asyncio
@mock.patch("aiohttp.ClientSession.post")
async def test_youcom_backend_fetch(mock_session_get):
    backend = YouComBackend(source="web")
    api_response = [
        {"title": "Fetch Result 1", "url": "https://www.example.com/fetch1", "html": "<div>Fetch Result 1 text</div>"},
    ]
    with mock.patch("os.environ.get", wraps=mock_os_environ_get):
        mock_session_get.return_value = MockAiohttpResponse(api_response, 200)
        async with ClientSession() as session:
            result = await backend.fetch(url="https://www.example.com/fetch1", session=session)
        assert result.title == "Fetch Result 1"
        assert result.text == "\nURL: https://www.example.com/fetch1\nFetch Result 1 text"


_real_os_environ_get = os.environ.get

def mock_exa_environ_get(name: str, default: Any = None):
    if name == "EXA_API_KEY":
        return "test_api_key"
    return _real_os_environ_get(name, default)

def test_exa_backend():
    backend = ExaBackend(source="web")
    assert backend.source == "web"

@pytest.mark.asyncio
@mock.patch("aiohttp.ClientSession.post")
async def test_exa_backend_search(mock_session_post):
    backend = ExaBackend(source="web")
    # real response from: POST https://api.exa.ai/search
    # {"query": "openai gpt-oss open source model", "numResults": 2, "contents": {"text": true, "summary": true}}
    api_response = {
        "results": [
            {
                "title": "openai/gpt-oss: gpt-oss-120b and gpt-oss-20b are two open-weight ...",
                "url": "https://github.com/openai/gpt-oss",
                "summary": "The webpage introduces gpt-oss, a series of open-weight language models developed by OpenAI, designed for high reasoning, agentic tasks, and versatile developer applications.",
            },
            {
                "title": "OpenAI Returns to Open-Source Roots with GPT-OSS Models | AI News",
                "url": "https://opentools.ai/news/openai-returns-to-open-source-roots-with-gpt-oss-models",
                "summary": "OpenAI has released two open-source language models, gpt-oss-120b and gpt-oss-20b, under the Apache 2.0 license, marking a return to its open-source roots.",
            },
        ]
    }
    with mock.patch("os.environ.get", wraps=mock_exa_environ_get):
        mock_session_post.return_value = MockAiohttpResponse(api_response, 200)
        async with ClientSession() as session:
            result = await backend.search(query="openai gpt-oss open source model", topn=2, session=session)
        assert result.title == "openai gpt-oss open source model"
        assert result.urls == {"0": "https://github.com/openai/gpt-oss", "1": "https://opentools.ai/news/openai-returns-to-open-source-roots-with-gpt-oss-models"}

@pytest.mark.asyncio
@mock.patch("aiohttp.ClientSession.post")
async def test_exa_backend_fetch(mock_session_post):
    backend = ExaBackend(source="web")
    # real response from: POST https://api.exa.ai/contents
    # {"urls": ["https://github.com/openai/gpt-oss"], "text": {"includeHtmlTags": true}}
    api_response = {
        "results": [
            {
                "title": "openai/gpt-oss",
                "url": "https://github.com/openai/gpt-oss",
                "text": "<h1>Repository: openai/gpt-oss</h1><p>gpt-oss-120b and gpt-oss-20b are two open-weight language models by OpenAI</p>",
            },
        ]
    }
    with mock.patch("os.environ.get", wraps=mock_exa_environ_get):
        mock_session_post.return_value = MockAiohttpResponse(api_response, 200)
        async with ClientSession() as session:
            result = await backend.fetch(url="https://github.com/openai/gpt-oss", session=session)
        assert result.title == "openai/gpt-oss"
        assert result.text == "\nURL: https://github.com/openai/gpt-oss\n# Repository: openai/gpt-oss \n\ngpt-oss-120b and gpt-oss-20b are two open-weight language models by OpenAI"


    