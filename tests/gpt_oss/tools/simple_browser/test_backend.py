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
@mock.patch("aiohttp.ClientSession.get")
async def test_youcom_backend_search_missing_results_key(mock_session_get):
    backend = YouComBackend(source="web")
    api_response = {"error": "rate limited"}
    with mock.patch("os.environ.get", wraps=mock_os_environ_get):
        mock_session_get.return_value = MockAiohttpResponse(api_response, 200)
        async with ClientSession() as session:
            result = await backend.search(query="test", topn=10, session=session)
        assert result.title == "test"
        assert result.urls == {}

@pytest.mark.asyncio
@mock.patch("aiohttp.ClientSession.get")
async def test_youcom_backend_search_null_results(mock_session_get):
    backend = YouComBackend(source="web")
    api_response = {"results": None}
    with mock.patch("os.environ.get", wraps=mock_os_environ_get):
        mock_session_get.return_value = MockAiohttpResponse(api_response, 200)
        async with ClientSession() as session:
            result = await backend.search(query="test", topn=10, session=session)
        assert result.title == "test"
        assert result.urls == {}

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


def mock_exa_environ_get(name: str, default: Any = "test_api_key"):
    assert name in ["EXA_API_KEY"]
    return default

def test_exa_backend():
    backend = ExaBackend(source="web")
    assert backend.source == "web"

@pytest.mark.asyncio
@mock.patch("aiohttp.ClientSession.post")
async def test_exa_backend_search(mock_session_post):
    backend = ExaBackend(source="web")
    api_response = {
        "results": [
            {"title": "Result 1", "url": "https://www.example.com/1", "summary": "Result 1 summary"},
            {"title": "Result 2", "url": "https://www.example.com/2", "summary": "Result 2 summary"},
        ]
    }
    with mock.patch("os.environ.get", wraps=mock_exa_environ_get):
        mock_session_post.return_value = MockAiohttpResponse(api_response, 200)
        async with ClientSession() as session:
            result = await backend.search(query="test", topn=10, session=session)
        assert result.title == "test"
        assert result.urls == {"0": "https://www.example.com/1", "1": "https://www.example.com/2"}

@pytest.mark.asyncio
@mock.patch("aiohttp.ClientSession.post")
async def test_exa_backend_fetch(mock_session_post):
    backend = ExaBackend(source="web")
    api_response = {
        "results": [
            {"title": "Fetch Result 1", "url": "https://www.example.com/fetch1", "text": "<div>Fetch Result 1 text</div>"},
        ]
    }
    with mock.patch("os.environ.get", wraps=mock_exa_environ_get):
        mock_session_post.return_value = MockAiohttpResponse(api_response, 200)
        async with ClientSession() as session:
            result = await backend.fetch(url="https://www.example.com/fetch1", session=session)
        assert result.title == "Fetch Result 1"
        assert result.text == "\nURL: https://www.example.com/fetch1\nFetch Result 1 text"


@pytest.mark.asyncio
@mock.patch("aiohttp.ClientSession.get")
async def test_youcom_backend_search_list_and_optional_snippets(mock_session_get):
    backend = YouComBackend(source="web", api_key="test_api_key")
    api_response = {
        "results": {
            "web": [
                {
                    "title": "Result With List Snippets",
                    "url": "https://www.example.com/list",
                    "snippets": ["First fragment.", "Second fragment."],
                },
                {
                    "title": "Result Without Snippets",
                    "url": "https://www.example.com/pdf",
                    "description": "Fallback web description",
                },
                {
                    "title": "Result With Empty Snippets",
                    "url": "https://www.example.com/empty",
                    "snippets": [],
                    "description": "Fallback when snippets empty",
                },
            ],
            "news": [
                {
                    "title": "News Without Description",
                    "url": "https://www.example.com/news",
                },
            ],
        }
    }
    mock_session_get.return_value = MockAiohttpResponse(api_response, 200)
    async with ClientSession() as session:
        result = await backend.search(query="test", topn=10, session=session)

    assert result.urls == {
        "0": "https://www.example.com/list",
        "1": "https://www.example.com/pdf",
        "2": "https://www.example.com/empty",
        "3": "https://www.example.com/news",
    }
    assert "First fragment. Second fragment." in result.text
    assert "['First fragment.'" not in result.text
    assert "Fallback web description" in result.text
    assert "Fallback when snippets empty" in result.text
    assert "[]" not in result.text

    mock_session_get.return_value = MockAiohttpResponse(
        {"results": {"web": None, "news": None}}, 200
    )
    async with ClientSession() as session:
        empty_sections = await backend.search(query="test", topn=10, session=session)
    assert empty_sections.urls == {}


@pytest.mark.asyncio
@mock.patch("aiohttp.ClientSession.get")
async def test_youcom_backend_search_escapes_html_in_results(mock_session_get):
    backend = YouComBackend(source="web", api_key="test_api_key")
    url_with_quote_and_entities = (
        "https://en.wikipedia.org/wiki/Bayes'_theorem?id=1&reg=us&sect=2"
    )
    api_response = {
        "results": {
            "web": [
                {
                    "title": "Why a < b & c > d",
                    "url": url_with_quote_and_entities,
                    "snippets": "Compare std::vector<int> with <a href='https://injected.example'>injected</a>",
                },
            ],
        }
    }
    mock_session_get.return_value = MockAiohttpResponse(api_response, 200)
    async with ClientSession() as session:
        result = await backend.search(query="bayes", topn=10, session=session)

    assert result.urls == {"0": url_with_quote_and_entities}
    assert "Why a < b & c > d" in result.text
    assert "std::vector<int>" in result.text


@pytest.mark.asyncio
@mock.patch("aiohttp.ClientSession.post")
async def test_exa_backend_search_optional_fields_and_html_escaping(mock_session_post):
    backend = ExaBackend(source="web", api_key="test_api_key")
    special_url = "https://en.wikipedia.org/wiki/Euler's_identity?x=1&copy=2"
    api_response = {
        "results": [
            {
                "title": "Template<T> & Friends",
                "url": special_url,
                "summary": "Uses <a href='https://injected.example'>raw tag</a> safely",
            },
            {
                "title": None,
                "url": "https://www.example.com/no-summary",
            },
        ]
    }
    mock_session_post.return_value = MockAiohttpResponse(api_response, 200)
    async with ClientSession() as session:
        result = await backend.search(query="euler", topn=10, session=session)

    assert result.urls == {
        "0": special_url,
        "1": "https://www.example.com/no-summary",
    }
    assert "Template<T> & Friends" in result.text
    assert "Uses <a href='https://injected.example'>raw tag</a> safely" in result.text
