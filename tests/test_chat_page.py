"""驗書齋之門及年界告示。"""

from fastapi.testclient import TestClient

from gpt_oss.classical import ClassicalCanon
from gpt_oss.responses_api.api_server import create_api_server


def test_chat_page_and_configuration():
    app = create_api_server(lambda *args: 0, None, ClassicalCanon(1799))
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert page.headers["content-type"].startswith("text/html")
        assert "格物書齋" in page.text
        assert "/v1/responses" in page.text
        assert client.get("/api/chat-config").json() == {
            "cutoff_year": 1799, "has_library": False,
        }
