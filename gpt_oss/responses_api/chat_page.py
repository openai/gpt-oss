"""開一書齋，以便問學。"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse


def open_shuzhai(app: FastAPI, cutoff_year: int, has_library: bool) -> None:
    """設問學之頁，明年界與藏書。"""
    @app.get("/", include_in_schema=False)
    async def shuzhai_page():
        return FileResponse(Path(__file__).with_name("chat.html"))

    @app.get("/api/chat-config", include_in_schema=False)
    async def shuzhai_catalogue():
        return {"cutoff_year": cutoff_year, "has_library": has_library}
