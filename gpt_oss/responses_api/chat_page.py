"""開一書齋，以便問學。"""

from pathlib import Path
from typing import Callable

from fastapi import FastAPI
from fastapi.responses import FileResponse


def open_shuzhai(app: FastAPI, cutoff_year: int, has_library: bool,
                model_ready: bool | Callable[[], bool] = True) -> None:
    """設問學之頁，明年界與藏書。"""
    @app.get("/", include_in_schema=False)
    async def shuzhai_page():
        return FileResponse(Path(__file__).with_name("chat.html"))

    @app.get("/api/chat-config", include_in_schema=False)
    async def shuzhai_catalogue():
        return {"cutoff_year": cutoff_year, "has_library": has_library,
                "model_ready": model_ready() if callable(model_ready) else model_ready}


if __name__ == "__main__":
    import argparse
    import uvicorn
    from fastapi import HTTPException

    parser = argparse.ArgumentParser(description="先開書齋之頁，候學者入席。")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    app = FastAPI()
    open_shuzhai(app, 1899, False, model_ready=False)

    @app.post("/v1/responses")
    async def waiting_for_scholar():
        raise HTTPException(status_code=503, detail="學者尚未入席，請先啟動模型服務。")

    uvicorn.run(app, host=args.host, port=args.port)
