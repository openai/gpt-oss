"""借遠方符術之力，守書齋問學之約。"""

import argparse
import os
from pathlib import Path
from typing import Literal

import aiohttp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from gpt_oss.classical import ClassicalCanon
from gpt_oss.classical_library import ClassicalLibrary
from .chat_page import open_shuzhai


class ShuzhaiMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["message"] = "message"
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=100000)


class ShuzhaiRequest(BaseModel):
    input: str | list[ShuzhaiMessage]
    max_output_tokens: int = Field(default=8192, ge=256, le=16384)
    stream: Literal[False] = False


def create_openai_chat(canon: ClassicalCanon | None = None,
                       library: ClassicalLibrary | None = None) -> FastAPI:
    """鑰藏於後臺；來客只得問學，不得改約。"""
    canon = canon or ClassicalCanon()
    app = FastAPI()
    open_shuzhai(app, canon.cutoff_year, library is not None,
                 model_ready=lambda: bool(os.getenv("OPENAI_API_KEY")))

    @app.post("/v1/responses")
    async def answer_scholar(body: ShuzhaiRequest):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise HTTPException(503, "尚未設定 OPENAI_API_KEY，請先於後臺 Secret 設定。")
        messages = ([{"role": "user", "content": body.input}] if isinstance(body.input, str)
                    else [{"role": m.role, "content": m.content} for m in body.input])
        if not messages or messages[-1]["role"] != "user":
            raise HTTPException(422, "須以來客之問作結。")
        if any(not message["content"].strip() for message in messages):
            raise HTTPException(422, "請書所問，不可空白。")
        if sum(len(m["content"]) for m in messages) > 100000:
            raise HTTPException(413, "此席文字過長，請另啟一席。")
        evidence = library.consult(messages[-1]["content"], canon.cutoff_year) if library else ""
        if evidence:
            messages[-1]["content"] += "\n\n" + evidence
        payload = {"model": os.getenv("OPENAI_MODEL", "gpt-4.1"),
                   "instructions": canon.instructions, "input": messages,
                   "max_output_tokens": body.max_output_tokens, "store": False}
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
                async with session.post("https://api.openai.com/v1/responses", json=payload,
                                        headers={"Authorization": f"Bearer {key}"}) as response:
                    if response.status != 200:
                        status = response.status
                        try:
                            failure = await response.json()
                        except (aiohttp.ContentTypeError, ValueError):
                            failure = {}
                        issue = failure.get("error", {}) if isinstance(failure, dict) else {}
                        signal = " ".join(str(issue.get(k, "")) for k in ("code", "type", "message")).lower()
                        detail = {401: "金鑰未通過驗證，請於 Secret 設定更正。",
                                  403: "此金鑰無權使用指定模型。",
                                  404: "指定模型不可用，請檢查 OPENAI_MODEL。",
                                  }.get(status, "遠端服務暫未能作答，請稍後再試。")
                        if status == 429:
                            if any(term in signal for term in ("insufficient_quota", "billing quota", "current quota", "run out of credits", "no balance", "exhausted")):
                                detail = "API 額度或儲值不足。請至 https://platform.openai.com/settings/organization/billing 檢查 API 帳務；ChatGPT 訂閱不含 API 額度。"
                            elif any(term in signal for term in ("rate_limit_exceeded", "requests per min", "tokens per min", "too many requests")):
                                detail = "API 請求速率受限。請稍候再試，並降低連續提問頻率。"
                            else:
                                detail = "OpenAI 回報 429，但未提供可辨認原因；可能是 API 額度／支出上限，亦可能是速率限制。請先查帳務與 API 用量限制，再重試。"
                        raise HTTPException(status if status in (401, 403, 429) else 502, detail)
                    result = await response.json()
        except (aiohttp.ClientError, TimeoutError, ValueError):
            raise HTTPException(504, "遠端連線中斷或逾時，請稍後重試。") from None
        # 只傳正文與用量，不洩後臺設定及內部推演。
        return {"id": result.get("id"), "status": result.get("status"),
                "output": [item for item in result.get("output", []) if item.get("type") == "message"],
                "usage": result.get("usage"), "error": {"message": "此次未得有效答覆。"} if result.get("error") else None}

    return app


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="開書齋，通遠方學者。")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--knowledge-cutoff-year", type=int, default=1899)
    parser.add_argument("--classical-library", type=Path)
    args = parser.parse_args()
    uvicorn.run(create_openai_chat(ClassicalCanon(args.knowledge_cutoff_year),
                ClassicalLibrary(args.classical_library) if args.classical_library else None),
                host=args.host, port=args.port)
