"""命學者製戲，存其原言，列疑以待覆核。"""

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path


GAME_PROMPT = """請製作一局可於瀏覽器遊玩的「九宮移數」之戲。
以一至八之籌置於九宮，留一空位。鄰空位之籌可移入，復歸順序則勝。
請依次交付：一、以繁體文言述規則與構思；二、以一個完整 html 程式碼區塊交付
可獨立開啟的成品，內含所需之 HTML、CSS、JavaScript，不借外部函式庫或網路資源；
三、以繁體文言列出驗法，明言哪些已驗、哪些未驗。
須有開局、重局、步數、勝負告示及觸控操作；開局必可解，且不得已是勝局。
說明、註釋與畫面文字皆用清楚之文言；自訂名稱取義於九宮與算籌。
現代保留字與介面依其原式。勿捏造此戲之古代出處。
請交付完整成品，不以省略號、待補段落或只有設計說明代替。
"""


def inspect_scroll(answer: str, response_status: str) -> tuple[list[str], str | None]:
    """只列可見之疑，不以字詞檢索斷定文章優劣。"""
    flags = []
    if response_status != "completed":
        flags.append(f"回覆未正常完成：{response_status}")
    blocks = re.findall(r"```html\s*\n([\s\S]*?)```", answer, re.I)
    game = blocks[0].strip() if len(blocks) == 1 else None
    if game is None:
        flags.append("未得唯一完整 html 區塊。")
    elif not re.search(r"</html\s*>", game, re.I):
        flags.append("HTML 缺少結尾，可能遭截斷。")
    if game and re.search(r"(?:src|href)\s*=\s*['\"](?:https?:)?//|\b(?:fetch|import)\s*\(", game, re.I):
        flags.append("遊戲可能依賴外部資源，須核查是否能獨立運行。")
    for word in ("人工智能", "人工智慧", "賦能", "沉浸式", "超讚", "搞定", "接下來我們", "讓我們一起"):
        if word in answer:
            flags.append(f"文風待覆核：出現「{word}」。")
    if re.search(r"TODO|FIXME|此處省略|自行補上", answer, re.I):
        flags.append("可能含未完成段落。")
    return flags, game


def consult_scholar(endpoint: str, model: str, messages: list[dict], token: str | None) -> dict:
    """向指定學者問策，不以別人之作冒充。"""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = {"model": model, "input": messages, "stream": False,
            "store": False, "max_output_tokens": 10000,
            "reasoning": {"effort": "medium"}}
    req = urllib.request.Request(endpoint, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)


def collect_words(result: dict) -> str:
    pieces = []
    for item in result.get("output", []):
        if item.get("type") != "message" or item.get("role") != "assistant":
            continue
        content = item.get("content", [])
        if isinstance(content, str):
            pieces.append(content)
        else:
            pieces.extend(part.get("text", "") for part in content
                          if part.get("type") in ("text", "output_text"))
    return "\n".join(pieces)


def run_trial(endpoint: str, model: str, destination: Path, repair: bool = False) -> dict:
    """存原稿與改稿；未能問得者，直書未得。"""
    destination.mkdir(parents=True, exist_ok=True)
    messages = [{"type": "message", "role": "user", "content": GAME_PROMPT}]
    report = {"endpoint": endpoint, "model": model, "status": "not_started", "rounds": [],
              "gameplay_verified": False, "classical_style_verified": False,
              "historical_knowledge_verified": False}
    (destination / "prompt.txt").write_text(GAME_PROMPT, encoding="utf-8")
    try:
        for index in range(2 if repair else 1):
            result = consult_scholar(endpoint, model, messages, os.getenv("CLASSICAL_EVAL_API_KEY"))
            (destination / f"response-{index + 1}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            answer = collect_words(result)
            (destination / f"answer-{index + 1}.md").write_text(answer, encoding="utf-8")
            flags, game = inspect_scroll(answer, result.get("status", "unknown"))
            if result.get("error") or not answer.strip():
                flags.append("未得有效正文；須核查模型服務。")
            if game:
                (destination / f"game-{index + 1}.html").write_text(game, encoding="utf-8")
            report["rounds"].append({"round": index + 1, "flags": flags,
                                     "game_extracted": game is not None})
            report["status"] = "needs_review"
            if not flags or not answer.strip() or result.get("error"):
                break
            messages.extend([
                {"type": "message", "role": "assistant", "content": answer},
                {"type": "message", "role": "user", "content":
                 "覆核得下列疑點，請逐項考訂；字詞未必有誤，須明辨其用法。\n" +
                 "\n".join(flags) + "\n請重交完整可玩之 html 成品，仍守前約。未實際試玩，不得稱已驗。"},
            ])
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        report["status"] = "blocked"
        report["reason"] = "模型服務無法連線或未回傳可解析的回覆；尚無模型成品可評。"
    (destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="以製戲驗學者，存稿以待考。")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/responses")
    parser.add_argument("--model", default="gpt-oss-20b")
    parser.add_argument("--output", type=Path, default=Path("data/classical/game-eval"))
    parser.add_argument("--repair", action="store_true")
    args = parser.parse_args()
    result = run_trial(args.endpoint, args.model, args.output, args.repair)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(2 if result["status"] == "blocked" else 0)
