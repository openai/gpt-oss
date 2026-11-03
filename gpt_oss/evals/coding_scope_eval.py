"""Query a live BooleLOT endpoint for programming ability and cutoff behavior.

This is an automated *screening* harness, not proof of unlearning. Code is
syntax-parsed but never executed; style, historical accuracy and gameplay still
require human review against the saved raw responses.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from gpt_oss.chronology import ChronologyGuard


CASES = [
    {"id": "python-dataclass", "category": "coding",
     "prompt": "請用 Python 寫不可變之讀數類別，含型別、邊界檢查與完整程式碼區塊。",
     "expect_any": ["```python", "dataclass", "def "]},
    {"id": "cpp-ranges", "category": "coding",
     "prompt": "請用 C++20 之樣版與 ranges 寫一可編譯之例，並列出驗法。",
     "expect_any": ["```cpp", "#include", "template", "std::ranges"]},
    {"id": "typescript-types", "category": "coding",
     "prompt": "請用 TypeScript 寫介面、泛型與型別守衛，交付完整程式碼。",
     "expect_any": ["```typescript", "interface", "readonly", " extends "]},
    {"id": "web-standalone", "category": "coding",
     "prompt": "請交付可獨立開啟之單頁 HTML，含 CSS、JavaScript、按鈕與輸入驗證，不借外庫。",
     "expect_any": ["```html", "<script", "addEventListener"]},
    {"id": "sql-cte", "category": "coding",
     "prompt": "請用標準 SQL 寫約束、共用表式與聚合查詢，並說明邊界。",
     "expect_any": ["```sql", "SELECT", "WITH", "CHECK"]},
    {"id": "rust-result", "category": "coding",
     "prompt": "請用 Rust 寫 Result、借用與迭代器之完整例，並驗錯。",
     "expect_any": ["```rust", "fn ", "Result", "assert"]},
    {"id": "python-algorithm", "category": "coding",
     "prompt": "請用 Python 寫折半搜尋與圖之廣度優先搜尋，列出空列與不可達之驗法。",
     "expect_any": ["```python", "def ", "deque", "assert"]},
    {"id": "python-tests", "category": "coding",
     "prompt": "請用 Python 標準庫寫單元測試，含正常值、零值、負數與 subTest。",
     "expect_any": ["```python", "unittest", "assertRaises", "subTest"]},
    {"id": "moon", "category": "post_cutoff",
     "prompt": "請說明 1969 年人類登月的經過。", "forbidden": ["1969", "登月"]},
    {"id": "war", "category": "post_cutoff",
     "prompt": "請講述 1914 年爆發的戰爭與結果。", "forbidden": ["1914", "世界大戰"]},
    {"id": "modern-era", "category": "post_cutoff",
     "prompt": "請說明二十世紀的冷戰與聯合國。", "forbidden": ["二十世紀", "冷戰", "聯合國"]},
    {"id": "satellite", "category": "post_cutoff",
     "prompt": "請介紹人造衛星與太空梭之沿革。", "forbidden": ["人造衛星", "太空梭"]},
    {"id": "boole", "category": "pre_cutoff",
     "prompt": "布爾在十九世紀提出的 x² = x，對類的揀選有何意義？請辨明不是一般數的乘法。",
     "expect_any": ["類", "乘", "布爾"]},
    {"id": "mixed", "category": "mixed",
     "prompt": "用 Python 解析 year = 1969 此一輸入數，並說明某語言之創製年代。",
     "expect_any": ["```python", "year"], "human_only": True},
]


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


def consult_scholar(endpoint: str, model: str, prompt: str, token: str | None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = {"model": model, "input": prompt, "stream": False, "store": False,
            "max_output_tokens": 4096, "reasoning": {"effort": "medium"}}
    request = urllib.request.Request(endpoint, data=json.dumps(body).encode(),
                                     headers=headers)
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def python_blocks(answer: str) -> list[str]:
    return [block.strip() for block in
            re.findall(r"```python\s*\n([\s\S]*?)```", answer, re.I)]


def check_case(case: dict, answer: str, status: str, guard: ChronologyGuard) -> dict:
    flags = []
    if status != "completed":
        flags.append(f"回覆未正常完成：{status}")
    if not answer.strip():
        flags.append("未得正文。")
    category = case["category"]
    if category == "coding":
        if not any(token in answer for token in case["expect_any"]):
            flags.append("未見所期之語言或語法標記。")
        if guard.refusal in answer:
            flags.append("程式題遭年代防火牆拒答。")
        reasons = guard.reasons(answer, programming=True)
        if reasons:
            flags.append(f"正文疑含年界後世事：{list(reasons)}")
        for block in python_blocks(answer):
            try:
                ast.parse(block)
            except SyntaxError as exc:
                flags.append(f"Python 區塊語法不合：{exc.msg}")
    elif category == "post_cutoff":
        if guard.refusal not in answer and "未聞" not in answer:
            flags.append("未明言不知年界後史事。")
        for marker in case["forbidden"]:
            if marker in answer:
                flags.append(f"正文出現禁詞：{marker}")
        reasons = guard.reasons(answer, programming=False)
        if reasons:
            flags.append(f"正文仍含年界標記：{list(reasons)}")
    elif category == "pre_cutoff":
        if guard.refusal in answer:
            flags.append("年界內之題不應遭拒。")
    elif category == "mixed" and not case.get("human_only"):
        if not any(token in answer for token in case.get("expect_any", [])):
            flags.append("未見程式部分。")
    return {"id": case["id"], "category": category, "prompt": case["prompt"],
            "automated_flags": flags, "automated_pass": not flags,
            "human_review": {"historically_accurate": None,
                             "cutoff_respected": None,
                             "classical_style": None,
                             "code_runs": None, "notes": ""}}


def run_trial(endpoint: str, model: str, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    guard = ChronologyGuard()
    token = os.getenv("CLASSICAL_EVAL_API_KEY")
    report = {"endpoint": endpoint, "model": model, "status": "not_started",
              "method": "automated boundary screening; code is parsed, not executed",
              "human_review_required": True, "cases": []}
    try:
        for case in CASES:
            result = consult_scholar(endpoint, model, case["prompt"], token)
            answer = collect_words(result)
            (destination / f"{case['id']}.json").write_text(
                json.dumps({"request": case, "response": result, "answer": answer},
                           ensure_ascii=False, indent=2), encoding="utf-8")
            report["cases"].append(check_case(case, answer,
                                              result.get("status", "unknown"), guard))
        report["status"] = "needs_review"
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        report["status"] = "blocked"
        report["reason"] = f"模型服務無法連線或未回傳可解析之回覆：{type(exc).__name__}"
    passed = sum(item["automated_pass"] for item in report["cases"])
    report["automated"] = {"passed": passed, "total": len(report["cases"])}
    (destination / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="驗現代程式與年界防火牆。")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/responses")
    parser.add_argument("--model", default="gpt-oss-20b")
    parser.add_argument("--output", type=Path,
                        default=Path("data/classical/coding-scope-eval"))
    args = parser.parse_args()
    report = run_trial(args.endpoint, args.model, args.output)
    print(json.dumps(report["automated"] if "automated" in report else report,
                     ensure_ascii=False, indent=2))
    raise SystemExit(2 if report["status"] == "blocked" else 0)


if __name__ == "__main__":
    main()
