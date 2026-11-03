"""Compile, parse and execute the reviewed modern-code examples.

Every lesson's code file is checked against the chronology firewall, then by a
toolchain appropriate to its language. Type/Rust parsers are optional; install
``.[coding]`` for full static coverage. The report never claims that a model
produced these examples: they are curated training material.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path

from gpt_oss.chronology import ChronologyGuard, post_cutoff_years
from gpt_oss.coding_library import CodingLibrary
from training.prepare_corpus import CODING_CATALOGUE, ROOT


class ScriptCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_script = False
        self.scripts: list[str] = []
        self.external = False

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.in_script = True
            if any(key in ("src", "href") for key, _ in attrs):
                self.external = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_script = False

    def handle_data(self, data):
        if self.in_script:
            self.scripts.append(data)


def _run(command: list[str], cwd: Path | None = None) -> tuple[bool, str]:
    try:
        completed = subprocess.run(command, cwd=cwd, capture_output=True,
                                   text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    detail = (completed.stdout + completed.stderr).strip()
    return completed.returncode == 0, detail[-1200:]


def _tree_sitter_parse(path: Path, language_module: str, function_name: str) -> tuple[str, bool, str]:
    try:
        from tree_sitter import Language, Parser
        module = __import__(language_module, fromlist=[function_name])
        language = Language(getattr(module, function_name)())
        tree = Parser(language).parse(path.read_bytes())
        if tree.root_node.has_error:
            return "tree-sitter", False, "語法樹含錯誤節點"
        return "tree-sitter", True, ""
    except ImportError:
        return "tree-sitter", None, "未安裝可選語法樹套件"
    except Exception as exc:  # pragma: no cover - binding-version diagnostics
        return "tree-sitter", False, f"{type(exc).__name__}: {exc}"


def _check_python(path: Path, temp: Path) -> tuple[str, bool, str]:
    ok, detail = _run([sys.executable, str(path)], cwd=ROOT)
    return "python-execute", ok, detail


def _check_javascript(path: Path, temp: Path) -> tuple[str, bool, str]:
    if not shutil.which("node"):
        return "node", None, "未找到 node"
    ok, detail = _run(["node", "--check", str(path)])
    if not ok:
        return "node --check", False, detail
    ok, detail = _run(["node", str(path)], cwd=ROOT)
    return "node-execute", ok, detail


def _check_cpp(path: Path, temp: Path) -> tuple[str, bool, str]:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        return "c++", None, "未找到 C++ 編譯器"
    binary = temp / f"{path.stem}.bin"
    ok, detail = _run([compiler, "-std=c++20", "-Wall", "-Wextra", "-O0",
                       str(path), "-o", str(binary)])
    if not ok:
        return "c++ compile", False, detail
    ok, detail = _run([str(binary)])
    return "c++ execute", ok, detail


def _check_typescript(path: Path, temp: Path) -> tuple[str, bool, str]:
    candidates = [shutil.which("tsc"), str(ROOT / "node_modules/.bin/tsc"),
                  str(ROOT / ".cache/npm/node_modules/.bin/tsc")]
    compiler = next((candidate for candidate in candidates
                     if candidate and Path(candidate).is_file()), None)
    if compiler:
        ok, detail = _run([compiler, "--strict", "--noEmit", "--target", "ES2022",
                           "--module", "esnext", str(path)])
        return "tsc", ok, detail
    method, ok, detail = _tree_sitter_parse(path, "tree_sitter_typescript", "language_typescript")
    if ok is None:
        return "typescript", None, detail
    return method, ok, detail


def _check_rust(path: Path, temp: Path) -> tuple[str, bool, str]:
    compiler = shutil.which("rustc")
    if compiler:
        metadata = temp / f"{path.stem}.rmeta"
        ok, detail = _run([compiler, "--edition", "2021", "--emit", "metadata",
                           "-o", str(metadata), str(path)])
        return "rustc", ok, detail
    method, ok, detail = _tree_sitter_parse(path, "tree_sitter_rust", "language")
    if ok is None:
        return "rust", None, detail
    return method, ok, detail


def _check_sql(path: Path, temp: Path) -> tuple[str, bool, str]:
    try:
        with sqlite3.connect(":memory:") as connection:
            connection.executescript(path.read_text(encoding="utf-8"))
        return "sqlite", True, ""
    except sqlite3.Error as exc:
        return "sqlite", False, str(exc)


def _check_html(path: Path, temp: Path) -> tuple[str, bool, str]:
    text = path.read_text(encoding="utf-8")
    parser = ScriptCollector()
    parser.feed(text)
    if parser.external or re.search(r"(?:src|href)\s*=\s*['\"](?:https?:)?//", text, re.I):
        return "html", False, "含外部資源"
    if re.search(r"\b(?:fetch|import)\s*\(", "\n".join(parser.scripts), re.I):
        return "html", False, "含外部載入呼叫"
    if "</html" not in text.casefold():
        return "html", False, "缺結尾"
    styles = re.findall(r"<style[^>]*>([\s\S]*?)</style>", text, re.I)
    for style in styles:
        if style.count("{") != style.count("}"):
            return "html/css", False, "樣式括號不合"
    if not shutil.which("node"):
        return "html", None, "未找到 node，無法驗腳本語法"
    for index, script in enumerate(parser.scripts):
        script_path = temp / f"{path.stem}-{index}.mjs"
        script_path.write_text(script, encoding="utf-8")
        ok, detail = _run(["node", "--check", str(script_path)])
        if not ok:
            return "node --check", False, detail
    return "html+node --check", True, ""


CHECKS = {".py": _check_python, ".js": _check_javascript, ".mjs": _check_javascript,
          ".cpp": _check_cpp, ".ts": _check_typescript, ".rs": _check_rust,
          ".sql": _check_sql, ".html": _check_html}


def verify_all(catalogue: Path = CODING_CATALOGUE, root: Path = ROOT) -> dict:
    catalogue = Path(catalogue).resolve()
    root = Path(root).resolve()
    payload = json.loads(catalogue.read_text(encoding="utf-8"))
    library = CodingLibrary(catalogue)
    guard = ChronologyGuard(payload["scope"]["historical_cutoff_year"])
    paths = sorted({root / lesson["code_file"] for lesson in payload["lessons"]})
    results = []
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        for path in paths:
            text = path.read_text(encoding="utf-8")
            reasons = list(guard.reasons(text, programming=True))
            years = post_cutoff_years(text)
            if reasons or years:
                results.append({"file": path.relative_to(root).as_posix(),
                                "status": "failed", "method": "chronology",
                                "detail": f"{reasons} {years}"})
                continue
            check = CHECKS.get(path.suffix)
            if check is None:
                results.append({"file": path.relative_to(root).as_posix(),
                                "status": "failed", "method": "dispatch",
                                "detail": "無對應驗法"})
                continue
            method, ok, detail = check(path, temp)
            status = "passed" if ok is True else "skipped" if ok is None else "failed"
            results.append({"file": path.relative_to(root).as_posix(),
                            "status": status, "method": method, "detail": detail.strip()})
    counts = {status: sum(item["status"] == status for item in results)
              for status in ("passed", "skipped", "failed")}
    return {"catalogue": catalogue.name, "cards": len(library.cards),
            "examples": len(paths), "counts": counts, "results": results,
            "note": "skipped 表示缺可選工具，非通過；failed 須修復。"}


def main() -> None:
    parser = argparse.ArgumentParser(description="驗校訂之現代程式例。")
    parser.add_argument("--catalogue", type=Path, default=CODING_CATALOGUE)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/classical/coding-examples-report.json")
    args = parser.parse_args()
    report = verify_all(args.catalogue)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report["counts"], ensure_ascii=False))
    print(f"報告：{args.output}")
    raise SystemExit(1 if report["counts"]["failed"] else 0)


if __name__ == "__main__":
    main()
