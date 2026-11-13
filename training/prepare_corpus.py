"""分經史之札與符術之課為習卷，別藏未見考題以待後驗。

史事之卷必有 `knowledge_year`；程式之卷只標 `knowledge_domain=
"programming"`，不冒充某年以前之古籍。兩者皆經年代與範圍查核。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from gpt_oss.chronology import ALLOWED_LANGUAGES, ChronologyGuard, post_cutoff_years
from gpt_oss.classical import ClassicalCanon


ROOT = Path(__file__).resolve().parents[1]
CLASSICAL_CATALOGUE = ROOT / "docs" / "classical_sources.json"
CODING_CATALOGUE = ROOT / "gpt_oss" / "data" / "coding_sources.json"
DEFAULT_OUTPUT = ROOT / "data" / "classical" / "training"
HELD_OUT = {"pg68907-images-3.epub", "pg77167-images-3.epub"}
FENCES = {".py": "python", ".js": "javascript", ".mjs": "javascript",
          ".ts": "typescript", ".cpp": "cpp", ".rs": "rust",
          ".sql": "sql", ".html": "html", ".css": "css"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reject_history(text: str, label: str, guard: ChronologyGuard) -> None:
    reasons = guard.reasons(text, programming=True)
    years = post_cutoff_years(text)
    if reasons or years:
        raise ValueError(f"{label} 混入年界後之事：{list(reasons)} {years}")


def _validate_coding_lesson(lesson: dict, catalogue: Path, guard: ChronologyGuard) -> tuple[str, dict]:
    required = ("id", "split", "source_id", "status", "languages", "topic",
                "prompt", "answer", "code_file", "acceptance")
    for field in required:
        if not lesson.get(field):
            raise ValueError(f"符術課缺 {field}：{lesson.get('id')}")
    if lesson["split"] not in {"train", "eval"}:
        raise ValueError("符術課之分卷不明。")
    if lesson["status"] != "reviewed_lesson":
        raise ValueError("未校符術課不可入卷。")
    if set(lesson["languages"]) - ALLOWED_LANGUAGES:
        raise ValueError(f"未許之程式語：{lesson['id']}")
    _reject_history(lesson["prompt"] + "\n" + lesson["answer"], lesson["id"], guard)

    relative = Path(lesson["code_file"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("程式例之路徑須在專案之內。")
    code_path = ROOT / relative
    if not code_path.is_file():
        raise FileNotFoundError(code_path)
    code = code_path.read_text(encoding="utf-8")
    _reject_history(code, str(relative), guard)
    if relative.suffix not in FENCES:
        raise ValueError(f"未識之程式檔：{relative}")
    return code, {
        "relative_path": relative.as_posix(),
        "fence": FENCES[relative.suffix],
        "sha256": sha256(code_path),
    }


def build_corpus(classical_source: Path = CLASSICAL_CATALOGUE,
                 coding_source: Path = CODING_CATALOGUE,
                 output: Path = DEFAULT_OUTPUT) -> dict:
    """依書源分習卷與考卷；史事與程式同卷而異其知識域。"""
    classical_source = Path(classical_source).resolve()
    coding_source = Path(coding_source).resolve()
    output = Path(output).resolve()
    catalogue = json.loads(classical_source.read_text(encoding="utf-8"))
    coding = json.loads(coding_source.read_text(encoding="utf-8"))
    guard = ChronologyGuard(coding["scope"]["historical_cutoff_year"])
    records: dict[str, list[dict]] = {"train": [], "eval": []}

    for entry in catalogue["sources"]:
        if entry["status"] != "reviewed_notes":
            continue
        for note in entry["notes"]:
            year = note.get("knowledge_year")
            if type(year) is not int or not 1 <= year <= 1899:
                raise ValueError("札記年代不合。")
            title, location = entry["title"], note["location"]
            text = note["text"].strip()
            if not text or not location:
                raise ValueError("札記正文或出處未備。")
            record = {
                "source_id": entry["file"], "source_title": title,
                "location": location, "knowledge_domain": "historical",
                "cutoff_exception": False, "knowledge_year": year,
                "messages": [
                    {"role": "system", "content": ClassicalCanon().instructions},
                    {"role": "user", "content": f"請據《{title}》所記，擇一義以文言說明，並辨其可考之處。"},
                    {"role": "assistant", "content": f"{text}\n\n所據：{title}，{location}。"},
                ],
            }
            records["eval" if entry["file"] in HELD_OUT else "train"].append(record)

    code_hashes: dict[str, str] = {}
    for lesson in coding["lessons"]:
        code, meta = _validate_coding_lesson(lesson, coding_source, guard)
        code_hashes[meta["relative_path"]] = meta["sha256"]
        assistant = (f"{lesson['answer'].strip()}\n\n"
                     f"```{meta['fence']}\n{code.strip()}\n```\n\n"
                     f"驗法：{'；'.join(lesson['acceptance'])}。")
        record = {
            "source_id": lesson["source_id"],
            "source_title": f"符術習卷：{lesson['topic']}",
            "location": f"{lesson['id']}；{meta['relative_path']}",
            "knowledge_domain": "programming",
            "cutoff_exception": True,
            "knowledge_year": None,
            "languages": lesson["languages"],
            "code_file": meta["relative_path"],
            "code_sha256": meta["sha256"],
            "messages": [
                {"role": "system", "content": ClassicalCanon().instructions},
                {"role": "user", "content": lesson["prompt"]},
                {"role": "assistant", "content": assistant},
            ],
        }
        records[lesson["split"]].append(record)

    train_sources = {r["source_id"] for r in records["train"]}
    eval_sources = {r["source_id"] for r in records["eval"]}
    if not records["train"] or not records["eval"]:
        raise ValueError("習卷與考卷皆須有材料。")
    if train_sources & eval_sources:
        raise ValueError("習卷與考卷不可含同一書源或題源。")
    for split, rows in records.items():
        domains = {r["knowledge_domain"] for r in rows}
        if domains != {"historical", "programming"}:
            raise ValueError(f"{split} 須兼含史事與程式，俾兩界皆可驗。")

    output.mkdir(parents=True, exist_ok=True)
    for split, rows in records.items():
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "catalogues": {
            "historical": {"path": classical_source.name, "sha256": sha256(classical_source)},
            "programming": {"path": coding_source.name, "sha256": sha256(coding_source)},
        },
        "code_examples": dict(sorted(code_hashes.items())),
        "train_count": len(records["train"]), "eval_count": len(records["eval"]),
        "train_sources": sorted(train_sources), "eval_sources": sorted(eval_sources),
        "domains": {split: {domain: sum(r["knowledge_domain"] == domain for r in rows)
                            for domain in ("historical", "programming")}
                    for split, rows in records.items()},
        "split_policy": "依書源或題源分卷；考卷之程式題與程式檔皆不見於習卷。",
        "note": "程式為明示特例；微調與提示不能保證自底座刪去一八九九年後之世事。",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="編札記與符術課為習題。")
    parser.add_argument("--classical-catalogue", type=Path, default=CLASSICAL_CATALOGUE)
    parser.add_argument("--coding-catalogue", type=Path, default=CODING_CATALOGUE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build_corpus(args.classical_catalogue, args.coding_catalogue,
                                  args.output), ensure_ascii=False, indent=2))
