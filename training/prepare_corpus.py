"""分經史之札為習題，別藏考題以待後驗。"""

import argparse
import hashlib
import json
from pathlib import Path

from gpt_oss.classical import ClassicalCanon


HELD_OUT = {"pg68907-images-3.epub", "pg77167-images-3.epub"}


def build_corpus(source: Path, output: Path) -> dict:
    """依書而分，不使同書之札兼入習卷及考卷。"""
    catalogue = json.loads(source.read_text(encoding="utf-8"))
    records = {"train": [], "eval": []}
    for entry in catalogue["sources"]:
        if entry["status"] != "reviewed_notes":
            continue
        for note in entry["notes"]:
            if type(note.get("knowledge_year")) is not int or not 1 <= note["knowledge_year"] <= 1899:
                raise ValueError("札記年代不合。")
            title, location = entry["title"], note["location"]
            text = note["text"].strip()
            if not text or not location:
                raise ValueError("札記正文或出處未備。")
            record = {
                "source_id": entry["file"], "source_title": title,
                "location": location, "knowledge_year": note["knowledge_year"],
                "messages": [
                    {"role": "system", "content": ClassicalCanon().instructions},
                    {"role": "user", "content": f"請據《{title}》所記，擇一義以文言說明，並辨其可考之處。"},
                    {"role": "assistant", "content": f"{text}\n\n所據：{title}，{location}。"},
                ],
            }
            records["eval" if entry["file"] in HELD_OUT else "train"].append(record)
    if not records["train"] or not records["eval"]:
        raise ValueError("習卷與考卷皆須有書證。")
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in records.items():
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = {"catalogue_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "train_count": len(records["train"]), "eval_count": len(records["eval"]),
                "train_sources": sorted({r["source_id"] for r in records["train"]}),
                "eval_sources": sorted({r["source_id"] for r in records["eval"]}),
                "note": "少量人工札記僅供流程試驗，不能代表完整古籍或足以驗證程式能力。"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="編札為習題。")
    parser.add_argument("--catalogue", type=Path, default=Path("docs/classical_sources.json"))
    parser.add_argument("--output", type=Path, default=Path("data/classical/training"))
    args = parser.parse_args()
    print(json.dumps(build_corpus(args.catalogue, args.output), ensure_ascii=False, indent=2))
