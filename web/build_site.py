"""集頁、學約與札記，成一書齋。"""

import json
import sys
from pathlib import Path

from gpt_oss.classical import ClassicalCanon


def build_scroll(destination: Path):
    root = Path(__file__).resolve().parents[1]
    sources = json.loads((root / "docs/classical_sources.json").read_text())["sources"]
    notes = []
    for source in sources:
        if source["status"] == "reviewed_notes":
            for note in source["notes"]:
                if note["knowledge_year"] <= 1899:
                    notes.append(f"【{source['title']}；{note['location']}】\n{note['text']}")
    scroll = {"html": (root / "gpt_oss/responses_api/chat.html").read_text(),
              "instructions": ClassicalCanon().instructions, "notes": "\n\n".join(notes)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text((root / "web/worker-template.mjs").read_text().replace(
        "__SCROLL__", json.dumps(scroll, ensure_ascii=False)))


if __name__ == "__main__":
    build_scroll(Path(sys.argv[1]))
