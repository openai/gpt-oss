"""編札記為索引，原卷不逕入問答。"""

import argparse
import json
import sqlite3
from pathlib import Path

from .classical_library import suoyin_terms


def compile_siku(catalogue: Path, destination: Path) -> int:
    """唯納已校札記；未校之卷留目而不入庫。"""
    sources = json.loads(catalogue.read_text(encoding="utf-8"))["sources"]
    passages = []
    for source in sources:
        if source["status"] != "reviewed_notes":
            continue
        for note in source["notes"]:
            year = note["knowledge_year"]
            if type(year) is not int or not 1 <= year <= 1899:
                raise ValueError(f"札記年界未明：{source['title']}")
            if not note["location"] or not note["text"]:
                raise ValueError("札記須有正文及出處。")
            terms = " ".join(suoyin_terms(
                source["title"] + " " + note["text"] + " " + note.get("search_terms", "")
            ))
            passages.append((source["title"], note["location"], year, note["text"], terms))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(destination) as conn:
        conn.execute("DROP TABLE IF EXISTS passages")
        conn.execute("CREATE VIRTUAL TABLE passages USING fts5("
                     "title UNINDEXED, location UNINDEXED, year UNINDEXED, text UNINDEXED, terms)")
        conn.executemany("INSERT INTO passages VALUES (?, ?, ?, ?, ?)", passages)
    return len(passages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="校札記，編書目。")
    parser.add_argument("catalogue", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(f"已編札記 {compile_siku(args.catalogue, args.destination)} 則。")
