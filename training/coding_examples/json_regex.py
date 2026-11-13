"""錄器為 JSON，以命名群組析簡牘之式。"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

PATTERN = re.compile(r"^(?P<power>river|ox|wind):(?P<turns>\d+)/(?P<buckets>\d+)$")


@dataclass(frozen=True)
class Record:
    power: str
    turns: int
    buckets: int


def parse(line: str) -> Record:
    match = PATTERN.fullmatch(line)
    if match is None:
        raise ValueError("簡牘格式不合。")
    return Record(match["power"], int(match["turns"]), int(match["buckets"]))


if __name__ == "__main__":
    record = parse("river:6/4")
    with TemporaryDirectory() as directory:
        path = Path(directory) / "reading.json"
        path.write_text(json.dumps(asdict(record), ensure_ascii=False), encoding="utf-8")
        assert json.loads(path.read_text(encoding="utf-8")) == {
            "power": "river", "turns": 6, "buckets": 4,
        }
    try:
        parse("river:-1/4")
    except ValueError:
        pass
    else:
        raise AssertionError("負數格式須遭拒。")
