"""Auditable source-grouped corpus preparation inspired by Vintage-LLM."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import re
from pathlib import Path


def prepare_corpus(input_path: Path, output_dir: Path, *, cutoff_year: int = 1899,
                   validation_fraction: float = 0.05, seed: int = 1899) -> dict:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    records, seen, raw_documents = [], set(), 0
    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw_documents += 1
            row = json.loads(line)
            text, year, source = row.get("text"), row.get("year"), str(row.get("source") or "").strip()
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Missing text at line {line_number}")
            if not isinstance(year, int) or year < 1 or year > cutoff_year:
                raise ValueError(f"Year must be an integer in 1..{cutoff_year} at line {line_number}")
            if not source:
                raise ValueError(f"Missing source at line {line_number}")
            normalized = re.sub(r"\s+", " ", text).strip()
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            records.append({"text": normalized, "year": year, "source": source, "sha256": digest})
    if len(records) < 2:
        raise ValueError("At least two unique documents are required")
    sources = sorted({row["source"] for row in records})
    if len(sources) < 2:
        raise ValueError("At least two distinct sources are required for source-safe splitting")
    rng = random.Random(seed)
    rng.shuffle(sources)
    valid_count = min(max(1, round(len(sources) * validation_fraction)), len(sources) - 1)
    valid_sources = set(sources[:valid_count])
    train = [row for row in records if row["source"] not in valid_sources]
    valid = [row for row in records if row["source"] in valid_sources]
    output_dir.mkdir(parents=True, exist_ok=True)

    def write(path, rows):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(
                    {key: row[key] for key in ("text", "year", "source")},
                    ensure_ascii=False,
                ) + "\n")

    train_path, valid_path = output_dir / "train.jsonl", output_dir / "valid.jsonl"
    write(train_path, train)
    write(valid_path, valid)
    manifest = {
        "cutoff_year": cutoff_year, "input": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "raw_documents": raw_documents, "unique_documents": len(records),
        "deduplicated_documents": raw_documents - len(records),
        "train_documents": len(train), "validation_documents": len(valid),
        "train_sources": sorted({row["source"] for row in train}),
        "validation_sources": sorted(valid_sources), "seed": seed,
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(valid_path.read_bytes()).hexdigest(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Prepare a source-safe vintage corpus")
    parser.add_argument("input", type=Path, help="JSONL rows with text, year and source")
    parser.add_argument("--output", type=Path, default=Path("data/vintage-corpus"))
    parser.add_argument("--cutoff-year", type=int, default=1899)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1899)
    args = parser.parse_args()
    result = prepare_corpus(args.input, args.output, cutoff_year=args.cutoff_year,
                            validation_fraction=args.validation_fraction, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
