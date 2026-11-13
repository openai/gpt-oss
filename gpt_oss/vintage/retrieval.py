"""HistLM BERT retrieval over a date-stamped, locally curated passage corpus.

The encoder produces embeddings only. It does not generate answers.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class HistoricalPassage:
    passage_id: str
    text: str
    year: int
    source: str


@dataclass(frozen=True)
class RetrievalHit:
    passage: HistoricalPassage
    score: float


class HistLMRetriever:
    """Semantic retrieval using a time-sliced HistLM BERT checkpoint.

    Corpus JSONL fields: passage_id, text, year, source. Date filtering happens
    before embedding and again at query time.
    """

    def __init__(
        self, corpus_path: str | Path, *,
        model_id: str = "Livingwithmachines/bert_1760_1900",
        start_year: int = 1760, end_year: int = 1900,
        device: str | None = None, batch_size: int = 16, max_length: int = 256,
    ):
        if start_year > end_year:
            raise ValueError("start_year must not exceed end_year")
        self.corpus_path = Path(corpus_path)
        self.model_id, self.start_year, self.end_year = model_id, start_year, end_year
        self.device, self.batch_size, self.max_length = device, batch_size, max_length
        self._passages = self._load_passages()
        self._tokenizer = self._model = self._matrix = None

    def _load_passages(self) -> list[HistoricalPassage]:
        if not self.corpus_path.is_file():
            raise FileNotFoundError(self.corpus_path)
        passages, seen = [], set()
        with self.corpus_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                year, text = row.get("year"), row.get("text")
                if not isinstance(year, int) or not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Invalid passage at {self.corpus_path}:{line_number}")
                if not self.start_year <= year <= self.end_year:
                    continue
                passage_id = str(row.get("passage_id") or f"line-{line_number}")
                if passage_id in seen:
                    raise ValueError(f"Duplicate passage_id: {passage_id}")
                seen.add(passage_id)
                passages.append(HistoricalPassage(
                    passage_id, text.strip(), year, str(row.get("source") or "unspecified")
                ))
        if not passages:
            raise ValueError("No passages fall inside the configured historical window")
        return passages

    def _load_encoder(self):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("HistLM retrieval requires torch and transformers.") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id)
        self._model.eval()
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self.device)

    def _embed(self, texts: list[str]):
        import torch
        vectors = []
        for offset in range(0, len(texts), self.batch_size):
            batch = texts[offset:offset + self.batch_size]
            encoded = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                hidden = self._model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu())
        return torch.cat(vectors, dim=0)

    def build_index(self):
        if self._model is None:
            self._load_encoder()
        if self._matrix is None:
            self._matrix = self._embed([item.text for item in self._passages])
        return len(self._passages)

    def search(self, query: str, *, top_k: int = 5,
               start_year: int | None = None, end_year: int | None = None):
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        lower = max(self.start_year, start_year if start_year is not None else self.start_year)
        upper = min(self.end_year, end_year if end_year is not None else self.end_year)
        if lower > upper:
            return []
        self.build_index()
        query_vector = self._embed([query])[0]
        scores = self._matrix @ query_vector
        hits = [
            RetrievalHit(item, float(scores[index]))
            for index, item in enumerate(self._passages)
            if lower <= item.year <= upper
        ]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]
