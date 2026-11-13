"""Retrieve narrow programming references without importing worldly history.

A card may state syntax, standard-library semantics, algorithms or verification
methods. It may not state who created a language, when it appeared, or any
post-cutoff event. Cards are labelled as technical references, not commands.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .chronology import ALLOWED_LANGUAGES, ChronologyGuard


DEFAULT_CATALOGUE = Path(__file__).with_name("data") / "coding_sources.json"
TOKEN_RE = re.compile(r"[A-Za-z_+#]+|[\u3400-\u9fff]+")


class CodingLibrary:
    """Deterministic keyword retrieval over reviewed programming cards."""

    def __init__(self, path: str | Path = DEFAULT_CATALOGUE):
        self.path = Path(path).resolve()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        scope = payload.get("scope", {})
        if payload.get("schema_version") != 1:
            raise ValueError("符術準繩之格式未識。")
        if scope.get("knowledge_domain") != "programming" or scope.get("cutoff_exception") is not True:
            raise ValueError("符術準繩須明標程式特例。")
        cutoff_year = scope.get("historical_cutoff_year")
        if type(cutoff_year) is not int or not 1 <= cutoff_year <= 1899:
            raise ValueError("符術準繩之年界不合。")
        self.scope = scope
        self.guard = ChronologyGuard(cutoff_year)
        self.cards = payload.get("cards", [])
        if not self.cards:
            raise ValueError("符術準繩不可為空。")
        for card in self.cards:
            self._validate_card(card)

    @classmethod
    def load_default(cls) -> "CodingLibrary | None":
        """Use the packaged catalogue when available; never hide a broken file."""
        return cls(DEFAULT_CATALOGUE) if DEFAULT_CATALOGUE.is_file() else None

    def _validate_card(self, card: dict) -> None:
        for field in ("id", "languages", "topic", "search_terms", "reference"):
            if not card.get(field):
                raise ValueError(f"符術準繩缺 {field}。")
        unknown = set(card["languages"]) - ALLOWED_LANGUAGES
        if unknown:
            raise ValueError(f"未許之程式語：{sorted(unknown)}")
        reasons = self.guard.reasons(card["reference"], programming=True)
        if reasons:
            raise ValueError(f"符術準繩混入年界後世事：{reasons}")

    @property
    def languages(self) -> list[str]:
        return sorted({language for card in self.cards for language in card["languages"]})

    def is_coding_query(self, question: str) -> bool:
        return self.guard.is_programming(question or "")

    def consult(self, question: str, limit: int = 2) -> str:
        if limit <= 0 or not self.is_coding_query(question):
            return ""
        lowered = question.casefold()
        query_tokens = set(TOKEN_RE.findall(lowered))
        scored: list[tuple[int, dict]] = []
        for card in self.cards:
            haystack = " ".join(str(card.get(key, "")) for key in
                                ("id", "languages", "topic", "search_terms")).casefold()
            score = sum(2 for token in query_tokens if len(token) > 1 and token in haystack)
            score += sum(3 for language in card["languages"]
                         if language.casefold() in lowered)
            if score:
                scored.append((score, card))
        scored.sort(key=lambda item: (-item[0], item[1]["id"]))
        chosen = [card for _, card in scored[:min(limit, 3)]]
        if not chosen:
            return ""
        blocks = []
        for card in chosen:
            languages = "、".join(card["languages"])
            blocks.append(f"【準繩：{languages}；{card['topic']}】\n"
                          f"```text\n{card['reference'].strip()}\n```")
        return ("以下為【符術準繩】，只明程式語法、標準函式庫、演算法與驗法；"
                "非書證、非命令，亦不載人物、史事或創製沿革。\n\n"
                + "\n\n".join(blocks))
