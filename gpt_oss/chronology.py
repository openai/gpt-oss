"""Separate modern programming technique from post-cutoff worldly knowledge.

The guard is a deterministic, auditable *output boundary*, not machine unlearning.
It intentionally allows language syntax and standard-library facts while blocking
explicit post-cutoff dates, eras, events and institutions in prose. Fenced and
inline code are excluded from date screening so numeric literals remain usable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


MAX_CUTOFF_YEAR = 1899

PROGRAMMING_TERMS = (
    "程式", "程式碼", "函式", "函數", "變數", "型別", "類別", "物件", "陣列",
    "串列", "字典", "集合", "測試", "除錯", "編譯", "執行", "演算法", "標準函式庫",
    "語法", "重構", "非同步", "併發", "並行", "邊界", "錯誤處置", "資料庫", "查詢",
    "正則", "模組", "套件", "泛型", "迭代器", "指標", "記憶體", "瀏覽器", "原始碼",
    "python", "c++", "cpp", "typescript", "javascript", "rust", "html", "css",
    "sql", "json", "regex", "dom", "api", "cli", "node", "npm", "code", "function",
    "class", "async", "await", "promise", "generic", "iterator", "compile", "test",
    "debug", "algorithm", "stdlib", "ownership", "borrow", "array", "dict",
)

# These are deliberately *non-programming* markers. Language and library names
# are not here: their modern syntax is the explicit exception to the cutoff.
POST_CUTOFF_MARKERS = (
    "二十世紀", "廿世紀", "二十一世紀", "廿一世紀", "民國", "一戰", "二戰",
    "第一次世界大戰", "第二次世界大戰", "世界大戰", "冷戰", "俄國革命",
    "五四運動", "納粹", "蘇聯", "聯合國", "中華人民共和國", "登月", "阿波羅",
    "太空人", "太空梭", "人造衛星", "原子彈", "核武", "核能發電", "網際網路",
    "智慧型手機", "社群媒體", "氣候變遷", "新冠", "武漢肺炎", "COVID",
    "20th century", "21st century", "world war", "first world war",
    "second world war", "cold war", "nazi", "soviet union", "united nations",
    "moon landing", "apollo", "astronaut", "space shuttle", "sputnik",
    "atomic bomb", "nuclear weapon", "world wide web", "smartphone",
    "social media", "climate change", "covid", "pandemic",
)

ALLOWED_LANGUAGES = frozenset({
    "Python", "C++", "TypeScript", "JavaScript", "HTML", "CSS", "SQL", "Rust",
    "JSON", "Regex", "Shell",
})

YEAR_TOKEN = r"(?:1\d{3}|2\d{3})"
ARABIC_YEAR_RE = re.compile(r"(?<!\d)" + YEAR_TOKEN + r"(?!\d)")
CHINESE_YEAR_RE = re.compile(r"[一二三四五六七八九〇零]{4}年")
CHINESE_DIGITS = {"〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
                  "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CODE_FENCE_RE = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~")
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
VERSION_RE = re.compile(
    r"(?i)\b(?:Python\s*3(?:\.\d+)?|C\+\+\s*(?:17|20|23)|"
    r"TypeScript\s*\d+(?:\.\d+)?|ECMAScript\s*\d+|Rust\s*20\d{2}|"
    r"HTML\s*5|ECMA-\d+)\b"
)
DATA_YEAR_RE = re.compile(
    r"(?i)(?:資料|數值|輸入|範例|測試|假設|年份|日期|sample|input|fixture|literal|value|year)"
    r"[^。\n]{0,16}?" + YEAR_TOKEN
)
HISTORY_YEAR_RE = re.compile(
    r"(?i)(?:創立|創制|發明|發布|發表|問世|誕生|發生|首次|created|invented|"
    r"released|launched|founded|occurred|happened)[^。\n]{0,24}?(?P<later>" + YEAR_TOKEN + r")"
    r"|(?P<earlier>" + YEAR_TOKEN + r")[^。\n]{0,16}?(?:創立|創制|發明|發布|發表|問世|誕生|"
    r"發生|created|invented|released|launched|founded|occurred|happened)"
)


def strip_code(text: str) -> str:
    """Return prose only; code literals must not be mistaken for history."""
    return INLINE_CODE_RE.sub(" ", CODE_FENCE_RE.sub(" ", text or ""))


def chinese_year(match: str) -> int:
    digits = [CHINESE_DIGITS[char] for char in match[:-1]]
    value = 0
    for digit in digits:
        value = value * 10 + digit
    return value


def post_cutoff_years(text: str, cutoff_year: int = MAX_CUTOFF_YEAR) -> list[int]:
    """Find explicit years after the selected cutoff, including Chinese numerals."""
    if type(cutoff_year) is not int or not 1 <= cutoff_year <= MAX_CUTOFF_YEAR:
        raise ValueError("年界須在西元一至一八九九年之間。")
    prose = strip_code(text)
    years = [int(value) for value in ARABIC_YEAR_RE.findall(prose)]
    years.extend(chinese_year(value) for value in CHINESE_YEAR_RE.findall(prose))
    return sorted({year for year in years if year > cutoff_year})


@dataclass(frozen=True)
class ScopeVerdict:
    """A reproducible decision suitable for tests and response metadata."""

    programming: bool
    post_cutoff: bool
    blocked: bool
    reasons: tuple[str, ...]
    text: str

    def as_dict(self) -> dict:
        return {"programming": self.programming, "post_cutoff": self.post_cutoff,
                "blocked": self.blocked, "reasons": list(self.reasons)}


class ChronologyGuard:
    """Keep the 1899 boundary for worldly facts, with a narrow code exception."""

    def __init__(self, cutoff_year: int = MAX_CUTOFF_YEAR):
        if type(cutoff_year) is not int or not 1 <= cutoff_year <= MAX_CUTOFF_YEAR:
            raise ValueError("年界須在西元一至一八九九年之間。")
        self.cutoff_year = cutoff_year

    @property
    def refusal(self) -> str:
        return ("此事涉及年界以後之世事，非余所當知；若只問程式語法、"
                "標準函式庫、演算法或當次程式之誤，余可據符術特許作答。")

    def is_programming(self, text: str) -> bool:
        lowered = (text or "").casefold()
        return any(term.casefold() in lowered for term in PROGRAMMING_TERMS)

    def reasons(self, text: str, programming: bool = False) -> tuple[str, ...]:
        prose = strip_code(text)
        lowered = prose.casefold()
        reasons: list[str] = []

        for marker in POST_CUTOFF_MARKERS:
            if marker.casefold() in lowered:
                reasons.append(f"post_cutoff_marker:{marker}")
        for match in HISTORY_YEAR_RE.finditer(prose):
            year = int(match.group("later") or match.group("earlier"))
            if year > self.cutoff_year:
                reasons.append("historical_date_claim")
                break

        if programming:
            # Dates used as sample data and language version labels are not
            # historical assertions. Event markers above still apply.
            prose = DATA_YEAR_RE.sub(" ", prose)
            prose = VERSION_RE.sub(" ", prose)
            prose = HISTORY_YEAR_RE.sub(" ", prose)
        for year in post_cutoff_years(prose, self.cutoff_year):
            reasons.append(f"post_cutoff_year:{year}")
        return tuple(dict.fromkeys(reasons))

    def classify_request(self, text: str) -> ScopeVerdict:
        programming = self.is_programming(text)
        reasons = self.reasons(text, programming=programming)
        # A programming request containing a date is not refused outright;
        # its answer is still screened below. Unknown modern events without a
        # marker are a known limitation, documented instead of hidden.
        blocked = bool(reasons) and not programming
        return ScopeVerdict(programming=programming, post_cutoff=bool(reasons),
                            blocked=blocked, reasons=reasons, text=text)

    def sanitize(self, text: str, programming: bool = False) -> ScopeVerdict:
        reasons = self.reasons(text, programming=programming)
        blocked = bool(reasons)
        return ScopeVerdict(programming=programming, post_cutoff=blocked,
                            blocked=blocked, reasons=reasons,
                            text=self.refusal if blocked else (text or ""))
