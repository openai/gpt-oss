import re
import sys
from typing import Optional


_PATTERNS = [
    # 0)"**Answer:** A" or "*Answers* – B", i.e. markdown‐wrapped "Answer(s)" with an unwrapped letter.
    re.compile(
        r'''(?ix)                   # case‐insensitive, ignore‐space
        (?:\*{1,2}|_{1,2})          # leading *…*  or _…_
        Answer[s]?                  #   Answer or Answers
        \s*[:\-–]?                  #   optional separator
        (?:\*{1,2}|_{1,2})          # closing wrapper
        \s*                         # optional space
        ([ABCD])\b                  # the actual letter
        ''',
        re.X
    ),

    # 0.1)
    re.compile(r'''(?ix)           # ignore case, allow verbose mode
        ^\s*                      # optional leading whitespace
        (?:\*{1,2}|_{1,2})?       # optional markdown wrapper
        Answer:?                   # the word 'answer' with an optional colon
        (?:\*{1,2}|_{1,2})?       # optional markdown wrapper again
        \s*:?\s*                  # optional colon with optional spaces
        (?:\*{1,2}|_{1,2})?       # optional markdown wrapper before letter
        ([ABCD])                 # capture the letter
        (?:\*{1,2}|_{1,2})?       # optional markdown wrapper after letter
        \s*                     # optional trailing whitespace, end of line
    ''', re.MULTILINE),

    # 1) Answer: (C)   or   Answers: (B)
    re.compile(r'\bAnswer[s]?\b\s*[:\-–]?\s*\(\s*([ABCD])\s*\)', re.IGNORECASE),

    # 2) Answer: C    or   Answers – D
    re.compile(r'\bAnswer[s]?\b\s*[:\-–]?\s*([ABCD])\b', re.IGNORECASE),

    # 3) Option B   or   Choice: C
    re.compile(r'\b(?:Option|Choice)\b\s*[:\-–]?\s*([ABCD])\b', re.IGNORECASE),

    # 7) LaTeX \boxed{...A...}, catches both \boxed{A} and
    #    \boxed{\text{A } 2.08\times10^{-6}\,\mathrm{m}} etc.
    re.compile(r'\\boxed\{[^}]*?([ABCD])[^}]*\}', re.MULTILINE | re.VERBOSE),

    # 7.5) LaTeX \boxed{\textbf{...C...}}
    re.compile(r'\\boxed\{[^}]*?\\textbf\{[^}]*?([ABCD])[^}]*\}[^}]*\}', re.MULTILINE | re.VERBOSE),

    # 7.51) LaTeX \boxed{\text{...C...}}
    re.compile(r'\\boxed\{[^}]*?\\text\{[^}]*?([ABCD])[^}]*\}[^}]*\}', re.MULTILINE | re.VERBOSE),

    # 4) bare singletons:  (A)  [B]
    re.compile(r'(?<![A-Za-z0-9])[\(\[]\s*([ABCD])\s*[\)\]](?![A-Za-z0-9])', re.VERBOSE),

    # 5) Markdown‐wrapped: *A*  **B**  _C_  __D__
    re.compile(r'(?<![A-Za-z0-9])(?:\*{1,2}|_{1,2})([ABCD])(?:\*{1,2}|_{1,2})(?![A-Za-z0-9])', re.VERBOSE),

    # 6) LaTeX \textbf{...C...}
    re.compile(r'\\textbf\{[^}]*?([ABCD])[^}]*\}', re.VERBOSE),

    # 8) markdown‐wrapped answer plus “)” plus description, e.g. **D) …**
    re.compile(r'''(?x)                        # ignore whitespace in pattern
        (?<![A-Za-z0-9])            # not preceded by word‐char
        (?:\*{1,2}|_{1,2})          # opening ** or __ or * or _
        \s*([ABCD])\)               # capture letter plus “)”
        [^*_\n]+?                   # some text inside wrapper
        (?:\*{1,2}|_{1,2})          # closing wrapper
        (?![A-Za-z0-9])             # not followed by word‐char
    '''),

    # 9) final fallback: a line that's exactly "A", "B.", "C)", "**D**", etc.
    re.compile(r'''(?x)^\s*
        (?:\*{1,2}|_{1,2})?     # optional markdown wrapper
        ([ABCD])                # capture group for letter
        (?:\*{1,2}|_{1,2})?     # optional closing markdown
        \s*[\.\)\-–:]?          # optional separator after the letter
        \s*.*$                  # allow any following text
    ''', re.MULTILINE),
]


def extract_abcd(text: str) -> Optional[str]:
    """
    Scan text (with Markdown/LaTeX wrappers intact) and return
    'A', 'B', 'C', or 'D' if a correct-answer declaration is found.
    Otherwise return None.
    """
    matches = []
    for prio, pat in enumerate(_PATTERNS):
        m = pat.search(text)
        if m:
            letter = m.group(1).upper()
            if letter in 'ABCD':
                matches.append((prio, m, letter))

    matches.sort(key=lambda triple: (
        triple[0],
        len(triple[1].group(0))
    ))
    for _, match, letter in matches:
        return letter
    return text.removeprefix('**')[:1]


def main() -> None:
    if len(sys.argv) > 1:
        # Process files
        for fn in sys.argv[1:]:
            try:
                with open(fn, encoding='utf8') as fp:
                    text = fp.read()
                ans = extract_abcd(text)
                print(f"{fn} ➜ {ans!r}")
            except (FileNotFoundError, IOError) as e:
                print(f"Error reading {fn}: {e}")
    else:
        # Read from stdin
        for line in sys.stdin:
            ans = extract_abcd(line.rstrip('\n'))
            print(f"{line.rstrip()} ➜ {ans!r}")


if __name__ == "__main__":
    main()

