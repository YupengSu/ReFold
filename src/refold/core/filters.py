"""Test-output noise filters, part of deduplication and under its rules (see dedup.py).

pytest: dot progress, PASSED-only lines and the warnings summary are dropped; failures,
tracebacks, assertion lines, file:line references and the summary are kept. Other runners
(go, cargo, jest/vitest/mocha): each run of lines matching a known PASS/progress pattern
becomes one marker line. These functions only decide what goes; dedup.py adds the
`[context-manager]` note in front and keeps the result only if it is shorter.
"""

from __future__ import annotations

import re

_SEPARATOR_RE = re.compile(r"^[=\-]{3,}")
_WARNINGS_HEADER_RE = re.compile(r"warnings summary", re.IGNORECASE)
_SUMMARY_COUNT_RE = re.compile(
    r"\b\d+\s+(?:failed|passed|error|errors|skipped|deselected|warning|warnings|xfailed|xpassed)\b",
    re.IGNORECASE,
)
_DOT_PROGRESS_BODY_RE = re.compile(r"^[.sFEXxpPw\s]+$")
_PERCENT_TAIL_RE = re.compile(r"\s*\[\s*\d+%\s*\]\s*$")
_ERROR_FAIL_KW_RE = re.compile(r"\b(?:error|fail)", re.IGNORECASE)
_KEEP_SUBSTRINGS = ("FAILED", "ERROR", "FAIL", "Traceback", "AssertionError", "assert ", "E   ")
_HEADER_KEEP = 3
_TRAILER_KEEP = 5
_MIN_OUTPUT_LINES = 3


def filter_pytest_output(obs: str) -> tuple[str, int, int] | None:
    """(the kept lines, how many, how many there were), or None if too little would be left."""
    lines = obs.splitlines(keepends=True)
    n = len(lines)
    if n == 0:
        return None
    keep: set[int] = set()
    in_warnings_block = False
    for i, line in enumerate(lines):
        body = line.rstrip("\r\n")
        if _SEPARATOR_RE.match(body):
            # Section separators are kept, and they open/close the warnings block.
            in_warnings_block = bool(_WARNINGS_HEADER_RE.search(body))
            keep.add(i)
            continue
        if in_warnings_block:
            if _ERROR_FAIL_KW_RE.search(body):
                keep.add(i)
            continue
        if not body.strip() or _is_dot_progress(body) or ("PASSED" in body and "FAILED" not in body):
            continue
        if (i < _HEADER_KEEP or i >= n - _TRAILER_KEEP
                or any(kw in body for kw in _KEEP_SUBSTRINGS)
                or _SUMMARY_COUNT_RE.search(body)
                or ".py:" in body
                or body.startswith(">")
                or line.startswith((" ", "\t"))):
            keep.add(i)
    if len(keep) < _MIN_OUTPUT_LINES:
        return None
    return "".join(lines[i] for i in sorted(keep)), len(keep), n


def _is_dot_progress(body: str) -> bool:
    s = _PERCENT_TAIL_RE.sub("", body)
    return not s.strip() or bool(_DOT_PROGRESS_BODY_RE.match(s))


_NOISE_RES = [
    re.compile(r"^\s*=== (RUN|PAUSE|CONT)\s"),  # go
    re.compile(r"^\s*--- PASS: .+\(\d"),  # go, top-level and subtests
    re.compile(r"^test \S+ \.\.\. ok$"),  # cargo
    re.compile(r"^\s*[✓✔] .+"),  # jest / vitest / mocha
]
_MIN_TOTAL_NOISE = 3


def filter_test_output(obs: str, run_note: str) -> str | None:
    """The output with each run of passing/progress lines replaced by `run_note` (a
    template with {n} and {s}), or None if there are too few such lines to bother."""
    lines = obs.split("\n")
    if sum(1 for line in lines if _is_noise(line)) < _MIN_TOTAL_NOISE:
        return None
    out: list[str] = []
    run = 0
    for line in lines + [None]:
        if line is not None and _is_noise(line):
            run += 1
            continue
        if run:
            out.append(run_note.format(n=run, s="s" if run > 1 else ""))
            run = 0
        if line is not None:
            out.append(line)
    return "\n".join(out)


def _is_noise(line: str) -> bool:
    return any(p.match(line) for p in _NOISE_RES)
