"""What a command is, and what the model wrote about it.

Two directions of parsing, both pure:

  command -> kind     is this a file read (and of which lines), a search, a test run,
                      a `restore`? Decides which dedup key and which noise filter a turn
                      gets.
  reply   -> labels   PROGRESS_t and STALE_t, the two lines at the top of a reply.

Anything a parser does not recognise returns None rather than a guess: an unrecognised
read is simply not deduplicated, which costs compression and never correctness.
"""

from __future__ import annotations

import re
import shlex
import sys
from dataclasses import dataclass

EOF = sys.maxsize


# ------------------------------------------------------------------ file reads
@dataclass(frozen=True)
class ReadSpec:
    path: str
    line_start: int  # 1-indexed, inclusive
    line_end: int  # 1-indexed, inclusive; EOF for open-ended
    tail_n: int | None = None  # "last N lines": the start line is unknown, so never deduped
    cd: str | None = None


_SED_RANGE_RE = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*p\s*$")
_AWK_NR_RANGE_RE = re.compile(r"^\s*NR\s*>=\s*(\d+)\s*&&\s*NR\s*<=\s*(\d+)\s*$")
_AWK_NR_SINGLE_RE = re.compile(r"^\s*NR\s*==\s*(\d+)\s*$")


def parse_read_command(cmd: str) -> ReadSpec | None:
    """`cat [-n] F`, `head/tail -n N F`, `sed -n 'A,Bp' F`, `awk 'NR>=A && NR<=B' F`,
    `nl -ba F`, optionally piped once into head/tail/sed and optionally preceded by
    `X && ... &&` segments (a `cd DIR` among them is tracked)."""
    if not cmd or not cmd.strip():
        return None
    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        return None
    if not tokens:
        return None

    cd: str | None = None
    if "&&" in tokens:
        segments: list[list[str]] = [[]]
        for t in tokens:
            if t == "&&":
                segments.append([])
            else:
                segments[-1].append(t)
        if any(not s for s in segments):
            return None
        for seg in segments[:-1]:
            if seg[0] == "cd" and len(seg) == 2:
                cd = seg[1]
        tokens = segments[-1]

    for forbidden in ("&&", "||", ";", ">", ">>", "<", "<<", "&"):
        if forbidden in tokens:
            return None
    if tokens.count("|") > 1:
        return None
    if "|" in tokens:
        i = tokens.index("|")
        left, right = tokens[:i], tokens[i + 1:]
    else:
        left, right = tokens, []

    spec = _parse_single(left)
    if spec is None:
        return None
    if right:
        spec = _apply_pipe(spec, right)
        if spec is None:
            return None
    if cd is not None:
        spec = ReadSpec(path=spec.path, line_start=spec.line_start, line_end=spec.line_end,
                        tail_n=spec.tail_n, cd=cd)
    return spec


def _parse_single(tokens: list[str]) -> ReadSpec | None:
    if not tokens:
        return None
    cmd, args = tokens[0], tokens[1:]
    if cmd == "cat":
        files = [a for a in args if not a.startswith("-")]
        if len(files) != 1:
            return None
        flags = [a for a in args if a.startswith("-")]
        if any(f not in ("-n", "-b", "-A", "-E", "-T", "-s", "-v") for f in flags):
            return None
        return ReadSpec(path=files[0], line_start=1, line_end=EOF)
    if cmd == "head":
        return _parse_head_tail(args, is_tail=False)
    if cmd == "tail":
        return _parse_head_tail(args, is_tail=True)
    if cmd == "sed":
        return _parse_sed(args)
    if cmd == "awk":
        if len(args) != 2:
            return None
        m = _AWK_NR_RANGE_RE.match(args[0])
        if m:
            return ReadSpec(path=args[1], line_start=int(m.group(1)), line_end=int(m.group(2)))
        m = _AWK_NR_SINGLE_RE.match(args[0])
        if m:
            ln = int(m.group(1))
            return ReadSpec(path=args[1], line_start=ln, line_end=ln)
        return None
    if cmd == "nl":
        files = [a for a in args if not a.startswith("-")]
        if len(files) != 1:
            return None
        return ReadSpec(path=files[0], line_start=1, line_end=EOF)
    return None


def _count_flag(args: list[str], i: int) -> tuple[int | None, int] | None:
    """Parse `-n N`, `-nN`, `--lines=N` or `-N` at args[i]. Returns (n, tokens consumed),
    (None, 0) when args[i] is not a count flag, or None when it is one but malformed."""
    a = args[i]
    try:
        if a == "-n" and i + 1 < len(args):
            return int(args[i + 1]), 2
        if a.startswith("-n") and len(a) > 2:
            return int(a[2:]), 1
        if a.startswith("--lines="):
            return int(a.split("=", 1)[1]), 1
    except ValueError:
        return None
    if a.startswith("-") and len(a) > 1 and a[1:].isdigit():
        return int(a[1:]), 1
    return None, 0


def _parse_head_tail(args: list[str], is_tail: bool) -> ReadSpec | None:
    n: int | None = None
    files: list[str] = []
    i = 0
    while i < len(args):
        got = _count_flag(args, i)
        if got is None:
            return None
        if got[1]:
            n, i = got[0], i + got[1]
            continue
        if args[i].startswith("-"):
            return None
        files.append(args[i])
        i += 1
    if n is None or n <= 0 or len(files) != 1:
        return None
    if is_tail:
        return ReadSpec(path=files[0], line_start=1, line_end=EOF, tail_n=n)
    return ReadSpec(path=files[0], line_start=1, line_end=n)


def _parse_sed(args: list[str]) -> ReadSpec | None:
    if len(args) < 3 or args[0] != "-n":
        return None
    m = _SED_RANGE_RE.match(args[1])
    if m is None:
        return None
    files = args[2:]
    if len(files) != 1 or files[0].startswith("-"):
        return None
    start, end = int(m.group(1)), int(m.group(2))
    if start < 1 or end < start:
        return None
    return ReadSpec(path=files[0], line_start=start, line_end=end)


def _apply_pipe(spec: ReadSpec, right: list[str]) -> ReadSpec | None:
    """Only a whole-file read may be piped, into `head/tail -n N` or `sed -n 'A,Bp'`."""
    if spec.line_start != 1 or spec.line_end != EOF or spec.tail_n is not None or not right:
        return None
    rhs_cmd, rhs_args = right[0], right[1:]
    if rhs_cmd == "sed":
        if len(rhs_args) == 2 and rhs_args[0] == "-n":
            m = _SED_RANGE_RE.match(rhs_args[1])
            if m:
                return ReadSpec(path=spec.path, line_start=int(m.group(1)),
                                line_end=int(m.group(2)), cd=spec.cd)
        return None
    if rhs_cmd not in ("head", "tail"):
        return None
    if any(not a.startswith("-") and not a.lstrip("-").isdigit() for a in rhs_args):
        return None
    n: int | None = None
    i = 0
    while i < len(rhs_args):
        got = _count_flag(rhs_args, i)
        if got is None or not got[1] or rhs_args[i].startswith("--lines="):
            return None
        n, i = got[0], i + got[1]
    if n is None or n <= 0:
        return None
    if rhs_cmd == "head":
        return ReadSpec(path=spec.path, line_start=1, line_end=n)
    return ReadSpec(path=spec.path, line_start=1, line_end=EOF, tail_n=n)


_OUTPUT_RE = re.compile(r"<output>\n?(.*?)\n?</output>", re.DOTALL)
_OUTPUT_HEAD_RE = re.compile(r"<output_head>\n?(.*?)\n?</output_head>", re.DOTALL)
_CAT_N_PREFIX_RE = re.compile(r"^\s*\d+\t")


def observation_lines(content: str, used_cat_n: bool) -> list[str] | None:
    """The file lines inside a tool message (`<output>`, or only the head of an elided
    `<output_head>`), newline-terminated, with `cat -n` numbering stripped."""
    m = (_OUTPUT_HEAD_RE if "<output_head>" in content else _OUTPUT_RE).search(content)
    if m is None:
        return None
    body = m.group(1)
    if body == "":
        return []
    lines = [line + "\n" for line in body.split("\n")]
    if used_cat_n:
        lines = [_CAT_N_PREFIX_RE.sub("", line) for line in lines]
    return lines


def used_cat_n(cmd: str) -> bool:
    stripped = cmd.lstrip()
    return stripped.startswith("cat") and (" -n " in f" {cmd} " or stripped.startswith("cat -n"))


# ------------------------------------------------------------------ searches
_SEARCH_COMMANDS = frozenset({"grep", "egrep", "fgrep", "rgrep", "find", "rg"})
_WRITE_REDIRECTS = frozenset({">", ">>", "<", "<<"})


def parse_search_command(cmd: str) -> str | None:
    """Canonical form of a pure search (grep/find/rg), or None.

    Strips one leading `cd DIR &&`, one trailing `| head/tail -N` and trailing `2>&1`,
    and normalises quoting through shlex. Anything that writes, or pipes into something
    other than a count-truncation, is not a pure search.
    """
    if not cmd or not cmd.strip():
        return None
    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        return None
    if not tokens:
        return None
    if len(tokens) >= 4 and tokens[0] == "cd" and tokens[2] == "&&":
        tokens = tokens[3:]
    if any(t in _WRITE_REDIRECTS for t in tokens):
        return None
    if "|" in tokens:
        pieces: list[list[str]] = [[]]
        for t in tokens:
            if t == "|":
                pieces.append([])
            else:
                pieces[-1].append(t)
        if any(p and p[0] == "tee" for p in pieces[1:]) or len(pieces) > 2:
            return None
        if len(pieces) == 2:
            tail = pieces[1]
            if not tail or tail[0] not in ("head", "tail") or not _is_count_args(tail[1:]):
                return None
        tokens = pieces[0]
    while tokens and tokens[-1] == "2>&1":
        tokens.pop()
    if not tokens or tokens[0] not in _SEARCH_COMMANDS:
        return None
    return " ".join(tokens)


def _is_count_args(args: list[str]) -> bool:
    if len(args) == 1:
        got = _count_flag(args, 0)
        return bool(got and got[1] == 1)
    if len(args) == 2 and args[0] == "-n":
        try:
            int(args[1])
            return True
        except ValueError:
            return False
    return False


# ------------------------------------------------------------------ test runs
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_CHAIN_OPS = frozenset({"&&", "||", ";", "|"})


def is_pytest_command(cmd: str) -> bool:
    """`pytest`, `py.test` or `python[3] -m pytest` at the head of any sub-command, after
    optional `VAR=val` assignments and one `cd DIR`."""
    if not cmd or not cmd.strip():
        return False
    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        return False
    starts = [0] + [i + 1 for i, t in enumerate(tokens) if t in _CHAIN_OPS and i + 1 < len(tokens)]
    for j in starts:
        while j < len(tokens) and _ENV_ASSIGN_RE.match(tokens[j]):
            j += 1
        if j < len(tokens) - 1 and tokens[j] == "cd":
            j += 2
        if j >= len(tokens):
            continue
        if tokens[j] in ("pytest", "py.test"):
            return True
        if tokens[j] in ("python", "python3") and j + 2 < len(tokens) \
                and tokens[j + 1] == "-m" and tokens[j + 2] == "pytest":
            return True
    return False


# Deliberately broad: the filter it enables only folds lines matching known PASS/progress
# patterns, so a false positive on output with no such lines changes nothing.
_TESTRUNNER_RES = [
    re.compile(r"(^|[;&|]\s*|&&\s*)go test\b"),
    re.compile(r"(^|[;&|]\s*|&&\s*)cargo test\b"),
    re.compile(r"\b(npx\s+)?(jest|vitest)\b"),
    re.compile(r"\b(npm|yarn|pnpm)\s+(run\s+)?test\b"),
    re.compile(r"\bmvn\b[^|;&]*\btest\b"),
]


def is_testrunner_command(cmd: str) -> bool:
    return bool(cmd and cmd.strip()) and any(p.search(cmd) for p in _TESTRUNNER_RES)


_RESTORE_RE = re.compile(r"restore\s+((?:step\s+)?#?\d+(?:\s*-\s*\d+)?(?:\s*[,\s]\s*#?\d+(?:\s*-\s*\d+)?)*)")


def parse_restore_command(cmd: str) -> list[int] | None:
    """Step ids of a `restore <step>` command (`restore 7`, `restore 3-5`, `restore 3, 9`),
    or None for any other command. The whole command must be the restore: the `restore`
    binary of dump/restore never takes a bare number, so nothing real is shadowed."""
    m = _RESTORE_RE.fullmatch(cmd.strip()) if cmd else None
    if m is None:
        return None
    ids: list[int] = []
    for lo, hi, one in re.findall(r"(\d+)\s*-\s*(\d+)|(\d+)", m.group(1)):
        for u in [int(one)] if one else range(int(lo), min(int(hi), int(lo) + 500) + 1):
            if u not in ids:
                ids.append(u)
    return ids


# ------------------------------------------------------------------ the model's labels
_PROGRESS_RE = re.compile(r"\s*PROGRESS\b\s*:?\s*[:|\-]?\s*(.*)", re.IGNORECASE)


def parse_progress(content: str | None) -> str | None:
    """PROGRESS_t: the one-line summary on the reply's `PROGRESS:` line, or None."""
    for line in (content or "").splitlines():
        m = _PROGRESS_RE.match(line)
        if m:
            return m.group(1).strip().strip("`").strip()
    return None


def parse_stale(content: str | None, max_lines: int = 4) -> list[int]:
    """STALE_t: step ids on a `STALE:` line among the first `max_lines` non-empty lines.

    Only the top of the reply is scanned, where the metadata lives, so a "stale" in the
    reasoning cannot match. `-`, `none`, `[]` and `n/a` mean nothing is named.
    """
    body = _metadata_line(content, "STALE", max_lines)
    return [int(x) for x in re.findall(r"\d+", body)] if body else []


def _metadata_line(content: str | None, key: str, max_lines: int) -> str:
    if not content:
        return ""
    seen = 0
    for line in content.splitlines():
        if not line.strip():
            continue
        seen += 1
        if seen > max_lines:
            break
        m = re.match(rf"\s*`?{key}`?\s*[-:–]\s*(.*)", line, re.IGNORECASE)
        if not m:
            continue
        body = m.group(1).strip().strip("`").strip()
        return "" if body.lower() in ("none", "-", "[]", "n/a") else body
    return ""
