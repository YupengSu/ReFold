"""Deduplication: d_i, the observation o_i with repeated spans replaced by stubs.

Each turn is compared only with turns before it, and the first copy is the one kept, so
an old turn's rendering never depends on a later one and, between boundary advances, the
prompt only grows at its end. What counts as a repeat:

  file read    a line with the same path, line number and bytes. All lines seen before:
               one stub naming the step that holds them; some: the new lines, with a stub
               for each run of old ones; none: in full.
  search       grep / find / rg: the same canonical command with the same output.
  any output   the same text, if it is at least 200 characters.

A stub for 20 lines or more keeps a signature skeleton (the top-level class and function
signatures, by file extension). Test runs (pytest; go / cargo / jest / vitest) also lose
their passing and progress lines (filters.py), under the same rules.

Every removal leaves a `[context-manager]` note (prompts.json, note_*) that names where the
same content still is and ends with the `restore <step>` that shows it again. A stub only
names a step still in the prompt: folded steps are not sources. A re-run renders in full:

  boundary    steps at or past b_t render as received, so an output the agent just asked
              for stays in full for at least k steps;
  once        once a read range, test command, search or output has been compressed, the
              next repeat of it renders in full and stays in full.

`reversible=False` (ablation): a repeat is compressed again, and folded steps stay
sources, so a stub can name a step the prompt no longer shows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from refold.core.filters import filter_pytest_output, filter_test_output


@dataclass(frozen=True)
class ReadMark:
    """What a file read showed, snapshotted when it ran: `content[i]` is line
    `line_start + i`. Never re-read from disk -- this is what the model saw."""

    path: str
    line_start: int
    line_end: int
    content: tuple[str, ...]


@dataclass(frozen=True)
class Turn:
    """One tool call and its observation, as the tool message was sent."""

    index: int
    command: str
    observation: str
    read: ReadMark | None = None
    search: tuple[str, str] | None = None  # (canonical command, sha1 of the output)
    pytest: bool = False
    testrunner: bool = False


@dataclass(frozen=True)
class Rendered:
    index: int
    body: str | None = None
    """The tool message body to send in place of the observation; None: as received."""


_OUTPUT_MIN_CHARS = 200

# A stub for content this long carries a signature skeleton, so the model can orient
# without restoring; shorter content stays a one-line stub.
_SKELETON_MIN_LINES = 20
_SKELETON_MAX_LINES = 12
_SKELETON_LINE_CAP = 160
_SIG_PATTERNS: dict[str, str] = {
    "py": r"^\s*(?:async\s+)?(?:def|class)\s+\w",
    "js": r"^\s*(?:export\s+(?:default\s+)?)?(?:async\s+)?function\b|^\s*class\s+\w"
          r"|^\s*(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?(?:function\b|\()|^\s*module\.exports\b",
    "go": r"^func\b|^type\s+\w",
    "java": r"^\s*(?:public|private|protected)\b.*\(",
    "php": r"^\s*(?:public|private|protected|static)?\s*function\s+\w|^\s*class\s+\w",
    "rs": r"^\s*(?:pub\s+)?(?:fn|struct|enum|trait|impl)\b",
}
for _ext in ("ts", "tsx", "jsx", "mjs"):
    _SIG_PATTERNS[_ext] = _SIG_PATTERNS["js"]


def skeleton_of(lines, path: str | None) -> str:
    """Definition lines for a known language; otherwise first line, count, last line."""
    ext = path.rsplit(".", 1)[-1].lower() if path and "." in path else ""
    pat = _SIG_PATTERNS.get(ext)
    picked: list[str] = []
    if pat is not None:
        rx = re.compile(pat)
        picked = [ln.rstrip("\n") for ln in lines if rx.match(ln)]
    if not picked:
        stripped = [ln.rstrip("\n") for ln in lines if ln.strip()]
        if len(stripped) < 2:
            return ""
        picked = [stripped[0], f"... ({len(stripped)} lines) ...", stripped[-1]]
    extra = len(picked) - _SKELETON_MAX_LINES
    if extra > 0:
        picked = picked[:_SKELETON_MAX_LINES] + [f"(+{extra} more)"]
    return "\n".join(ln[:_SKELETON_LINE_CAP] for ln in picked)


def _with_skeleton(note: str, lines, path: str | None) -> str:
    """The note, plus the signature skeleton as indented detail lines for long content."""
    if len(lines) < _SKELETON_MIN_LINES:
        return note
    skel = skeleton_of(lines, path)
    return note + "\n" + "\n".join("  " + ln for ln in skel.split("\n")) if skel else note


_RETURNCODE_RE = re.compile(r"<returncode>(-?\d+)</returncode>")


def _framed(original: str, body: str) -> str:
    """`body` in the shell frame of the tool message it came from."""
    m = _RETURNCODE_RE.search(original)
    return f"<returncode>{m.group(1) if m else 0}</returncode>\n<output>\n{body}\n</output>"


def render(turns: list[Turn], folded: set[int], boundary: int, notes: dict[str, str], *,
           reversible: bool = True) -> list[Rendered]:
    """d_i for every turn not in `folded`; turns with index >= `boundary` as received.

    `notes` are the note templates from prompts.json.
    """
    dedup = _Deduplicator(notes, reversible)
    out = []
    for turn in turns:
        if turn.index in folded and reversible:
            continue
        out.append(dedup.render(turn, protected=turn.index >= boundary))
    return [r for r in out if r.index not in folded]  # folded turns were only sources


class _Deduplicator:
    """One pass over the turns in order: what has been shown, and what has already been
    compressed once (with `reversible`, a repeat of that renders in full)."""

    def __init__(self, notes: dict[str, str], reversible: bool):
        self.notes, self.reversible = notes, reversible
        self.line_text: dict[tuple[str, int], str] = {}  # (path, line) -> text last shown
        self.line_step: dict[tuple[str, int], int] = {}  # (path, line) -> the step that showed it
        self.first_search: dict[tuple[str, str], int] = {}  # (command, output sha1) -> first step
        self.first_output: dict[str, int] = {}  # output text -> first step
        self.compressed: set[tuple] = set()  # (kind, key) compressed once already

    def render(self, turn: Turn, protected: bool) -> Rendered:
        if turn.read is not None and turn.read.content:
            return self._read(turn, protected)
        if (turn.pytest or turn.testrunner) and not protected:
            rendered = self._test_run(turn)
            if rendered is not None:
                return rendered
        if turn.search is not None:
            return self._search(turn, protected)
        if len(turn.observation) >= _OUTPUT_MIN_CHARS:
            return self._output(turn, protected)
        return Rendered(turn.index)

    def _compressed_once(self, kind: str, key) -> None:
        if self.reversible:
            self.compressed.add((kind, key))

    def _test_run(self, turn: Turn) -> Rendered | None:
        """Passing and progress lines dropped, the first time this command's output is."""
        if ("test", turn.command) in self.compressed:
            return None
        body = _filtered(turn, self.notes)
        if body is None:
            return None
        self._compressed_once("test", turn.command)
        return Rendered(turn.index, body)

    def _search(self, turn: Turn, protected: bool) -> Rendered:
        first = self.first_search.get(turn.search)
        if first is not None and not protected and ("search", turn.search) not in self.compressed:
            self._compressed_once("search", turn.search)
            return Rendered(turn.index, self.notes["note_search"].format(first=first, step=turn.index))
        self.first_search.setdefault(turn.search, turn.index)
        return Rendered(turn.index)

    def _output(self, turn: Turn, protected: bool) -> Rendered:
        obs = turn.observation
        first = self.first_output.get(obs)
        if first is not None and not protected and ("output", obs) not in self.compressed:
            self._compressed_once("output", obs)
            note = self.notes["note_output"].format(first=first, step=turn.index)
            return Rendered(turn.index, _with_skeleton(note, obs.splitlines(), None))
        if first is None:
            self.first_output[obs] = turn.index
        return Rendered(turn.index)

    def _read(self, turn: Turn, protected: bool) -> Rendered:
        """All lines seen before: one stub; some: the new lines with a stub per run of old
        ones; none: in full. Every line read updates what was last shown."""
        mark = turn.read
        novel: dict[int, str] = {}
        for offset, line in enumerate(mark.content):
            key = (mark.path, mark.line_start + offset)
            if self.line_text.get(key) != line:
                novel[key[1]] = line
                self.line_text[key] = line
                self.line_step[key] = turn.index
        rng = (mark.path, mark.line_start, mark.line_end)
        if protected or len(novel) == len(mark.content) or (not novel and ("read", rng) in self.compressed):
            return Rendered(turn.index)
        self._compressed_once("read", rng)
        if not novel:
            first = min(self.line_step.get((mark.path, mark.line_start + o), turn.index)
                        for o in range(len(mark.content)))
            note = self.notes["note_read"].format(path=mark.path, start=mark.line_start, end=mark.line_end,
                                                  first=first, step=turn.index)
            return Rendered(turn.index, _with_skeleton(note, mark.content, mark.path))
        body = _partial(mark, novel, self.line_step, self.notes["note_partial_run"])
        return Rendered(turn.index, self.notes["note_partial"].format(step=turn.index) + "\n"
                        + _framed(turn.observation, body))


def _filtered(turn: Turn, notes: dict[str, str]) -> str | None:
    """A test run's output with its passing/progress lines omitted, under its note; None
    when that would not make it shorter."""
    if turn.pytest:
        got = filter_pytest_output(turn.observation)
        if got is None:
            return None
        kept, n_kept, total = got
        body = notes["note_pytest"].format(omitted=total - n_kept, total=total, step=turn.index) + "\n" + kept
    else:
        kept = filter_test_output(turn.observation, notes["note_tests_run"])
        if kept is None:
            return None
        body = notes["note_tests"].format(step=turn.index) + "\n" + kept
    return body if len(body) < len(turn.observation) else None


def _partial(mark: ReadMark, novel: dict[int, str], seen_turn: dict, run_note: str) -> str:
    """The new lines, with each run of already-shown lines replaced by a note naming the
    step that showed them."""
    parts: list[str] = []
    n = len(mark.content)
    i = 0
    while i < n:
        ln = mark.line_start + i
        if ln in novel:
            parts.append(novel[ln])
            i += 1
            continue
        start = ln
        while i < n and (mark.line_start + i) not in novel:
            i += 1
        end = mark.line_start + i - 1
        first = min(seen_turn[(mark.path, x)] for x in range(start, end + 1))
        parts.append(run_note.format(start=start, end=end, first=first) + "\n")
    return "".join(parts)
