"""Folding: each run of finished steps behind the boundary becomes one note.

For a step i in S_t with i < b_t, the reply and its tool output leave the prompt and

    f_i = PROGRESS_i ⊕ e_i

takes their place, where e_i collects key details from the removed output: the function
signatures of a file it read, the first hits of a search or listing. Notes of adjacent
steps share one line:

    [context-manager] steps <i>-<j> folded: <PROGRESS_i> · ... · <PROGRESS_j>. `restore <i>-<j>` shows them again.
      <file>: <extracted function signatures>
      $ <search command> -> <retained search hits>

The details are mechanical, so a fold costs no model call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from refold.core.dedup import Turn, skeleton_of

_LISTING_COMMANDS = (["grep"], ["rg"], ["find"], ["ls"], ["tree"])
_OUTPUT_RE = re.compile(r"<output>\n?(.*?)</output>", re.DOTALL)


@dataclass(frozen=True)
class Fold:
    start: int
    end: int
    note: str


def folds(turns: list[Turn], stale: set[int], progress: list[str], boundary: int,
          notes: dict[str, str]) -> list[Fold]:
    """The maximal runs of steps in `stale`, cut at the boundary. Rebuilt from S_t every
    step, so a run that grows as the boundary advances gets one note, not two."""
    n, horizon = len(turns), boundary - 1
    out, i = [], 0
    while i < n:
        if i not in stale:
            i += 1
            continue
        j = i
        while j + 1 < n and j + 1 in stale:
            j += 1
        end = min(j, horizon)
        if i <= end:
            ids = f"{i}" if i == end else f"{i}-{end}"
            note = notes["note_fold"].format(
                steps=f"step{'s' if i != end else ''} {ids}", ids=ids, them="them" if i != end else "it",
                progress=" · ".join(s for s in progress[i:end + 1] if s).rstrip(". ") or notes["no_progress"])
            detail = details(turns[i:end + 1])
            out.append(Fold(i, end, f"{note}\n{detail}" if detail else note))
        i = j + 1
    return out


def details(turns: list[Turn]) -> str:
    """e_i for a run of folded steps: signatures of the files read, the first hits of
    searches and listings; indented, at most 12 lines."""
    lines: list[str] = []
    for t in turns:
        if not t.observation.strip():
            continue
        if t.read is not None and t.read.content:
            skel = skeleton_of(t.read.content, t.read.path)
            if skel:
                lines.append(f"  {t.read.path.rsplit('/', 1)[-1]}: {skel.replace(chr(10), '; ')[:200]}")
        elif t.search is not None or t.command.strip().split()[:1] in _LISTING_COMMANDS:
            m = _OUTPUT_RE.search(t.observation)
            hits = [ln for ln in (m.group(1) if m else t.observation).splitlines() if ln.strip()][:4]
            if hits:
                lines.append(f"  $ {t.command.strip()[:40]} -> " + " | ".join(h[:70] for h in hits))
    return "\n".join(lines[:12])
