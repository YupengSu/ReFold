"""ReFold: deduplication and folding at the rendering layer, for any agent loop.

    h_t = p_sys ⊕ p_task ⊕ z_1 ⊕ ... ⊕ z_t,  z_i = a_i ⊕ o_i    the history, never edited
    d_i  = o_i with repeated spans stubbed                     dedup.py
    a_t  = PROGRESS_t ⊕ <STALE_t> ⊕ ã_t                         prompts.json: protocol
    S_t  = S_{t-1} ∪ (STALE_t ∩ {t-k, ..., t-1})               observe()
    f_i  = PROGRESS_i ⊕ e_i                                     fold.py
    b_t  = max(0, k ⌊(t - k) / k⌋)                              boundary()
    c_t  = p_sys ⊕ p_task ⊕ ⨁_i  f_i        if i ≤ b_t and i ∈ S_t
                                 a_i ⊕ o_i  if i > b_t
                                 a_i ⊕ d_i  otherwise           render()

Each step runs OBSERVE (read the reply, update S_t) and RENDER (build c_t from h_t). The
history is never edited, so every removal is reversible: `restore <step>` is answered from
the history and never reaches the shell. Re-running the original command gives the current
state instead, and that output renders in full.

The engine imports no agent framework. It reads the history as a list of chat messages
({"role", "content"}: system, the task, then each reply followed by the tool messages of
the calls it issued) from the `messages` callable it is given, and returns the prompt in
the same form. An adapter feeds it: refold.strategies.refold.agent for mini-swe-agent.

Indexing. A turn is one tool call and its output. Steps are numbered from 0, as the
`[step N]` labels and STALE show them to the model: step s renders in full iff s >= b_t,
and the reply that issued step t-1 may name steps t-1-k .. t-2.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field

from refold.core import dedup, fold, parse
from refold.core.dedup import ReadMark, Rendered, Turn

PROMPTS: dict[str, str] = json.loads((Path(__file__).parent / "prompts.json").read_text())

# The instance templates open every reply with this sentence; the PROGRESS/STALE protocol
# takes its place, so the model is asked for one opening, not two.
THOUGHT_ANCHOR = "Include a THOUGHT section explaining your reasoning and what you're trying to accomplish"
_RESTORE_MAX = 10  # steps per `restore` call
_RESTORE_CLAUSE = re.compile(r" `restore \{\w+\}` shows (?:it|them|all of it|\{them\}) again\.")


class ReFoldKnobs(BaseModel):
    chunk_size: int = Field(3, ge=1)
    """k: the boundary advances k steps at a time, and STALE may name only the last k steps.
    The method's one knob; everything below switches part of it off or replaces it."""

    # --- ablation switches: the defaults are ReFold ---------------------------------
    dedup: bool = True
    """False: every observation renders as received."""
    fold: bool = True
    """False: STALE is still asked for and parsed, but nothing is folded."""
    chunked: bool = True
    """False: the boundary moves every step, b_t = t - k."""
    reversible: bool = True
    """False: `restore` is neither offered nor answered, notes do not say how to see the
    content again, a repeat of something removed is compressed again, and folded steps
    stay dedup sources."""
    stale_window: int | None = Field(None, ge=0)
    """How many steps back STALE may name. None: k; 0: any earlier step. The prompt states it."""
    stale_source: Literal["agent", "random", "all"] = "agent"
    """Which steps enter S_t. `agent`: the ones the reply names. `random`: as many new steps
    as the reply names, drawn uniformly from the window. `all`: every step in the window.
    The prompt asks for STALE either way."""
    seed: int = 0
    """Seed of stale_source=random; every task and step draws from its own stream."""


def prompt_layer(k: int, reversible: bool = True, window: int | None = None) -> tuple[str, str]:
    """(the disclosure appended to the system prompt, the protocol that replaces the
    THOUGHT item) for chunk size k. The protocol states the STALE window: k steps, or
    `window` when it is set (0: any earlier step)."""
    w = k if window is None else window
    window_text = (PROMPTS["protocol_window_any"] if w == 0 else
                   PROMPTS["protocol_window"].replace("{k}", str(w)).replace("{plural}", "" if w == 1 else "s"))
    protocol = (PROMPTS["protocol"].replace("{restore}", PROMPTS["protocol_restore"] if reversible else "")
                .replace("{window}", window_text))
    return PROMPTS["disclosure" if reversible else "disclosure_irreversible"], protocol


def inject_prompts(system: str, task: str, knobs) -> tuple[str, str]:
    """The prompt layer on a framework's own texts: the disclosure appended to the system
    prompt, and the PROGRESS/STALE protocol in place of the THOUGHT item (or appended)."""
    disclosure, protocol = prompt_layer(knobs.chunk_size, knobs.reversible, knobs.stale_window)
    if protocol.strip() not in task:
        task = task.replace(THOUGHT_ANCHOR, protocol.strip(), 1) if THOUGHT_ANCHOR in task else task + "\n" + protocol
    if "[context-manager]" not in system:
        system += disclosure
    return system, task


class ReFoldEngine:
    def __init__(self, knobs, messages: Callable[[], list[dict]], cwd: str = "/"):
        """`knobs`: any object with the ReFoldKnobs fields; `messages`: returns the history
        h_t; `cwd`: the directory a relative path in a command resolves against (every
        command runs in a fresh shell that starts there, so a `cd` does not carry over)."""
        self.config = knobs
        self._messages = messages
        rev = knobs.reversible
        self.notes = PROMPTS if rev else {n: _RESTORE_CLAUSE.sub("", t) if n.startswith("note_") else t
                                          for n, t in PROMPTS.items()}
        self.folded_steps: Callable[[], set[int]] = self.base_folded_steps
        """The steps the prompt drops; an adapter may widen it (compact + ReFold)."""
        self.reset(cwd)

    @property
    def messages(self) -> list[dict]:
        return self._messages()

    def reset(self, cwd: str = "/") -> None:
        self.turns: list[Turn] = []
        self.tool_msg: list[int] = []  # step -> index of its tool message
        self.reply_msg: list[int] = []  # step -> index of the reply that issued it
        self.progress: list[str] = []  # step -> PROGRESS of that reply ("" if it had none)
        self.has_progress: list[bool] = []  # step -> did that reply carry a PROGRESS line
        self.stale: set[int] = set()  # S_t
        self.agent_stale: set[int] = set()  # what the replies named, when stale_source is not `agent`
        self.folds: list[fold.Fold] = []
        self.removed: dict[int, str] = {}  # step -> "folded" | "compressed" (deduplicated) in the last prompt
        self.cwd: str = cwd
        self.labels = {"replies": 0, "progress_missing": 0, "stale_out_of_window": 0}
        self.restores = dict.fromkeys(("calls", "folded", "compressed", "shown", "invalid"), 0)
        self.reruns = dict.fromkeys(("folded", "folded_identical", "compressed", "compressed_identical"), 0)

    # ------------------------------------------------------------------ OBSERVE
    def observe(self, commands: list[str], observations: list[dict]) -> None:
        """OBSERVE: append the new turns to the record, read PROGRESS_t and
        STALE_t from the reply that issued them, and update S_t. The reply and then the
        `observations` are the last messages of the history."""
        reply_idx = len(self.messages) - len(observations) - 1
        for offset, (cmd, obs) in enumerate(zip(commands, observations)):
            self._count_rerun(cmd, obs.get("content", "") or "")
            self.turns.append(self._turn(len(self.turns), cmd, obs.get("content", "") or ""))
            self.tool_msg.append(reply_idx + 1 + offset)
            self.reply_msg.append(reply_idx)
        t = len(self.turns)
        if t == 0:
            return
        content = self.messages[self.reply_msg[-1]].get("content", "") or ""
        progress = parse.parse_progress(content)
        self.labels["replies"] += 1
        self.labels["progress_missing"] += progress is None
        while len(self.progress) < t:  # every turn of this reply shares its PROGRESS
            self.progress.append(progress or "")
            self.has_progress.append(progress is not None)
        w = self.config.chunk_size if self.config.stale_window is None else self.config.stale_window
        lo, hi = (0 if w == 0 else t - 1 - w), t - 2  # the last w steps before this one
        named = parse.parse_stale(content)
        self.labels["stale_out_of_window"] += sum(not lo <= s <= hi for s in named)
        if self.config.stale_source == "agent":
            self.stale.update(s for s in named if lo <= s <= hi)
        else:
            new = {s for s in named if lo <= s <= hi and s not in self.stale}
            self.agent_stale |= new
            window = [s for s in range(max(lo, 0), hi + 1) if s not in self.stale]
            if self.config.stale_source == "random":
                task = hashlib.sha1((self.messages[1].get("content") or "").encode()).hexdigest()
                window = random.Random(f"{self.config.seed}:{task}:{t}").sample(window, min(len(new), len(window)))
            self.stale.update(window)
        self.folds = fold.folds(self.turns, self.stale, self.progress, self.boundary(), self.notes)

    def _turn(self, index: int, cmd: str, obs: str) -> Turn:
        """Which dedup key a turn gets: a file read, a search, or a test run."""
        read = self._read_mark(cmd, obs)
        pytest = bool(cmd) and parse.is_pytest_command(cmd)
        testrunner, search = False, None
        if cmd and read is None and not pytest:
            testrunner = parse.is_testrunner_command(cmd)
            canonical = parse.parse_search_command(cmd)
            if canonical is not None:
                search = (canonical, hashlib.sha1(obs.encode("utf-8")).hexdigest())
        return Turn(index=index, command=cmd, observation=obs, read=read, search=search, pytest=pytest,
                    testrunner=testrunner)

    def _read_mark(self, cmd: str, obs: str) -> ReadMark | None:
        spec = parse.parse_read_command(cmd) if cmd else None
        if spec is None or spec.tail_n is not None:
            return None
        base = os.path.join(self.cwd, spec.cd) if spec.cd else self.cwd  # `cd DIR && cat f`
        path = os.path.normpath(os.path.join(base, spec.path)) if spec.path else ""
        if not os.path.isabs(path):
            return None
        lines = parse.observation_lines(obs, used_cat_n=parse.used_cat_n(cmd))
        if lines and lines[-1] == "\n":
            lines = lines[:-1]  # the empty line the <output>\n...\n</output> split leaves
        if not lines:
            return None
        return ReadMark(path=path, line_start=spec.line_start, line_end=spec.line_start + len(lines) - 1,
                        content=tuple(lines))

    def _count_rerun(self, cmd: str, obs: str) -> None:
        """A re-issue of the command of a step the last prompt showed folded or omitted:
        recovery by re-running. Its latest earlier copy decides; if that one
        was shown in full, this is a plain repeat."""
        if not cmd.strip():
            return
        prev = next((t for t in reversed(self.turns) if t.command.strip() == cmd.strip()), None)
        how = self.removed.get(prev.index) if prev is not None else None
        if how is not None:
            self.reruns[how] += 1
            self.reruns[f"{how}_identical"] += prev.observation == obs

    # ------------------------------------------------------------------ restore
    def restore(self, steps: list[int]) -> str:
        """`restore <step>`: the original turn z_i = a_i ⊕ o_i of each named
        step -- its agent message, command and tool observation -- exactly as first sent,
        from the history, without running anything. Judged against the last prompt sent,
        which the reply was written from."""
        self.restores["calls"] += 1
        blocks = []
        for s in steps[:_RESTORE_MAX]:
            how = self.removed.get(s) if 0 <= s < len(self.turns) else "invalid"
            self.restores[how or "shown"] += 1
            if how == "invalid":
                blocks.append(PROMPTS["restore_invalid"].format(step=s, last=len(self.turns) - 1))
            elif how is None:
                blocks.append(PROMPTS["restore_shown"].format(step=s))
            else:
                reply = (self.messages[self.reply_msg[s]].get("content") or "").strip()
                blocks.append("\n".join([PROMPTS["restore_head"].format(step=s), *([reply] if reply else []),
                                         f"$ {self.turns[s].command}",
                                         self.messages[self.tool_msg[s]].get("content") or ""]))
        if len(steps) > _RESTORE_MAX:
            blocks.append(PROMPTS["restore_cap"].format(max=_RESTORE_MAX))
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------ RENDER
    def boundary(self) -> int:
        """b_t, with t the number of steps so far. Steps s >= b_t render exactly as
        received. Unchunked (ablation), the boundary moves every step: b_t = t - k."""
        k, t = self.config.chunk_size, len(self.turns)
        return max(0, k * ((t - k) // k)) if self.config.chunked else max(0, t - k)

    def base_folded_steps(self) -> set[int]:
        """Steps whose reply and output leave the prompt: those in S_t behind the boundary
        (i ≤ b_t and i ∈ S_t). They are no dedup source either, so no stub can
        point at a step the model cannot see."""
        if not self.config.fold:
            return set()
        return {i for f in self.folds for i in range(f.start, f.end + 1)}

    def render(self) -> list[dict]:
        """RENDER: c_t = R(h_t), rebuilt from the history every call.
        Between boundary advances it is the previous prompt plus the new messages."""
        messages = self.messages
        if not self.turns:
            return [dict(m) for m in messages]
        folded = self.folded_steps()
        if self.config.dedup:
            rendered = dedup.render(self.turns, folded, self.boundary(), self.notes, reversible=self.config.reversible)
        else:
            rendered = [Rendered(t.index) for t in self.turns]
        body = {r.index: r.body for r in rendered}
        self.removed = {**{r.index: "compressed" for r in rendered if r.body is not None},
                        **dict.fromkeys(folded, "folded")}

        step_of = {m: s for s, m in enumerate(self.tool_msg)}
        skip_msgs = {self.tool_msg[s] for s in folded}
        steps_of_reply: dict[int, set[int]] = {}
        for s, r in enumerate(self.reply_msg):
            steps_of_reply.setdefault(r, set()).add(s)
        skip_msgs |= {r for r, steps in steps_of_reply.items() if steps <= folded}  # the reply goes too
        notes = self._fold_notes(folded)
        remind_at, remind = self._reminder()

        out: list[dict] = []
        for i, msg in enumerate(messages):
            if i in skip_msgs:
                continue
            orig = msg.get("content", "") or ""
            content = orig
            if i in step_of:
                s = step_of[i]
                if body.get(s) is not None:
                    content = body[s]
                if isinstance(content, str):
                    content = f"[step {s}]\n{content}"
            if i in notes:
                content = f"{notes[i]}\n{content}"
            if i == remind_at and isinstance(content, str):
                content = f"{content}\n\n{remind}"
            out.append({**msg, "content": content} if content is not orig else dict(msg))
        return out

    def _fold_notes(self, folded: set[int]) -> dict[int, str]:
        """msg index -> the fold notes f_i to prepend: each note goes on the tool message of
        the first surviving step after its run."""
        notes: dict[int, str] = {}
        if not self.config.fold:
            return notes
        for f in self.folds:
            after = f.end + 1
            while after < len(self.tool_msg) and after in folded:
                after += 1
            if after >= len(self.tool_msg):
                continue
            idx = self.tool_msg[after]
            notes[idx] = f"{notes[idx]}\n{f.note}" if idx in notes else f.note
        return notes

    def _reminder(self) -> tuple[int, str]:
        """(msg index, text) of the format reminder appended to the newest observation when
        the last reply skipped its PROGRESS line; firmer after repeated misses."""
        if not self.has_progress or self.has_progress[-1]:
            return -1, ""
        misses = len(self.has_progress) - 1 - max((i for i, ok in enumerate(self.has_progress) if ok), default=-1)
        text = PROMPTS["remind_gentle"] if misses <= 1 else PROMPTS["remind_harsh"].format(misses=misses)
        return self.tool_msg[-1], text

    # ------------------------------------------------------------------ record
    def record(self) -> dict:
        """What ReFold did on this task (a trajectory's info.refold)."""
        return {"steps": len(self.turns), "stale": sorted(self.stale), **self.labels,
                **({"agent_stale": sorted(self.agent_stale)} if self.config.stale_source != "agent" else {}),
                "folds": [{"start": f.start, "end": f.end, "note": f.note} for f in self.folds],
                "restores": dict(self.restores), "refetches": dict(self.reruns)}
