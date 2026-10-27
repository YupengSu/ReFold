"""A per-call log of a running cell, to watch a run as it happens and to replay it afterwards.

`refold run --live` sets REFOLD_LIVE_DIR to OUT/live, and the runner then wraps the agent
class with `with_live_log`: every task appends one JSON line per model call to
OUT/live/<instance_id>.jsonl. The trajectory lands only when a task ends; this log lands
as each call returns, so a dashboard can show every session's context while it grows and
folds. It only reads what the agent already holds and never changes what the model is
sent (tests/test_live.py checks the prompts are identical with and without it).

    {"ev": "start", "task": ID, "t": T}
    {"ev": "call", "task": ID, "step": N, "t0": T, "t1": T, "prompt": p, "cached": c, "completion": g,
     "sent_chars": ..., "hist_chars": ..., "turns": ..., "folded": ..., "compressed": ...}
    {"ev": "end", "task": ID, "t": T, "exit_status": "Submitted", "steps": N}

step is the 1-based model call; t0 is when the rendered prompt went to the model and t1
when the reply came back (wall clock, seconds). prompt, cached and completion are the
server's usage counts. cached is null when the response has no prompt_tokens_details:
vLLM leaves them out when nothing was served from the prefix cache (as on a task's first
call) and when it runs without --enable-prompt-tokens-details. sent_chars is the length of the prompt
as rendered, hist_chars that of the full history h_t it was rendered from, so the two are
equal for `base`, and prompt * hist_chars / sent_chars estimates what the full history
would have cost. turns is the number of tool turns so far; folded and compressed count the
steps ReFold removed from this prompt (0 for every other arm).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from minisweagent.exceptions import FormatError

LIVE_ENV = "REFOLD_LIVE_DIR"


def _chars(messages: list[dict]) -> int:
    n = 0
    for m in messages:
        c = m.get("content")
        n += len(c) if isinstance(c, str) else len(json.dumps(c, ensure_ascii=False)) if c else 0
        for tc in m.get("tool_calls") or []:
            n += len(((tc or {}).get("function") or {}).get("arguments") or "")
    return n


def _usage(message: dict) -> tuple[int | None, int | None, int | None]:
    r = (message.get("extra") or {}).get("response")
    u = r.get("usage") if isinstance(r, dict) else None
    if not isinstance(u, dict):
        return None, None, None
    det = u.get("prompt_tokens_details")
    cached = det.get("cached_tokens") if isinstance(det, dict) else None
    return u.get("prompt_tokens"), cached, u.get("completion_tokens")


def with_live_log(agent_class: type, live_dir: str | os.PathLike) -> type:
    """The agent class, appending one line per model call to LIVE_DIR/<instance_id>.jsonl.
    The instance id is the `instance_id` attribute the runner's progress wrapper sets."""
    root = Path(live_dir)

    class Live(agent_class):
        _live_lock = threading.Lock()

        def _live(self, rec: dict) -> None:
            task = getattr(self, "instance_id", "") or f"task-{id(self):x}"
            line = json.dumps({**rec, "task": task}) + "\n"
            with self._live_lock:
                root.mkdir(parents=True, exist_ok=True)
                with (root / f"{task}.jsonl").open("a") as f:
                    f.write(line)

        def run(self, task: str = "", **kwargs):
            self._live({"ev": "start", "t": time.time()})
            try:
                return super().run(task=task, **kwargs)
            finally:
                last = (self.messages[-1].get("extra") or {}) if self.messages else {}
                self._live({"ev": "end", "t": time.time(), "exit_status": last.get("exit_status", ""),
                            "steps": self.n_calls})

        def render(self) -> list[dict]:
            out = super().render()
            removed = getattr(self, "removed", None) or {}
            self._live_call = {"t0": time.time(), "sent_chars": _chars(out), "hist_chars": _chars(self.messages),
                               "turns": len(getattr(self, "turns", None) or []),
                               "folded": sum(1 for v in removed.values() if v == "folded"),
                               "compressed": sum(1 for v in removed.values() if v == "compressed")}
            return out

        def _live_reply(self, message: dict) -> None:
            pre = getattr(self, "_live_call", None)
            if pre is None:  # the strategy queried without rendering
                return
            p, c, g = _usage(message)
            self._live({"ev": "call", "step": self.n_calls, **pre, "t1": time.time(),
                        "prompt": p, "cached": c, "completion": g})
            self._live_call = None

        def query(self) -> dict:
            try:
                message = super().query()
            except FormatError as e:  # billed, then rejected: the call still happened
                if e.messages:
                    self._live_reply(e.messages[0])
                raise
            self._live_reply(message)
            return message

    Live.__name__ = Live.__qualname__ = agent_class.__name__
    Live.__module__ = agent_class.__module__  # so the trajectory records the real class
    return Live
