"""compact -- threshold-triggered summarisation, as Claude Code and Codex do it.

Once the estimated prompt passes `trigger_ratio` (75%) of the budget
B, the oldest `fraction` (60%) of the history not yet summarised is sent to the same
served model, which returns a summary of at most max(2048, B/8) tokens, and the summary
replaces that history for good. The `keep_recent` (6) newest turns are kept, fewer when
they do not fit. At the full window the trigger is never reached and compact runs as
base.

Compaction acts outside the window, on history too old to keep, so it also composes with
ReFold acting inside it (compact + ReFold; `--strategy refold --compact`):

    CompactAgent         compact            cuts raw messages
    ReFoldCompactAgent   compact + ReFold   cuts whole turns and adds them to ReFold's
                                            dropped set, so no stub can point into
                                            summarised history

Both share one trigger and one summariser (CompactionMixin). The history stays
append-only: the summary lives in the render, never in `self.messages`.
"""

from __future__ import annotations

import json

from minisweagent.agents.default import AgentConfig

from refold.agent import RenderingAgent
from refold.registry import StrategySpec, register_strategy
from refold.strategies.refold.agent import ReFoldAgent, ReFoldConfig

SUMMARY_PROMPT = (
    "Summarise the following portion of a software-engineering agent's history into "
    "AT MOST {cap} tokens. If a summary so far is given, produce ONE summary that "
    "replaces it and also covers the new material -- do not restate it and append. "
    "Keep file paths, symbol names, test names, error messages and any conclusion "
    "reached. Prose only, no tool calls.\n\n"
)


class CompactConfig(AgentConfig):
    context_budget: int = 245_760
    """B, usable context in tokens (the served window minus the 16,384-token reply)."""
    trigger_ratio: float = 0.75
    fraction: float = 0.6
    """Share of the not-yet-summarised history handed to one summarisation call."""
    keep_recent: int = 6
    """Newest turns never summarised (halved when they alone do not fit)."""


def _chars(msgs: list[dict]) -> int:
    """Characters of content plus the JSON of tool_calls (content alone is ~55%)."""
    return sum(len(m.get("content") or "") + (len(json.dumps(m["tool_calls"])) if m.get("tool_calls") else 0)
               for m in msgs)


class CompactionMixin:
    """Trigger, summariser and bookkeeping. A subclass provides `render()` and
    `_compact_once(keep_recent) -> bool` (True if it cut something)."""

    _MIN_SUMMARY_CHARS = 200
    _PRIOR_CPT = 3.2  # chars/token until the first real prompt_tokens; below the 3.4-3.8 measured
    _MAX_COMPACTIONS_PER_STEP = 8
    _SUMMARY_FLOOR = 2048  # the 2048 of max(2048, B/8): Qwen3.6 reasons ~1.1k tokens before it writes
    _MIN_SPAN_TURNS = 4

    def _init_compaction(self) -> None:
        self.compactions = self.degenerate = 0
        self.summary_calls: list[dict] = []  # usage of every summarisation request
        self._summary: str | None = None
        self._observed_cpt: float | None = None

    # -- the budget -------------------------------------------------------------
    def _est_tokens(self, msgs: list[dict]) -> int:
        return int(_chars(msgs) / (self._observed_cpt or self._PRIOR_CPT))

    def _trigger(self) -> int:
        return int(self.config.context_budget * self.config.trigger_ratio)

    def _summary_cap(self) -> int:
        """max(2048, B/8). Below the model's reasoning cost the reply comes back
        finish_reason=length with empty content, and an empty summary would drop the span
        instead of compressing it."""
        return max(self._SUMMARY_FLOOR, self.config.context_budget // 8)

    def _room(self, prev: str) -> int:
        """Characters of span body that still leave room for the summary itself."""
        cap = self._summary_cap()
        return max(1000, int((self.config.context_budget - cap) * (self._observed_cpt or self._PRIOR_CPT))
                   - len(SUMMARY_PROMPT) - len(prev))

    # -- the loop ---------------------------------------------------------------
    def before_query(self) -> None:
        """Compact until the render fits. If nothing is left to cut at this keep depth and
        it still does not fit, the protected tail gives way, as a compacting agent
        guarantees the prompt fits rather than failing on its own constant."""
        keep = self.config.keep_recent
        for _ in range(self._MAX_COMPACTIONS_PER_STEP):
            if not self._compact_once(keep):
                if keep <= 1 or self._est_tokens(self.render()) <= self._trigger():
                    return
                keep = max(1, keep // 2)

    def after_query(self, rendered: list[dict], message: dict) -> None:
        """Calibrate chars/token from what the server counted. The first measurement
        replaces the prior; later ones only tighten it (overestimating is the safe side)."""
        super().after_query(rendered, message)
        usage = ((message.get("extra") or {}).get("response") or {}).get("usage") or {}
        pt = usage.get("prompt_tokens") or 0
        if pt > 0:
            r = _chars(rendered) / pt
            self._observed_cpt = r if self._observed_cpt is None else min(self._observed_cpt, r)

    def _summarise_span(self, body: str) -> bool:
        """Summarise `body` into the running summary. False (and nothing advances) if the
        reply is degenerate: a failed summary must not delete the span it was given."""
        prev = f"Summary so far (supersede it, do not repeat it):\n{self._summary}\n\n" if self._summary else ""
        room = self._room(prev)
        if len(body) > room:  # the request itself must fit; keep the newest part
            body = f"...[{len(body) - room} chars of older detail omitted]...\n" + body[-room:]
        cap = self._summary_cap()
        # Instruction LAST: placed first, ~20k characters above the end, the model carried
        # on transcribing the body instead of summarising it.
        resp = self._summarise(prev + body + "\n\n" + SUMMARY_PROMPT.format(cap=cap), cap)
        self.cost += resp["extra"]["cost"]
        got = (resp["content"] or "").strip()
        if len(got) < self._MIN_SUMMARY_CHARS:
            self.degenerate += 1
            return False
        self.compactions += 1
        self._summary = got
        return True

    def _summarise(self, text: str, cap: int) -> dict:
        """One capped summarisation call, NOT routed through the action parser.

        LitellmModel.query parses tool calls and raises FormatError on a prose reply, so
        the summariser goes through upstream's private `_query`. It sends the bash tool
        schema along, as every reported compaction number did.
        """
        msgs = [{"role": "user", "content": text}]
        raw = self.model._query(msgs, max_tokens=cap)
        self._record_usage(raw)
        content = raw.choices[0].message.content or ""
        if len(content.strip()) < self._MIN_SUMMARY_CHARS:  # reasoned past the cap: retry at 2x
            raw = self.model._query(msgs, max_tokens=cap * 2)
            self._record_usage(raw)
            content = raw.choices[0].message.content or content
        try:
            cost = self.model._calculate_cost(raw).get("cost", 0.0)
        except Exception:
            cost = 0.0
        return {"content": content, "extra": {"cost": cost}}

    def _record_usage(self, raw) -> None:
        """A summary call counts toward tokens and cost like any other."""
        usage = getattr(raw, "usage", None)
        details = getattr(usage, "prompt_tokens_details", None)
        self.summary_calls.append({
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "cached_tokens": int(getattr(details, "cached_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        })

    def _summary_note(self) -> dict:
        return {"role": "user", "content": f"[compacted history]\n{self._summary}"}

    def serialize(self, *extra) -> dict:
        record = {"compactions": self.compactions, "degenerate": self.degenerate,
                  "summary_calls": self.summary_calls}
        return super().serialize({"info": {"compact": record}}, *extra)


class CompactAgent(CompactionMixin, RenderingAgent):
    """compact. Cuts raw messages; a summarised span is never shown again."""

    config_class = CompactConfig

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._init_compaction()
        self._summ_upto = 0

    def _compact_once(self, keep_recent: int) -> bool:
        if self._est_tokens(self.render()) <= self._trigger():
            return False
        if sum(m.get("role") == "tool" for m in self.messages) <= keep_recent:
            return False
        # The pool is the not-yet-compacted messages outside the protected tail (about two
        # messages per turn). Message indices, not tool indices, so a run of FormatError
        # retries is compactable too.
        pool = list(range(self._summ_upto, max(0, len(self.messages) - 2 * keep_recent)))
        if not pool:
            return False
        upto = pool[min(int(len(pool) * self.config.fraction), len(pool) - 1)] + 1
        while upto < len(self.messages) and self.messages[upto].get("role") == "tool":
            upto += 1  # never separate a tool call from its reply
        if upto <= self._summ_upto:
            return False
        span = self.messages[self._summ_upto:upto]
        if not self._summarise_span("\n".join(f"[{m.get('role')}] {m.get('content') or ''}" for m in span)):
            return False
        self._summ_upto = upto
        return True

    def render(self) -> list[dict]:
        if self._summary is None:
            return list(self.messages)
        # Keep the system message AND the task: dropping the latter took the tool-call
        # contract away and the model stopped calling tools.
        head, seen_user = [], False
        for m in self.messages[: self._summ_upto]:
            if m.get("role") == "system" or (m.get("role") == "user" and not seen_user):
                seen_user = seen_user or m.get("role") == "user"
                head.append(m)
        return head + [self._summary_note()] + list(self.messages[self._summ_upto:])


class ReFoldCompactConfig(CompactConfig, ReFoldConfig):
    pass


class ReFoldCompactAgent(CompactionMixin, ReFoldAgent):
    """compact + ReFold. The cut is a turn horizon added to ReFold's dropped set."""

    config_class = ReFoldCompactConfig

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._init_compaction()
        self._summ_turn_upto = 0

    def folded_steps(self) -> set[int]:
        return super().folded_steps() | set(range(self._summ_turn_upto))

    def render(self) -> list[dict]:
        msgs = super().render()
        if self._summary is None:
            return msgs
        head = 0  # after the system message and the task
        while head < min(2, len(msgs)) and msgs[head].get("role") in ("system", "user"):
            head += 1
        return msgs[:head] + [self._summary_note()] + msgs[head:]

    def _compact_once(self, keep_recent: int) -> bool:
        if self._est_tokens(self.render()) <= self._trigger():
            return False
        n = len(self.turns)
        if n - self._summ_turn_upto <= keep_recent:
            return False
        pool = list(range(self._summ_turn_upto, n - keep_recent))
        if not pool:
            return False
        upto = pool[min(int(len(pool) * self.config.fraction), len(pool) - 1)] + 1
        # A span of 1-3 turns is so short the summariser reasons past its cap and returns
        # nothing; cut a worthwhile span or none.
        upto = max(upto, min(self._summ_turn_upto + self._MIN_SPAN_TURNS, n - keep_recent))
        if upto <= self._summ_turn_upto:
            return False
        body = "\n".join(f"[turn {t.index}] $ {t.command}\n{t.observation}"
                         for t in self.turns[self._summ_turn_upto:upto])
        if not self._summarise_span(body):
            return False
        self._summ_turn_upto = upto
        return True

    def serialize(self, *extra) -> dict:
        return super().serialize({"info": {"compact": {"turns_summarised": self._summ_turn_upto}}}, *extra)


register_strategy(StrategySpec(
    name="compact",
    agent="refold.strategies.compact.CompactAgent",
    summary="an extra model call once the prompt passes 75% of the budget -> a summary",
))
