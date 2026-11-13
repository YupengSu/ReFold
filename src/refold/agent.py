"""The agent loop every strategy shares, and the one method a strategy overrides.

A strategy decides what the model sees and nothing else. It subclasses `RenderingAgent`
and overrides `render()`, which returns the messages for this call. `self.messages` is
the append-only interaction history h_t in every arm: `render()` returns a
rewritten copy and never edits it, so the full history is always there to render from.

Two optional hooks bracket the call: `before_query()` runs before the call is counted
(compaction decides here), `after_query()` once the reply is back. The time
spent in `before_query()` and `render()` is the strategy's own per-step overhead; it is recorded on each reply as `extra.overhead_s`.

With `cache_breakpoints` on, the prompt also carries prompt-cache breakpoints on the
messages `cache_anchors()` names, for APIs that cache only at marked points (Anthropic).
"""

from __future__ import annotations

import time

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import LimitsExceeded

MAX_CACHE_BREAKPOINTS = 4  # Anthropic's limit per request


class RenderingConfig(AgentConfig):
    cache_breakpoints: bool = False
    """Mark prompt-cache breakpoints (`cache_control`) on the messages `cache_anchors()`
    names. Anthropic caches only at marked points; vLLM caches every prefix and needs none."""


def with_cache_control(msgs: list[dict], anchors: list[int]) -> list[dict]:
    """A copy of `msgs` with a cache breakpoint on each message in `anchors`, in the form
    litellm carries to Anthropic: on a tool message itself (it lands on the tool_result
    block), otherwise on the last text block. The messages themselves are not edited."""
    out = list(msgs)
    for i in anchors:
        m, mark = dict(out[i]), {"type": "ephemeral"}
        content = m.get("content")
        if m.get("role") == "tool":
            m["cache_control"] = mark
        elif isinstance(content, str) and content:
            m["content"] = [{"type": "text", "text": content, "cache_control": mark}]
        elif isinstance(content, list) and content:
            m["content"] = [*content[:-1], {**content[-1], "cache_control": mark}]
        else:
            continue
        out[i] = m
    return out


class RenderingAgent(DefaultAgent):
    """mini-swe-agent's loop with the prompt produced by `render()`.

    With the default `render()` this is upstream's append-only loop: the `base` strategy.
    A subclass names its knobs by setting `config_class`.
    """

    config_class: type[AgentConfig] = RenderingConfig

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("config_class", type(self).config_class)
        super().__init__(*args, **kwargs)

    def render(self) -> list[dict]:
        """The messages to send this call. Must not mutate `self.messages`."""
        return list(self.messages)

    def before_query(self) -> None:
        pass

    def after_query(self, rendered: list[dict], message: dict) -> None:
        pass

    def cache_anchors(self, rendered: list[dict]) -> list[int]:
        """The messages of `rendered` that carry a cache breakpoint (at most four). An
        append-only prompt needs the newest one only; the task is marked too, so a prompt
        rewritten after it still reads the task back."""
        task = next((i for i, m in enumerate(rendered) if m.get("role") == "user"), None)
        return sorted({i for i in (task, len(rendered) - 1) if i is not None and i >= 0})

    def query(self) -> dict:
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded({"role": "exit", "content": "LimitsExceeded",
                                  "extra": {"exit_status": "LimitsExceeded", "submission": ""}})
        start = time.perf_counter()
        self.before_query()
        self.n_calls += 1
        rendered = self.render()
        sent = rendered
        if getattr(self.config, "cache_breakpoints", False):
            sent = with_cache_control(rendered, self.cache_anchors(rendered)[-MAX_CACHE_BREAKPOINTS:])
        overhead = time.perf_counter() - start
        message = self.model.query(sent)
        message.setdefault("extra", {})["overhead_s"] = overhead
        self.cost += message["extra"].get("cost", 0.0)
        self.after_query(rendered, message)
        self.add_messages(message)
        return message
