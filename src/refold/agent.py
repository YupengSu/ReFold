"""The agent loop every strategy shares, and the one method a strategy overrides.

A strategy decides what the model sees and nothing else. It subclasses `RenderingAgent`
and overrides `render()`, which returns the messages for this call. `self.messages` is
the append-only interaction history h_t in every arm: `render()` returns a
rewritten copy and never edits it, so the full history is always there to render from.

Two optional hooks bracket the call: `before_query()` runs before the call is counted
(compaction decides here), `after_query()` once the reply is back. The time
spent in `before_query()` and `render()` is the strategy's own per-step overhead; it is recorded on each reply as `extra.overhead_s`.
"""

from __future__ import annotations

import time

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import LimitsExceeded


class RenderingAgent(DefaultAgent):
    """mini-swe-agent's loop with the prompt produced by `render()`.

    With the default `render()` this is upstream's append-only loop: the `base` strategy.
    A subclass names its knobs by setting `config_class`.
    """

    config_class: type[AgentConfig] = AgentConfig

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

    def query(self) -> dict:
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded({"role": "exit", "content": "LimitsExceeded",
                                  "extra": {"exit_status": "LimitsExceeded", "submission": ""}})
        start = time.perf_counter()
        self.before_query()
        self.n_calls += 1
        rendered = self.render()
        overhead = time.perf_counter() - start
        message = self.model.query(rendered)
        message.setdefault("extra", {})["overhead_s"] = overhead
        self.cost += message["extra"].get("cost", 0.0)
        self.after_query(rendered, message)
        self.add_messages(message)
        return message
