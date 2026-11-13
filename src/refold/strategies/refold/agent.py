"""ReFold on mini-swe-agent: the adapter of refold.core.engine to its agent loop.

ReFold itself is ReFoldEngine (refold/core/engine.py). This adapter gives it mini-swe-agent's
history (self.messages, never edited), puts the prompt layer into mini-swe-agent's
templates, answers `restore <step>` instead of running it, and OBSERVEs each turn after
its commands ran. The engine's state reads through as the agent's own attributes
(agent.turns, agent.stale, ...).
"""

from __future__ import annotations

from typing import Any

from refold.agent import RenderingAgent, RenderingConfig
from refold.core import parse
from refold.core.engine import PROMPTS, ReFoldEngine, ReFoldKnobs, inject_prompts

__all__ = ["PROMPTS", "ReFoldAgent", "ReFoldConfig"]


class ReFoldConfig(RenderingConfig, ReFoldKnobs):
    """mini-swe-agent's agent config plus ReFold's knobs (refold.core.engine.ReFoldKnobs)."""


_ENGINE_STATE = ("turns", "tool_msg", "reply_msg", "progress", "has_progress", "stale", "agent_stale", "folds",
                 "removed", "cwd", "labels", "restores", "reruns", "notes")


def _engine_attr(name: str) -> property:
    return property(lambda self: getattr(self.engine, name), lambda self, v: setattr(self.engine, name, v))


class ReFoldAgent(RenderingAgent):
    config_class = ReFoldConfig

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.config.system_template, self.config.instance_template = inject_prompts(
            self.config.system_template, self.config.instance_template, self.config)
        self.engine = ReFoldEngine(self.config, lambda: self.messages, cwd=self._cwd())
        self.engine.folded_steps = self.folded_steps  # a subclass may widen it (compact + ReFold)
        self._marked_boundary = 0  # the boundary the last cache breakpoints were placed for
        self._history_of: dict[int, int] = {}  # id of a rendered message -> its history index

    for _name in _ENGINE_STATE:
        locals()[_name] = _engine_attr(_name)
    del _name

    def _cwd(self) -> str:
        # Every command runs in a fresh shell that starts in the environment's working
        # directory, so that is what a relative path resolves against.
        return getattr(getattr(self.env, "config", None), "cwd", "") or "/"

    def run(self, task: str = "", **kwargs: Any) -> dict:
        self.engine.reset(self._cwd())
        self._marked_boundary, self._history_of = 0, {}
        return super().run(task=task, **kwargs)

    # ------------------------------------------------------------------ OBSERVE
    def execute_actions(self, message: dict) -> list[dict]:
        """Run the reply's commands -- `restore <step>` is answered from the history, not the
        shell -- then OBSERVE the turn. PROGRESS/STALE lines at the start of a command are
        taken out before it runs and read as the reply's labels; the history keeps the reply
        as the model wrote it."""
        labels, actions = _split_labels((message.get("extra") or {}).get("actions") or [])
        restores = [parse.parse_restore_command(_unwrap(a.get("command", ""))) if self.config.reversible else None
                    for a in actions]
        outputs = [self.env.execute(a) if steps is None else {"output": "", "returncode": 0, "exception_info": ""}
                   for a, steps in zip(actions, restores)]
        observations = self.model.format_observation_messages(message, outputs, self.get_template_vars())
        for obs, steps in zip(observations, restores):
            if steps is not None:  # the logged turn in place of the shell frame, not wrapped in a second one
                obs["content"] = self.restore(steps)
        added = self.add_messages(*observations)
        self.observe([_unwrap(a.get("command", "")) for a in actions], observations, labels)
        return added

    def observe(self, commands: list[str], observations: list[dict], labels: str = "") -> None:
        self.engine.observe(commands, observations, labels)

    def restore(self, steps: list[int]) -> str:
        return self.engine.restore(steps)

    # ------------------------------------------------------------------ RENDER
    def boundary(self) -> int:
        return self.engine.boundary()

    def folded_steps(self) -> set[int]:
        return self.engine.base_folded_steps()

    def render(self) -> list[dict]:
        """The engine's render, with each kept reply's `tool_calls` trimmed to the calls whose
        tool message stays. A reply stays while any of its steps does; the call of a folded
        step leaves with its output, so every tool call the model sees has its answer, which
        strict chat APIs require."""
        msgs = self.engine.render()
        kept = {m.get("tool_call_id") for m in msgs if m.get("role") == "tool"}
        out = [{**m, "tool_calls": [tc for tc in m["tool_calls"] if tc["id"] in kept]}
               if m.get("tool_calls") and any(tc["id"] not in kept for tc in m["tool_calls"]) else m
               for m in msgs]
        # by identity, so a subclass that inserts messages (compact + ReFold) keeps the map
        self._history_of = {id(m): h for m, h in zip(out, self.engine.rendered_from)}
        return out

    def cache_anchors(self, rendered: list[dict]) -> list[int]:
        """The task, the newest message, and the last message before the boundary: the one
        this prompt is rendered with and the one the previous prompt was. Nothing before the
        boundary changes until it advances, and an advance rewrites from the old boundary on
        (a step behind it is already final), so the old boundary's mark is where the
        advancing prompt reads the cache back, and the new one is where the next will."""
        b, prev = self.boundary(), self._marked_boundary
        self._marked_boundary = b
        anchors = set(super().cache_anchors(rendered))
        for step in {prev, b}:
            if 0 < step < len(self.turns):  # step 0: the task, already marked
                before = [j for j, m in enumerate(rendered)
                          if self._history_of.get(id(m), len(self.messages)) < self.reply_msg[step]]
                if before:
                    anchors.add(before[-1])
        return sorted(anchors)

    # ------------------------------------------------------------------ record
    def serialize(self, *extra_dicts) -> dict:
        return super().serialize({"info": {"refold": self.engine.record()}}, *extra_dicts)


def _split_labels(actions: list[dict]) -> tuple[str, list[dict]]:
    """The PROGRESS/STALE lines at the start of the actions' commands, and the actions with
    them taken out. A command that is nothing but labels runs as it is, so the model sees
    from the shell's error that it issued no command."""
    found, out = [], []
    for a in actions:
        cmd = a.get("command")
        labels, rest = parse.split_labels(cmd) if isinstance(cmd, str) else ("", "")
        if labels:
            found.append(labels)
            if rest.strip():
                a = {**a, "command": rest}
        out.append(a)
    return "\n".join(found), out


def _unwrap(cmd: Any) -> str:
    if not isinstance(cmd, str):
        return ""
    if len(cmd) >= 2 and cmd[0] == cmd[-1] and cmd[0] in ("'", '"'):
        return cmd[1:-1]
    return cmd
