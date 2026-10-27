"""ReFold on mini-swe-agent: the adapter of refold.core.engine to its agent loop.

ReFold itself is ReFoldEngine (refold/core/engine.py). This adapter gives it mini-swe-agent's
history (self.messages, never edited), puts the prompt layer into mini-swe-agent's
templates, answers `restore <step>` instead of running it, and OBSERVEs each turn after
its commands ran. The engine's state reads through as the agent's own attributes
(agent.turns, agent.stale, ...).
"""

from __future__ import annotations

from typing import Any

from minisweagent.agents.default import AgentConfig

from refold.agent import RenderingAgent
from refold.core import parse
from refold.core.engine import PROMPTS, ReFoldEngine, ReFoldKnobs, inject_prompts

__all__ = ["PROMPTS", "ReFoldAgent", "ReFoldConfig"]


class ReFoldConfig(AgentConfig, ReFoldKnobs):
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

    for _name in _ENGINE_STATE:
        locals()[_name] = _engine_attr(_name)
    del _name

    def _cwd(self) -> str:
        # Every command runs in a fresh shell that starts in the environment's working
        # directory, so that is what a relative path resolves against.
        return getattr(getattr(self.env, "config", None), "cwd", "") or "/"

    def run(self, task: str = "", **kwargs: Any) -> dict:
        self.engine.reset(self._cwd())
        return super().run(task=task, **kwargs)

    # ------------------------------------------------------------------ OBSERVE
    def execute_actions(self, message: dict) -> list[dict]:
        """Run the reply's commands -- `restore <step>` is answered from the history, not the
        shell -- then OBSERVE the turn."""
        actions = (message.get("extra") or {}).get("actions") or []
        restores = [parse.parse_restore_command(_unwrap(a.get("command", ""))) if self.config.reversible else None
                    for a in actions]
        outputs = [self.env.execute(a) if steps is None else {"output": "", "returncode": 0, "exception_info": ""}
                   for a, steps in zip(actions, restores)]
        observations = self.model.format_observation_messages(message, outputs, self.get_template_vars())
        for obs, steps in zip(observations, restores):
            if steps is not None:  # the logged turn in place of the shell frame, not wrapped in a second one
                obs["content"] = self.restore(steps)
        added = self.add_messages(*observations)
        self.observe([_unwrap(a.get("command", "")) for a in actions], observations)
        return added

    def observe(self, commands: list[str], observations: list[dict]) -> None:
        self.engine.observe(commands, observations)

    def restore(self, steps: list[int]) -> str:
        return self.engine.restore(steps)

    # ------------------------------------------------------------------ RENDER
    def boundary(self) -> int:
        return self.engine.boundary()

    def folded_steps(self) -> set[int]:
        return self.engine.base_folded_steps()

    def render(self) -> list[dict]:
        return self.engine.render()

    # ------------------------------------------------------------------ record
    def serialize(self, *extra_dicts) -> dict:
        return super().serialize({"info": {"refold": self.engine.record()}}, *extra_dicts)


def _unwrap(cmd: Any) -> str:
    if not isinstance(cmd, str):
        return ""
    if len(cmd) >= 2 and cmd[0] == cmd[-1] and cmd[0] in ("'", '"'):
        return cmd[1:-1]
    return cmd
