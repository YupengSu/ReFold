"""A scripted model and a real local shell: agent loops with no network and no GPU."""

from __future__ import annotations

import json
import os

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel


class ScriptedModel(LitellmModel):
    """Replays `script`, a list of (reply content, command[, extra tool args]), and keeps
    every prompt it was sent; a list of commands issues them as parallel tool calls of one
    reply. A real LitellmModel underneath, so observations are formatted by upstream's own
    templates."""

    def __init__(self, script):
        super().__init__(model_name="scripted/test", cost_tracking="ignore_errors")
        self.script = list(script)
        self.prompts: list[list[dict]] = []
        self.n_summaries = 0

    def _query(self, messages, **kw):  # compaction's summariser
        from types import SimpleNamespace

        self.n_summaries += 1
        text = "SUMMARY: the agent read the modules and listed a directory. " * 6
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=40))

    def _calculate_cost(self, response):
        return {"cost": 0.0}

    def query(self, messages, **kw):
        self.prompts.append(json.loads(json.dumps(messages)))
        step = self.script[min(len(self.prompts) - 1, len(self.script) - 1)]
        content, command, extra_args = (*step, {})[:3]
        commands = command if isinstance(command, list) else [command]
        cids = [f"call_{len(self.prompts)}" + (f"_{j}" if len(commands) > 1 else "") for j in range(len(commands))]
        return {"role": "assistant", "content": content,
                "tool_calls": [{"id": cid, "type": "function",
                                "function": {"name": "bash", "arguments": json.dumps({"command": c, **extra_args})}}
                               for cid, c in zip(cids, commands)],
                "extra": {"actions": [{"command": c, "tool_call_id": cid} for cid, c in zip(cids, commands)],
                          "cost": 0.0}}


TEMPLATES = {
    "system_template": "You are an agent.\n",
    "instance_template": "Task: {{task}}\n1. Include a THOUGHT section explaining your reasoning and what "
                         "you're trying to accomplish\n2. Provide a bash tool call",
}


def assert_every_tool_call_answered(prompt: list[dict]) -> None:
    """Every tool call of every reply has its tool message, and no tool message lacks its
    call (strict chat APIs such as Anthropic's and DeepSeek's answer HTTP 400 otherwise)."""
    calls = {tc["id"] for m in prompt for tc in m.get("tool_calls") or []}
    answered = {m["tool_call_id"] for m in prompt if m.get("role") == "tool"}
    assert calls <= answered, f"tool calls without their tool message: {sorted(calls - answered)}"
    assert answered <= calls, f"tool messages without their call: {sorted(answered - calls)}"


def make_agent(agent_cls, script, cwd, **cfg):
    return agent_cls(ScriptedModel(script), LocalEnvironment(cwd=str(cwd)),
                     **{**TEMPLATES, "step_limit": len(script), "cost_limit": 0, **cfg})
