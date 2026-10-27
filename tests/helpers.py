"""A scripted model and a real local shell: agent loops with no network and no GPU."""

from __future__ import annotations

import json
import os

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel


class ScriptedModel(LitellmModel):
    """Replays `script`, a list of (reply content, command[, extra tool args]), and keeps
    every prompt it was sent. A real LitellmModel underneath, so observations are
    formatted by upstream's own templates."""

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
        cid = f"call_{len(self.prompts)}"
        return {"role": "assistant", "content": content,
                "tool_calls": [{"id": cid, "type": "function",
                                "function": {"name": "bash", "arguments": json.dumps({"command": command, **extra_args})}}],
                "extra": {"actions": [{"command": command, "tool_call_id": cid}], "cost": 0.0}}


TEMPLATES = {
    "system_template": "You are an agent.\n",
    "instance_template": "Task: {{task}}\n1. Include a THOUGHT section explaining your reasoning and what "
                         "you're trying to accomplish\n2. Provide a bash tool call",
}


def make_agent(agent_cls, script, cwd, **cfg):
    return agent_cls(ScriptedModel(script), LocalEnvironment(cwd=str(cwd)),
                     **{**TEMPLATES, "step_limit": len(script), "cost_limit": 0, **cfg})
