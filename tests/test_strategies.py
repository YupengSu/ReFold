"""Every arm runs a real agent loop, sends a well-formed prompt, and removes something.

Each agent class is imported by the dotted path the runner uses and stepped with a
scripted model over a real local shell, so an error anywhere on the render path is a red
test here rather than a crash in the first real run.
"""

from __future__ import annotations

import importlib

import pytest
from helpers import make_agent

from refold.cli.run import build_config
from refold.registry import available, get_strategy

ARMS = [(s, False) for s in available()] + [(s, True) for s in available() if get_strategy(s).compact_agent]

# Budgets small enough that the threshold-driven arms fire within a short script.
OVERRIDES = {"compact": {"context_budget": 3000}}
COMPACT_OVERRIDES = {"context_budget": 3000}


def chars(msgs: list[dict]) -> int:
    return sum(len(m.get("content") or "") + sum(len(tc["function"]["arguments"]) for tc in m.get("tool_calls") or [])
               for m in msgs)


def script(i):
    return [(f"PROGRESS: read module {i % 6}\nSTALE: -", f"cat mod{i % 6:02d}.py") for i in range(14)]


def build(name, compact, corpus):
    cfg = build_config(name, "scripted/test", compact=compact)
    mod, _, cls = cfg["agent_path"].rpartition(".")
    agent = make_agent(getattr(importlib.import_module(mod), cls), script(0), corpus,
                       **OVERRIDES.get(name, {}), **(COMPACT_OVERRIDES if compact else {}))
    return agent


@pytest.mark.parametrize("name,compact", ARMS)
def test_every_arm_runs_to_the_step_limit_and_removes_something(name, compact, corpus):
    agent = build(name, compact, corpus)
    exit_status = agent.run("Investigate the modules.")["exit_status"]
    assert exit_status == "LimitsExceeded"  # LocalEnvironment never submits
    prompts = agent.model.prompts
    assert len(prompts) == 14
    for p in prompts:  # a valid chat history: system first, every tool call answered
        assert p[0]["role"] == "system"
        open_calls = set()
        for m in p:
            open_calls |= {tc["id"] for tc in m.get("tool_calls") or []}
            if m["role"] == "tool":
                assert m["tool_call_id"] in open_calls, "a tool reply without its call"
                open_calls.discard(m["tool_call_id"])
        assert not open_calls
    # the last prompt against the history it was rendered from (everything before the last reply)
    last_reply = max(i for i, m in enumerate(agent.messages) if m["role"] == "assistant")
    raw, rendered = chars(agent.messages[:last_reply]), chars(prompts[-1])
    if name == "base":
        assert rendered == raw
    else:
        assert rendered < raw, f"{name}{'+compact' if compact else ''} removed nothing"
    info = agent.serialize()["info"]
    if compact or name == "compact":
        assert info["compact"]["compactions"] >= 1 and agent.model.n_summaries >= 1
        assert info["compact"]["summary_calls"][0]["prompt_tokens"] == 100


def test_compaction_never_edits_the_log(corpus):
    agent = build("refold", True, corpus)
    agent.run("go")
    tools = [m for m in agent.messages if m["role"] == "tool"]
    assert len(tools) == 14 and all("[compacted history]" not in (m.get("content") or "") for m in agent.messages)

