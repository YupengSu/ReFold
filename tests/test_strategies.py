"""Every arm runs a real agent loop, sends a well-formed prompt, and removes something.

Each agent class is imported by the dotted path the runner uses and stepped with a
scripted model over a real local shell, so an error anywhere on the render path is a red
test here rather than a crash in the first real run.
"""

from __future__ import annotations

import importlib

import pytest
from helpers import assert_every_tool_call_answered, make_agent

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


def parallel_script():
    """Two tool calls per reply; each reply names only the first of the previous reply's
    two steps stale, so folding splits replies."""
    return [(f"PROGRESS: read modules {r % 6} and {(r + 1) % 6}\nSTALE: {2 * r - 2 if r else '-'}",
             [f"cat mod{r % 6:02d}.py", f"cat mod{(r + 1) % 6:02d}.py"]) for r in range(14)]


@pytest.mark.parametrize("name,compact", ARMS)
def test_with_parallel_tool_calls_every_prompt_answers_every_call(name, compact, corpus):
    cfg = build_config(name, "scripted/test", compact=compact)
    mod, _, cls = cfg["agent_path"].rpartition(".")
    agent = make_agent(getattr(importlib.import_module(mod), cls), parallel_script(), corpus,
                       **OVERRIDES.get(name, {}), **(COMPACT_OVERRIDES if compact else {}))
    agent.run("Investigate the modules.")
    assert len(agent.model.prompts) == 14
    for p in agent.model.prompts:
        assert_every_tool_call_answered(p)
    if name == "refold" and not compact:  # the case under test happened: a reply was split
        assert any(len(m.get("tool_calls") or []) == 1 for p in agent.model.prompts for m in p)


def test_compaction_may_cut_inside_a_reply_and_keeps_its_later_calls(corpus):
    """One reply's ten tool calls are the whole history behind the protected tail: the cut
    lands inside it, and the reply stays with the calls past the cut."""
    script = [("PROGRESS: read ten modules\nSTALE: -", [f"cat mod{i:02d}.py" for i in range(10)]),
              *[(f"PROGRESS: filler {j}\nSTALE: -", f"echo filler-{j}") for j in range(5)]]
    cfg = build_config("refold", "scripted/test", compact=True)
    mod, _, cls = cfg["agent_path"].rpartition(".")
    agent = make_agent(getattr(importlib.import_module(mod), cls), script, corpus, **COMPACT_OVERRIDES)
    agent.run("Investigate the modules.")
    assert agent.serialize()["info"]["compact"]["compactions"] >= 1
    for p in agent.model.prompts:
        assert_every_tool_call_answered(p)
    kept = [len(m["tool_calls"]) for p in agent.model.prompts for m in p
            if "read ten modules" in (m.get("content") or "") and m.get("tool_calls")]
    assert any(0 < n < 10 for n in kept)  # cut inside the reply, which stayed with its later calls


def test_compaction_never_edits_the_log(corpus):
    agent = build("refold", True, corpus)
    agent.run("go")
    tools = [m for m in agent.messages if m["role"] == "tool"]
    assert len(tools) == 14 and all("[compacted history]" not in (m.get("content") or "") for m in agent.messages)

