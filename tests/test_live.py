"""`refold run --live`: the per-call log, and that it changes nothing the model is sent."""

from __future__ import annotations

import json

import pytest
from helpers import ScriptedModel, make_agent
from test_refold import filler, read

from refold.agent import RenderingAgent
from refold.live import LIVE_ENV, with_live_log
from refold.strategies.refold.agent import ReFoldAgent

SCRIPT = [read(0), read(0), read(1), read(2), ("PROGRESS: done with the reads\nSTALE: 2, 3", "echo x"),
          ("PROGRESS: check\nSTALE: -", "echo y"), *filler(8)]


class UsageModel(ScriptedModel):
    """The scripted model with a vLLM-style usage block on every reply."""

    def query(self, messages, **kw):
        m = super().query(messages, **kw)
        n = sum(len(x.get("content") or "") for x in messages) // 4
        m["extra"]["response"] = {"usage": {"prompt_tokens": n, "completion_tokens": 7,
                                            "prompt_tokens_details": {"cached_tokens": n // 2}}}
        return m


def run(cls, corpus, live_dir=None):
    agent_cls = with_live_log(cls, live_dir) if live_dir else cls
    agent = make_agent(agent_cls, SCRIPT, corpus)
    agent.model = UsageModel(SCRIPT)
    agent.instance_id = "demo__task-1"
    agent.run("Investigate the modules.")
    return agent


def api(prompts):
    return [[{k: v for k, v in m.items() if k != "extra"} for m in p] for p in prompts]


@pytest.mark.parametrize("cls", [RenderingAgent, ReFoldAgent])
def test_the_log_changes_no_prompt(cls, corpus, tmp_path):
    plain = run(cls, corpus)
    logged = run(cls, corpus, tmp_path / "live")
    assert api(logged.model.prompts) == api(plain.model.prompts)
    assert type(logged).__name__ == cls.__name__ and type(logged).__module__ == cls.__module__


def test_one_line_per_call_between_start_and_end(corpus, tmp_path):
    agent = run(ReFoldAgent, corpus, tmp_path / "live")
    lines = [json.loads(x) for x in (tmp_path / "live" / "demo__task-1.jsonl").read_text().splitlines()]
    calls = [x for x in lines if x["ev"] == "call"]
    assert lines[0]["ev"] == "start" and lines[-1]["ev"] == "end" and len(calls) == agent.n_calls
    assert [c["step"] for c in calls] == list(range(1, agent.n_calls + 1))
    assert all(c["t0"] <= c["t1"] and c["completion"] == 7 and c["cached"] == c["prompt"] // 2 for c in calls)
    assert {x["task"] for x in lines} == {"demo__task-1"}
    assert lines[-1]["steps"] == agent.n_calls and lines[-1]["exit_status"]
    # ReFold folds steps 2 and 3 once the boundary passes them: the prompt then is shorter than the history
    assert any(c["folded"] for c in calls) and any(c["sent_chars"] < c["hist_chars"] for c in calls)


def test_base_sends_its_whole_history(corpus, tmp_path):
    run(RenderingAgent, corpus, tmp_path / "live")
    calls = [json.loads(x) for x in (tmp_path / "live" / "demo__task-1.jsonl").read_text().splitlines()]
    calls = [c for c in calls if c["ev"] == "call"]
    assert calls and all(c["sent_chars"] == c["hist_chars"] and c["folded"] == 0 for c in calls)


def test_the_runner_wraps_the_agent_only_when_asked(monkeypatch, tmp_path):
    from refold.benchmarks.runners.common import with_progress

    monkeypatch.delenv(LIVE_ENV, raising=False)
    plain = with_progress(RenderingAgent).__mro__
    monkeypatch.setenv(LIVE_ENV, str(tmp_path))
    logged = with_progress(RenderingAgent).__mro__
    assert len(logged) == len(plain) + 1
    assert any("_live" in vars(c) for c in logged) and not any("_live" in vars(c) for c in plain)


def test_live_is_recorded_in_the_plan(monkeypatch, tmp_path, capsys):
    from refold.cli import main
    from refold.cli import run as run_cmd

    class Bench:
        runner_module, config = "refold.benchmarks.runners.swebench", tmp_path / "swebench.yaml"

        def data_flags(self):
            return ["--flavor", "verified"]

        def provenance(self):
            return {}

        def load(self, subset=None, limit=None):
            return ["a__b-1"]

    monkeypatch.setattr(run_cmd, "get_benchmark", lambda name: Bench())

    def plan(*flags):
        assert main(["run", "--strategy", "base", "--model", "m", "-o", str(tmp_path / "o"), *flags, "--dry-run"]) == 0
        out = capsys.readouterr().out
        return json.loads(out[: out.rindex("}\n") + 1])

    assert plan("--live")["live"] is True
    assert "live" not in plan()  # a run without --live writes the plan it always wrote
