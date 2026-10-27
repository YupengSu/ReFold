"""Benchmarks and runners, without datasets or containers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from refold.benchmarks import available, get_benchmark
from refold.benchmarks.base import HF_DATASETS, SETS, pick, subset_ids
from refold.benchmarks.runners import swebench as sb_runner
from refold.benchmarks.runners import terminalbench as tb_runner
from refold.benchmarks.runners.common import stop_when_container_gone, with_progress
from refold.benchmarks.swebench import SweBenchVerified


def test_all_five_benchmarks_are_registered():
    assert available() == ["msb", "swebench", "swebenchpro", "tbench1", "tbench21"]


@pytest.mark.parametrize("f", sorted(p.name for p in SETS.glob("*.json")))
def test_every_benchmark_ships_fifty_tasks(f):
    assert len(set(subset_ids(f, "all"))) == 50


def test_the_verified_sets_nest():
    every = set(subset_ids("sets.json", "all"))
    assert every < set(subset_ids("sets.json", "a100")) and len(subset_ids("sets.json", "a100")) == 100  # w = 64


def test_a_subset_must_be_complete_in_the_data():
    every = subset_ids("sets.json", "all")
    assert pick(every + ["x"], "sets.json", "all", 3) == every[:3]
    assert pick(list("abc"), "sets.json", None, 2) == ["a", "b"]  # no subset: the whole split
    with pytest.raises(KeyError, match="1 of the 50 tasks of subset 'all' are not in the data"):
        pick(every[1:], "sets.json", "all", None)  # a dataset that lost a task fails loudly
    with pytest.raises(KeyError, match="Available: all"):
        subset_ids("sets.json", "nope")


def test_every_hf_dataset_is_pinned_to_a_revision():
    assert set(HF_DATASETS) == {"verified", "pro", "msb"}
    assert all(len(rev) == 40 and int(rev, 16) >= 0 for _, _, rev in HF_DATASETS.values())
    assert get_benchmark("swebenchpro").provenance()["revision"] == HF_DATASETS["pro"][2]


def test_run_id_follows_the_predictions_not_the_path(tmp_path):
    (tmp_path / "preds.json").write_text('{"a": 1}')
    first = SweBenchVerified.run_id(tmp_path)
    (tmp_path / "preds.json").write_text('{"a": 2}')
    assert SweBenchVerified.run_id(tmp_path) != first


def test_terminal_bench_is_scored_from_results_json(tmp_path, monkeypatch):
    monkeypatch.setenv("TBENCH21_TASKS", str(tmp_path))
    (tmp_path / "results.json").write_text(json.dumps({"a": {"resolved": True}, "b": {"resolved": False}}))
    r = get_benchmark("tbench21").score(tmp_path)
    assert (r.resolved, r.submitted, r.resolved_ids) == (1, 2, ("a",))


def test_pro_scoring_says_where_the_harness_comes_from(tmp_path, monkeypatch):
    monkeypatch.delenv("SWEBENCH_PRO_HARNESS", raising=False)
    with pytest.raises(FileNotFoundError, match="SWEBENCH_PRO_HARNESS"):
        get_benchmark("swebenchpro").score(tmp_path)


# ---------------------------------------------------------------- runners
def test_pro_and_msb_instances_get_their_image_and_working_directory():
    inst, env = sb_runner._pro({"instance_id": "p", "dockerhub_tag": "t1", "problem_statement": '"quoted"'})
    assert inst["image_name"] == "jefzda/sweap-images:t1" and inst["problem_statement"] == "quoted"
    assert env == {"cwd": "/app", "run_args": ["--rm", "--entrypoint", ""]}
    inst, env = sb_runner._msb({"instance_id": "m", "org": "BurntSushi", "repo": "ripgrep", "number": 7,
                                "title": "T", "body": "B", "resolved_issues": [{"number": 3, "title": "I"}]})
    assert inst["image_name"] == "mswebench/burntsushi_m_ripgrep:pr-7" and env == {"cwd": "/home/ripgrep"}
    assert inst["problem_statement"].startswith("T\n\nB") and "linked issue #3" in inst["problem_statement"]


def test_the_progress_wrapper_keeps_the_class_identity():
    from refold.strategies.refold.agent import ReFoldAgent

    wrapped = with_progress(ReFoldAgent)
    assert issubclass(wrapped, ReFoldAgent)
    assert f"{wrapped.__module__}.{wrapped.__name__}" == "refold.strategies.refold.agent.ReFoldAgent"


def test_a_removed_container_ends_the_task():
    from minisweagent.exceptions import InterruptAgentFlow

    class Agent:
        n_calls, cost = 0, 0.0

        def __init__(self, output):
            self.output = output

        def step(self):
            return [{"role": "tool", "content": self.output}]

    progress = SimpleNamespace(update_instance_status=lambda *a: None)
    gone = "<returncode>1</returncode>\n<output>\nError response from daemon: No such container: 6294\n</output>"
    with pytest.raises(InterruptAgentFlow) as e:
        with_progress(Agent)(gone, progress_manager=progress).step()
    assert e.value.messages[0]["extra"]["exit_status"] == "ContainerGone"
    assert with_progress(Agent)("ok", progress_manager=progress).step()[0]["content"] == "ok"
    with pytest.raises(InterruptAgentFlow):  # the Terminal-Bench runner has no progress manager
        stop_when_container_gone(Agent)(gone).step()


def test_terminal_bench_task_loaders(tmp_path):
    t1 = tmp_path / "tb1" / "task-a"
    t1.mkdir(parents=True)
    (t1 / "task.yaml").write_text("instruction: do it\nmax_test_timeout_sec: 30\n")
    (t1 / "Dockerfile").write_text("FROM x\nWORKDIR /srv\nWORKDIR /srv/app\n")
    [a] = tb_runner.load_tasks_tb1(tmp_path / "tb1")
    assert (a["instance_id"], a["workdir"], a["image_name"]) == ("task-a", "/srv/app", "tbench-task-task-a")
    t21 = tmp_path / "tb21" / "task-b"
    t21.mkdir(parents=True)
    (t21 / "task.toml").write_text('[environment]\ndocker_image = "img:1"\n')
    (t21 / "instruction.md").write_text("do it\n")
    [b] = tb_runner.load_tasks_tb21(tmp_path / "tb21")
    assert (b["image_name"], b["problem_statement"]) == ("img:1", "do it")


@pytest.mark.parametrize("out,ok", [
    ("===== 3 passed in 0.1s =====", True), ("===== 1 failed, 2 passed =====", False),
    ("===== no tests ran =====", False), ("bash: pytest: command not found", False)])
def test_terminal_bench_1_verdict_from_the_pytest_summary(out, ok):
    assert tb_runner.parse_pytest_resolved(out) is ok


def test_the_swebench_runner_never_writes_into_the_shared_config(monkeypatch, tmp_path):
    """Upstream sets config["environment"]["image"] in place; with worker threads sharing
    one config, each instance must get its own copy."""
    import minisweagent.run.benchmarks.swebench as sw

    seen = []
    monkeypatch.setattr(sw, "get_sb_environment", lambda cfg, inst: seen.append(cfg) or SimpleNamespace())
    monkeypatch.setattr(sw, "process_instance", lambda inst, out, cfg, pm: sw.get_sb_environment(cfg, inst))
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: [{"instance_id": "m", "org": "o", "repo": "r",
                                                                    "number": 1, "title": "t", "body": "b"}])
    monkeypatch.setattr(sb_runner, "run_batch", lambda insts, work, out, w: [work(i, None) for i in insts])
    cfgfile = tmp_path / "c.yaml"
    cfgfile.write_text("environment:\n  cwd: /home\n")
    sb_runner.main(flavor="msb", filter_spec="", output=str(tmp_path / "o"), workers=1, model="m",
                   model_class=None, agent_class="refold.agent.RenderingAgent", config_spec=[str(cfgfile)],
                   redo_existing=False)
    assert seen[0]["environment"]["cwd"] == "/home/r"
