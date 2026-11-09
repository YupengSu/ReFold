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


# ---------------------------------------------------------------- Terminal-Bench verdicts (tb run, Harbor)
SUMMARY = "=========================== short test summary info ============================\n"


@pytest.mark.parametrize("lines,tests", [
    ("PASSED ../tests/test_outputs.py::test_a\nPASSED ../tests/test_outputs.py::test_b\n",
     {"test_a": True, "test_b": True}),
    ("PASSED t.py::test_a\nFAILED t.py::test_b - AssertionError: wrong\n", {"test_a": True, "test_b": False}),
    ("SKIPPED t.py::test_s\nXFAIL t.py::test_x\n", {"test_s": True, "test_x": True}),
    ("XPASS t.py::test_x\nERROR t.py::test_e\n", {"test_x": False, "test_e": False}),
    ("PASSED t.py::test_a\nwarnings and other lines\n===== 1 passed in 0.1s =====\n", {"test_a": True}),
    ("", {})])  # a summary with no test lines: all([]), so tb run calls it resolved
def test_terminal_bench_1_reads_tests_as_tb_run_does(lines, tests):
    assert tb_runner.parse_pytest_tests("collected 2 items\n" + SUMMARY + lines) == tests


@pytest.mark.parametrize("out", ["===== 3 passed in 0.1s =====", "bash: pytest: command not found", ""])
def test_terminal_bench_1_without_a_short_summary_has_no_verdict(out):
    assert tb_runner.parse_pytest_tests(out) is None  # tb run's PARSE_ERROR


@pytest.mark.parametrize("reward_json,reward_txt,verdict", [
    (None, "1", (True, None)), (None, "0\n", (False, None)), (None, "1.0", (True, None)),
    (None, "0.5", (False, None)), (None, "", (None, "empty reward.txt")),
    (None, "pass", (None, "unparsable reward.txt")), (None, "nan", (None, "unparsable reward.txt")),
    (None, None, (None, "no reward file")),
    ('{"reward": 1}', None, (True, None)), ("1", None, (True, None)), ('{"accuracy": 1.0}', None, (True, None)),
    ('{"reward": 0}', "1", (False, None)),  # reward.json is read first
    ("", "1", (None, "empty reward.json")), ("{", None, (None, "unparsable reward.json")),
    ('{"reward": "1"}', None, (None, "unparsable reward.json")), ("{}", None, (None, "unparsable reward.json")),
    ('{"a": 1, "b": 0}', None, (None, "several rewards in reward.json"))])
def test_terminal_bench_2_reads_the_reward_as_harbor_does(reward_json, reward_txt, verdict):
    assert tb_runner.reward_verdict(reward_json, reward_txt) == verdict


def _stub_docker(monkeypatch, test_stdout="", files=None, timeout=False, calls=None, test_stderr=""):
    """subprocess.run as `docker` would answer: the test script prints `test_stdout` and
    `test_stderr` (or times out); `cat` returns the file from `files`, absent files fail."""
    files = dict(files or {})

    def run(argv, **kw):
        if calls is not None:
            calls.append(argv)
        if argv[:2] == ["docker", "exec"] and "bash" in argv:
            if timeout:
                raise tb_runner.subprocess.TimeoutExpired(argv, kw.get("timeout"), output=b"partial", stderr=None)
            return SimpleNamespace(returncode=0, stdout=test_stdout, stderr=test_stderr)
        if argv[:2] == ["docker", "exec"] and argv[3] == "cat":
            return SimpleNamespace(returncode=0 if argv[4] in files else 1, stdout=files.get(argv[4], ""), stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(tb_runner.subprocess, "run", run)


ENV = SimpleNamespace(container_id="c", config=SimpleNamespace(cwd="/app"))


def test_terminal_bench_2_has_no_pytest_fallback_and_drops_stale_rewards(monkeypatch, tmp_path):
    inst, calls = {"task_dir": str(tmp_path), "max_test_timeout_sec": 1}, []
    _stub_docker(monkeypatch, SUMMARY + "PASSED t.py::test_a\n===== 1 passed =====", calls=calls)
    assert tb_runner.score_tb21(ENV, inst)[::2] == (None, "no reward file")  # Harbor reads only the reward file
    rm = calls.index(["docker", "exec", "c", "rm", "-f", "/logs/verifier/reward.txt", "/logs/verifier/reward.json"])
    assert rm < next(i for i, a in enumerate(calls) if "bash" in a)  # before the tests run
    _stub_docker(monkeypatch, "", {"/logs/verifier/reward.txt": "1\n"})
    assert tb_runner.score_tb21(ENV, inst).resolved is True


@pytest.mark.parametrize("stdout,stderr,reward,warning", [
    # test.sh could not install uv, so pytest never started, and it wrote 0 all the same
    ("", "curl: (60) SSL certificate problem\n/tests/test.sh: line 13: uvx: command not found\n", "0",
     "no pytest short test summary"),
    (SUMMARY + "FAILED t.py::test_a - x\n===== 1 failed =====", "", "0", None),  # the tests ran and failed
    ("", "", "1", None)])
def test_terminal_bench_2_marks_a_zero_reward_without_a_pytest_summary(monkeypatch, tmp_path, stdout, stderr,
                                                                        reward, warning):
    _stub_docker(monkeypatch, stdout, {"/logs/verifier/reward.txt": reward}, test_stderr=stderr)
    v = tb_runner.score_tb21(ENV, {"task_dir": str(tmp_path), "max_test_timeout_sec": 1})
    assert (v.resolved, v.reason, v.warning) == (reward == "1", None, warning)  # Harbor's verdict is kept


@pytest.mark.parametrize("score,inst", [
    (tb_runner.score_tb21, {"max_test_timeout_sec": 1}),
    (tb_runner.score_tb1, {"max_test_timeout_sec": 1, "workdir": "/w"})])
def test_a_test_timeout_is_no_verdict(monkeypatch, tmp_path, score, inst):
    _stub_docker(monkeypatch, timeout=True, files={"/logs/verifier/reward.txt": "0"})
    v = score(ENV, {**inst, "task_dir": str(tmp_path)})
    assert (v.resolved, v.reason) == (None, "test timeout") and v.output.endswith("partial")


def test_terminal_bench_1_scores_the_short_summary(monkeypatch, tmp_path):
    inst = {"task_dir": str(tmp_path), "max_test_timeout_sec": 1, "workdir": "/w"}
    _stub_docker(monkeypatch, SUMMARY + "PASSED t.py::test_a\nFAILED t.py::test_b - x\n")
    assert tb_runner.score_tb1(ENV, inst).resolved is False
    _stub_docker(monkeypatch, "===== 2 passed =====")
    assert tb_runner.score_tb1(ENV, inst)[::2] == (None, "no pytest short test summary")


PYTEST_ONE_TEST = """============================= test session starts ==============================
collected 1 item

../tests/test_outputs.py .                                               [100%]

==================================== PASSES ====================================
=========================== short test summary info ============================
PASSED ../tests/test_outputs.py::test_success
============================== 1 passed in 0.01s ===============================
"""


def test_terminal_bench_1_reads_a_task_with_one_test(monkeypatch, tmp_path):
    # `pytest -rA` with one test puts its entry right above the final line. The old regex took
    # the entry for the summary line, found no "N passed" in it and failed the task.
    inst = {"task_dir": str(tmp_path), "max_test_timeout_sec": 1, "workdir": "/w"}
    _stub_docker(monkeypatch, PYTEST_ONE_TEST)
    assert tb_runner.score_tb1(ENV, inst).resolved is True
    _stub_docker(monkeypatch, PYTEST_ONE_TEST.replace("PASSED", "FAILED").replace("1 passed", "1 failed"))
    assert tb_runner.score_tb1(ENV, inst).resolved is False


def test_terminal_bench_1_copies_tests_only_when_the_task_has_them(monkeypatch, tmp_path):
    # swe-bench-astropy-2 keeps its tests in run-tests.sh: copying a missing tests/ raised
    inst, calls = {"task_dir": str(tmp_path), "max_test_timeout_sec": 1, "workdir": "/w"}, []
    _stub_docker(monkeypatch, PYTEST_ONE_TEST, calls=calls)
    assert tb_runner.score_tb1(ENV, inst).resolved is True
    assert not any(a[:2] == ["docker", "cp"] and a[2].endswith("/tests/.") for a in calls)
    assert ["docker", "cp", f"{tmp_path}/run-tests.sh", "c:/run-tests.sh"] in calls
    (tmp_path / "tests").mkdir()
    calls.clear()
    tb_runner.score_tb1(ENV, inst)
    assert ["docker", "cp", f"{tmp_path}/tests/.", "c:/tests/"] in calls


def test_the_saved_tail_ends_with_the_pytest_summary(monkeypatch, tmp_path):
    # the test scripts write their install logs to stderr; stdout, with the summary, comes last
    inst = {"task_dir": str(tmp_path), "max_test_timeout_sec": 1, "workdir": "/w"}
    _stub_docker(monkeypatch, PYTEST_ONE_TEST, test_stderr=" + yarl==1.25.1\n" * 500)
    assert tb_runner.score_tb1(ENV, inst).output[-4000:].endswith(PYTEST_ONE_TEST)


# ---------------------------------------------------------------- Terminal-Bench: one task end to end
class Timeout(Exception):  # named like litellm's request timeout
    pass


class ContextWindowExceededError(Exception):
    pass


def _run_task(monkeypatch, tmp_path, flavor, agent_does, scored=tb_runner.Verdict(True, "ok"), config=None,
              prepare=lambda inst: "/app"):
    """process_instance with a stub image, container, model and agent. `agent_does`: an
    exit status to return or an exception to raise. Returns (results record, traj info,
    the container's kwargs, whether the tests ran)."""
    seen = {"env": {}, "info": None, "scored": False}

    class Env:
        def __init__(self, **kw):
            seen["env"] = kw
            self.container_id, self.config = "c", SimpleNamespace(cwd=kw.get("cwd"))

        def cleanup(self):
            pass

    class Agent:
        def __init__(self, model, env, **kw):
            pass

        def run(self, task):
            if isinstance(agent_does, Exception):
                raise agent_does
            return {"exit_status": agent_does}

        def save(self, path, extra):
            seen["info"] = extra["info"]

    def score(env, inst):
        seen["scored"] = True
        if isinstance(scored, Exception):
            raise scored
        return scored

    _, _, run_args, _ = tb_runner.FLAVORS[flavor]
    monkeypatch.setattr(tb_runner, "DockerEnvironment", Env)
    monkeypatch.setattr(tb_runner, "get_model", lambda config: None)
    monkeypatch.setitem(tb_runner.FLAVORS, flavor, (None, prepare, run_args, score))
    progress = SimpleNamespace(on_instance_start=lambda *a: None, update_instance_status=lambda *a: None,
                               on_instance_end=lambda *a: None)
    tb_runner.process_instance({"instance_id": "t", "image_name": "img", "problem_statement": "p"}, tmp_path,
                               config or {}, Agent, flavor, progress)
    record = json.loads((tmp_path / "results.json").read_text())["t"]
    return record, seen["info"], seen["env"], seen["scored"]


@pytest.mark.parametrize("flavor", ["tb1", "tb21"])
def test_the_tests_still_run_after_the_agent_fails(monkeypatch, tmp_path, flavor):
    record, info, _, scored = _run_task(monkeypatch, tmp_path, flavor, ContextWindowExceededError("too long"))
    assert scored and record == {"instance_id": "t", "exit_status": "ContextWindowExceededError", "resolved": True}
    assert info["resolved"] is True and "ContextWindowExceededError" in info["traceback"]


def test_a_model_api_error_leaves_a_tb21_task_without_a_verdict(monkeypatch, tmp_path):
    record, info, _, scored = _run_task(monkeypatch, tmp_path, "tb21", Timeout("600 s"))
    assert scored and record["resolved"] is None and record["verifier_error"] == "agent API error: Timeout"
    assert info["test_verdict"] is True  # what the tests said, kept for the record
    record, _, _, _ = _run_task(monkeypatch, tmp_path, "tb1", Timeout("600 s"))  # tb run keeps the verdict
    assert record["resolved"] is True and "verifier_error" not in record


@pytest.mark.parametrize("agent_does,scored,reason", [
    ("ContainerGone", tb_runner.Verdict(True), "container gone"),
    ("Submitted", tb_runner.subprocess.CalledProcessError(1, ["docker", "cp"]), "scoring failed: CalledProcessError"),
    ("Submitted", tb_runner.Verdict(None, "", "no reward file"), "no reward file")])
def test_no_verdict_is_recorded_with_its_reason(monkeypatch, tmp_path, agent_does, scored, reason):
    record, info, _, ran = _run_task(monkeypatch, tmp_path, "tb21", agent_does, scored)
    assert record["resolved"] is None and record["verifier_error"] == reason == info["verifier_error"]
    assert ran is (agent_does != "ContainerGone")


def test_a_zero_reward_without_a_pytest_summary_is_recorded(monkeypatch, tmp_path):
    scored = tb_runner.Verdict(False, "uvx: command not found", warning="no pytest short test summary")
    record, info, _, _ = _run_task(monkeypatch, tmp_path, "tb21", "Submitted", scored)
    assert record == {"instance_id": "t", "exit_status": "Submitted", "resolved": False,
                      "verifier_warning": "no pytest short test summary"}
    assert info["verifier_warning"] == "no pytest short test summary" and "verifier_error" not in info


def test_a_task_that_never_started_is_not_scored(monkeypatch, tmp_path):
    def prepare(inst):
        raise RuntimeError("docker pull failed")

    record, info, _, scored = _run_task(monkeypatch, tmp_path, "tb21", "Submitted", prepare=prepare)
    assert not scored and info is None  # no agent, so no trajectory
    assert record == {"instance_id": "t", "exit_status": "RuntimeError", "resolved": None,
                      "verifier_error": "setup failed: RuntimeError"}


def test_verifier_errors_are_counted_apart(monkeypatch, tmp_path, capsys):
    from refold.cli import _score

    monkeypatch.setenv("TBENCH21_TASKS", str(tmp_path))
    (tmp_path / "results.json").write_text(json.dumps({
        "a": {"resolved": True}, "b": {"resolved": False},
        "c": {"resolved": None, "verifier_error": "no reward file"}}))
    r = get_benchmark("tbench21").score(tmp_path)
    assert (r.resolved, r.submitted, r.verifier_error, r.verifier_error_ids) == (1, 3, 1, ("c",))
    _score(tmp_path, "tbench21")
    assert capsys.readouterr().out.startswith(
        "resolved 1 / 3 (1 verifier errors)\nverifier errors (without them: 1 / 2): c")
    d = json.loads((tmp_path / "resolved.json").read_text())
    assert (d["n_resolved"], d["n_verifier_error"], d["verifier_error"], d["resolved"]) == (1, 1, ["c"], ["a"])


def test_verifier_warnings_are_listed_but_count_as_unresolved(monkeypatch, tmp_path, capsys):
    from refold.cli import _score

    monkeypatch.setenv("TBENCH21_TASKS", str(tmp_path))
    (tmp_path / "results.json").write_text(json.dumps({
        "a": {"resolved": True}, "b": {"resolved": False},
        "w": {"resolved": False, "verifier_warning": "no pytest short test summary"}}))
    r = get_benchmark("tbench21").score(tmp_path)
    assert (r.resolved, r.submitted, r.verifier_error, r.verifier_warning_ids) == (1, 3, 0, ("w",))
    _score(tmp_path, "tbench21")
    assert capsys.readouterr().out.startswith(
        "resolved 1 / 3\nreward 0 without a pytest summary, tests may not have run: w")
    d = json.loads((tmp_path / "resolved.json").read_text())
    assert (d["n_verifier_warning"], d["verifier_warning"], d["n_verifier_error"]) == (1, ["w"], 0)


@pytest.mark.parametrize("flavor,user,expected", [
    ("tb21", ["--rm", "-e", "A=1"], ["--rm", "--entrypoint", "", "--rm", "-e", "A=1"]),  # the user's flags come last
    ("tb21", None, ["--rm", "--entrypoint", ""]),
    ("tb1", ["-e", "A=1"], ["-e", "A=1"])])  # tb1 sets none of its own
def test_the_users_run_args_are_kept(monkeypatch, tmp_path, flavor, user, expected):
    config = {"environment": {"run_args": user}} if user else {}
    _, _, env, _ = _run_task(monkeypatch, tmp_path, flavor, "Submitted", config=config)
    assert env.get("run_args") == expected and env["image"] == "img"


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


def test_the_swebench_runner_keeps_the_users_run_args(monkeypatch, tmp_path):
    import minisweagent.run.benchmarks.swebench as sw

    seen = []
    monkeypatch.setattr(sw, "get_sb_environment", lambda cfg, inst: seen.append(cfg) or SimpleNamespace())
    monkeypatch.setattr(sw, "process_instance", lambda inst, out, cfg, pm: sw.get_sb_environment(cfg, inst))
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: [{"instance_id": "p", "dockerhub_tag": "t",
                                                                    "problem_statement": "x"}])
    monkeypatch.setattr(sb_runner, "run_batch", lambda insts, work, out, w: [work(i, None) for i in insts])
    cfgfile = tmp_path / "c.yaml"
    cfgfile.write_text("environment:\n  run_args: ['-e', 'A=1']\n")
    sb_runner.main(flavor="pro", filter_spec="", output=str(tmp_path / "o"), workers=1, model="m",
                   model_class=None, agent_class="refold.agent.RenderingAgent", config_spec=[str(cfgfile)],
                   redo_existing=False)
    assert seen[0]["environment"]["run_args"] == ["--rm", "--entrypoint", "", "-e", "A=1"]
