"""`refold report` computes the run metrics from a cell's trajectories."""

from __future__ import annotations

import csv
import json

import pytest

from refold.cli import main
from refold.cli.report import cell_report


def traj(calls, exit_status="Submitted", **info):
    """A trajectory whose model calls are (prompt, cached, completion, overhead_s), one minute apart."""
    msgs, t = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], 1000.0
    for p, c, g, o in calls:
        usage = {"prompt_tokens": p, "completion_tokens": g, "prompt_tokens_details": {"cached_tokens": c}}
        msgs.append({"role": "assistant", "content": "x",
                     "extra": {"response": {"usage": usage}, "timestamp": t, "overhead_s": o}})
        t += 60
        msgs.append({"role": "tool", "content": "y", "extra": {"timestamp": t}})
    return {"info": {"exit_status": exit_status, **info}, "messages": msgs}


@pytest.fixture
def cell(tmp_path):
    c = tmp_path / "cell"
    for name, d in {
        "a": traj([(1000, 800, 100, 0.001), (2000, 1800, 200, 0.002)]),
        "b": traj([(5000, 0, 50, 0.004)], "ContextWindowExceededError",
                  compact={"compactions": 2, "summary_calls": [
                      {"prompt_tokens": 1000, "cached_tokens": 0, "completion_tokens": 100}]}),
    }.items():
        (c / name).mkdir(parents=True)
        (c / name / f"{name}.traj.json").write_text(json.dumps(d))
    (c / "plan.json").write_text(json.dumps({"model_name": "openai/Qwen/Qwen3.6-27B", "context_budget": 16384}))
    (c / "resolved.json").write_text(json.dumps({"n_resolved": 1, "resolved": ["a"]}))
    rows = [{"t": 0, "vllm:kv_cache_usage_perc": 0.1, "vllm:request_queue_time_seconds_sum": 0,
             "vllm:request_queue_time_seconds_count": 0, "vllm:num_preemptions_total": 0},
            {"t": 5, "vllm:kv_cache_usage_perc": 0.5, "vllm:request_queue_time_seconds_sum": 3,
             "vllm:request_queue_time_seconds_count": 2, "vllm:num_preemptions_total": 1},
            {"t": 10, "vllm:kv_cache_usage_perc": 0.3, "vllm:request_queue_time_seconds_sum": 6,
             "vllm:request_queue_time_seconds_count": 4, "vllm:num_preemptions_total": 2}]
    (c / "server_metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return c


def test_every_metric_follows_its_formula(cell):
    row, tasks, calls = cell_report(cell)
    cost_a = 100 * (0.30 * 400 + 0.15 * 2600 + 2.70 * 300) / 1e6  # pi_u (p - c) + pi_c c + pi_g g, in cents
    cost_b = 100 * (0.30 * 6000 + 2.70 * 150) / 1e6  # the summary call counts like any other
    kv = sorted(64 * peak / 2**20 + 0.14 for peak in (2000, 5000))  # kappa max_j p_ij + s
    assert row["N"] == 2 and row["Resolve"] == 1 and row["Exhausted"] == 1 and row["Timeouts"] == 0
    assert row["Steps"] == 2  # the median of 2 and 1, half rounded up
    assert row["Tokens (k)"] == pytest.approx((3300 + 6150) / 2 / 1000)
    assert row["Cost (¢)"] == pytest.approx((cost_a + cost_b) / 2)
    assert row["Cache hit (%)"] == pytest.approx(100 * 2600 / 9000)
    assert row["Duration (min)"] == pytest.approx(1.5) and row["KV cache (GiB)"] == pytest.approx(sum(kv) / 2)
    assert row["Compactions"] == 2 and row["Overhead (ms/step)"] == pytest.approx(7 / 3)
    assert (row["KV peak (%)"], row["Queue (s)"], row["Preemptions"]) == (pytest.approx(50), 1.5, 2)
    assert [t["resolved"] for t in tasks] == [1, 0] and sum(c["auxiliary"] for c in calls) == 1


def test_a_model_without_known_rates_needs_them(cell):
    (cell / "plan.json").write_text(json.dumps({"model_name": "some/other-model"}))
    row, _, _ = cell_report(cell)
    assert row["Cost (¢)"] is None and row["KV cache (GiB)"] is None
    row, _, _ = cell_report(cell, price="1,0.1,5", kv="10,0.5")
    assert row["Cost (¢)"] > 0 and row["KV cache (GiB)"] > 0.5


def test_the_cli_prints_a_table_and_writes_the_per_task_record(cell, tmp_path, capsys):
    assert main(["report", str(cell), "--tasks", str(tmp_path / "t.csv"), "--calls", str(tmp_path / "c.csv")]) == 0
    out = capsys.readouterr().out
    assert out.startswith("| cell | N | Resolve |") and "| 50.0 |" in out
    assert [r["task"] for r in csv.DictReader((tmp_path / "t.csv").open())] == ["a", "b"]
    assert len(list(csv.DictReader((tmp_path / "c.csv").open()))) == 4
