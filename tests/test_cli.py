"""`refold run` resolves a cell into the exact runner command, and checks what it can."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest

from refold.benchmarks import get_benchmark
from refold.cli import main
from refold.cli import run as run_cmd
from refold.cli.run import FULL_BUDGET, build_command, build_config


class _Bench:  # a benchmark without a dataset download
    runner_module, config = "refold.benchmarks.runners.swebench", Path("/x/swebench.yaml")

    def data_flags(self):
        return ["--flavor", "verified"]


def test_compact_is_a_strategy_and_refold_composes_with_it():
    assert build_config("compact", "m")["agent_path"] == "refold.strategies.compact.CompactAgent"
    cfg = build_config("refold", "m", compact=True)
    assert cfg["agent_path"] == "refold.strategies.compact.ReFoldCompactAgent" and cfg["arm"] == "compact+refold"
    with pytest.raises(ValueError, match="--strategy compact"):
        build_config("base", "m", compact=True)


def test_the_context_budget_defaults_to_the_full_window_and_sets_the_arms_that_trigger_on_it():
    assert FULL_BUDGET == 262_144 - 16_384
    assert build_config("compact", "m")["agent"] == {"context_budget": FULL_BUDGET}
    assert build_config("refold", "m", compact=True, context_budget=16384)["agent"] == {"context_budget": 16384}
    cfg = build_config("refold", "m", context_budget=16384)  # the server enforces it; the plan records it
    assert cfg["agent"] == {} and cfg["context_budget"] == 16384


def test_local_serving_is_a_model_string_and_a_base():
    cfg = build_config("refold", "Qwen/Qwen3.6-27B", api_base="http://h:1/v1")
    assert cfg["model_name"] == "openai/Qwen/Qwen3.6-27B"
    assert cfg["model_kwargs"] == {"temperature": 0.0, "api_base": "http://h:1/v1", "api_key": "dummy"}
    assert build_config("refold", "deepseek/deepseek-chat")["model_kwargs"] == {}


def test_knobs_are_coerced_and_checked_against_the_arm():
    cfg = build_config("refold", "m",
                       overrides=["chunk_size=5", "agent.fold=false", "environment.container_timeout=6h"])
    assert cfg["agent"] == {"chunk_size": 5, "fold": False}
    cmd = build_command(["i"], cfg, Path("/o"), 1, _Bench())
    assert "environment.container_timeout=6h" in cmd and "agent.chunk_size=5" in cmd
    with pytest.raises(ValueError, match="key=value"):
        build_config("refold", "m", overrides=["chunk_size"])
    with pytest.raises(ValueError, match="has no knob 'recent_k'; its knobs are .*chunk_size"):
        build_config("refold", "m", overrides=["recent_k=5"])  # a typo or an old name must not pass silently


def test_the_command_layers_the_benchmark_config_and_anchors_the_filter():
    cfg = build_config("refold", "m", model_class="my.Model")
    cmd = build_command(["a__b-1", "a__b-12"], cfg, Path("/o"), 4, _Bench())
    cs = [cmd[i + 1] for i, x in enumerate(cmd) if x == "-c"]
    assert cs[0] == "/x/swebench.yaml"
    rx = cmd[cmd.index("--filter") + 1]
    assert re.match(rx, "a__b-1") and not re.match(rx, "a__b-10")
    assert cmd[cmd.index("--model-class") + 1] == "my.Model"
    assert "agent.step_limit=250" in cs and "model.model_kwargs.max_tokens=16384" in cs
    assert "model.model_kwargs.timeout=600" in cs


@pytest.mark.parametrize("name", ["swebench", "swebenchpro", "msb", "tbench1", "tbench21"])
def test_each_benchmark_runs_its_own_prompt(name, monkeypatch, tmp_path):
    monkeypatch.setenv("TBENCH1_TASKS", str(tmp_path))
    monkeypatch.setenv("TBENCH21_TASKS", str(tmp_path))
    b = get_benchmark(name)
    assert b.config.is_file() and b.config.name == {"swebench": "swebench.yaml", "swebenchpro": "swebenchpro.yaml",
                                                    "msb": "multiswebench.yaml"}.get(name, "terminalbench.yaml")
    assert b.data_flags()[:1] == ["--flavor"]


def test_the_server_must_serve_the_budget_plus_the_reply(monkeypatch):
    def serving(max_model_len):
        body = json.dumps({"data": [{"id": "m", "max_model_len": max_model_len}]}).encode()
        monkeypatch.setattr(run_cmd.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(body))

    serving(16384 + 16384)
    run_cmd.check_server("http://h:1/v1", 16384)
    serving(FULL_BUDGET + 16384)
    with pytest.raises(ValueError, match="--max-model-len 32768"):
        run_cmd.check_server("http://h:1/v1", 16384)


def test_server_metrics_are_summed_over_label_sets():
    text = "\n".join([
        "# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.",
        'vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.25',
        'vllm:num_preemptions_total{engine="0",model_name="m"} 3.0',
        'vllm:request_queue_time_seconds_sum{engine="0",model_name="m"} 1.5',
        'vllm:request_queue_time_seconds_count{engine="0",model_name="m"} 6.0',
        'vllm:request_queue_time_seconds_bucket{engine="0",le="0.3",model_name="m"} 6.0',
        'vllm:prompt_tokens_total{engine="0",model_name="m"} 100.0',
        'vllm:prompt_tokens_total{engine="1",model_name="m"} 50.0'])
    assert run_cmd.parse_metrics(text) == {
        "vllm:kv_cache_usage_perc": 0.25, "vllm:num_preemptions_total": 3.0,
        "vllm:request_queue_time_seconds_sum": 1.5, "vllm:request_queue_time_seconds_count": 6.0,
        "vllm:prompt_tokens_total": 150.0}


def test_errors_are_one_line_not_a_traceback(capsys):
    assert main(["run", "--strategy", "base", "--compact", "--model", "m", "-o", "/tmp/x", "--dry-run"]) == 2
    assert "--compact runs a strategy beneath compaction" in capsys.readouterr().err


def test_strategies_lists_base_compact_and_refold(capsys):
    main(["strategies"])
    names = [line.split()[0] for line in capsys.readouterr().out.splitlines()]
    assert names == ["base", "compact", "refold"]
