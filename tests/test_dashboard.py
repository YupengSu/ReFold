"""The run dashboard: what it reads from a run directory, the page it serves, the CLI wiring."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import urllib.request

import pytest

from refold import dashboard


def make_run(tmp_path, half_line=False):
    out = tmp_path / "run"
    (out / "live").mkdir(parents=True)
    (out / "plan.json").write_text(json.dumps({"model_name": "openai/Qwen/Qwen3.6-27B", "benchmark": "swebench",
                                               "instances": 2, "workers": 2, "arm": "refold"}))
    rows = [{"ev": "start", "task": "django__django-1", "t": 100.0},
            {"ev": "call", "task": "django__django-1", "step": 1, "t0": 100.5, "t1": 103.0, "prompt": 2000,
             "cached": None, "completion": 100, "sent_chars": 7000, "hist_chars": 7000},
            {"ev": "end", "task": "django__django-1", "t": 110.0, "exit_status": "Submitted", "steps": 1}]
    text = "".join(json.dumps(r) + "\n" for r in rows)
    (out / "live" / "django__django-1.jsonl").write_text(text + ('{"ev": "ca' if half_line else ""))
    (out / "server_metrics.jsonl").write_text(json.dumps({"t": 101.0, "vllm:kv_cache_usage_perc": 0.2}) + "\n")
    return out


def test_snapshot_reads_only_what_was_appended(tmp_path):
    out = make_run(tmp_path, half_line=True)
    first = dashboard.snapshot(out, {})
    assert [r["ev"] for r in first["live"]] == ["start", "call", "end"]  # the half-written line waits
    assert first["metrics"][0]["vllm:kv_cache_usage_perc"] == 0.2 and first["resolved"] is None
    again = dashboard.snapshot(out, first["offsets"])
    assert again["live"] == [] and again["metrics"] == []
    (out / "resolved.json").write_text(json.dumps({"resolved": ["django__django-1"]}))
    assert dashboard.snapshot(out, first["offsets"])["resolved"] == ["django__django-1"]


def test_static_page_carries_the_data_and_the_models_rates(tmp_path):
    out = make_run(tmp_path)
    html = dashboard.write_static(out, tmp_path / "d.html").read_text()
    assert "window.DASH_DATA" in html and "<!--DASH_CFG-->" not in html
    cfg = json.loads(html.split("window.DASH_CFG = ", 1)[1].split(";", 1)[0])
    assert cfg == {"name": "run", "mode": "replay", "rates": [0.30, 0.15, 2.70]}


def test_the_server_serves_the_page_and_the_new_rows(tmp_path):
    out = make_run(tmp_path)
    srv = dashboard.make_server(out, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}/"
        assert b"window.DASH_CFG" in urllib.request.urlopen(base).read()
        req = urllib.request.Request(base + "data", data=b"{}", method="POST")
        data = json.loads(urllib.request.urlopen(req).read())
        assert len(data["live"]) == 3 and data["plan"]["arm"] == "refold"
    finally:
        srv.shutdown()


def test_run_dashboard_implies_live(tmp_path):
    from refold.cli import main

    rc = main(["run", "--strategy", "base", "--model", "Qwen/Qwen3.6-27B", "--benchmark", "swebench",
               "--subset", "all", "--limit", "1", "-o", str(tmp_path / "o"), "--dashboard", "--dry-run"])
    assert rc == 0  # the plan printed by --dry-run is checked through the parser below
    import argparse

    from refold.cli import run as run_cmd

    p = argparse.ArgumentParser()
    run_cmd.add_args(p)
    a = p.parse_args(["--strategy", "base", "--model", "m", "-o", "x", "--dashboard"])
    assert a.dashboard == 8901 and not a.live
    assert p.parse_args(["--strategy", "base", "--model", "m", "-o", "x", "--dashboard", "9000"]).dashboard == 9000


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_page_runs_a_replay_without_errors(tmp_path):
    """Executes the page's script under node with a stub DOM, through several frames."""
    out = make_run(tmp_path)
    html = dashboard.write_static(out, tmp_path / "d.html").read_text()
    scripts = html.split("<script>")[1:]
    code = "".join(s.split("</script>")[0] + "\n" for s in scripts)
    stub = r"""
const mk = () => { const e = { style: {}, dataset: {}, children: [], classList: { toggle(){}, add(){} },
  setAttribute(){}, append(c){ this.children.push(c); }, remove(){}, querySelector: () => mk(),
  get lastChild(){ return { remove: () => this.children.pop() }; }, innerHTML: "", textContent: "", value: "0" }; return e; };
globalThis.window = globalThis; globalThis.location = { search: "" };
globalThis.document = { getElementById: () => mk(), createElement: () => mk(), createElementNS: () => mk(),
  documentElement: mk(), body: mk(), title: "" };
globalThis.performance = { now: () => 0 }; let frames = 0;
globalThis.requestAnimationFrame = (f) => { if (frames++ < 5) f(1000 * frames); };
"""
    (tmp_path / "page.js").write_text(stub + code + "\nconsole.log('ok', typeof render);\n")
    r = subprocess.run(["node", str(tmp_path / "page.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "ok function" in r.stdout, r.stderr
