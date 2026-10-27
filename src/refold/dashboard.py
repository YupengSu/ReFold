"""refold dashboard -- a live page for one run, from the logs `refold run --live` writes.

    refold dashboard OUT [--port 8901]          # serve it; reach it with ssh -L 8901:127.0.0.1:8901 HOST
    refold dashboard OUT --static OUT.html      # one file with the data inside; replays the run
    refold run ... --dashboard [PORT]           # log live and serve this page while the run lasts

The page polls OUT every two seconds and reads only what was appended since the last poll:
OUT/live/<task>.jsonl (one line per model call, refold/live.py), OUT/server_metrics.jsonl
(the server's /metrics) and plan.json, plus resolved.json once the cell is scored. It
shows every session's context as it grows (and folds, under ReFold), the KV-cache pool,
the request queue, spend so far, and three time series. It serves on 127.0.0.1 only.
"""

from __future__ import annotations

import argparse
import http.server
import json
import socketserver
import threading
from pathlib import Path

PAGE = Path(__file__).with_name("dashboard.html")
DEFAULT_PORT = 8901

# $ per million tokens (uncached input, cached input, completion): the rates `refold report` uses.
RATES = {"qwen3.6-35b-a3b": (0.10, 0.05, 0.90), "qwen3.6-27b": (0.30, 0.15, 2.70)}


def _read_from(path: Path, off: int) -> tuple[int, list[dict]]:
    """Complete lines appended after byte `off`; a half-written last line waits for the next poll."""
    if not path.exists():
        return off, []
    with path.open("rb") as f:
        f.seek(off)
        chunk = f.read()
    end = chunk.rfind(b"\n") + 1
    rows = []
    for line in chunk[:end].splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return off + end, rows


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def snapshot(out: Path, offsets: dict[str, int]) -> dict:
    """What was appended since `offsets`, and the new offsets."""
    calls, metrics, new = [], [], {}
    for p in sorted((out / "live").glob("*.jsonl")) + [out / "server_metrics.jsonl"]:
        key = str(p.relative_to(out))
        new[key], rows = _read_from(p, offsets.get(key, 0))
        (metrics if p.name == "server_metrics.jsonl" else calls).extend(rows)
    resolved = _json(out / "resolved.json")
    return {"plan": _json(out / "plan.json"), "live": calls, "metrics": metrics, "offsets": new,
            "resolved": resolved.get("resolved") if resolved else None}


def _config(out: Path, mode: str) -> dict:
    model = str(_json(out / "plan.json").get("model_name", "")).lower()
    rates = next((v for k, v in RATES.items() if k in model), None)
    return {"name": out.resolve().name, "mode": mode, "rates": rates}


def page(out: Path, mode: str, data: dict | None = None) -> str:
    inject = f"<script>window.DASH_CFG = {json.dumps(_config(out, mode))};"
    if data is not None:
        inject += f"window.DASH_DATA = {json.dumps(data, separators=(',', ':'))};"
    return PAGE.read_text(encoding="utf-8").replace("<!--DASH_CFG-->", inject + "</script>")


def write_static(out: Path, dest: Path) -> Path:
    dest.write_text(page(out, "replay", snapshot(out, {})), encoding="utf-8")
    return dest


def make_server(out: Path, port: int) -> socketserver.ThreadingTCPServer:
    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(page(out, "live").encode(), "text/html; charset=utf-8")

        def do_POST(self):  # /data: the offsets the page holds in, the new rows out
            n = int(self.headers.get("Content-Length") or 0)
            offsets = json.loads(self.rfile.read(n) or b"{}")
            self._send(json.dumps(snapshot(out, offsets), separators=(",", ":")).encode(), "application/json")

        def log_message(self, *a):
            pass

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    return Server(("127.0.0.1", port), Handler)


def serve_in_background(out: Path, port: int) -> socketserver.ThreadingTCPServer:
    """For `refold run --dashboard`: the page lives as long as the run's process."""
    srv = make_server(out, port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("out", type=Path, help="the run's output directory (refold run -o)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--static", type=Path, metavar="OUT.html", help="write one replayable file instead of serving")


def main(args) -> int:
    if not args.out.is_dir():
        raise FileNotFoundError(f"{args.out} is not a directory")
    if args.static:
        dest = write_static(args.out, args.static)
        print(f"wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return 0
    with make_server(args.out, args.port) as srv:
        print(f"dashboard: http://127.0.0.1:{args.port}/  ({args.out})", flush=True)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0
