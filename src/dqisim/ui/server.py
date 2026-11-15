"""Local web UI: forms generated from the component registry, jobs run as niced subprocesses.

Binds to 127.0.0.1 only; rejects foreign Host headers (DNS rebinding) and non-JSON POSTs
(cross-site form posts). At most `max_jobs` (default 1, hard cap 2) experiments run at once;
further submissions queue.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .. import registry, safety
from .. import spec as specmod

HERE = Path(__file__).parent


class JobManager:
    def __init__(self, results_root: Path, plugins: list[str], max_jobs: int = 1):
        self.root = results_root
        self.plugins = plugins
        self.max_jobs = max(1, min(max_jobs, safety.MAX_WORKERS))
        self.procs: dict[str, subprocess.Popen] = {}
        self.queue: list[tuple[str, Path]] = []
        self.lock = threading.Lock()
        threading.Thread(target=self._pump, daemon=True).start()

    def submit(self, spec: dict) -> str:
        from ..runner import new_run_dir

        out = new_run_dir(spec.get("name", "ui"), self.root)
        safety.check_disk(out, 0)
        out.mkdir(parents=True)
        (out / "spec.json").write_text(json.dumps(spec, indent=2))
        (out / "status.json").write_text(json.dumps({"state": "queued", "done": 0, "total": 0}))
        with self.lock:
            self.queue.append((out.name, out))
        return out.name

    def _launch(self, name: str, out: Path) -> None:
        cmd = [sys.executable, "-m", "dqisim", "run", str(out / "spec.json"), "--out", str(out), "--quiet"]
        for p in self.plugins:
            cmd += ["--plugin", p]
        env = dict(os.environ)
        threads = spec_threads(out)
        if threads:
            env["DQISIM_THREADS"] = str(threads)
        log = open(out / "log.txt", "w")
        self.procs[name] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)  # inherits nice/QoS; child re-applies its own limits

    def _pump(self) -> None:
        while True:
            with self.lock:
                for n, p in list(self.procs.items()):
                    if p.poll() is not None:
                        del self.procs[n]
                        st = read_json(self.root / n / "status.json") or {}
                        if st.get("state") in ("running", "queued"):
                            st.update(state="failed", reason=f"exit code {p.returncode} (see log.txt)")
                            (self.root / n / "status.json").write_text(json.dumps(st))
                while self.queue and len(self.procs) < self.max_jobs:
                    self._launch(*self.queue.pop(0))
            time.sleep(0.5)

    def cancel(self, name: str) -> bool:
        with self.lock:
            for i, (n, _) in enumerate(self.queue):
                if n == name:
                    self.queue.pop(i)
                    _mark(self.root / name, "cancelled")
                    return True
            p = self.procs.get(name)
        if p:
            p.terminate()
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
            _mark(self.root / name, "cancelled")
            return True
        return False


def _mark(d: Path, state: str):
    st = read_json(d / "status.json") or {}
    st["state"] = state
    (d / "status.json").write_text(json.dumps(st))


def spec_threads(out: Path) -> int | None:
    s = read_json(out / "spec.json") or {}
    t = (s.get("resources") or {}).get("threads")
    return int(t) if t else None


def read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def make_handler(jobs: JobManager, port: int):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _guard(self) -> bool:
            if self.headers.get("Host") not in allowed_hosts:
                self._send(HTTPStatus.FORBIDDEN, {"error": "bad host"})
                return False
            return True

        def do_GET(self):
            if not self._guard():
                return
            path = urlparse(self.path).path
            try:
                if path == "/":
                    return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
                if path == "/api/catalog":
                    return self._send(200, {"components": registry.catalog(), "presets": specmod.presets(),
                                            "default": specmod.DEFAULT_SPEC, "budget": safety.Budget().to_dict()})
                if path.startswith("/api/preset/"):
                    return self._send(200, specmod.load(path.rsplit("/", 1)[1]))
                if path == "/api/status":
                    return self._send(200, safety.status())
                if path == "/api/jobs":
                    return self._send(200, list_jobs(jobs))
                if path.startswith("/api/jobs/"):
                    return self._send(200, job_detail(jobs.root / Path(path).name))
            except Exception as e:  # noqa: BLE001
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._guard():
                return
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                return self._send(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "JSON only"})
            n = int(self.headers.get("Content-Length") or 0)
            if n > 1_000_000:
                return self._send(413, {"error": "too large"})
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                path = urlparse(self.path).path
                if path == "/api/validate":
                    s = specmod.merge(specmod.DEFAULT_SPEC, body)
                    from ..cli import validate

                    validate(s)
                    return self._send(200, {"ok": True, "points": len(specmod.expand(s)) * len(specmod.seeds(s)),
                                            "toml": specmod.to_toml(s)})
                if path == "/api/run":
                    s = specmod.merge(specmod.DEFAULT_SPEC, body)
                    from ..cli import validate

                    validate(s)
                    return self._send(200, {"job": jobs.submit(s)})
                if path.startswith("/api/jobs/") and path.endswith("/cancel"):
                    return self._send(200, {"ok": jobs.cancel(Path(path).parent.name)})
            except (ValueError, safety.ResourceExceeded, KeyError, TypeError) as e:
                return self._send(400, {"error": str(e)})
            self._send(404, {"error": "not found"})

    return H


def list_jobs(jobs: JobManager) -> list[dict]:
    out = []
    if jobs.root.exists():
        for d in sorted(jobs.root.iterdir(), reverse=True):
            if d.is_dir() and (d / "status.json").exists():
                st = read_json(d / "status.json") or {}
                out.append({"id": d.name, "state": st.get("state"), "done": st.get("done"), "total": st.get("total")})
    return out[:200]


def job_detail(d: Path) -> dict:
    from ..runner import load_rows, summarize

    if not d.exists():
        return {"error": "no such job"}
    rows = load_rows(d)
    log = d / "log.txt"
    return {"id": d.name, "status": read_json(d / "status.json"), "spec": read_json(d / "spec.json"),
            "summary": summarize(rows) if rows else [],
            "log_tail": log.read_text()[-4000:] if log.exists() else ""}


def serve(port: int = 8765, plugins: list[str] | None = None, results_root: str | None = None,
          max_jobs: int = 1, block: bool = True):
    from ..runner import RESULTS_ROOT

    safety.apply_priority()
    for p in plugins or []:
        registry.load_plugin(p)
    registry.load_builtins()
    root = Path(results_root) if results_root else RESULTS_ROOT
    root.mkdir(parents=True, exist_ok=True)
    jobs = JobManager(root, plugins or [], max_jobs)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(jobs, port))
    httpd.jobs = jobs
    print(f"dqisim UI on http://127.0.0.1:{port}  (results in {root}; Ctrl-C to stop)", flush=True)
    if not block:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return httpd
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for p in list(jobs.procs.values()):
            p.terminate()
