import json
import socket
import time
import urllib.error
import urllib.request

import pytest


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _req(port, path, body=None, headers=None):
    h = {"Host": f"127.0.0.1:{port}", **(headers or {})}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        h.setdefault("Content-Type", "application/json")
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers=h)
    with urllib.request.urlopen(r, timeout=30) as resp:
        return json.loads(resp.read())


def test_ui_end_to_end(tmp_path):
    from dqisim.ui.server import serve

    port = _free_port()
    httpd = serve(port=port, results_root=str(tmp_path / "res"), block=False)
    try:
        cat = _req(port, "/api/catalog")
        assert {"problem", "decoder", "weights", "evaluator", "baseline"} <= set(cat["components"])
        assert "quickstart" in cat["presets"]
        spec = {"name": "ui-test", "seeds": [0], "problem": {"name": "xorsat-regular", "n": 14, "k": 3, "D": 6},
                "dqi": {"ell": 2, "decoder": {"name": "bp"}, "weights": {"name": "optimal-ideal"}},
                "evaluators": [{"name": "exact"}], "baselines": [{"name": "exact"}]}
        v = _req(port, "/api/validate", spec)
        assert v["ok"] and v["points"] == 1
        job = _req(port, "/api/run", spec)["job"]
        for _ in range(120):
            d = _req(port, f"/api/jobs/{job}")
            if d["status"]["state"] in ("done", "failed", "aborted"):
                break
            time.sleep(0.5)
        assert d["status"]["state"] == "done", d.get("log_tail")
        assert d["summary"][0]["stats"]["evaluator:exact.frac"]["mean"] > 0.5
        assert d["status"]["safety"]["nice"] == 19
        with pytest.raises(urllib.error.HTTPError) as e:
            _req(port, "/api/catalog", headers={"Host": "evil.example:80"})
        assert e.value.code == 403
        with pytest.raises(urllib.error.HTTPError) as e:
            _req(port, "/api/run", spec, headers={"Content-Type": "text/plain"})
        assert e.value.code == 415
        with pytest.raises(urllib.error.HTTPError) as e:
            _req(port, "/api/validate", {**spec, "dqi": {**spec["dqi"], "decoder": {"name": "nope"}}})
        assert e.value.code == 400
    finally:
        httpd.shutdown()
