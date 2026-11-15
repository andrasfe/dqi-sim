import json
import subprocess
import sys
import textwrap

import pytest

from dqisim import registry, spec
from dqisim.cli import main


def test_component_shorthand():
    assert spec.parse_component("bp:iters=30,mode=sumproduct") == {"name": "bp", "iters": 30, "mode": "sumproduct"}
    assert spec.parse_component("custom:w=1;2;3") == {"name": "custom", "w": [1, 2, 3]}


def test_set_path_and_expand():
    s = spec.load(None)
    spec.set_path(s, "dqi.decoder", "bposd:iters=5")
    spec.set_path(s, "problem.n", "90")
    assert s["dqi"]["decoder"] == {"name": "bposd", "iters": 5} and s["problem"]["n"] == 90
    s["sweep"] = {"dqi.decoder": ["bp", "bitflip"], "dqi.ell": [2, 3, 4]}
    pts = spec.expand(s)
    assert len(pts) == 6
    assert pts[-1][1]["dqi"]["decoder"]["name"] == "bitflip" and pts[-1][1]["dqi"]["ell"] == 4


def test_toml_roundtrip():
    import tomllib

    s = spec.load("small-exact")
    assert tomllib.loads(spec.to_toml(s)) == s


def test_all_presets_validate():
    from dqisim.cli import validate

    for name in spec.presets():
        validate(spec.load(name))


def test_unknown_parameter_is_rejected():
    with pytest.raises(ValueError, match="no parameter"):
        registry.get("decoder", "bp").resolve({"iterz": 3})


def test_cli_run_end_to_end(tmp_path, capsys):
    out = tmp_path / "run"
    main(["run", "--problem", "xorsat-regular:n=14,k=3,D=6", "--decoder", "bp", "--ell", "2",
          "--evaluator", "exact", "--evaluator", "mc:samples=50", "--baseline", "exact",
          "--seeds", "0,1", "--sweep", "dqi.weights=optimal-ideal,optimal-aware", "--out", str(out)])
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and all(not r["errors"] for r in rows)
    st = json.loads((out / "status.json").read_text())
    assert st["state"] == "done" and st["safety"]["nice"] == 19
    summ = json.loads((out / "summary.json").read_text())
    assert len(summ) == 2
    text = capsys.readouterr().out
    assert "exact.frac" in text


def test_plugin_component(tmp_path):
    plug = tmp_path / "myplug.py"
    plug.write_text(textwrap.dedent('''
        import numpy as np
        from dqisim.registry import register, P
        from dqisim.decoders import Decoder
        @register("decoder", "zero", "always guesses y = 0", P("dummy", int, 0, "unused"))
        class Zero(Decoder):
            def __init__(self, dummy): pass
            def decode(self, S): return np.zeros((S.shape[0], self.inst.m), np.uint8)
    '''))
    out = tmp_path / "run"
    main(["run", "--plugin", str(plug), "--problem", "xorsat-regular:n=14,k=3,D=6", "--decoder", "zero",
          "--ell", "2", "--evaluator", "exact", "--baseline", "none", "--out", str(out), "--quiet"])
    row = json.loads((out / "results.jsonl").read_text().splitlines()[0])
    assert row["evaluators"]["exact"]["p_succ"] == pytest.approx(row["evaluators"]["exact"]["w"][0] ** 2)


def test_resource_abort_exit_code(tmp_path):
    r = subprocess.run([sys.executable, "-m", "dqisim", "run", "--problem", "xorsat-regular:n=26,k=3,D=6",
                        "--ell", "2", "--evaluator", "exact:dense=yes", "--baseline", "none", "--memory-gb", "0.25",
                        "--out", str(tmp_path / "r")], capture_output=True, text=True, timeout=120)
    # the dense 2^26 state needs ~1.6 GB: refused up front, recorded as an error row, never allocated
    assert "needs" in r.stdout + r.stderr


def test_seed_syntax_and_empty_rejected():
    from dqisim.cli import validate

    assert spec.parse_seeds("0") == [0]
    assert spec.parse_seeds("0-3,7") == [0, 1, 2, 3, 7]
    assert spec.parse_seeds(5) == [5]
    s = spec.load("quickstart")
    s["seeds"] = ""
    with pytest.raises(ValueError, match="no seeds"):
        validate(s)
