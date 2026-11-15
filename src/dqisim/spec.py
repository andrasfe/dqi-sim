"""Experiment specs: one nested dict (TOML/JSON on disk) describing the whole pipeline.

    name, seeds
    problem    = {name = "xorsat-regular", n = 120, k = 3, D = 6}
    dqi        = {ell = "auto", postselect = true, ell_auto_eps = 0.1,
                  weights = {name = "optimal-ideal"}, decoder = {name = "bp", iters = 50}}
    evaluators = [{name = "semicircle"}, {name = "mc", samples = 200}]
    baselines  = [{name = "sa", sweeps = 500}]
    sweep      = {"dqi.decoder" = ["bp", "bposd"], "problem.n" = [60, 120]}
    resources  = {memory_gb = 4, walltime_s = 3600, workers = 1}

Every field is addressable by a dotted path for `--set` and `sweep`. A string value on a
component path is shorthand: "bp:iters=30,mode=sumproduct" (lists use ';': "custom:w=1;2;3").
"""
from __future__ import annotations

import copy
import itertools
import json
import tomllib
from pathlib import Path

COMPONENT_PATHS = {"problem": "problem", "dqi.weights": "weights", "dqi.decoder": "decoder"}
LIST_PATHS = {"evaluators": "evaluator", "baselines": "baseline"}

DEFAULT_SPEC = {
    "name": "experiment",
    "seeds": [0],
    "problem": {"name": "xorsat-regular", "n": 60, "k": 3, "D": 6},
    "dqi": {
        "ell": "auto",
        "postselect": True,
        "ell_auto_eps": 0.1,
        "weights": {"name": "optimal-ideal"},
        "decoder": {"name": "bp"},
    },
    "evaluators": [{"name": "semicircle"}, {"name": "ideal"}, {"name": "mc"}],
    "baselines": [{"name": "sa"}],
    "sweep": {},
    "resources": {},
}

PRESET_DIR = Path(__file__).parent / "presets"


def parse_scalar(s: str):
    if not isinstance(s, str):
        return s
    t = s.strip()
    if t.lower() in ("true", "false"):
        return t.lower() == "true"
    for conv in (int, float):
        try:
            return conv(t)
        except ValueError:
            pass
    if t.startswith("[") or t.startswith("{"):
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            pass
    return t


def parse_component(s: str | dict) -> dict:
    """'bp:iters=30,mode=minsum' -> {'name': 'bp', 'iters': 30, 'mode': 'minsum'}."""
    if isinstance(s, dict):
        return dict(s)
    name, _, rest = str(s).partition(":")
    out = {"name": name.strip()}
    for part in filter(None, (p.strip() for p in rest.split(","))):
        k, eq, v = part.partition("=")
        if not eq:
            raise ValueError(f"bad component parameter {part!r} in {s!r} (want key=value)")
        v = v.strip()
        out[k.strip()] = [parse_scalar(x) for x in v.split(";")] if ";" in v else parse_scalar(v)
    return out


def load(path: str | Path | None) -> dict:
    spec = copy.deepcopy(DEFAULT_SPEC)
    if not path:
        return spec
    p = Path(path)
    if not p.exists() and (PRESET_DIR / f"{path}.toml").exists():
        p = PRESET_DIR / f"{path}.toml"
    data = tomllib.loads(p.read_text()) if p.suffix == ".toml" else json.loads(p.read_text())
    return merge(spec, data)


def merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k in ("problem",) or k in LIST_PATHS or k == "sweep":
            out[k] = copy.deepcopy(v)  # replace components wholesale
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            if k == "dqi":
                for kk, vv in v.items():
                    out["dqi"][kk] = copy.deepcopy(vv)
            else:
                out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_path(spec: dict, path: str, value) -> dict:
    """Set a dotted path. Component paths accept shorthand strings; a new component
    name resets that component's parameters."""
    keys = path.split(".")
    if path in COMPONENT_PATHS and not isinstance(value, dict):
        node = spec
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = parse_component(value)
        return spec
    if path in LIST_PATHS:
        vals = value if isinstance(value, list) else [v for v in str(value).split("+") if v]
        spec[path] = [parse_component(v) for v in vals]
        return spec
    node = spec
    for k in keys[:-1]:
        if isinstance(node, list):
            node = node[int(k)]
        else:
            node = node.setdefault(k, {})
    last = keys[-1]
    v = parse_scalar(value) if isinstance(value, str) else value
    if isinstance(node, list):
        node[int(last)] = v
    else:
        node[last] = v
    return spec


def expand(spec: dict) -> list[tuple[dict, dict]]:
    """Cartesian product of the sweep. Returns [(overrides, concrete spec)]."""
    sweep = spec.get("sweep") or {}
    if not sweep:
        return [({}, copy.deepcopy(spec))]
    keys = list(sweep)
    vals = [sweep[k] if isinstance(sweep[k], list) else [sweep[k]] for k in keys]
    points = []
    for combo in itertools.product(*vals):
        s = copy.deepcopy(spec)
        s["sweep"] = {}
        ov = {}
        for k, v in zip(keys, combo):
            set_path(s, k, v)
            ov[k] = v
        points.append((ov, s))
    return points


def parse_seeds(s) -> list[int]:
    """'3' -> [3]; '0,1,2' -> [0,1,2]; '0-4' -> [0..4]. Lists pass through."""
    if isinstance(s, int):
        return [s]
    if isinstance(s, list):
        return [int(x) for x in s]
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        a, dash, b = part.partition("-")
        out.extend(range(int(a), int(b) + 1) if dash else [int(a)])
    return out


def seeds(spec: dict) -> list[int]:
    out = parse_seeds(spec.get("seeds", [0]))
    if not out:
        raise ValueError("no seeds given (use e.g. 0, 0,1,2 or 0-9)")
    return out


def label_components(items: list[dict]) -> list[tuple[str, dict]]:
    """Unique display labels: explicit 'label', else name (+#i on duplicates)."""
    names = [c.get("label") or c["name"] for c in items]
    out = []
    for i, (n, c) in enumerate(zip(names, items)):
        lab = n if names.count(n) == 1 else f"{n}#{names[:i].count(n) + 1}"
        out.append((lab, c))
    return out


def presets() -> dict[str, str]:
    out = {}
    for p in sorted(PRESET_DIR.glob("*.toml")):
        first = p.read_text().splitlines()[0] if p.stat().st_size else ""
        out[p.stem] = first.lstrip("# ").strip()
    return out


def to_toml(spec: dict) -> str:
    """Minimal TOML writer for specs (scalars, lists, nested tables, arrays of tables)."""

    def val(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, list):
            return "[" + ", ".join(val(x) for x in v) + "]"
        if isinstance(v, dict):
            return "{" + ", ".join(f"{json.dumps(k) if '.' in k else k} = {val(x)}" for k, x in v.items()) + "}"
        raise TypeError(type(v))

    lines = []
    for k, v in spec.items():
        if not isinstance(v, (dict, list)) or (isinstance(v, list) and not any(isinstance(x, dict) for x in v)):
            lines.append(f"{k} = {val(v)}")
    for k, v in spec.items():
        if isinstance(v, dict):
            lines.append(f"\n[{k}]")
            for kk, vv in v.items():
                key = json.dumps(kk) if "." in kk else kk
                lines.append(f"{key} = {val(vv)}")
        elif isinstance(v, list) and any(isinstance(x, dict) for x in v):
            for item in v:
                lines.append(f"\n[[{k}]]")
                for kk, vv in item.items():
                    lines.append(f"{kk} = {val(vv)}")
    return "\n".join(lines) + "\n"
