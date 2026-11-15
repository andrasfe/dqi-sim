"""Component registry: the single source of truth for what can be swapped.

Each pipeline stage is a *kind* (problem, weights, decoder, evaluator, baseline).
A component registers a factory plus typed parameter specs. The CLI (`dqisim list`,
`--decoder bp:iters=30`) and the web UI (forms generated from `/api/components`)
both read the same specs, so a new component shows up everywhere once registered.

Add your own without touching the package:

    # my_decoder.py
    from dqisim.registry import register, P
    @register("decoder", "my-bp", "BP with my twist", P("iters", int, 30, "iterations"))
    class MyBP: ...

    dqisim run spec.toml --plugin my_decoder.py
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

KINDS = ("problem", "weights", "decoder", "evaluator", "baseline")


@dataclass
class P:
    """Parameter spec."""

    name: str
    type: type | str
    default: Any
    help: str = ""
    choices: list | None = None
    min: float | None = None
    max: float | None = None

    def coerce(self, value: Any) -> Any:
        if value is None:
            return None
        t = self.type
        if t == "int|float|str":  # e.g. ell: 12, 0.05 (fraction of m) or "auto"
            if isinstance(value, (int, float)):
                return value
            s = str(value).strip()
            for conv in (int, float):
                try:
                    return conv(s)
                except ValueError:
                    pass
            return s
        if t == "list[float]":
            if isinstance(value, str):
                return [float(x) for x in value.replace(";", ",").split(",") if x.strip()]
            return [float(x) for x in value]
        if t is bool:
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if t is int:
            return int(float(value))
        if t is float:
            return float(value)
        if t is str:
            v = str(value)
            if self.choices and v not in self.choices:
                raise ValueError(f"{self.name}={v!r} not in {self.choices}")
            return v
        return value

    def schema(self) -> dict:
        t = self.type if isinstance(self.type, str) else self.type.__name__
        d = {"name": self.name, "type": t, "default": self.default, "help": self.help}
        if self.choices:
            d["choices"] = self.choices
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        return d


@dataclass
class Component:
    kind: str
    name: str
    help: str
    factory: Callable
    params: list[P] = field(default_factory=list)

    def resolve(self, given: dict) -> dict:
        given = dict(given)
        given.pop("name", None)
        given.pop("label", None)
        out = {}
        names = {p.name for p in self.params}
        unknown = set(given) - names
        if unknown:
            raise ValueError(
                f"{self.kind} '{self.name}' has no parameter(s) {sorted(unknown)}; "
                f"valid: {sorted(names)}"
            )
        for p in self.params:
            out[p.name] = p.coerce(given[p.name]) if p.name in given else p.default
        return out

    def build(self, given: dict):
        params = self.resolve(given)
        obj = self.factory(**params)
        try:
            obj.params = params
            obj.component = self.name
        except AttributeError:  # pragma: no cover
            pass
        return obj

    def schema(self) -> dict:
        return {"name": self.name, "help": self.help, "params": [p.schema() for p in self.params]}


_REGISTRY: dict[str, dict[str, Component]] = {k: {} for k in KINDS}


def register(kind: str, name: str, help: str = "", *params: P):
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")

    def deco(factory):
        _REGISTRY[kind][name] = Component(kind, name, help, factory, list(params))
        return factory

    return deco


def get(kind: str, name: str) -> Component:
    load_builtins()
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        raise ValueError(f"unknown {kind} {name!r}; available: {sorted(_REGISTRY[kind])}") from None


def build(kind: str, spec: dict | str):
    if isinstance(spec, str):
        spec = {"name": spec}
    return get(kind, spec["name"]).build(spec)


def catalog() -> dict:
    load_builtins()
    return {k: [c.schema() for c in sorted(v.values(), key=lambda c: c.name)] for k, v in _REGISTRY.items()}


_loaded = [False]


def load_builtins() -> None:
    if _loaded[0]:
        return
    _loaded[0] = True
    for mod in ("problems", "weights", "decoders", "evaluators", "baselines"):
        importlib.import_module(f"dqisim.{mod}")


def load_plugin(path: str) -> None:
    """Import a user module (file path or dotted module name) that calls `register`."""
    load_builtins()
    p = Path(path)
    if p.suffix == ".py" and p.exists():
        spec = importlib.util.spec_from_file_location(f"dqisim_plugin_{p.stem}", p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    else:
        importlib.import_module(path)
