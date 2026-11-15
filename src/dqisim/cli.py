"""Command-line interface.

  dqisim list [kind]                     components and their parameters
  dqisim presets                         bundled experiment specs
  dqisim spec [SPEC] [overrides]         print the resolved spec as TOML (save, edit, rerun)
  dqisim run  [SPEC] [overrides]         run (sweeps x seeds), results/<stamp>-<name>/
  dqisim report RUN_DIR [--metric M]     table of means ± s.e. per sweep point
  dqisim ui [--port 8765]                local web UI (127.0.0.1 only)
  dqisim status                          what the safety layer sees right now

Overrides (all combine with a SPEC file or preset):
  --problem xorsat-regular:n=200,k=3,D=6   --decoder bposd:iters=20   --weights product:t=0.4
  --evaluator mc:samples=400 (repeatable)  --baseline sa:sweeps=2000 (repeatable)
  --ell auto|12|0.08   --seeds 0,1,2 | 0-9   --set dqi.postselect=false (any dotted path)
  --sweep dqi.decoder=bp,bposd,bitflip     --sweep problem.n=50,100,200
  --memory-gb 4 --walltime 3600 --workers 1 --threads 2 --plugin my_components.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import registry, safety
from . import spec as specmod


def _split_list(s: str) -> list:
    # split on commas that are not inside a component's parameter list ("bp:iters=3,mode=x")
    if ":" in s:
        return [x for x in s.split("|") if x]
    return [specmod.parse_scalar(x) for x in s.split(",") if x != ""]


def build_spec(a) -> dict:
    for p in a.plugin or []:
        registry.load_plugin(p)
    s = specmod.load(a.spec)
    if a.name:
        s["name"] = a.name
    if a.problem:
        specmod.set_path(s, "problem", a.problem)
    if a.decoder:
        specmod.set_path(s, "dqi.decoder", a.decoder)
    if a.weights:
        specmod.set_path(s, "dqi.weights", a.weights)
    if a.ell is not None:
        specmod.set_path(s, "dqi.ell", a.ell)
    if a.evaluator:
        s["evaluators"] = [specmod.parse_component(e) for e in a.evaluator]
    if a.baseline:
        s["baselines"] = [] if a.baseline == ["none"] else [specmod.parse_component(b) for b in a.baseline]
    if a.seeds:
        s["seeds"] = specmod.parse_seeds(a.seeds)
    for kv in a.set or []:
        k, _, v = kv.partition("=")
        specmod.set_path(s, k.strip(), v)
    for kv in a.sweep or []:
        k, _, v = kv.partition("=")
        s.setdefault("sweep", {})[k.strip()] = _split_list(v)
    res = s.setdefault("resources", {})
    for flag, key in (("memory_gb", "memory_gb"), ("walltime", "walltime_s"), ("workers", "workers")):
        val = getattr(a, flag, None)
        if val is not None:
            res[key] = val
    validate(s)
    return s


def validate(s: dict) -> None:
    """Fail fast on unknown components/parameters and empty specs before any work starts."""
    specmod.seeds(s)
    for _, pt in specmod.expand(s):
        registry.get("problem", pt["problem"]["name"]).resolve(pt["problem"])
        registry.get("weights", pt["dqi"]["weights"]["name"]).resolve(pt["dqi"]["weights"])
        registry.get("decoder", pt["dqi"]["decoder"]["name"]).resolve(pt["dqi"]["decoder"])
        for e in pt.get("evaluators", []):
            registry.get("evaluator", e["name"]).resolve(e)
        for b in pt.get("baselines", []):
            registry.get("baseline", b["name"]).resolve(b)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("spec", nargs="?", help="spec file (.toml/.json) or preset name")
    p.add_argument("--name")
    p.add_argument("--problem")
    p.add_argument("--decoder")
    p.add_argument("--weights")
    p.add_argument("--ell")
    p.add_argument("--evaluator", action="append")
    p.add_argument("--baseline", action="append", help="repeatable; 'none' for no baselines")
    p.add_argument("--seeds")
    p.add_argument("--set", action="append", metavar="PATH=VALUE")
    p.add_argument("--sweep", action="append", metavar="PATH=V1,V2,...",
                   help="component values use '|' as separator: dqi.decoder='bp:iters=10|bposd'")
    p.add_argument("--plugin", action="append")
    p.add_argument("--memory-gb", dest="memory_gb", type=float)
    p.add_argument("--walltime", type=float)
    p.add_argument("--workers", type=int)
    p.add_argument("--threads", type=int, help="BLAS/numba threads (max 4); re-executes with DQISIM_THREADS")


def cmd_list(a) -> None:
    for p in a.plugin or []:
        registry.load_plugin(p)
    cat = registry.catalog()
    for kind in registry.KINDS:
        if a.kind and a.kind != kind:
            continue
        print(f"\n== {kind} ==")
        for c in cat[kind]:
            print(f"  {c['name']:<16} {c['help']}")
            for p in c["params"]:
                extra = f" {p['choices']}" if p.get("choices") else ""
                print(f"      {p['name']:<14} {p['type']:<12} = {p['default']!r:<12} {p['help']}{extra}")


def cmd_presets(a) -> None:
    for k, v in specmod.presets().items():
        print(f"  {k:<26} {v}")


def cmd_spec(a) -> None:
    print(specmod.to_toml(build_spec(a)), end="")


def cmd_run(a) -> None:
    from . import runner

    s = build_spec(a)
    n_pts = len(specmod.expand(s)) * len(specmod.seeds(s))
    b = safety.Budget.from_dict(s.get("resources"))
    print(f"dqisim run '{s['name']}': {n_pts} point-seeds | nice {b.nice}, threads {os.environ.get('OMP_NUM_THREADS')}, "
          f"workers {b.workers}, memory {b.memory_gb:g} GB, walltime {b.walltime_s:g} s", flush=True)
    if a.dry_run:
        print(specmod.to_toml(s))
        return
    try:
        out = runner.run(s, Path(a.out) if a.out else None, plugins=a.plugin, quiet=a.quiet)
    except safety.ResourceExceeded as e:
        print(f"ABORTED (resource guard): {e}", file=sys.stderr)
        sys.exit(2)
    print(f"\nresults: {out}")
    _report(out, None)


def _report(run_dir: Path, metric: str | None) -> None:
    from . import runner

    summ = runner.summarize(runner.load_rows(run_dir))
    if not summ:
        print("no rows")
        return
    keys = sorted({k for s in summ for k in s["stats"]})
    if metric:
        keys = [k for k in keys if metric in k]
    else:
        keys = [k for k in keys if k.endswith(".frac") or k == "ell" or k.endswith(".p_succ")]
    ovkeys = sorted({k for s in summ for k in s["overrides"]})
    head = ovkeys + ["seeds"] + [k.replace("evaluator:", "").replace("baseline:", "B:") for k in keys]
    rows = []
    for s in summ:
        r = [str(s["overrides"].get(k, "")) for k in ovkeys] + [str(s["seeds"])]
        for k in keys:
            st = s["stats"].get(k)
            r.append("" if not st else (f"{st['mean']:.4f}" + (f"±{st['se']:.4f}" if st["n"] > 1 else "")))
        rows.append(r)
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(head)]
    print("  ".join(h.ljust(w) for h, w in zip(head, widths)))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    errs = sorted({e for s in summ for e in s["errors"]})
    for e in errs:
        print(f"  !! {e}")


def cmd_report(a) -> None:
    if a.json:
        from . import runner

        print(json.dumps(runner.summarize(runner.load_rows(Path(a.run_dir))), indent=2))
    else:
        _report(Path(a.run_dir), a.metric)


def cmd_ui(a) -> None:
    from .ui.server import serve

    serve(port=a.port, plugins=a.plugin or [], results_root=a.results)


def cmd_status(a) -> None:
    safety.apply_priority()
    print(json.dumps(safety.status(), indent=2))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="dqisim", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list", help="list components")
    p.add_argument("kind", nargs="?", choices=registry.KINDS)
    p.add_argument("--plugin", action="append")
    p.set_defaults(fn=cmd_list)
    p = sub.add_parser("presets", help="list bundled specs")
    p.set_defaults(fn=cmd_presets)
    p = sub.add_parser("spec", help="print resolved spec as TOML")
    _common(p)
    p.set_defaults(fn=cmd_spec)
    p = sub.add_parser("run", help="run an experiment")
    _common(p)
    p.add_argument("--out", help="output directory (default results/<stamp>-<name>)")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("report", help="summarise a run directory")
    p.add_argument("run_dir")
    p.add_argument("--metric", help="substring filter on metric names, e.g. p_succ or frac_nopost")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("ui", help="local web UI")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--plugin", action="append")
    p.add_argument("--results", help="results root (default ./results)")
    p.set_defaults(fn=cmd_ui)
    p = sub.add_parser("status", help="show priority/limits")
    p.set_defaults(fn=cmd_status)
    a = ap.parse_args(argv)

    threads = getattr(a, "threads", None)
    if threads and str(min(max(threads, 1), 4)) != os.environ.get("OMP_NUM_THREADS"):
        os.environ["DQISIM_THREADS"] = str(threads)
        os.execv(sys.executable, [sys.executable, "-m", "dqisim", *(argv if argv is not None else sys.argv[1:])])
    try:
        a.fn(a)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
