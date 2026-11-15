"""Run an experiment spec: sweep points x seeds, evaluators + baselines, under the guard."""
from __future__ import annotations

import concurrent.futures as cf
import json
import math
import multiprocessing as mp
import os
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from . import registry, safety
from .evaluators import Context
from .spec import expand, label_components, seeds

RESULTS_ROOT = Path(os.environ.get("DQISIM_RESULTS", Path.cwd() / "results"))


def _key(d) -> str:
    return json.dumps(d, sort_keys=True, default=str)


def new_run_dir(name: str, root: Path | None = None) -> Path:
    root = Path(root or RESULTS_ROOT)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in name)[:60]
    d = root / f"{stamp}-{safe}"
    i = 1
    while d.exists():
        d = root / f"{stamp}-{safe}-{i}"
        i += 1
    return d


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def run_group(task: dict, emit=None) -> list[dict]:
    """All sweep points that share one (problem spec, seed): instance built once,
    baselines run once per distinct baseline spec."""
    registry.load_builtins()
    for p in task.get("plugins", []):
        registry.load_plugin(p)
    seed = task["seed"]
    rows = []
    t0 = time.perf_counter()
    try:
        prob = registry.build("problem", task["problem"])
        inst = prob.generate(np.random.default_rng([seed, 0]))
    except (ValueError, safety.ResourceExceeded) as e:
        rows = [{"point": pi, "overrides": ov, "seed": seed, "errors": [f"problem: {e}"]}
                for pi, ov, _ in task["points"]]
        if emit:
            emit(rows)
            return []
        return rows
    baseline_cache: dict[str, dict] = {}
    for pi, ov, spec in task["points"]:
        safety.checkpoint()
        row = {"point": pi, "overrides": ov, "seed": seed, "instance": inst.summary(),
               "evaluators": {}, "baselines": {}, "errors": []}
        dqi = spec["dqi"]
        try:
            ctx = Context(inst, np.random.default_rng([seed, 1, pi]), dqi.get("ell", "auto"),
                          registry.build("weights", dqi["weights"]), dqi["decoder"],
                          bool(dqi.get("postselect", True)), float(dqi.get("ell_auto_eps", 0.1)), probe_seed=seed)
            for label, ev in label_components(spec.get("evaluators", [])):
                t = time.perf_counter()
                try:
                    res = registry.build("evaluator", ev).run(ctx)
                    res["time_s"] = round(time.perf_counter() - t, 4)
                    row["evaluators"][label] = res
                except (ValueError, safety.ResourceExceeded, np.linalg.LinAlgError) as e:
                    if safety._state.breach:
                        raise
                    row["errors"].append(f"evaluator {label}: {e}")
            if "ell" in ctx.cache:
                row["ell"] = ctx.cache["ell"]
        except (ValueError, safety.ResourceExceeded) as e:
            if safety._state.breach:
                raise
            row["errors"].append(f"dqi: {e}")
        for label, bl in label_components(spec.get("baselines", [])):
            k = _key(bl)
            if k not in baseline_cache:
                try:
                    baseline_cache[k] = registry.build("baseline", bl).run(inst, np.random.default_rng([seed, 2]))
                except (ValueError, safety.ResourceExceeded) as e:
                    if safety._state.breach:
                        raise
                    baseline_cache[k] = {"error": str(e)}
            row["baselines"][label] = baseline_cache[k]
        row["time_s"] = round(time.perf_counter() - t0, 3)
        t0 = time.perf_counter()
        if emit:
            emit([_jsonable(row)])
        else:
            rows.append(_jsonable(row))
    return rows


def _worker_init(budget: dict):
    safety.set_budget(safety.Budget.from_dict(budget))
    safety.apply_priority()
    safety.reset_clock()


def plan(spec: dict, plugins: list[str] | None = None) -> list[dict]:
    points = expand(spec)
    groups: dict[tuple, dict] = {}
    for pi, (ov, s) in enumerate(points):
        for seed in seeds(spec):
            key = (_key(s["problem"]), seed)
            g = groups.setdefault(key, {"seed": seed, "problem": s["problem"], "points": [], "plugins": plugins or []})
            g["points"].append((pi, ov, s))
    return list(groups.values())


def run(spec: dict, out_dir: Path | None = None, plugins: list[str] | None = None, quiet: bool = False,
        on_progress=None) -> Path:
    budget = safety.Budget.from_dict(spec.get("resources"))
    out = Path(out_dir) if out_dir else new_run_dir(spec.get("name", "experiment"))
    tasks = plan(spec, plugins)
    total = sum(len(t["points"]) for t in tasks)
    with safety.guarded(budget, out) as wd:
        out.mkdir(parents=True, exist_ok=True)
        safety.safe_write(out / "spec.json", json.dumps(spec, indent=2, default=str))
        status = {"state": "running", "done": 0, "total": total, "started": time.time(), "errors": 0,
                  "safety": safety.status()}

        def write_status():
            status["safety"] = safety.status()
            status["peak_rss_mb"] = round(wd.peak_rss / safety.MB, 1)
            (out / "status.json").write_text(json.dumps(status, default=str))

        write_status()

        def consume(rows):
            for r in rows:
                safety.safe_write(out / "results.jsonl", json.dumps(r) + "\n", mode="a")
                status["done"] += 1
                status["errors"] += len(r.get("errors", []))
                if not quiet:
                    _print_row(r, status["done"], total)
            write_status()
            if on_progress:
                on_progress(status)

        try:
            if budget.workers > 1 and len(tasks) > 1:
                ctx = mp.get_context("spawn")
                with cf.ProcessPoolExecutor(budget.workers, mp_context=ctx, initializer=_worker_init,
                                            initargs=(budget.to_dict(),)) as ex:
                    futs = [ex.submit(run_group, t) for t in tasks]
                    for f in cf.as_completed(futs):
                        safety.checkpoint()
                        consume(f.result())
            else:
                for t in tasks:
                    run_group(t, emit=consume)
            status["state"] = "done"
        except safety.ResourceExceeded as e:
            status["state"] = "aborted"
            status["reason"] = str(e)
            raise
        except KeyboardInterrupt:
            status["state"] = "aborted"
            status["reason"] = safety._state.breach or "interrupted"
            raise
        except Exception as e:
            status["state"] = "failed"
            status["reason"] = f"{type(e).__name__}: {e}"
            status["traceback"] = traceback.format_exc()
            raise
        finally:
            status["finished"] = time.time()
            write_status()
            if (out / "results.jsonl").exists():
                summ = summarize(load_rows(out))
                safety.safe_write(out / "summary.json", json.dumps(summ, indent=2))
    return out


def load_rows(run_dir: Path) -> list[dict]:
    p = Path(run_dir) / "results.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def summarize(rows: list[dict]) -> list[dict]:
    """Mean and standard error per sweep point of every evaluator/baseline scalar."""
    by = defaultdict(list)
    for r in rows:
        by[r["point"]].append(r)
    out = []
    for pi in sorted(by):
        rs = by[pi]
        acc = defaultdict(list)
        for r in rs:
            for kind in ("evaluators", "baselines"):
                for lab, res in (r.get(kind) or {}).items():
                    for k, v in res.items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None:
                            acc[f"{kind[:-1]}:{lab}.{k}"].append(v)
            if "ell" in r:
                acc["ell"].append(r["ell"])
        stats = {}
        for k, vs in acc.items():
            a = np.asarray(vs, float)
            stats[k] = {"mean": float(a.mean()), "se": float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0,
                        "n": int(a.size)}
        out.append({"point": pi, "overrides": rs[0]["overrides"], "seeds": len(rs),
                    "errors": sorted({e for r in rs for e in r.get("errors", [])}), "stats": stats})
    return out


def _print_row(r: dict, done: int, total: int) -> None:
    parts = [f"[{done}/{total}]"]
    if r.get("overrides"):
        parts.append(" ".join(f"{k}={v}" for k, v in r["overrides"].items()))
    parts.append(f"seed={r['seed']}")
    if "ell" in r:
        parts.append(f"ℓ={r['ell']}")
    for lab, res in (r.get("evaluators") or {}).items():
        if "frac" in res and res["frac"] is not None:
            parts.append(f"{lab}={res['frac']:.4f}" + (f"(p_succ {res['p_succ']:.1e}!)" if res.get("warning") else ""))
    for lab, res in (r.get("baselines") or {}).items():
        if res.get("frac") is not None:
            parts.append(f"{lab}={res['frac']:.4f}")
    for e in r.get("errors", []):
        parts.append(f"!! {e}")
    print("  ".join(parts), flush=True)
