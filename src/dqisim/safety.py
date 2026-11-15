"""Resource safety: keep experiments from degrading or taking down the host.

Layers, from cheapest to last resort:

1. Priority: `nice 19` plus (macOS) background QoS, which also throttles disk I/O,
   or (Linux) idle I/O class via `ionice` when available.
2. Thread caps: set in `dqisim/__init__.py` before numpy/numba load.
3. Pre-flight estimates: every evaluator states its peak memory; work that would
   exceed the budget is refused before it allocates anything.
4. Cooperative checkpoints: long loops call `checkpoint()`, which enforces the
   wall-time budget, pauses while the machine is busy (load average) or short of
   memory, and raises `ResourceExceeded` when the budget is gone.
5. Watchdog thread: samples RSS and system memory every second; on breach it
   interrupts the main thread and hard-exits if it does not stop within a grace period.
6. Disk: outputs only go to the results directory, writes are size-checked against
   a results-directory cap and a free-space floor.
"""
from __future__ import annotations

import _thread
import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import psutil

from . import THREADS

GB = 1024**3
MB = 1024**2
MAX_WORKERS = 2  # hard cap on concurrent worker processes


class ResourceExceeded(RuntimeError):
    """Raised when a job would exceed, or has exceeded, its resource budget."""


@dataclass
class Budget:
    threads: int = THREADS
    workers: int = 1
    memory_gb: float = 4.0  # per-process RSS cap
    min_free_mem_gb: float = 8.0  # pause/abort if system available memory drops below
    min_free_disk_gb: float = 20.0  # refuse writes if free disk drops below
    results_cap_mb: float = 500.0  # cap on total size of the results directory
    walltime_s: float = 3600.0  # per job
    max_load: float = 0.75  # pause new work while 1-min load avg > max_load * ncpu
    max_pause_s: float = 300.0  # longest we wait for the machine to calm down
    nice: int = 19
    background_qos: bool = True

    def clamp(self) -> "Budget":
        self.threads = max(1, min(int(self.threads), 4))
        self.workers = max(1, min(int(self.workers), MAX_WORKERS))
        self.memory_gb = max(0.25, float(self.memory_gb))
        self.nice = max(0, min(int(self.nice), 19))
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Budget":
        d = dict(d or {})
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known).clamp()


@dataclass
class _State:
    budget: Budget = field(default_factory=Budget)
    started: float = field(default_factory=time.monotonic)
    breach: str | None = None
    priority_applied: bool = False
    pauses_s: float = 0.0


_state = _State()


def budget() -> Budget:
    return _state.budget


def set_budget(b: Budget) -> None:
    _state.budget = b.clamp()


# ----------------------------------------------------------------------------- priority


def apply_priority(b: Budget | None = None) -> dict:
    """Lower CPU and I/O priority of this process (inherited by children)."""
    b = b or _state.budget
    info = {}
    try:
        cur = os.getpriority(os.PRIO_PROCESS, 0)
        if cur < b.nice:
            os.nice(b.nice - cur)
        info["nice"] = os.getpriority(os.PRIO_PROCESS, 0)
    except OSError as e:  # pragma: no cover
        info["nice_error"] = str(e)
    if b.background_qos:
        if sys.platform == "darwin" and hasattr(os, "PRIO_DARWIN_BG"):
            try:
                os.setpriority(os.PRIO_DARWIN_PROCESS, 0, os.PRIO_DARWIN_BG)
                info["darwin_background"] = True
            except OSError as e:  # pragma: no cover
                info["darwin_background_error"] = str(e)
        elif sys.platform.startswith("linux") and shutil.which("ionice"):  # pragma: no cover
            r = subprocess.run(["ionice", "-c3", "-p", str(os.getpid())], capture_output=True)
            info["ionice_idle"] = r.returncode == 0
    _state.priority_applied = True
    return info


# ----------------------------------------------------------------------------- memory


def rss_bytes() -> int:
    return psutil.Process().memory_info().rss


def available_bytes() -> int:
    return psutil.virtual_memory().available


def preflight(bytes_needed: float, what: str) -> None:
    """Refuse work whose estimated peak memory does not fit in the budget."""
    b = _state.budget
    cap = b.memory_gb * GB
    have = cap - rss_bytes()
    if bytes_needed > have:
        raise ResourceExceeded(
            f"{what}: needs ~{bytes_needed / MB:,.0f} MB but only {max(have, 0) / MB:,.0f} MB "
            f"left of the {b.memory_gb:g} GB budget (raise resources.memory_gb or shrink the instance)"
        )
    sys_free = available_bytes() - b.min_free_mem_gb * GB
    if bytes_needed > sys_free:
        raise ResourceExceeded(
            f"{what}: needs ~{bytes_needed / MB:,.0f} MB but the system would drop below its "
            f"{b.min_free_mem_gb:g} GB free-memory floor"
        )


# ----------------------------------------------------------------------------- checkpoints


def _busy() -> str | None:
    b = _state.budget
    try:
        load1 = os.getloadavg()[0]
    except OSError:  # pragma: no cover
        load1 = 0.0
    ncpu = os.cpu_count() or 1
    if load1 > b.max_load * ncpu:
        return f"load {load1:.1f} > {b.max_load:.2f} x {ncpu} cores"
    if available_bytes() < b.min_free_mem_gb * GB:
        return f"system free memory below {b.min_free_mem_gb:g} GB"
    return None


_last_busy_check = [0.0]


def checkpoint() -> None:
    """Call from long loops. Cheap: real checks run at most every 0.5 s."""
    if _state.breach:
        raise ResourceExceeded(_state.breach)
    now = time.monotonic()
    if now - _last_busy_check[0] < 0.5:
        return
    _last_busy_check[0] = now
    b = _state.budget
    if now - _state.started > b.walltime_s:
        raise ResourceExceeded(f"wall-time budget of {b.walltime_s:g} s exhausted")
    reason = _busy()
    if reason:
        waited = 0.0
        while reason and waited < b.max_pause_s:
            time.sleep(2.0)
            waited += 2.0
            reason = _busy()
        _state.pauses_s += waited
        if reason:
            raise ResourceExceeded(f"machine stayed busy for {b.max_pause_s:g} s ({reason})")


def reset_clock() -> None:
    _state.started = time.monotonic()
    _state.breach = None


# ----------------------------------------------------------------------------- watchdog


class Watchdog(threading.Thread):
    def __init__(self, interval: float = 1.0, grace: float = 10.0):
        super().__init__(daemon=True, name="dqisim-watchdog")
        self.interval = interval
        self.grace = grace
        self._stop = threading.Event()
        self.peak_rss = 0

    def run(self) -> None:
        b = _state.budget
        while not self._stop.wait(self.interval):
            rss = rss_bytes()
            self.peak_rss = max(self.peak_rss, rss)
            msg = None
            if rss > b.memory_gb * GB:
                msg = f"RSS {rss / MB:,.0f} MB exceeded the {b.memory_gb:g} GB budget"
            elif available_bytes() < 0.5 * b.min_free_mem_gb * GB:
                msg = f"system free memory fell below {0.5 * b.min_free_mem_gb:g} GB"
            if msg:
                _state.breach = msg
                sys.stderr.write(f"[dqisim watchdog] {msg}; stopping\n")
                _thread.interrupt_main()
                if self._stop.wait(self.grace):
                    return
                sys.stderr.write("[dqisim watchdog] main thread did not stop; hard exit\n")
                os._exit(3)

    def stop(self) -> None:
        self._stop.set()


@contextmanager
def guarded(b: Budget | None = None, results_dir: Path | None = None):
    """Apply priority, start the watchdog, check the disk; yields the Watchdog."""
    if b is not None:
        set_budget(b)
    apply_priority()
    reset_clock()
    if results_dir is not None:
        check_disk(results_dir, 0)
    wd = Watchdog()
    wd.start()
    try:
        yield wd
    except KeyboardInterrupt:
        if _state.breach:
            raise ResourceExceeded(_state.breach) from None
        raise
    finally:
        wd.stop()


# ----------------------------------------------------------------------------- disk


_size_cache: dict[Path, tuple[int, float]] = {}


def dir_size(path: Path) -> int:
    total = 0
    if path.exists():
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    return total


def check_disk(results_dir: Path, bytes_to_write: int) -> None:
    b = _state.budget
    results_dir = Path(results_dir)
    probe = results_dir if results_dir.exists() else results_dir.parent
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free - bytes_to_write < b.min_free_disk_gb * GB:
        raise ResourceExceeded(f"disk free {free / GB:.1f} GB would fall below {b.min_free_disk_gb:g} GB")
    root = results_dir.parent if results_dir.name else results_dir
    now = time.monotonic()
    cached = _size_cache.get(root)
    if cached and now - cached[1] < 10.0:
        used = cached[0]
    else:
        used = dir_size(root) if root.exists() else 0
    _size_cache[root] = (used + bytes_to_write, cached[1] if cached and now - cached[1] < 10.0 else now)
    if used + bytes_to_write > b.results_cap_mb * MB:
        raise ResourceExceeded(
            f"results directory {root} would exceed its {b.results_cap_mb:g} MB cap "
            f"({used / MB:.1f} MB used); prune old runs or raise resources.results_cap_mb"
        )


def safe_write(path: Path, text: str, mode: str = "w") -> None:
    data = text.encode()
    check_disk(Path(path).parent, len(data))
    with open(path, mode) as f:
        f.write(text)


def status() -> dict:
    b = _state.budget
    try:
        load = os.getloadavg()
    except OSError:  # pragma: no cover
        load = (0, 0, 0)
    return {
        "pid": os.getpid(),
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "threads_env": os.environ.get("OMP_NUM_THREADS"),
        "rss_mb": round(rss_bytes() / MB, 1),
        "system_available_gb": round(available_bytes() / GB, 1),
        "load_avg": [round(x, 2) for x in load],
        "cpus": os.cpu_count(),
        "budget": b.to_dict(),
        "paused_s": _state.pauses_s,
    }
