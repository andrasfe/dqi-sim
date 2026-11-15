import os
import subprocess
import sys

import pytest

from dqisim import safety


def test_thread_caps_set_before_numpy():
    out = subprocess.run([sys.executable, "-c", "import dqisim, os; print(os.environ['OMP_NUM_THREADS'], os.environ['NUMBA_NUM_THREADS'])"],
                         capture_output=True, text=True, env={**os.environ, "DQISIM_THREADS": "99"})
    assert out.stdout.split() == ["4", "4"]  # hard cap


def test_priority_applied_in_child():
    code = "from dqisim import safety; safety.apply_priority(); import os; print(os.getpriority(os.PRIO_PROCESS, 0))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "19"


def test_preflight_refuses_oversized_allocation():
    with pytest.raises(safety.ResourceExceeded):
        safety.preflight(10 * safety.GB, "test")


def test_walltime_enforced():
    safety.set_budget(safety.Budget(walltime_s=0.0, max_load=10.0, min_free_mem_gb=1))
    safety.reset_clock()
    safety._last_busy_check[0] = 0
    with pytest.raises(safety.ResourceExceeded):
        safety.checkpoint()


def test_results_cap_enforced(tmp_path):
    safety.set_budget(safety.Budget(results_cap_mb=0.001, min_free_disk_gb=1))
    safety._size_cache.clear()
    d = tmp_path / "res" / "run"
    d.mkdir(parents=True)
    with pytest.raises(safety.ResourceExceeded):
        safety.safe_write(d / "x.txt", "x" * 5000)


def test_budget_clamps_workers_and_threads():
    b = safety.Budget.from_dict({"workers": 16, "threads": 64})
    assert b.workers == 2 and b.threads == 4


def test_watchdog_kills_runaway_memory():
    code = (
        "from dqisim import safety\n"
        "import numpy as np, time\n"
        "b = safety.Budget(memory_gb=0.3, min_free_mem_gb=1)\n"
        "try:\n"
        "    with safety.guarded(b):\n"
        "        xs = []\n"
        "        for _ in range(200):\n"
        "            xs.append(np.ones(10_000_000)); time.sleep(0.05)\n"
        "    print('survived')\n"
        "except safety.ResourceExceeded as e:\n"
        "    print('stopped', e)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert "stopped" in out.stdout and "survived" not in out.stdout
