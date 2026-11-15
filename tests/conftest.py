import os

import pytest

os.environ.setdefault("DQISIM_THREADS", "2")


@pytest.fixture(autouse=True)
def _budget(tmp_path, monkeypatch):
    from dqisim import safety

    monkeypatch.setenv("DQISIM_RESULTS", str(tmp_path / "results"))
    safety.set_budget(safety.Budget(memory_gb=4, min_free_mem_gb=2, min_free_disk_gb=5, max_load=10.0))
    safety.reset_clock()
    yield
