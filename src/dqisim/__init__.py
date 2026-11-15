"""dqisim: a decoder-in-the-loop simulator for Decoded Quantum Interferometry variants.

Importing this package caps the thread pools of numpy/BLAS/numba *before* they are
loaded, so every entry point (CLI, UI, tests, `python -m dqisim`) starts with a small
CPU footprint. Override with DQISIM_THREADS (hard-capped at 4).
"""
import os as _os

_MAX_THREADS = 4


def _thread_cap() -> int:
    try:
        n = int(_os.environ.get("DQISIM_THREADS", "2"))
    except ValueError:
        n = 2
    return max(1, min(n, _MAX_THREADS))


THREADS = _thread_cap()
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    _os.environ[_var] = str(THREADS)

__version__ = "0.1.0"
