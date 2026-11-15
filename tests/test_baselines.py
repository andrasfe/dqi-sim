import numpy as np

from dqisim import registry

from helpers import instance


def test_exact_is_an_upper_bound_and_heuristics_reach_it_on_tiny():
    inst = instance("xorsat-random", n=16, m=40, k=3, seed=5)
    rng = np.random.default_rng(0)
    opt = registry.build("baseline", "exact").run(inst, rng)["frac"]
    for b in ["sa", "sa-degree", "walksat", "prange", "random"]:
        f = registry.build("baseline", b).run(inst, np.random.default_rng(1))["frac"]
        assert f <= opt + 1e-12
    assert registry.build("baseline", {"name": "sa", "sweeps": 2000}).run(inst, rng)["frac"] == opt


def test_planted_noiseless_is_fully_satisfiable():
    inst = instance("xorsat-regular", n=60, k=3, D=6, v="planted", noise=0.0)
    f = registry.build("baseline", "prange").run(inst, np.random.default_rng(0))["frac"]
    assert f == 1.0
