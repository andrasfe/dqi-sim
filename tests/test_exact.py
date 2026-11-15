import math

import numpy as np
import pytest

from dqisim import gf2, registry
from dqisim.weights import ideal_fraction, ideal_top

from helpers import ctx, instance


def _perfect_instance():
    # find a small instance whose dual distance exceeds 2ℓ+1 for ℓ=1 (no syndrome collisions)
    for seed in range(200):
        inst = instance("xorsat-random", seed=seed, n=14, m=20, k=3)
        H = inst.dense_B().T
        if gf2.min_distance_bruteforce(H, 3) is None:
            return inst
    pytest.skip("no collision-free instance found")


def test_perfect_regime_matches_paper_formula():
    inst = _perfect_instance()
    r = registry.build("evaluator", {"name": "exact"}).run(ctx(inst, 1, "ml"))
    assert r["syndrome_collisions"] == 0
    assert r["p_succ"] == pytest.approx(1.0)
    assert r["frac"] == pytest.approx(ideal_fraction(inst.m, 1), abs=1e-12)
    assert r["frac_nopost"] == pytest.approx(r["frac_post"], abs=1e-12)
    assert r["frac_dense"] == pytest.approx(r["frac"], abs=1e-12)


def test_dqi_state_equals_polynomial_of_objective():
    """Independent check of signs/conventions: amplitude(x) ∝ Σ_k w_k/sqrt(C(m,k)) e_k(f_1(x),...,f_m(x))."""
    inst = _perfect_instance()
    m, n, ell = inst.m, inst.n, 1
    w = ideal_top(m, ell)[1]
    X = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(np.uint8)
    B = inst.dense_B()
    fi = 1 - 2 * ((X.astype(int) @ B.T.astype(int) + inst.v) % 2)  # +1 when satisfied
    E = np.zeros((ell + 1, 2**n))
    E[0] = 1
    for i in range(m):
        for k in range(ell, 0, -1):
            E[k] += E[k - 1] * fi[:, i]
    P = sum(w[k] / math.sqrt(math.comb(m, k)) * E[k] for k in range(ell + 1))
    prob = P**2 / (P**2).sum()
    s = (fi == 1).sum(axis=1)
    r = registry.build("evaluator", {"name": "exact"}).run(ctx(inst, ell, "ml"))
    assert (prob * s).sum() / m == pytest.approx(r["frac_dense"], abs=1e-12)


@pytest.mark.parametrize("decoder", ["ml", "bp", "bposd", "bitflip"])
def test_quadratic_form_equals_dense_distribution(decoder):
    inst = instance(n=14, k=3, D=6, seed=3)
    r = registry.build("evaluator", {"name": "exact", "dense": "yes"}).run(ctx(inst, 3, decoder))
    assert r["syndrome_collisions"] > 0  # the interesting regime
    assert r["frac_dense"] == pytest.approx(r["frac_post"], abs=1e-10)
    assert 0 < r["p_succ"] <= 1


def test_aware_weights_never_worse_than_paper_weights():
    inst = instance(n=16, k=3, D=6, seed=4)
    r = registry.build("evaluator", {"name": "exact"}).run(ctx(inst, 4, "bp"))
    assert r["frac_aware_weights"] >= r["frac_paper_weights"] - 1e-12


def test_exact_refuses_huge_enumeration():
    from dqisim.safety import ResourceExceeded

    inst = instance(n=200, k=3, D=6)
    with pytest.raises(ResourceExceeded):
        registry.build("evaluator", {"name": "exact", "max_entries": 1000}).run(ctx(inst, 5, "bp"))
