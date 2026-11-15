import numpy as np
import pytest

from dqisim import registry
from dqisim.evaluators import sample_supports, supports_to_bits, syndromes_bits

from helpers import instance


def _setup(name, inst, prior=0.02, ell=2, **kw):
    d = registry.build("decoder", {"name": name, **kw})
    d.setup(inst, prior, ell)
    return d


@pytest.mark.parametrize("name", ["bp", "bposd", "bitflip"])
def test_low_weight_errors_decode(name):
    inst = instance(n=200, k=3, D=6)
    sup = sample_supports(np.random.default_rng(0), inst.m, 1, 200)
    Y = _setup(name, inst).decode(syndromes_bits(inst, sup))
    assert (Y == supports_to_bits(sup, inst.m)).all(axis=1).mean() > 0.99


def test_bposd_always_returns_valid_syndrome_solution():
    inst = instance(n=60, k=3, D=6)
    sup = sample_supports(np.random.default_rng(1), inst.m, 25, 50)  # far beyond the radius
    S = syndromes_bits(inst, sup)
    Y = _setup("bposd", inst, prior=0.2).decode(S)
    H = inst.dense_B().T
    assert ((Y @ H.T) % 2 == S).all()


def test_ml_returns_minimum_weight_coset_leader():
    inst = instance(n=12, k=3, D=6)
    d = _setup("ml", inst, ell=2)
    sup = sample_supports(np.random.default_rng(2), inst.m, 2, 100)
    Y = d.decode(syndromes_bits(inst, sup))
    w = Y.sum(axis=1)
    assert (w <= 2).all() and (w >= 1).all()


def test_sumproduct_and_minsum_agree_on_easy_errors():
    inst = instance(n=200, k=3, D=6)
    sup = sample_supports(np.random.default_rng(3), inst.m, 3, 100)
    S = syndromes_bits(inst, sup)
    a = _setup("bp", inst, mode="minsum").decode(S)
    b = _setup("bp", inst, mode="sumproduct").decode(S)
    assert (a == b).all(axis=1).mean() > 0.97
