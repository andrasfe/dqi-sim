import numpy as np
import pytest

from dqisim import registry

from helpers import ctx, instance


@pytest.mark.parametrize("decoder", ["bp", "bitflip"])
def test_mc_agrees_with_exact_on_average(decoder):
    """MC drops collision cross-terms, which vanish in expectation over random v."""
    diffs, psucc = [], []
    for seed in range(6):
        inst = instance(n=20, k=3, D=6, seed=seed)
        ex = registry.build("evaluator", {"name": "exact", "dense": "no"}).run(ctx(inst, 3, decoder, seed=seed))
        mc = registry.build("evaluator", {"name": "mc", "samples": 1500, "k_points": 0}).run(ctx(inst, 3, decoder, seed=seed))
        diffs.append(mc["frac"] - ex["frac"])
        psucc.append(mc["p_succ"] - ex["p_succ"])
    assert abs(np.mean(diffs)) < 0.01
    assert abs(np.mean(psucc)) < 0.03


def test_mc_with_perfect_decoder_matches_ideal():
    inst = instance(n=400, k=3, D=6)
    r = registry.build("evaluator", {"name": "mc", "samples": 100}).run(ctx(inst, 3, "bp"))
    ideal = registry.build("evaluator", {"name": "ideal"}).run(ctx(inst, 3, "bp"))
    assert r["p_succ"] == pytest.approx(1.0, abs=1e-9)
    assert r["frac"] == pytest.approx(ideal["frac"], abs=1e-9)


def test_auto_ell_probe_is_reasonable():
    inst = instance(n=300, k=3, D=6)
    c = ctx(inst, "auto", "bp")
    ell = c.ell
    assert 1 <= ell < inst.m // 2
    r = registry.build("evaluator", {"name": "mc", "samples": 200}).run(c)
    assert r["ell_auto"] is True


def test_auto_ell_is_paired_across_sweep_points():
    inst = instance(n=200, k=3, D=6)
    ells = {ctx(inst, "auto", "bp", weights=w, seed=s).ell for s, w in [(1, "optimal-ideal"), (2, "product:t=0.3"), (3, "single")]}
    assert len(ells) == 1


def test_hopeless_postselection_is_flagged():
    inst = instance(n=200, k=3, D=6)
    r = registry.build("evaluator", {"name": "mc", "samples": 60, "k_points": 8}).run(ctx(inst, 0.2, "bitflip"))
    assert r["p_succ"] < 1e-3 and "warning" in r and r["repetitions"] > 1e3
