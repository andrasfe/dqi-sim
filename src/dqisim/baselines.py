"""Classical baselines run on the same instances, so DQI numbers are never quoted alone.

The paper's lesson: DQI+BP beat generic heuristics on its tuned instances, but a
tailored classical algorithm (degree-aware annealing) beat DQI+BP. Keep both kinds here.
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit

from . import safety
from .gf2 import fwht, rows_to_int, rref_solve
from .problems import Instance
from .registry import P, register


class Baseline:
    def run(self, inst: Instance, rng: np.random.Generator) -> dict:  # pragma: no cover
        raise NotImplementedError


def _timed(fn):
    def wrapper(self, inst, rng):
        safety.checkpoint()
        t = time.perf_counter()
        out = fn(self, inst, rng)
        out["time_s"] = round(time.perf_counter() - t, 4)
        return out

    return wrapper


@njit(cache=True)
def _anneal(n, m, var_ptr, var_cl, clause_ptr, clause_vars, v, deg, sweeps, beta0, beta1, gamma0, x, seed):
    np.random.seed(seed)
    par = np.zeros(m, np.int8)
    for c in range(m):
        p = 0
        for t in range(clause_ptr[c], clause_ptr[c + 1]):
            p ^= x[clause_vars[t]]
        par[c] = p
    sat = 0
    for c in range(m):
        if par[c] == v[c]:
            sat += 1
    best = sat
    bestx = x.copy()
    w = np.ones(m)
    for s in range(sweeps):
        frac = s / max(sweeps - 1, 1)
        beta = beta0 + (beta1 - beta0) * frac
        gamma = gamma0 * (1.0 - frac)
        if gamma0 != 0.0:
            for c in range(m):
                w[c] = deg[c] ** gamma
        for _ in range(n):
            j = np.random.randint(n)
            dw = 0.0
            dc = 0
            for t in range(var_ptr[j], var_ptr[j + 1]):
                c = var_cl[t]
                if par[c] == v[c]:
                    dw -= w[c]
                    dc -= 1
                else:
                    dw += w[c]
                    dc += 1
            if dw >= 0.0 or np.random.random() < np.exp(beta * dw):
                x[j] ^= 1
                for t in range(var_ptr[j], var_ptr[j + 1]):
                    par[var_cl[t]] ^= 1
                sat += dc
                if sat > best:
                    best = sat
                    bestx[:] = x
    return best, bestx


@njit(cache=True)
def _walksat(n, m, var_ptr, var_cl, clause_ptr, clause_vars, v, flips, noise, x, seed):
    np.random.seed(seed)
    par = np.zeros(m, np.int8)
    for c in range(m):
        p = 0
        for t in range(clause_ptr[c], clause_ptr[c + 1]):
            p ^= x[clause_vars[t]]
        par[c] = p
    sat = 0
    for c in range(m):
        if par[c] == v[c]:
            sat += 1
    best = sat
    for _ in range(flips):
        if sat == m:
            break
        c = np.random.randint(m)
        tries = 0
        while par[c] == v[c] and tries < 64:
            c = np.random.randint(m)
            tries += 1
        lo, hi = clause_ptr[c], clause_ptr[c + 1]
        if np.random.random() < noise:
            j = clause_vars[lo + np.random.randint(hi - lo)]
        else:
            j = clause_vars[lo]
            bd = -m
            for t in range(lo, hi):
                jj = clause_vars[t]
                d = 0
                for u in range(var_ptr[jj], var_ptr[jj + 1]):
                    cc = var_cl[u]
                    d += -1 if par[cc] == v[cc] else 1
                if d > bd:
                    bd = d
                    j = jj
        for u in range(var_ptr[j], var_ptr[j + 1]):
            cc = var_cl[u]
            sat += -1 if par[cc] == v[cc] else 1
            par[cc] ^= 1
        x[j] ^= 1
        if sat > best:
            best = sat
    return best


def _arrays(inst: Instance):
    vp, vc = inst.var_csr()
    return (inst.n, inst.m, vp, vc, inst.clause_ptr.astype(np.int64), inst.clause_vars.astype(np.int64),
            inst.v.astype(np.int8))


@register("baseline", "sa", "Simulated annealing (Metropolis, linear β schedule), best over restarts",
          P("sweeps", int, 500, "sweeps per restart", min=1), P("restarts", int, 1, "independent restarts", min=1),
          P("beta0", float, 0.1, "initial inverse temperature"), P("beta1", float, 4.0, "final inverse temperature"))
class SimulatedAnnealing(Baseline):
    gamma0 = 0.0

    def __init__(self, sweeps, restarts, beta0, beta1, **kw):
        self.sweeps, self.restarts, self.beta0, self.beta1 = sweeps, restarts, beta0, beta1

    @_timed
    def run(self, inst, rng):
        a = _arrays(inst)
        deg = inst.clause_degrees().astype(np.float64)
        best = 0
        for _ in range(self.restarts):
            safety.checkpoint()
            x = rng.integers(0, 2, inst.n).astype(np.int8)
            b, _ = _anneal(*a, deg, self.sweeps, self.beta0, self.beta1, self.gamma0, x, int(rng.integers(1 << 31)))
            best = max(best, int(b))
        return {"frac": best / inst.m}


@register("baseline", "sa-degree", "Degree-aware annealing: clause weights deg^γ with γ annealed γ0 → 0 "
                                   "(in the spirit of the paper's tailored 'irregular annealing'; not their exact rule)",
          P("sweeps", int, 500, "sweeps per restart", min=1), P("restarts", int, 1, "independent restarts", min=1),
          P("beta0", float, 0.1, "initial inverse temperature"), P("beta1", float, 4.0, "final inverse temperature"),
          P("gamma0", float, -1.0, "initial degree exponent (negative favours short clauses)"))
class DegreeAnnealing(SimulatedAnnealing):
    def __init__(self, sweeps, restarts, beta0, beta1, gamma0):
        super().__init__(sweeps, restarts, beta0, beta1)
        self.gamma0 = gamma0


@register("baseline", "walksat", "WalkSAT-style focused local search on unsatisfied clauses",
          P("flips", int, 200_000, "flip budget", min=1), P("noise", float, 0.3, "random-walk probability", min=0, max=1))
class WalkSat(Baseline):
    def __init__(self, flips, noise):
        self.flips, self.noise = flips, noise

    @_timed
    def run(self, inst, rng):
        x = rng.integers(0, 2, inst.n).astype(np.int8)
        b = _walksat(*_arrays(inst), self.flips, self.noise, x, int(rng.integers(1 << 31)))
        return {"frac": int(b) / inst.m}


@register("baseline", "prange", "Prange-style: satisfy a random maximal independent set of constraints exactly "
                                "(Gaussian elimination), best over trials",
          P("trials", int, 10, "random trials", min=1))
class Prange(Baseline):
    def __init__(self, trials):
        self.trials = trials

    @_timed
    def run(self, inst, rng):
        safety.preflight(inst.m * inst.n * 3, "prange dense B")
        B = inst.dense_B()
        best = 0
        for _ in range(self.trials):
            safety.checkpoint()
            order = rng.permutation(inst.m)
            _, rows, _ = rref_solve(B.T, np.zeros(inst.n, np.uint8), order)
            x, _, ok = rref_solve(B[rows], inst.v[rows])
            best = max(best, int(inst.satisfied(x[None])[0]))
        return {"frac": best / inst.m}


@register("baseline", "random", "Best of N uniformly random assignments",
          P("samples", int, 1000, "assignments", min=1))
class RandomAssign(Baseline):
    def __init__(self, samples):
        self.samples = samples

    @_timed
    def run(self, inst, rng):
        best = 0
        for a in range(0, self.samples, 256):
            X = rng.integers(0, 2, (min(256, self.samples - a), inst.n), dtype=np.uint8)
            best = max(best, int(inst.satisfied(X).max()))
        return {"frac": best / inst.m}


@register("baseline", "exact", "True optimum by evaluating all 2^n assignments with one Walsh-Hadamard transform (n ≤ 30)")
class BruteForce(Baseline):
    @_timed
    def run(self, inst, rng):
        if inst.n > 30:
            raise safety.ResourceExceeded("exact baseline: n > 30")
        safety.preflight(2 * 8 * 2**inst.n, f"exact baseline 2^{inst.n}")
        u = np.zeros(2**inst.n)
        np.add.at(u, rows_to_int(inst.dense_B()).astype(np.int64), 1.0 - 2.0 * inst.v)
        f = fwht(u)
        return {"frac": float((inst.m + f.max()) / 2 / inst.m)}
