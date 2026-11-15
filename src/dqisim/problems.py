"""Problem instances: max-XORSAT (max-LINSAT over F_2), stored sparsely.

Constraint i is satisfied by x ∈ F_2^n iff  b_i · x = v_i  (mod 2).
DQI views B^T (n x m) as the parity-check matrix of the code C^⊥ = ker B^T, so each
constraint is a code bit and each variable is a parity check.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import safety
from .registry import P, register


@dataclass
class Instance:
    n: int
    m: int
    clause_ptr: np.ndarray  # (m+1,) CSR row pointer
    clause_vars: np.ndarray  # (nnz,) variable indices
    v: np.ndarray  # (m,) uint8 right-hand side
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ views
    @property
    def nnz(self) -> int:
        return int(self.clause_vars.size)

    def clause_degrees(self) -> np.ndarray:
        return np.diff(self.clause_ptr)

    def var_degrees(self) -> np.ndarray:
        return np.bincount(self.clause_vars, minlength=self.n)

    def edges(self) -> tuple[np.ndarray, np.ndarray]:
        """(constraint index, variable index) per nonzero of B."""
        return np.repeat(np.arange(self.m), self.clause_degrees()), self.clause_vars

    def dense_B(self) -> np.ndarray:
        safety.preflight(self.m * self.n, "dense B")
        B = np.zeros((self.m, self.n), np.uint8)
        r, c = self.edges()
        np.bitwise_xor.at(B, (r, c), 1)
        return B

    def var_csr(self) -> tuple[np.ndarray, np.ndarray]:
        """Variable -> incident constraints (CSR), for local search."""
        r, c = self.edges()
        order = np.argsort(c, kind="stable")
        ptr = np.zeros(self.n + 1, np.int64)
        np.cumsum(np.bincount(c, minlength=self.n), out=ptr[1:])
        return ptr, r[order].astype(np.int64)

    # ------------------------------------------------------------------ objective
    def satisfied(self, X: np.ndarray) -> np.ndarray:
        """Number of satisfied constraints for each row of X (batch, n)."""
        X = np.atleast_2d(X).astype(np.uint8)
        out = np.empty(X.shape[0], np.int64)
        step = max(1, int(2e7 // max(self.nnz, 1)))
        starts = self.clause_ptr[:-1]
        for a in range(0, X.shape[0], step):
            par = np.add.reduceat(X[a : a + step][:, self.clause_vars], starts, axis=1) & 1
            out[a : a + step] = (par == self.v).sum(axis=1)
        return out

    def summary(self) -> dict:
        cd, vd = self.clause_degrees(), self.var_degrees()
        return {
            "n": self.n,
            "m": self.m,
            "nnz": self.nnz,
            "clause_degree_mean": float(cd.mean()),
            "var_degree_mean": float(vd.mean()),
            "isolated_vars": int((vd == 0).sum()),
            **self.meta,
        }


def _from_clause_lists(n: int, clauses: list[np.ndarray] | np.ndarray, v: np.ndarray, meta: dict) -> Instance:
    if isinstance(clauses, np.ndarray) and clauses.ndim == 2:
        deg = np.full(clauses.shape[0], clauses.shape[1])
        flat = clauses.reshape(-1)
    else:
        deg = np.array([len(c) for c in clauses])
        flat = np.concatenate([np.asarray(c) for c in clauses]) if len(clauses) else np.zeros(0, int)
    ptr = np.zeros(len(deg) + 1, np.int64)
    np.cumsum(deg, out=ptr[1:])
    return Instance(n, len(deg), ptr, flat.astype(np.int64), v.astype(np.uint8), meta)


def _rhs(kind: str, m: int, rng, inst_parity=None, noise: float = 0.0) -> np.ndarray:
    if kind == "random":
        return rng.integers(0, 2, m, dtype=np.uint8)
    if kind == "ones":
        return np.ones(m, np.uint8)
    if kind == "planted":
        v = inst_parity()
        flips = (rng.random(m) < noise).astype(np.uint8)
        return v ^ flips
    raise ValueError(f"unknown v={kind!r}")


def _finish(n, clauses, rng, v_kind, noise, meta) -> Instance:
    inst = _from_clause_lists(n, clauses, np.zeros(len(clauses), np.uint8), meta)
    planted = rng.integers(0, 2, n, dtype=np.uint8)
    inst.v = _rhs(v_kind, inst.m, rng, lambda: _parity(inst, planted), noise)
    inst.meta = {**meta, "v": v_kind}
    return inst


def _parity(inst: Instance, x: np.ndarray) -> np.ndarray:
    return (np.add.reduceat(x[inst.clause_vars], inst.clause_ptr[:-1]) & 1).astype(np.uint8)


V_PARAM = P("v", str, "random", "right-hand side: random | planted (B x* + noise) | ones (MaxCut for k=2)",
            choices=["random", "planted", "ones"])
NOISE_PARAM = P("noise", float, 0.1, "flip probability for v=planted", min=0, max=0.5)


@register("problem", "xorsat-random", "Random max-k-XORSAT: m clauses of k distinct variables",
          P("n", int, 60, "variables", min=2), P("m", int, 120, "constraints", min=1),
          P("k", int, 3, "variables per clause", min=1), V_PARAM, NOISE_PARAM)
class RandomXorsat:
    def __init__(self, n, m, k, v, noise):
        self.n, self.m, self.k, self.v, self.noise = n, m, k, v, noise

    def generate(self, rng) -> Instance:
        if self.k > self.n:
            raise ValueError("k > n")
        C = np.sort(rng.integers(0, self.n, (self.m, self.k)), axis=1)
        for _ in range(1000):
            bad = np.flatnonzero((np.diff(C, axis=1) == 0).any(axis=1)) if self.k > 1 else np.zeros(0, int)
            if bad.size == 0:
                break
            C[bad] = np.sort(rng.integers(0, self.n, (bad.size, self.k)), axis=1)
        return _finish(self.n, C, rng, self.v, self.noise, {"ensemble": "random", "k": self.k})


def _configuration(var_deg: np.ndarray, clause_deg: np.ndarray, rng) -> list[np.ndarray]:
    """Configuration model: match variable sockets to clause sockets; repair repeats by swaps."""
    sockets = np.repeat(np.arange(var_deg.size), var_deg)
    rng.shuffle(sockets)
    ptr = np.zeros(clause_deg.size + 1, np.int64)
    np.cumsum(clause_deg, out=ptr[1:])
    clause_of = np.repeat(np.arange(clause_deg.size), clause_deg)
    for _ in range(50):
        key = clause_of * (var_deg.size + 1) + sockets
        order = np.argsort(key, kind="stable")
        dup = order[1:][np.diff(key[order]) == 0]
        if dup.size == 0:
            break
        partners = rng.integers(0, sockets.size, dup.size)
        sockets[dup], sockets[partners] = sockets[partners], sockets[dup].copy()
    clauses = []
    for i in range(clause_deg.size):
        # any repeat that survived the swaps cancels in pairs (x ⊕ x = 0)
        vals, cnt = np.unique(sockets[ptr[i] : ptr[i + 1]], return_counts=True)
        c = vals[cnt % 2 == 1]
        clauses.append(c if c.size else vals[:1])
    return clauses


@register("problem", "xorsat-regular", "Gallager ensemble: every clause has k variables, every variable sits in D clauses (m = nD/k)",
          P("n", int, 60, "variables", min=2), P("k", int, 3, "variables per clause", min=1),
          P("D", int, 6, "clauses per variable", min=1), V_PARAM, NOISE_PARAM)
class RegularXorsat:
    def __init__(self, n, k, D, v, noise):
        if (n * D) % k:
            raise ValueError(f"n*D={n * D} must be divisible by k={k}")
        self.n, self.k, self.D, self.v, self.noise = n, k, D, v, noise

    def generate(self, rng) -> Instance:
        m = self.n * self.D // self.k
        clauses = _configuration(np.full(self.n, self.D), np.full(m, self.k), rng)
        return _finish(self.n, clauses, rng, self.v, self.noise,
                       {"ensemble": "regular", "k": self.k, "D": self.D})


def _sample_degrees(dist: list[float], size: int, rng) -> np.ndarray:
    p = np.asarray(dist, float)
    p = p / p.sum()
    return rng.choice(np.arange(1, p.size + 1), size=size, p=p)


@register("problem", "xorsat-irregular", "Irregular degree distributions (the paper tunes these to hurt simulated annealing)",
          P("n", int, 60, "variables", min=2), P("m", int, 120, "constraints", min=1),
          P("clause_dist", "list[float]", [0, 0, 1.0], "P(clause degree = 1, 2, 3, ...)"),
          P("var_dist", "list[float]", [0, 0, 0, 0, 0, 1.0], "relative P(variable degree = 1, 2, ...) (rescaled to match sockets)"),
          V_PARAM, NOISE_PARAM)
class IrregularXorsat:
    def __init__(self, n, m, clause_dist, var_dist, v, noise):
        self.n, self.m, self.cd, self.vd, self.v, self.noise = n, m, clause_dist, var_dist, v, noise

    def generate(self, rng) -> Instance:
        cdeg = np.minimum(_sample_degrees(self.cd, self.m, rng), self.n)
        vdeg = _sample_degrees(self.vd, self.n, rng)
        total = int(cdeg.sum())
        # rescale variable degrees to the clause socket count, keeping the shape
        vdeg = np.maximum(1, np.round(vdeg * total / vdeg.sum())).astype(int)
        diff = total - int(vdeg.sum())
        while diff:
            idx = rng.integers(0, self.n, abs(diff))
            if diff > 0:
                np.add.at(vdeg, idx, 1)
            else:
                ok = idx[vdeg[idx] > 1]
                np.add.at(vdeg, ok, -1)
            diff = total - int(vdeg.sum())
        clauses = _configuration(vdeg, cdeg, rng)
        return _finish(self.n, clauses, rng, self.v, self.noise, {"ensemble": "irregular"})


@register("problem", "xorsat-file", "Load an instance from .npz (keys: B dense m x n, or clause_ptr/clause_vars/n; and v)",
          P("path", str, "", "path to .npz"))
class FileXorsat:
    def __init__(self, path):
        self.path = path

    def generate(self, rng) -> Instance:
        d = np.load(self.path)
        if "B" in d:
            B = d["B"].astype(np.uint8)
            clauses = [np.flatnonzero(row) for row in B]
            return _from_clause_lists(B.shape[1], clauses, d["v"], {"ensemble": "file", "path": self.path})
        return Instance(int(d["n"]), d["clause_ptr"].size - 1, d["clause_ptr"], d["clause_vars"], d["v"].astype(np.uint8),
                        {"ensemble": "file", "path": self.path})
