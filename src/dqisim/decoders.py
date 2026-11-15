"""Syndrome decoders for C^⊥ = ker B^T: given s = B^T y, recover y.

In DQI the decoder runs reversibly *in superposition*, so it cannot know |y| (that
register was uncomputed). Every decoder here is therefore configured once with a fixed
channel prior (default ℓ/m) and then applied identically to every syndrome.

Interface: `setup(inst, prior)` once, then `decode(S)` with S (batch, n) uint8 → Y (batch, m).
`bytes_per_item(inst)` lets evaluators size batches to the memory budget.
"""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np

from . import safety
from .gf2 import pack_rows, rref_solve
from .problems import Instance
from .registry import P, register

CLIP = 50.0


def unpack_rows(packed: np.ndarray, nbits: int) -> np.ndarray:
    return np.unpackbits(np.ascontiguousarray(packed).view(np.uint8), axis=1, bitorder="little")[:, :nbits]


class Decoder:
    def setup(self, inst: Instance, prior: float, ell: int | None = None) -> None:
        self.inst = inst
        self.prior = float(np.clip(prior, 1e-6, 0.5 - 1e-6))

    def bytes_per_item(self, inst: Instance) -> int:
        return 16 * (inst.m + inst.n)

    def decode(self, S: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class _Graph:
    """Tanner-graph index structures shared by message-passing decoders."""

    def __init__(self, inst: Instance):
        bit, chk = inst.edges()  # edges sorted by bit (constraint)
        self.E = bit.size
        self.m, self.n = inst.m, inst.n
        self.bit, self.chk = bit, chk
        self.bit_starts = inst.clause_ptr[:-1]
        self.perm_c = np.argsort(chk, kind="stable")  # bit-order -> check-order
        self.inv_c = np.argsort(self.perm_c)
        chk_sorted = chk[self.perm_c]
        self.checks = np.unique(chk_sorted)  # non-empty checks
        self.chk_starts = np.searchsorted(chk_sorted, self.checks)
        self.chk_deg = np.diff(np.append(self.chk_starts, self.E))
        self.chk_of_edge_c = np.repeat(np.arange(self.checks.size), self.chk_deg)  # check-order
        self.bit_deg = np.diff(inst.clause_ptr)

    def syndrome_ok(self, Y: np.ndarray, Sc: np.ndarray) -> np.ndarray:
        par = np.add.reduceat(Y[:, self.bit][:, self.perm_c], self.chk_starts, axis=1) & 1
        return (par == Sc).all(axis=1)


@register("decoder", "bp", "Belief propagation on the Tanner graph of B^T (syndrome form)",
          P("iters", int, 50, "maximum iterations", min=1),
          P("mode", str, "minsum", "message rule", choices=["minsum", "sumproduct"]),
          P("alpha", float, 0.8, "min-sum normalisation (ignored for sum-product)", min=0.1, max=1.0),
          P("prior", float, 0.0, "fixed channel error prior; 0 = use ℓ/m"))
class BP(Decoder):
    def __init__(self, iters, mode, alpha, prior):
        self.iters, self.mode, self.alpha, self.prior_override = iters, mode, alpha, prior

    def setup(self, inst, prior, ell=None):
        super().setup(inst, self.prior_override or prior, ell)
        self.g = _Graph(inst)

    def bytes_per_item(self, inst):
        return 8 * 12 * inst.nnz + 16 * (inst.m + inst.n)

    def run(self, S: np.ndarray):
        """Returns (Y, converged, posterior LLR)."""
        g = self.g
        S = np.asarray(S, np.uint8)
        Sc = S[:, g.checks]
        B = S.shape[0]
        L0 = np.log((1 - self.prior) / self.prior)
        Y = np.zeros((B, g.m), np.uint8)
        post = np.full((B, g.m), L0)
        done = np.zeros(B, bool)
        live = np.arange(B)
        Q = np.full((B, g.E), L0)  # bit -> check messages, bit order
        for _ in range(self.iters):
            safety.checkpoint()
            Qc = Q[:, g.perm_c]
            neg = Qc < 0
            par = (np.add.reduceat(neg.astype(np.int32), g.chk_starts, axis=1) & 1) ^ Sc[live]
            sign = 1.0 - 2.0 * (par[:, g.chk_of_edge_c] ^ neg)
            if self.mode == "minsum":
                A = np.abs(Qc)
                min1 = np.minimum.reduceat(A, g.chk_starts, axis=1)
                eq = A == min1[:, g.chk_of_edge_c]
                min2 = np.minimum.reduceat(np.where(eq, np.inf, A), g.chk_starts, axis=1)
                ties = np.add.reduceat(eq.astype(np.int32), g.chk_starts, axis=1) >= 2
                min2 = np.where(ties, min1, min2)
                mag = np.where(eq, min2[:, g.chk_of_edge_c], min1[:, g.chk_of_edge_c]) * self.alpha
            else:
                T = np.clip(np.abs(np.tanh(Qc / 2)), 1e-300, 1 - 1e-15)
                lt = np.log(T)
                tot = np.add.reduceat(lt, g.chk_starts, axis=1)[:, g.chk_of_edge_c] - lt
                mag = 2 * np.arctanh(np.minimum(np.exp(tot), 1 - 1e-15))
            Rc = np.clip(sign * mag, -CLIP, CLIP)
            R = Rc[:, g.inv_c]
            total = L0 + np.add.reduceat(R, g.bit_starts, axis=1)
            Yl = (total < 0).astype(np.uint8)
            ok = g.syndrome_ok(Yl, Sc[live])
            if ok.any():
                idx = live[ok]
                Y[idx], post[idx], done[idx] = Yl[ok], total[ok], True
            keep = ~ok
            if not keep.any():
                return Y, done, post
            live = live[keep]
            Y[live], post[live] = Yl[keep], total[keep]
            Q = total[keep][:, g.bit] - R[keep]
        return Y, done, post

    def decode(self, S):
        return self.run(S)[0]


@register("decoder", "bposd", "BP followed by ordered-statistics decoding (OSD-0) when BP fails",
          P("iters", int, 30, "BP iterations", min=1),
          P("mode", str, "minsum", "BP message rule", choices=["minsum", "sumproduct"]),
          P("alpha", float, 0.8, "min-sum normalisation", min=0.1, max=1.0),
          P("prior", float, 0.0, "fixed channel error prior; 0 = use ℓ/m"))
class BPOSD(BP):
    def setup(self, inst, prior, ell=None):
        super().setup(inst, prior, ell)
        safety.preflight(inst.n * inst.m * 2, "BP-OSD dense parity-check matrix")
        self.H = inst.dense_B().T.copy()

    def bytes_per_item(self, inst):
        return super().bytes_per_item(inst) + inst.n * inst.m // 4

    def decode(self, S):
        Y, done, post = self.run(S)
        for b in np.flatnonzero(~done):
            safety.checkpoint()
            order = np.argsort(post[b], kind="stable")  # most likely flipped bits first
            y, _, ok = rref_solve(self.H, S[b], order)
            if ok:
                Y[b] = y
        return Y


@register("decoder", "bitflip", "Gallager hard-decision bit flipping (cheap, weak baseline)",
          P("iters", int, 50, "maximum iterations", min=1))
class BitFlip(Decoder):
    def __init__(self, iters):
        self.iters = iters

    def setup(self, inst, prior, ell=None):
        super().setup(inst, prior, ell)
        self.g = _Graph(inst)

    def decode(self, S):
        g = self.g
        Sc = np.asarray(S, np.uint8)[:, g.checks]
        B = Sc.shape[0]
        Y = np.zeros((B, g.m), np.uint8)
        for _ in range(self.iters):
            safety.checkpoint()
            par = np.add.reduceat(Y[:, g.bit][:, g.perm_c], g.chk_starts, axis=1) & 1
            unsat = (par ^ Sc).astype(np.int32)
            if not unsat.any():
                break
            cnt = np.add.reduceat(unsat[:, g.chk_of_edge_c][:, g.inv_c], g.bit_starts, axis=1)
            score = cnt - g.bit_deg / 2.0
            best = score.max(axis=1, keepdims=True)
            flip = (score == best) & (best > 0)
            Y ^= flip.astype(np.uint8)
        return Y


def enumerate_supports(m: int, k: int, chunk: int = 200_000):
    """Yield (batch, k) arrays of all k-subsets of range(m) in lexicographic order."""
    it = combinations(range(m), k)
    while True:
        block = list(_take(it, chunk))
        if not block:
            return
        yield np.array(block, dtype=np.int64).reshape(len(block), k)


def _take(it, n):
    for _ in range(n):
        try:
            yield next(it)
        except StopIteration:
            return


def count_upto(m: int, ell: int) -> int:
    return sum(comb(m, k) for k in range(ell + 1))


def syndromes_packed(Bp: np.ndarray, supports: np.ndarray) -> np.ndarray:
    """XOR of packed rows of B over each support → (batch, W) packed syndromes."""
    if supports.shape[1] == 0:
        return np.zeros((supports.shape[0], Bp.shape[1]), np.uint64)
    return np.bitwise_xor.reduce(Bp[supports], axis=1)


@register("decoder", "ml", "Exact minimum-weight (coset-leader) decoding up to weight t by table lookup; "
                           "the best any classical decoder can do. Small instances only.",
          P("t", int, 0, "table radius; 0 = use ℓ", min=0))
class MinWeight(Decoder):
    def __init__(self, t):
        self.t = t

    def setup(self, inst, prior, ell=None):
        super().setup(inst, prior, ell)
        t = self.t or ell or 1
        N = count_upto(inst.m, t)
        W = max(1, (inst.n + 63) // 64)
        safety.preflight(N * (8 * W + 120), f"ML table ({N:,} entries)")
        self.Bp = pack_rows(self._dense(inst))
        self.table: dict[bytes, np.ndarray] = {}
        for k in range(t + 1):
            for sup in enumerate_supports(inst.m, k):
                safety.checkpoint()
                syn = syndromes_packed(self.Bp, sup)
                for row, s in zip(sup, syn):
                    self.table.setdefault(s.tobytes(), row)
        self.radius = t

    @staticmethod
    def _dense(inst):
        return inst.dense_B()

    def decode(self, S):
        Sp = pack_rows(S)
        Y = np.zeros((S.shape[0], self.inst.m), np.uint8)
        for b, s in enumerate(Sp):
            row = self.table.get(s.tobytes())
            if row is not None:
                Y[b, row] = 1
        return Y
