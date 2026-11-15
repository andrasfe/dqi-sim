"""Evaluators: ways of computing what a DQI variant achieves on an instance.

All of them return the expected satisfied fraction `frac` for the configured weights,
plus diagnostics. Fidelity / cost ladder:

  semicircle  asymptotic law (paper Eq. 6)                       O(1)
  ideal       finite-m optimum with perfect decoding (G=I, F=A)  O(ℓ)
  mc          Monte Carlo decoder profile → (F, G)               O(k_points * samples * decode)
  exact       enumerate every |y| ≤ ℓ, decode each, exact (F, G)  O(Σ_k C(m,k) * decode)
              optional dense FWHT over all 2^n x: full output distribution and sampling
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import safety
from .decoders import (Decoder, count_upto, enumerate_supports, pack_rows, syndromes_packed,
                       unpack_rows)
from .problems import Instance
from .registry import P, build, register
from .weights import (A_matrix, OptimalIdeal, _Weights, evaluate, ideal_fraction, log_binom,
                      semicircle, top_generalized)


# ----------------------------------------------------------------------------- context


@dataclass
class Context:
    """Everything an evaluator needs for one (instance, DQI configuration, seed)."""

    inst: Instance
    rng: np.random.Generator
    ell_spec: int | float | str
    weights: _Weights
    decoder_spec: dict
    postselect: bool = True
    ell_auto_eps: float = 0.1
    probe_seed: int = 0  # auto-ℓ probe depends only on this, so sweep points stay paired
    cache: dict = field(default_factory=dict)

    def decoder(self, ell: int | None) -> Decoder:
        key = ("decoder", ell)
        if key not in self.cache:
            dec = build("decoder", self.decoder_spec)
            prior = (ell or 1) / self.inst.m
            dec.setup(self.inst, prior, ell)
            self.cache[key] = dec
        return self.cache[key]

    @property
    def ell(self) -> int:
        if "ell" not in self.cache:
            self.cache["ell"] = self._resolve_ell()
        return self.cache["ell"]

    def _resolve_ell(self) -> int:
        s, m = self.ell_spec, self.inst.m
        if isinstance(s, (int, np.integer)) and not isinstance(s, bool):
            ell = int(s)
        elif isinstance(s, float) and 0 < s < 1:
            ell = int(round(s * m))
        elif isinstance(s, float):
            ell = int(s)
        elif s == "auto":
            ell = probe_radius(self, self.ell_auto_eps)
            self.cache["ell_auto"] = True
        else:
            raise ValueError(f"dqi.ell must be an int, a fraction of m in (0,1), or 'auto'; got {s!r}")
        return max(1, min(ell, m))


def batch_size(dec: Decoder, inst: Instance, cap: int = 4096) -> int:
    b = safety.budget()
    room = 0.25 * (b.memory_gb * safety.GB - safety.rss_bytes())
    per = max(dec.bytes_per_item(inst), 1)
    return int(max(8, min(cap, room // per)))


def sample_supports(rng, m: int, k: int, N: int) -> np.ndarray:
    if k == 0:
        return np.zeros((N, 0), np.int64)
    if k > m // 4:
        return np.argsort(rng.random((N, m)), axis=1)[:, :k]
    S = np.sort(rng.integers(0, m, (N, k)), axis=1)
    while True:
        bad = np.flatnonzero((np.diff(S, axis=1) == 0).any(axis=1)) if k > 1 else np.zeros(0, int)
        if bad.size == 0:
            return S
        S[bad] = np.sort(rng.integers(0, m, (bad.size, k)), axis=1)


def syndromes_bits(inst: Instance, supports: np.ndarray) -> np.ndarray:
    """Syndrome bits B^T y for each support (sparse; works for any n)."""
    N, k = supports.shape
    if k == 0:
        return np.zeros((N, inst.n), np.uint8)
    deg = inst.clause_degrees()
    starts = inst.clause_ptr[supports]  # (N, k)
    lens = deg[supports]
    rows = np.repeat(np.arange(N), lens.sum(axis=1))
    flat_starts = np.repeat(starts.ravel(), lens.ravel())
    offs = np.arange(lens.sum()) - np.repeat(np.cumsum(lens.ravel()) - lens.ravel(), lens.ravel())
    cols = inst.clause_vars[flat_starts + offs]
    cnt = np.bincount(rows * inst.n + cols, minlength=N * inst.n)
    return (cnt.reshape(N, inst.n) & 1).astype(np.uint8)


def supports_to_bits(supports: np.ndarray, m: int) -> np.ndarray:
    Y = np.zeros((supports.shape[0], m), np.uint8)
    if supports.shape[1]:
        np.put_along_axis(Y, supports, 1, axis=1)
    return Y


def decode_ok(ctx: Context, dec: Decoder, supports: np.ndarray) -> np.ndarray:
    out = np.empty(supports.shape[0], bool)
    bs = batch_size(dec, ctx.inst)
    for a in range(0, supports.shape[0], bs):
        safety.checkpoint()
        sup = supports[a : a + bs]
        S = syndromes_bits(ctx.inst, sup)
        Y = dec.decode(S)
        out[a : a + bs] = (Y == supports_to_bits(sup, ctx.inst.m)).all(axis=1)
    return out


def probe_radius(ctx: Context, eps: float, samples: int = 64) -> int:
    """Largest k with decoder failure rate ≤ eps (exponential + binary search)."""
    m = ctx.inst.m
    dec = ctx.decoder(None)
    rng = np.random.default_rng([ctx.probe_seed, 7])

    def fail(k):
        return 1.0 - decode_ok(ctx, dec, sample_supports(rng, m, k, samples)).mean()

    lo, hi = 0, 1
    while hi <= m // 2 and fail(hi) <= eps:
        lo, hi = hi, hi * 2
    hi = min(hi, m // 2 + 1)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fail(mid) <= eps:
            lo = mid
        else:
            hi = mid
    return max(lo, 1)


def _weights(ctx: Context, m, ell, F=None, G=None):
    return ctx.weights.choose(m, ell, F, G)


def _common(ctx: Context) -> dict:
    m, ell = ctx.inst.m, ctx.ell
    return {"ell": ell, "ell_over_m": ell / m, "ell_auto": bool(ctx.cache.get("ell_auto")),
            "semicircle": semicircle(ell / m), "ideal_finite": ideal_fraction(m, ell)}


# ----------------------------------------------------------------------------- evaluators


class Evaluator:
    def run(self, ctx: Context) -> dict:  # pragma: no cover
        raise NotImplementedError


@register("evaluator", "semicircle", "Asymptotic semicircle law at ℓ/m (perfect decoding, optimal weights)")
class Semicircle(Evaluator):
    def run(self, ctx):
        c = _common(ctx)
        return {**c, "frac": c["semicircle"], "p_succ": 1.0}


@register("evaluator", "ideal", "Finite m, perfect decoding (G = I, F = A) with the configured weights")
class Ideal(Evaluator):
    def run(self, ctx):
        m, ell = ctx.inst.m, ctx.ell
        A = A_matrix(m, ell)
        w = _weights(ctx, m, ell, A, np.eye(ell + 1))
        r = evaluate(w, A, np.eye(ell + 1), m)
        return {**_common(ctx), **r, "w": w.tolist()}


@register("evaluator", "mc", "Monte Carlo decoder profile: failure rate ε_k and neighbour success π_k on a k-grid, "
                             "interpolated into (F, G). Exact in expectation over random v.",
          P("samples", int, 200, "decodes per grid point", min=8),
          P("k_points", int, 24, "grid points in 1..ℓ (0 = every k)", min=0))
class MonteCarlo(Evaluator):
    def __init__(self, samples, k_points):
        self.samples, self.k_points = samples, k_points

    def profile(self, ctx: Context):
        m, ell = ctx.inst.m, ctx.ell
        dec = ctx.decoder(ell)
        if self.k_points == 0 or ell <= self.k_points:
            grid = np.arange(1, ell + 1)
        else:
            grid = np.unique(np.r_[1, 2, np.round(np.linspace(1, ell, self.k_points)).astype(int)])
        succ = np.zeros(ell + 1)
        trials = np.zeros(ell + 1)
        succ[0] = trials[0] = 1.0  # the zero syndrome always decodes to zero
        pi_pts = []
        for k in grid:
            sup = sample_supports(ctx.rng, m, int(k), self.samples)
            drop = ctx.rng.integers(0, k, self.samples)
            keep = np.ones_like(sup, bool)
            keep[np.arange(self.samples), drop] = False
            sup_minus = sup[keep].reshape(self.samples, k - 1)
            ok = decode_ok(ctx, dec, sup)
            ok_minus = decode_ok(ctx, dec, sup_minus) if k > 1 else np.ones(self.samples, bool)
            succ[k] += ok.sum()
            trials[k] += self.samples
            succ[k - 1] += ok_minus.sum()
            trials[k - 1] += self.samples
            pi_pts.append((k, float((ok & ok_minus).mean())))
        have = np.flatnonzero(trials > 0)
        good = np.interp(np.arange(ell + 1), have, succ[have] / trials[have])
        pk, pv = zip(*pi_pts)
        pi = np.interp(np.arange(1, ell + 1), pk, pv)
        pi = np.minimum(pi, np.minimum(good[1:], good[:-1]))
        se = np.sqrt(np.maximum(good * (1 - good), 1e-12) / np.maximum(trials, 1))
        return good, pi, grid, se

    def run(self, ctx):
        m, ell = ctx.inst.m, ctx.ell
        good, pi, grid, se = self.profile(ctx)
        F = np.diag(np.sqrt(np.arange(1, ell + 1) * (m - np.arange(1, ell + 1) + 1)) * pi, 1)
        F = F + F.T
        G = np.diag(good)
        w = _weights(ctx, m, ell, F, G)
        r = evaluate(w, F, G, m)
        ideal_w = OptimalIdeal().choose(m, ell)
        aware_w = top_generalized(F, G)
        eps = 1 - good
        k50 = int(np.argmax(eps >= 0.5)) if (eps >= 0.5).any() else None
        idx = np.unique(np.round(np.linspace(0, ell, min(ell + 1, 60))).astype(int))
        return {**_common(ctx), **r,
                "frac_paper_weights": evaluate(ideal_w, F, G, m)["frac"],
                "p_succ_paper_weights": evaluate(ideal_w, F, G, m)["p_succ"],
                "frac_aware_weights": evaluate(aware_w, F, G, m)["frac"],
                "p_succ_aware_weights": evaluate(aware_w, F, G, m)["p_succ"],
                "decoder_k50": k50,
                "profile": {"k": idx.tolist(), "eps": eps[idx].round(5).tolist(), "eps_se": se[idx].round(5).tolist()},
                "assumes_random_v": True}


@register("evaluator", "exact", "Enumerate every error |y| ≤ ℓ, decode each syndrome, build exact (F, G) with all "
                                "syndrome collisions; post-selected and non-post-selected. Optional dense 2^n output distribution.",
          P("max_entries", int, 2_000_000, "refuse if Σ_k≤ℓ C(m,k) exceeds this", min=1),
          P("dense", str, "auto", "full 2^n distribution via FWHT: auto (n ≤ 22) | yes | no", choices=["auto", "yes", "no"]),
          P("shots", int, 0, "if dense: sample this many x and report the best (0 = off)", min=0))
class Exact(Evaluator):
    def __init__(self, max_entries, dense, shots):
        self.max_entries, self.dense, self.shots = max_entries, dense, shots

    def enumerate(self, ctx: Context):
        inst, ell = ctx.inst, ctx.ell
        m, n = inst.m, inst.n
        N = count_upto(m, ell)
        if N > self.max_entries:
            raise safety.ResourceExceeded(
                f"exact: Σ_k≤{ell} C({m},k) = {N:,} > max_entries {self.max_entries:,}; lower ℓ or use mc")
        W, We = max(1, (n + 63) // 64), max(1, (m + 63) // 64)
        safety.preflight(N * (8 * (W + We) + 24) * 3, f"exact enumeration ({N:,} errors)")
        dec = ctx.decoder(ell)
        Bp = pack_rows(inst.dense_B()) if inst.n * inst.m <= 5e8 else None
        if Bp is None:
            raise safety.ResourceExceeded("exact: instance too large for packed rows")
        vbits = inst.v.astype(np.uint8)
        K = np.empty(N, np.int16)
        Sg = np.empty((N, W), np.uint64)
        Eg = np.empty((N, We), np.uint64)
        sign = np.empty(N, np.int8)
        pos = 0
        bs = batch_size(dec, inst)
        for k in range(ell + 1):
            for sup in enumerate_supports(m, k, chunk=bs):
                safety.checkpoint()
                b = sup.shape[0]
                syn = syndromes_packed(Bp, sup)
                Y = dec.decode(unpack_rows(syn, n))
                Ytrue = supports_to_bits(sup, m)
                K[pos : pos + b] = k
                Sg[pos : pos + b] = syn
                Eg[pos : pos + b] = pack_rows(Y ^ Ytrue)
                sign[pos : pos + b] = 1 - 2 * (vbits[sup].sum(axis=1) & 1) if k else 1
                pos += b
        return K, Sg, Eg, sign, Bp

    @staticmethod
    def _F(K, keys, sign, Bp, v, m, ell, W, select):
        """F[k,k'] = Σ_i (-1)^{v_i} Σ_{y,y' in group, s_y' = s_y ⊕ b_i} sign_y sign_y' / sqrt(C(m,k) C(m,k'))."""
        idx = np.flatnonzero(select)
        if idx.size == 0:
            return np.zeros((ell + 1, ell + 1))
        Kk, sg, kk = K[idx], sign[idx].astype(float), np.ascontiguousarray(keys[idx])
        void = np.dtype((np.void, 8 * kk.shape[1]))
        kv = kk.view(void).ravel()
        order = np.argsort(kv, kind="stable")
        sorted_v = kv[order]
        lb = log_binom(m, np.arange(ell + 1))
        scale = np.exp(-0.5 * lb)
        Fm = np.zeros((ell + 1, ell + 1))
        for i in range(m):
            safety.checkpoint()
            q = kk.copy()
            q[:, :W] ^= Bp[i]
            qv = q.view(void).ravel()
            loc = np.searchsorted(sorted_v, qv)
            loc[loc >= sorted_v.size] = 0
            hit = sorted_v[loc] == qv
            if not hit.any():
                continue
            j = order[loc[hit]]
            val = (1 - 2 * int(v[i])) * sg[hit] * sg[j]
            np.add.at(Fm, (Kk[hit], Kk[j]), val)
        return Fm * np.outer(scale, scale)

    def run(self, ctx):
        inst, ell = ctx.inst, ctx.ell
        m, n = inst.m, inst.n
        K, Sg, Eg, sign, Bp = self.enumerate(ctx)
        W = Sg.shape[1]
        good = ~Eg.any(axis=1)
        counts = np.bincount(K, minlength=ell + 1).astype(float)
        good_k = np.bincount(K[good], minlength=ell + 1).astype(float)
        G = np.diag(good_k / counts)
        F_post = self._F(K, Sg, sign, Bp, inst.v, m, ell, W, good)
        F_all = self._F(K, np.concatenate([Sg, Eg], axis=1), sign, Bp, inst.v, m, ell, W, np.ones_like(good))
        Fsel = F_post if ctx.postselect else F_all
        Gsel = G if ctx.postselect else np.eye(ell + 1)
        w = _weights(ctx, m, ell, Fsel, Gsel)
        post = evaluate(w, F_post, G, m)
        nopost = evaluate(w, F_all, np.eye(ell + 1), m)
        ideal_w = OptimalIdeal().choose(m, ell)
        aware_w = top_generalized(Fsel, Gsel)
        uniq = np.unique(np.ascontiguousarray(Sg).view(np.dtype((np.void, 8 * W))).ravel()).size
        out = {**_common(ctx),
               "frac": post["frac"] if ctx.postselect else nopost["frac"],
               "p_succ": post["p_succ"], "repetitions": post["repetitions"],
               **({"warning": post["warning"]} if "warning" in post and ctx.postselect else {}),
               "frac_post": post["frac"], "frac_nopost": nopost["frac"],
               "frac_paper_weights": evaluate(ideal_w, Fsel, Gsel, m)["frac"],
               "frac_aware_weights": evaluate(aware_w, Fsel, Gsel, m)["frac"],
               "eps": (1 - good_k / counts).round(6).tolist(),
               "entries": int(K.size), "syndrome_collisions": int(K.size - uniq),
               "w": np.asarray(w).round(6).tolist()}
        dense = self.dense == "yes" or (self.dense == "auto" and n <= 22)
        if dense:
            out.update(self._dense(ctx, K, Sg, sign, good, w, Bp))
        return out

    def _dense(self, ctx, K, Sg, sign, good, w, Bp):
        inst = ctx.inst
        m, n = inst.m, inst.n
        if n > 30:
            raise safety.ResourceExceeded("dense: n > 30")
        safety.preflight(3 * 8 * 2**n, f"dense 2^{n} state")
        coef = np.asarray(w)[K] * np.exp(-0.5 * log_binom(m, K.astype(float))) * sign
        if not ctx.postselect:
            return {"dense_note": "dense distribution is only computed for the post-selected branch"}
        a = np.zeros(2**n)
        np.add.at(a, Sg[good, 0].astype(np.int64), coef[good])
        amp = _fwht(a)
        prob = amp**2
        prob /= prob.sum()
        u = np.zeros(2**n)
        brow = Bp[:, 0].astype(np.int64)
        np.add.at(u, brow, 1.0 - 2.0 * inst.v)
        f = np.rint(_fwht(u)).astype(np.int64)
        s = (m + f) // 2
        hist = np.bincount(s, weights=prob, minlength=m + 1)
        smax = int(s.max())
        res = {"dense": True, "frac_dense": float((prob * s).sum() / m), "opt_frac": smax / m,
               "p_opt": float(prob[s == smax].sum()),
               "p_opt_uniform": float((s == smax).mean()),
               "hist": hist.round(8).tolist()}
        if self.shots:
            xs = ctx.rng.choice(2**n, size=self.shots, p=prob)
            res["best_of_shots_frac"] = float(s[xs].max() / m)
            res["shots"] = self.shots
        return res


def _fwht(a):
    from .gf2 import fwht

    return fwht(a)
