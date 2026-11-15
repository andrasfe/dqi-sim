"""DQI polynomial weights and the quadratic-form picture of DQI performance.

DQI prepares  Σ_k w_k / sqrt(C(m,k)) Σ_{|y|=k} (-1)^{v·y} |B^T y>  and measures x after a
Hadamard transform. For max-XORSAT, with f(x) = #sat - #unsat, every evaluator in this
package reduces DQI to two (ℓ+1)x(ℓ+1) matrices:

    post-selection success probability   P_succ = w^T G w            (|w| = 1)
    expected objective (post-selected)   <f>    = w^T F w / w^T G w
    expected satisfied fraction          <s>/m  = 1/2 + <f> / (2m)

Perfect decoding with 2ℓ+1 < d^⊥ gives G = I and F = A, the paper's tridiagonal matrix
(off-diagonal sqrt(k(m-k+1))). Imperfect decoders and syndrome collisions change F and G;
the evaluators estimate or compute them. Weight strategies then pick w from (F, G).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import eigh, eigh_tridiagonal
from scipy.special import gammaln

from .registry import P, register


def a_offdiag(m: int, ell: int) -> np.ndarray:
    k = np.arange(1, ell + 1, dtype=float)
    return np.sqrt(k * (m - k + 1))


def A_matrix(m: int, ell: int, d: float = 0.0) -> np.ndarray:
    """Paper's A^(m,ℓ,d): diag(0, d, 2d, ..., ℓd), off-diagonal sqrt(k(m-k+1))."""
    A = np.diag(np.arange(ell + 1) * d)
    off = a_offdiag(m, ell)
    A += np.diag(off, 1) + np.diag(off, -1)
    return A


def ideal_top(m: int, ell: int, d: float = 0.0) -> tuple[float, np.ndarray]:
    """Largest eigenvalue/eigenvector of A (fast tridiagonal solver)."""
    if ell == 0:
        return 0.0, np.ones(1)
    lam, vec = eigh_tridiagonal(np.arange(ell + 1) * d, a_offdiag(m, ell), select="i",
                                select_range=(ell, ell))
    w = vec[:, 0]
    return float(lam[0]), w * np.sign(w.sum() or 1.0)


def ideal_fraction(m: int, ell: int, r_over_p: float = 0.5, p: int = 2) -> float:
    """Finite-m expected satisfied fraction with optimal weights and perfect decoding.

    General max-LINSAT over F_p with r satisfying values per constraint (Lemma 9.2 form):
        <s> = m r/p + sqrt(r(p-r))/p * λ_max(A^(m,ℓ,d)),  d = (p-2r)/sqrt(r(p-r)).
    """
    r = r_over_p * p
    d = (p - 2 * r) / math.sqrt(r * (p - r))
    lam, _ = ideal_top(m, ell, d)
    return min(1.0, r / p + math.sqrt(r * (p - r)) / p * lam / m)


def semicircle(ell_over_m: float, r_over_p: float = 0.5) -> float:
    """Asymptotic semicircle law (paper Eq. 6)."""
    mu, rho = ell_over_m, r_over_p
    if rho > 1 - mu:
        return 1.0
    return (math.sqrt(mu * (1 - rho)) + math.sqrt(rho * (1 - mu))) ** 2


def evaluate(w: np.ndarray, F: np.ndarray, G: np.ndarray, m: int) -> dict:
    w = np.asarray(w, float)
    w = w / np.linalg.norm(w)
    g = float(w @ G @ w)
    f = float(w @ F @ w) / g if g > 1e-300 else float("nan")
    out = {"frac": 0.5 + f / (2 * m), "f": f, "p_succ": g,
           # expected circuit runs per accepted sample when post-selecting
           "repetitions": 1.0 / g if g > 1e-300 else float("inf")}
    if g < 1e-3:
        out["warning"] = f"post-selection success {g:.1e}: frac is conditional on a near-impossible event"
    return out


def top_generalized(F: np.ndarray, G: np.ndarray) -> np.ndarray:
    """argmax_w  w^T F w / w^T G w, on the coordinates where G is non-negligible."""
    diag = np.diag(G)
    keep = np.flatnonzero(diag > 1e-12 * max(diag.max(), 1e-300))
    Fs, Gs = F[np.ix_(keep, keep)], G[np.ix_(keep, keep)]
    Gs = Gs + 1e-14 * np.eye(len(keep)) * np.trace(Gs) / len(keep)
    lam, vec = eigh(Fs, Gs, subset_by_index=[len(keep) - 1, len(keep) - 1])
    w = np.zeros(F.shape[0])
    w[keep] = vec[:, 0]
    w = w * np.sign(w.sum() or 1.0)
    return w / np.linalg.norm(w)


def log_binom(m: int, k: np.ndarray) -> np.ndarray:
    return gammaln(m + 1) - gammaln(k + 1) - gammaln(m - k + 1)


# ----------------------------------------------------------------------------- strategies


class _Weights:
    needs_response = False

    def choose(self, m: int, ell: int, F: np.ndarray | None = None, G: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _norm(w):
        w = np.asarray(w, float)
        return w / np.linalg.norm(w)


@register("weights", "optimal-ideal", "Paper's choice: principal eigenvector of A (assumes perfect decoding)")
class OptimalIdeal(_Weights):
    def choose(self, m, ell, F=None, G=None):
        return self._norm(ideal_top(m, ell)[1])


@register("weights", "optimal-aware", "Decoder-aware: maximise w^T F w / w^T G w for the evaluator's measured response "
                                      "(falls back to optimal-ideal when no response is available)")
class OptimalAware(_Weights):
    needs_response = True

    def choose(self, m, ell, F=None, G=None):
        if F is None or G is None:
            return OptimalIdeal().choose(m, ell)
        return top_generalized(F, G)


@register("weights", "product", "Gibbs-like P(f) = Π_i (1 + t f_i) truncated at degree ℓ: w_k ∝ t^k sqrt(C(m,k))",
          P("t", float, 0.3, "tilt t ∈ (0, 1]; amplitude ∝ (1+t)^#sat (1-t)^#unsat before truncation", min=1e-6, max=1))
class Product(_Weights):
    def __init__(self, t):
        self.t = t

    def choose(self, m, ell, F=None, G=None):
        k = np.arange(ell + 1)
        logw = k * math.log(self.t) + 0.5 * log_binom(m, k)
        return self._norm(np.exp(logw - logw.max()))


@register("weights", "single", "All weight on degree ℓ (a pure Dicke shell)")
class Single(_Weights):
    def choose(self, m, ell, F=None, G=None):
        w = np.zeros(ell + 1)
        w[-1] = 1.0
        return w


@register("weights", "uniform", "Equal weight on every degree 0..ℓ")
class Uniform(_Weights):
    def choose(self, m, ell, F=None, G=None):
        return self._norm(np.ones(ell + 1))


@register("weights", "custom", "Explicit weights w_0..w_ℓ (padded/truncated to ℓ+1, then normalised)",
          P("w", "list[float]", [1.0, 1.0], "comma-separated weights"))
class Custom(_Weights):
    def __init__(self, w):
        self.w = w

    def choose(self, m, ell, F=None, G=None):
        w = np.zeros(ell + 1)
        n = min(ell + 1, len(self.w))
        w[:n] = self.w[:n]
        return self._norm(w)
