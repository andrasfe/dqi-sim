"""GF(2) primitives: Walsh-Hadamard transform, bit-packed elimination, encodings."""
from __future__ import annotations

import numpy as np

from . import safety


def fwht(a: np.ndarray) -> np.ndarray:
    """In-place unnormalised Walsh-Hadamard transform of a length-2^n vector."""
    n = a.size
    if n & (n - 1):
        raise ValueError("length must be a power of two")
    h = 1
    while h < n:
        v = a.reshape(-1, 2, h)
        x = v[:, 0, :].copy()
        v[:, 0, :] += v[:, 1, :]
        v[:, 1, :] = x - v[:, 1, :]
        h *= 2
        safety.checkpoint()
    return a


def rows_to_int(B: np.ndarray) -> np.ndarray:
    """Encode each row of a 0/1 matrix (≤63 columns) as an integer, bit j = column j."""
    n = B.shape[1]
    if n > 63:
        raise ValueError("integer encoding needs ≤63 columns")
    w = (np.uint64(1) << np.arange(n, dtype=np.uint64)).astype(np.uint64)
    return (B.astype(np.uint64) * w).sum(axis=1).astype(np.uint64)


def pack_rows(X: np.ndarray) -> np.ndarray:
    """Pack 0/1 rows into uint64 words (little-endian bit order within the row)."""
    X = np.ascontiguousarray(X, dtype=np.uint8)
    nb = X.shape[1]
    W = max(1, (nb + 63) // 64)
    pad = W * 64 - nb
    if pad:
        X = np.concatenate([X, np.zeros((X.shape[0], pad), np.uint8)], axis=1)
    bytes_ = np.packbits(X, axis=1, bitorder="little")
    return bytes_.view(np.uint64).reshape(X.shape[0], W)


def rref_solve(H: np.ndarray, s: np.ndarray, order: np.ndarray | None = None):
    """Solve H y = s over GF(2), putting y's support on the earliest independent columns
    of `order` (an information set). Returns (y, pivots, consistent).

    H: (r, m) 0/1; s: (r,) 0/1. Bit-packed row operations, O(rank * r * m / 8).
    """
    r, m = H.shape
    if order is None:
        order = np.arange(m)
    A = np.concatenate([H[:, order], s.reshape(-1, 1)], axis=1).astype(np.uint8)
    A = np.packbits(A, axis=1)  # big-endian bits within each byte
    row = 0
    pivots = []
    for c in range(m):
        if row == r:
            break
        byte, bit = c >> 3, np.uint8(0x80 >> (c & 7))
        hits = np.flatnonzero(A[row:, byte] & bit)
        if hits.size == 0:
            continue
        p = row + hits[0]
        if p != row:
            A[[row, p]] = A[[p, row]]
        mask = (A[:, byte] & bit) != 0
        mask[row] = False
        A[mask] ^= A[row]
        pivots.append(c)
        row += 1
        if (c & 255) == 0:
            safety.checkpoint()
    aug_byte, aug_bit = m >> 3, np.uint8(0x80 >> (m & 7))
    aug = (A[:, aug_byte] & aug_bit) != 0
    consistent = not aug[row:].any()
    y = np.zeros(m, np.uint8)
    piv = np.asarray(pivots, dtype=np.int64)
    if piv.size:
        y[order[piv]] = aug[: piv.size]
    return y, order[piv] if piv.size else piv, consistent


def rank(H: np.ndarray) -> int:
    _, piv, _ = rref_solve(H, np.zeros(H.shape[0], np.uint8))
    return int(piv.size)


def min_distance_bruteforce(H: np.ndarray, max_weight: int) -> int | None:
    """Minimum weight of a nonzero y with H y = 0, searched up to max_weight (tests only)."""
    from itertools import combinations

    cols = [H[:, j].astype(np.uint8) for j in range(H.shape[1])]
    for w in range(1, max_weight + 1):
        for c in combinations(range(H.shape[1]), w):
            acc = np.zeros(H.shape[0], np.uint8)
            for j in c:
                acc ^= cols[j]
            if not acc.any():
                return w
    return None
