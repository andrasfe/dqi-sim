import math

import numpy as np
import pytest

from dqisim.weights import A_matrix, ideal_fraction, ideal_top, semicircle, top_generalized


def test_semicircle_reproduces_paper_opi_number():
    # paper: n/p = 1/10, r/p = 1/2  ->  ℓ/m ≈ n/(2p) = 1/20  ->  <s>/m ≈ 0.7179
    assert semicircle(1 / 20, 0.5) == pytest.approx(0.7179, abs=1e-4)


def test_semicircle_saturates():
    assert semicircle(0.6, 0.5) == 1.0
    assert semicircle(0.0, 0.5) == pytest.approx(0.5)


@pytest.mark.parametrize("mu", [0.02, 0.05, 0.2])
@pytest.mark.parametrize("rho,p", [(0.5, 2), (0.3, 10), (0.7, 10)])
def test_finite_eigenvalue_converges_to_semicircle(mu, rho, p):
    m = 40000
    ell = int(mu * m)
    assert ideal_fraction(m, ell, rho, p) == pytest.approx(semicircle(mu, rho), abs=5e-3)


def test_tridiagonal_solver_matches_dense():
    m, ell = 50, 9
    lam, w = ideal_top(m, ell)
    ev = np.linalg.eigvalsh(A_matrix(m, ell))
    assert lam == pytest.approx(ev[-1])
    assert np.linalg.norm(w) == pytest.approx(1)


def test_generalized_reduces_to_ideal_when_G_is_identity():
    m, ell = 30, 6
    A = A_matrix(m, ell)
    w = top_generalized(A, np.eye(ell + 1))
    assert abs(w @ ideal_top(m, ell)[1]) == pytest.approx(1, abs=1e-8)
