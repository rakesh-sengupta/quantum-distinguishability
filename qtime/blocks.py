"""Exact symmetry-reduced (block) calculations for one driven qubit.

When H0 is diagonal and only qubit 1 is driven, the configuration r of the
other N-1 qubits is conserved. The prior |+>^N then splits into independent
two-level problems for qubit 1 with local field b(r) = h_1 + sum_j K_1j r_j,
weighted by the probability of r (uniform). Couplings among the spectators only
add r-dependent phases that are common to both branches and cancel.

For uniform couplings K_1j = K the field depends on r only through
m = sum_j r_j, which is binomially distributed; this makes exact calculations
for N ~ 10^4-10^5 qubits cheap. Nothing here simulates a generic 2^N-dimensional
many-body state.
"""
import numpy as np
from scipy.stats import binom
from scipy.integrate import quad

PLUS = np.array([1, 1], complex) / np.sqrt(2)


# ---------------------------------------------------------------- blocks
def uniform_blocks(N, K, h):
    """Local fields b_m = h + K m and weights P_n(m) for uniform couplings."""
    n = N - 1
    k = np.arange(n + 1)
    m = n - 2 * k
    P = binom.pmf(k, n, 0.5)
    keep = P > 1e-300
    return h + K * m[keep], P[keep], m[keep]


def U2(b, v, t):
    """Vectorised exp(-i t (b Z + v X)) for arrays b (v scalar or array)."""
    b = np.asarray(b, float)
    v = np.broadcast_to(np.asarray(v, float), b.shape)
    Om = np.sqrt(b * b + v * v)
    c = np.cos(Om * t)
    s = np.where(Om > 0, np.sin(Om * t) / np.where(Om > 0, Om, 1), t)
    row0 = np.stack([c - 1j * s * b, -1j * s * v], -1)
    row1 = np.stack([-1j * s * v, c + 1j * s * b], -1)
    return np.stack([row0, row1], -2)


def _apply(U, psi):
    if psi.ndim == 1:
        return np.einsum("mij,j->mi", U, psi)
    return np.einsum("mij,mj->mi", U, psi)


def branch_states(b, tau, du, kind="in", u0=0.0):
    """Per-block states of qubit 1 in the two branches.

    kind="in":  u0 vs u0+du for one step.
    kind="ord": (u0 then u0+du) vs (u0+du then u0).
    """
    Ua, Ub = U2(b, u0, tau), U2(b, u0 + du, tau)
    if kind == "in":
        return _apply(Ua, PLUS), _apply(Ub, PLUS)
    return _apply(Ub, _apply(Ua, PLUS)), _apply(Ua, _apply(Ub, PLUS))


def exact_quantities(b, P, tau, du, kind="in"):
    """Exact full-state and restricted-readout distinguishabilities.

    Returns dict with
      T      : trace distance of the full N-qubit states (Helstrom, single shot)
      Et     : E_r t_r, the largest single-shot distinguishability of any readout
               that measures the spectators in the computational basis
      Et2    : E_r t_r^2 (block, 'incoherent' part of T^2)
      varz   : Var_r z_r  (coherent, relative-phase part of T^2); T^2 = Et2 + varz
      Tloc   : trace distance of the reduced states of qubit 1
    """
    a, c = branch_states(b, tau, du, kind)
    z = np.einsum("mi,mi->m", a.conj(), c)
    t2 = np.clip(1 - np.abs(z) ** 2, 0, None)
    Ez = np.sum(P * z)
    T = np.sqrt(max(0.0, 1 - abs(Ez) ** 2))
    ra = np.einsum("m,mi,mj->ij", P, a, a.conj())
    rc = np.einsum("m,mi,mj->ij", P, c, c.conj())
    Tloc = 0.5 * np.abs(np.linalg.eigvalsh(ra - rc)).sum()
    return dict(T=T, Et=float(np.sum(P * np.sqrt(t2))), Et2=float(np.sum(P * t2)),
                varz=float(np.sum(P * np.abs(z - Ez) ** 2)), Tloc=float(Tloc))


# ---------------------------------------------------------------- kernels
def fker(w, tau):
    w = np.asarray(w, float)
    out = np.full(w.shape, tau, complex)
    nz = np.abs(w) > 1e-14
    out[nz] = (np.exp(1j * w[nz] * tau) - 1) / (1j * w[nz])
    return out


def gker(w, tau):
    return fker(w, tau) * (np.exp(1j * np.asarray(w) * tau) - 1)


def L_first_order(b, P, tau, du, kind="in"):
    """Exact finite-N leading term L_in (or L_ord) for the prior |+>^N.

    L^2 = du^2 ( E|k(2b)|^2 - (E Re k(2b))^2 ),  k = f or g.
    (Prop. 1(a) with <A^2> = E|k|^2 and <A> = E Re k.)
    """
    k = fker(2 * b, tau) if kind == "in" else gker(2 * b, tau)
    return abs(du) * np.sqrt(max(0.0, np.sum(P * np.abs(k) ** 2) - np.sum(P * k.real) ** 2))


# ---------------------------------------------------------------- asymptotics
def S_trunc(K, tau, h, n_parity, kind="in", M=200001):
    """S_n (kind='in') or S_n^ord (kind='ord') by direct summation over m = n (mod 2)."""
    m = np.arange(-M, M + 1)
    m = m[(m % 2) == n_parity]
    b = h + K * m
    if kind == "tilde":                       # S~_n of Proposition 2(iv)
        return np.minimum(tau ** 2, 1 / np.maximum(b * b, 1e-300)).sum()
    if kind == "in":
        bb = np.where(b == 0, 1, b)
        phi = np.where(np.abs(b) > 1e-14, np.sin(b * tau) ** 2 / bb ** 2, tau ** 2)
    else:
        bb = np.where(b == 0, 1, np.abs(b))
        phi = np.where(np.abs(b) > 1e-14, (2 * np.sin(b * tau) ** 2 / bb) ** 2, 0.0)
    return phi.sum()


def S_poisson(K, tau, h, n_parity, kind="in"):
    """Closed form of S_n from Poisson summation (Corollary 2).

    S_n = (1/2|K|) sum_{|j| <= J} phihat(pi j/|K|) (-1)^{j n} cos(pi j h/K),
    phihat(k) = pi (tau - |k|/2)_+             for the input kernel |f|^2,
    phihat(k) = (pi/2)[4(2tau-|k|)_+ - (4tau-|k|)_+]  for the order kernel |g|^2.
    """
    K = abs(K)
    if kind == "in":
        hat = lambda k: np.pi * max(tau - abs(k) / 2, 0.0)
        kmax = 2 * tau
    else:
        hat = lambda k: 0.5 * np.pi * (4 * max(2 * tau - abs(k), 0.0) - max(4 * tau - abs(k), 0.0))
        kmax = 4 * tau
    J = int(np.floor(kmax * K / np.pi))
    tot = hat(0.0)
    for j in range(1, J + 1):
        tot += 2 * hat(np.pi * j / K) * np.cos(np.pi * j * h / K) * (-1) ** (j * n_parity)
    return tot / (2 * K)


def asym(N, K, tau, h, du, kind="in"):
    """Asymptotic formula of Theorem 3 / Corollary 5."""
    n = N - 1
    return abs(du) * (2 / (np.pi * n)) ** 0.25 * np.sqrt(S_poisson(K, tau, h, n % 2, kind))


def E_mu2(b, P, tau):
    """E min(tau^2, 1/b^2), the quantity in Proposition 2(iii)."""
    return float(np.sum(P * np.minimum(tau ** 2, 1 / np.maximum(b * b, 1e-300))))


# ---------------------------------------------------------------- general couplings
def L_fourier(Ks, h, tau, du):
    """Proposition 3: L_in for |+>^N and arbitrary couplings K_1j via the
    characteristic function chi(k) = e^{ikh} prod_j cos(k K_1j) of the local field."""
    Ks = np.asarray(Ks, float)

    def rechi(k):
        return np.cos(k * h) * np.prod(np.cos(k * Ks))

    pts = [0.0]
    E2 = quad(lambda k: (tau - k / 2) * rechi(k), 0, 2 * tau, limit=4000, points=pts)[0]
    E1 = quad(lambda k: 0.5 * rechi(k), 0, 2 * tau, limit=4000, points=pts)[0]
    return abs(du) * np.sqrt(max(0.0, E2 - E1 ** 2))


def L_bruteforce(Ks, h, tau, du):
    """Same quantity by enumerating all 2^(N-1) spectator configurations (small N)."""
    Ks = np.asarray(Ks, float)
    n = len(Ks)
    r = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
    b = h + r @ Ks
    return L_first_order(b, np.full(b.shape, 2.0 ** -n), tau, du)
