"""Dense/sparse tools for small networks (N <= 6).

Model of the manuscript with uniform all-to-all couplings:
    H0 = -Delta * sum_i Z_i + K * sum_{i<j} Z_i Z_j      (K = beta*J, h_i = -Delta)
    V(u) = sum_{i in S} u_i X_i   (transverse, enc="x")  or  sum u_i Z_i  (enc="z")
Amplitude damping L_k = sigma^-_k at rate gamma; pure dephasing L_k = Z_k at rate
gphi/2 (coherence decay rate gphi).
Convention: basis bit 0 = spin up (s=+1); sigma^- maps s=+1 -> s=-1; qubit 0 is the
most significant bit of the configuration index. Qubit 0 here is "qubit 1" in the paper.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import expm, sqrtm, eigh

sx = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
sz_ = np.array([1., -1.])
sm = sp.csr_matrix(np.array([[0, 0], [1, 0]], dtype=complex))  # QuTiP sigmam convention (matches manuscript numerics): damps s=+1 -> s=-1
I2 = sp.identity(2, format="csr", dtype=complex)


def op_on(op, i, N):
    out = sp.identity(1, format="csr", dtype=complex)
    for k in range(N):
        out = sp.kron(out, op if k == i else I2, format="csr")
    return out


def spins(N):
    """s[config, i] in {+1,-1}; config index uses qubit 0 as most significant bit."""
    idx = np.arange(2 ** N)
    bits = (idx[:, None] >> (N - 1 - np.arange(N))[None, :]) & 1
    return 1 - 2 * bits


def diag_energies(N, beta, J=1.0, Delta=0.1):
    s = spins(N)
    M = s.sum(1)
    zz = (M ** 2 - N) / 2.0  # sum_{i<j} s_i s_j
    return -Delta * M + beta * J * zz


def hamiltonian(N, beta, inputs, J=1.0, Delta=0.1, enc="x"):
    """inputs: dict {qubit: amplitude}"""
    H = sp.diags(diag_energies(N, beta, J, Delta)).astype(complex).tocsr()
    for i, a in inputs.items():
        if enc == "x":
            H = H + a * op_on(sx, i, N)
        else:
            H = H + a * sp.diags(spins(N)[:, i].astype(complex))
    return H.tocsr()


def flip_freqs(N, beta, i, J=1.0, Delta=0.1):
    """omega_i(s) = E_{s^(i)} - E_s for every config s."""
    E = diag_energies(N, beta, J, Delta)
    flip = np.arange(2 ** N) ^ (1 << (N - 1 - i))
    return E[flip] - E


def fker(w, dt):
    out = np.full(w.shape, dt, dtype=complex)
    nz = np.abs(w) > 1e-12
    out[nz] = (np.exp(1j * w[nz] * dt) - 1) / (1j * w[nz])
    return out


# ---------- bounds (first order, trace distance) ----------
def bound_collective_guess(dI, N, beta, dt, J=1.0):
    """Collective-field heuristic 4|dI| dt/(2+Omega_N dt), Omega_N = beta J N.

    NOT a theorem of the manuscript. It encodes the 'collective-field guess'
    discussed in the Introduction and is kept only as a comparison model;
    Corollary 1 of the manuscript shows that no bound of this form can hold
    for all prior states.
    """
    return 4 * np.sum(np.abs(dI)) * dt / (2 + beta * J * N * dt)


def bound_opnorm(dI_dict, N, beta, dt, J=1.0, Delta=0.1):
    """Correct worst-case bound: sum |dI_i| max_s |f(omega_i(s))|."""
    return sum(abs(d) * np.abs(fker(flip_freqs(N, beta, i, J, Delta), dt)).max()
               for i, d in dI_dict.items())


def bound_state(dI_dict, p, N, beta, dt, J=1.0, Delta=0.1):
    """State-dependent bound: sum |dI_i| sqrt(E_p |f(omega_i(s))|^2), p = diag(rho)."""
    return sum(abs(d) * np.sqrt(np.sum(p * np.abs(fker(flip_freqs(N, beta, i, J, Delta), dt)) ** 2))
               for i, d in dI_dict.items())


# ---------- distances ----------
def tdist_pure(a, b):
    return np.sqrt(max(0.0, 1 - abs(np.vdot(a, b)) ** 2))


def infid_pure(a, b):
    return 1 - abs(np.vdot(a, b)) ** 2


def tdist(r, s):
    w = np.linalg.eigvalsh((r - s + (r - s).conj().T) / 2)
    return 0.5 * np.abs(w).sum()


def fidelity(r, s):
    w, v = eigh((r + r.conj().T) / 2)
    w = np.clip(w, 0, None)
    sr = (v * np.sqrt(w)) @ v.conj().T
    m = sr @ s @ sr
    ev = np.clip(np.linalg.eigvalsh((m + m.conj().T) / 2), 0, None)
    return float(np.sum(np.sqrt(ev)) ** 2)


# ---------- Lindblad evolution ----------
def liouvillian(H, N, gamma, gphi=0.0):
    d = 2 ** N
    Id = sp.identity(d, format="csr", dtype=complex)
    # row-major vec: vec(A rho B) = (A kron B^T) vec(rho)
    L = -1j * (sp.kron(H, Id) - sp.kron(Id, H.T))
    def add_jump(Lk, rate):
        nonlocal L
        LdL = (Lk.conj().T @ Lk).tocsr()
        L = L + rate * (sp.kron(Lk, Lk.conj()) - 0.5 * sp.kron(LdL, Id) - 0.5 * sp.kron(Id, LdL.T))
    if gamma > 0:
        for k in range(N):
            add_jump(op_on(sm, k, N), gamma)
    if gphi > 0:  # pure dephasing, L = sz, rate gphi/2 -> coherence decay gphi
        for k in range(N):
            add_jump(sp.diags(spins(N)[:, k].astype(complex)).tocsr(), gphi / 2)
    return L.tocsr()


def evolve(rho, L, dt, N):
    d = 2 ** N
    if d <= 64:
        return (expm((L * dt).toarray()) @ rho.reshape(-1)).reshape(d, d)
    return expm_multiply(L * dt, rho.reshape(-1)).reshape(d, d)


def dephase(rho):
    return np.diag(np.diag(rho))


def plus_state(N):
    return np.ones(2 ** N, dtype=complex) / np.sqrt(2 ** N)


def zero_state(N):
    v = np.zeros(2 ** N, dtype=complex); v[0] = 1; return v
