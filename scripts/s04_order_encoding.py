"""Proposition 1(b) post-measurement formula, Proposition 6 (Zeno), Theorem 4 window,
Theorem 5 (longitudinal encoding with damping and dephasing; local readout).
Random-number order is identical to the original n5_misc.py.  -> results/s04_order_encoding.json"""
from _common import check, save
import numpy as np
from scipy.linalg import expm
from qtime.core import (op_on, sx, fker, flip_freqs, hamiltonian, tdist, tdist_pure, dephase,
                        liouvillian, plus_state, diag_energies, spins)

rng = np.random.default_rng(3); Delta = 0.1; res = {}

# ---- Proposition 1(b): L = (|du|/2) sum_s |f(w_i(s))| |p(s)-p(s^(i))| for diagonal priors
N, K, tau, du = 4, 2.0, 0.5, 0.01; errs = []
for t in range(200):
    p = rng.dirichlet(np.ones(2 ** N)); rho = np.diag(p).astype(complex)
    F = np.diag(fker(flip_freqs(N, K, 0, 1.0, Delta), tau)); A = du * (op_on(sx, 0, N).toarray() @ F)
    L = 0.5 * np.linalg.svd(A @ rho - rho @ A, compute_uv=False).sum()
    flip = np.arange(2 ** N) ^ (1 << (N - 1)); pred = 0.5 * du * np.sum(np.abs(np.diag(F)) * np.abs(p - p[flip]))
    errs.append(abs(L - pred) / pred)
res["postmeas_formula_maxrelerr"] = float(max(errs))
check(max(errs) < 1e-12, f"Prop. 1(b) exact for 200 diagonal priors (max rel. err {max(errs):.1e})")

# ---- Proposition 6 (Zeno): M dephasing steps inside tau; branches u=0 vs u'=du
N, K, tau = 4, 2.0, 1.0; p = rng.dirichlet(np.ones(2 ** N)); rho0 = np.diag(p).astype(complex); zen = []
for du in [0.2, 0.5]:
    for M in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        dt = tau / M; Ub = expm(-1j * hamiltonian(N, K, {0: du}, 1.0, Delta).toarray() * dt); r = rho0.copy()
        for _ in range(M):
            r = dephase(Ub @ r @ Ub.conj().T)
        T = tdist(r, rho0)                     # branch u=0 leaves a diagonal prior unchanged
        bound = 1.5 * (du * tau) ** 2 / M if du * dt <= 0.5 else np.nan
        zen.append(dict(du=du, M=M, T=T, bound=bound))
        if not np.isnan(bound):
            check(T <= bound, f"Zeno bound du={du} M={M}: {T:.2e} <= {bound:.2e}")
res["zeno"] = zen

# ---- Theorem 4 window: basis-state prior, order sensitivity vs x = |b| tau
N, tau, du = 4, 1.0, 0.005; win = []
for K in np.linspace(0.02, 3.5, 70):
    psi = np.zeros(2 ** N, complex); psi[0] = 1
    U0 = expm(-1j * hamiltonian(N, K, {}, 1.0, Delta).toarray() * tau)
    U1 = expm(-1j * hamiltonian(N, K, {0: du}, 1.0, Delta).toarray() * tau)
    T = tdist_pure(U1 @ U0 @ psi, U0 @ U1 @ psi); b = -Delta + K * (N - 1); x = abs(b) * tau
    win.append(dict(x=x, T=T, theory=du * tau * 2 * np.sin(x) ** 2 / x))
res["window"] = win

# ---- Theorem 5(i),(ii): longitudinal encoding with amplitude damping and dephasing
N, K, tau = 4, 2.0, 0.7
v = rng.normal(size=2 ** N) + 1j * rng.normal(size=2 ** N); v /= np.linalg.norm(v); rho = np.outer(v, v.conj())
a = {0: 0.3, 1: -0.2}; b = {0: -0.1, 1: 0.4}


def two_step(enc):
    La = liouvillian(hamiltonian(N, K, a, 1.0, Delta, enc), N, 0.3, 0.2)
    Lb = liouvillian(hamiltonian(N, K, b, 1.0, Delta, enc), N, 0.3, 0.2)
    Ea, Eb = expm((La * tau).toarray()), expm((Lb * tau).toarray())
    r1 = (Eb @ (Ea @ rho.reshape(-1))).reshape(16, 16); r2 = (Ea @ (Eb @ rho.reshape(-1))).reshape(16, 16)
    return tdist(r1, r2), Ea, Eb


res["longit_order_T"], Ea, Eb = two_step("z")
res["transv_order_T"] = two_step("x")[0]
ra = (Ea @ rho.reshape(-1)).reshape(16, 16); rb = (Eb @ rho.reshape(-1)).reshape(16, 16)
res["longit_readoutTV"] = float(0.5 * np.abs(np.real(np.diag(ra - rb))).sum()); res["longit_T"] = float(tdist(ra, rb))
print(f"  longitudinal order T = {res['longit_order_T']:.1e}; transverse = {res['transv_order_T']:.3f}; "
      f"longitudinal one step: full T = {res['longit_T']:.3f}, Z-readout TV = {res['longit_readoutTV']:.1e}")
check(res["longit_order_T"] < 1e-12, "Theorem 5(i): longitudinal inputs are order-blind with damping+dephasing")
check(res["longit_readoutTV"] < 1e-12, "Theorem 5(ii): longitudinal inputs invisible to Z readout")

# ---- Theorem 5(iv): local readout of a longitudinal imprint
loc = []; tau, du = 0.5, 0.05
for K in [0.3, 1.0]:
    for N in range(2, 13):
        psi = plus_state(N); E = diag_energies(N, K, 1.0, Delta); s0 = spins(N)[:, 0]
        x1 = np.exp(-1j * E * tau) * psi; x2 = np.exp(-1j * (E + du * s0) * tau) * psi
        red = lambda v: v.reshape(2, -1) @ v.reshape(2, -1).conj().T
        loc.append(dict(K=K, N=N, T=tdist(red(x1), red(x2)), pred=abs(np.cos(2 * K * tau)) ** (N - 1) * abs(np.sin(du * tau))))
dev = max(abs(d["T"] - d["pred"]) for d in loc)
res["local"] = loc; res["local_maxdev"] = dev
check(dev < 1e-14, f"Theorem 5(iv) local readout formula (max dev {dev:.1e})")
save("s04_order_encoding.json", res)
