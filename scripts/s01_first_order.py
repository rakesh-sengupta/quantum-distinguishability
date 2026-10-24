"""Theorem 1 (explicit remainder), Theorem 2 (population bound) and the remainder of
Theorem 4 (order) on random priors, N = 3-5.  -> results/s01_first_order.json"""
from _common import check, save
import numpy as np
from scipy.linalg import expm
from qtime.core import op_on, sx, fker, flip_freqs, hamiltonian, tdist, bound_state

Delta = 0.1


def kernel_op(d, N, K, tau, kind="f"):
    A = np.zeros((2 ** N, 2 ** N), complex)
    for i, v in d.items():
        w = flip_freqs(N, K, i, 1.0, Delta)
        f = fker(w, tau)
        k = f if kind == "f" else f * (np.exp(1j * w * tau) - 1)
        A += v * (op_on(sx, i, N).toarray() @ np.diag(k))
    return A


def half_tnorm_comm(A, rho):
    return 0.5 * np.linalg.svd(A @ rho - rho @ A, compute_uv=False).sum()


# ---------------------------------------------------------------- Part A: Theorem 2 (1296 cases)
rng = np.random.default_rng(7)
LB, small_rel = [], []
for N in [3, 4, 5]:
    for K in [0.5, 2.0, 8.0]:
        for tau in [0.5, 1.0]:
            for unorm in [0.0, 0.1, 0.5]:
                for dnorm in [0.01, 0.1]:
                    for trial in range(12):
                        S = [0, 1]
                        u = dict(zip(S, rng.normal(size=2))); s = sum(abs(v) for v in u.values()) or 1
                        u = {i: unorm * v / s for i, v in u.items()}
                        d = dict(zip(S, rng.normal(size=2))); s = sum(abs(v) for v in d.values())
                        d = {i: dnorm * v / s for i, v in d.items()}
                        up = {i: u[i] + d[i] for i in S}
                        if trial % 3 == 0:
                            v = rng.normal(size=2 ** N) + 1j * rng.normal(size=2 ** N); v /= np.linalg.norm(v); rho = np.outer(v, v.conj())
                        elif trial % 3 == 1:
                            w = rng.normal(size=(2 ** N, 3)) + 1j * rng.normal(size=(2 ** N, 3)); rho = w @ w.conj().T; rho /= np.trace(rho)
                        else:
                            k = rng.integers(2 ** N); rho = np.zeros((2 ** N, 2 ** N), complex); rho[k, k] = 1
                        Ua = expm(-1j * hamiltonian(N, K, u, 1.0, Delta).toarray() * tau)
                        Ub = expm(-1j * hamiltonian(N, K, up, 1.0, Delta).toarray() * tau)
                        T = tdist(Ua @ rho @ Ua.conj().T, Ub @ rho @ Ub.conj().T)
                        L = half_tnorm_comm(kernel_op(d, N, K, tau), rho)
                        B = bound_state(d, np.real(np.diag(rho)), N, K, tau, 1.0, Delta)
                        LB.append(L / B)
                        if unorm == 0 and dnorm == 0.01:
                            small_rel.append(abs(T - L) / L)
LB = np.array(LB)
print(f"Theorem 2: {LB.size} cases, median L/B = {np.median(LB):.3f}, min = {LB.min():.3f}, max = {LB.max():.4f}")
check(np.all(LB <= 1 + 1e-9), "Theorem 2 population bound L_in <= B never violated")
print(f"  small inputs (u=0, |du|_1=0.01): median |T-L|/L = {np.median(small_rel):.1e}")

# ---------------------------------------------------------------- Part B: Theorems 1 and 4 remainders (972 cases each)
rng = np.random.default_rng(11)


def rand_state(N, k):
    if k == 0:
        v = rng.normal(size=2 ** N) + 1j * rng.normal(size=2 ** N); v /= np.linalg.norm(v); return np.outer(v, v.conj())
    if k == 1:
        w = rng.normal(size=(2 ** N, 3)) + 1j * rng.normal(size=(2 ** N, 3)); r = w @ w.conj().T; return r / np.trace(r)
    r = np.zeros((2 ** N, 2 ** N), complex); j = rng.integers(2 ** N); r[j, j] = 1; return r


Rs, Ro = [], []
for N in [3, 4, 5]:
    for K in [0.5, 2.0, 8.0]:
        for tau in [0.5, 1.0]:
            for un in [0.0, 0.1, 0.5]:
                for dn in [0.01, 0.1, 0.4]:
                    for t in range(6):
                        S = [0, 1]; x = rng.normal(size=2)
                        u = {i: (un * x[j] / np.abs(x).sum()) for j, i in enumerate(S)}
                        y = rng.normal(size=2)
                        d = {i: (dn * y[j] / np.abs(y).sum()) for j, i in enumerate(S)}
                        up = {i: u[i] + d[i] for i in S}
                        rho = rand_state(N, t % 3)
                        Ua = expm(-1j * hamiltonian(N, K, u, 1.0, Delta).toarray() * tau)
                        Ub = expm(-1j * hamiltonian(N, K, up, 1.0, Delta).toarray() * tau)
                        T = tdist(Ua @ rho @ Ua.conj().T, Ub @ rho @ Ub.conj().T)
                        L = half_tnorm_comm(kernel_op(d, N, K, tau), rho)
                        a = dn * tau
                        R = un * dn * tau ** 2 + a ** 2 * (1 + a / 2 + a ** 2 / 8)          # Eq. (13)
                        Rs.append(abs(T - L) / R)
                        Wab, Wba = Ub @ Ua, Ua @ Ub                                      # a=u then b=up, and reverse
                        To = tdist(Wab @ rho @ Wab.conj().T, Wba @ rho @ Wba.conj().T)
                        Lo = half_tnorm_comm(kernel_op(d, N, K, tau, "g"), rho)
                        Lam0 = max(un, sum(abs(v) for v in up.values())); c = 2 * tau * dn
                        R2 = 4 * Lam0 * dn * tau ** 2 + c ** 2 * (1 + c / 2 + c ** 2 / 8)   # Eq. (order remainder)
                        Ro.append(abs(To - Lo) / R2)
Rs, Ro = np.array(Rs), np.array(Ro)
print(f"Theorem 1: {Rs.size} cases, max |T-L|/R = {Rs.max():.3f}, median {np.median(Rs):.3f}")
print(f"Theorem 4: {Ro.size} cases, max |T-L|/R = {Ro.max():.3f}, median {np.median(Ro):.3f}")
check(np.all(Rs <= 1 + 1e-9), "Theorem 1 remainder never violated")
check(np.all(Ro <= 1 + 1e-9), "Theorem 4 remainder never violated")

save("s01_first_order.json", dict(thm2_n=int(LB.size), thm2_median=float(np.median(LB)), thm2_min=float(LB.min()),
                                   small_rel_median=float(np.median(small_rel)),
                                   thm1_n=int(Rs.size), thm1_max=float(Rs.max()),
                                   thm4_n=int(Ro.size), thm4_max=float(Ro.max())))
