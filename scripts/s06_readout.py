"""Readout: what a restricted measurement can access.
 (1) Proposition 7: Z-basis order-blindness for diagonal priors and one transverse input qubit, and its
     failure for two input qubits or a coherent prior;
 (2) sequence-order discrimination benchmark with finite shots (basis prior);
 (3) Proposition 8: |+>^N prior, Helstrom vs spectator-Z / local readouts (input and order), N up to 40001.
-> results/s06_readout.json"""
from _common import check, save
import numpy as np
from scipy.linalg import expm
from scipy.stats import norm
from qtime.core import hamiltonian, op_on, sx, tdist, plus_state
from qtime.blocks import uniform_blocks, exact_quantities

Delta = 0.1
out = {}

# ---------------------------------------------------------------- (1) Z-blindness
rng = np.random.default_rng(31)


def z_tv_order(N, K, tau, a, b, rho):
    Ua = expm(-1j * hamiltonian(N, K, a, 1.0, Delta).toarray() * tau)
    Ub = expm(-1j * hamiltonian(N, K, b, 1.0, Delta).toarray() * tau)
    r1 = Ub @ Ua @ rho @ (Ub @ Ua).conj().T; r2 = Ua @ Ub @ rho @ (Ua @ Ub).conj().T
    return 0.5 * np.abs(np.real(np.diag(r1 - r2))).sum(), tdist(r1, r2)


worst1, multi, coh = 0.0, [], []
for N in [3, 4, 5]:
    for K in [0.5, 2.0, 8.0]:
        for tau in [0.5, 1.0]:
            for t in range(10):
                p = rng.dirichlet(np.ones(2 ** N)); rho = np.diag(p).astype(complex)
                a = {0: rng.normal()}; b = {0: rng.normal()}
                worst1 = max(worst1, z_tv_order(N, K, tau, a, b, rho)[0])
                a2 = {0: rng.normal(), 1: rng.normal()}; b2 = {0: rng.normal(), 1: rng.normal()}
                multi.append(z_tv_order(N, K, tau, a2, b2, rho)[0])
                psi = plus_state(N); coh.append(z_tv_order(N, K, tau, a, b, np.outer(psi, psi.conj()))[0])
print(f"(1) Z-readout TV between orders: one input, diagonal priors: max {worst1:.1e}; "
      f"two inputs: median {np.median(multi):.2e}; coherent prior |+>^N: median {np.median(coh):.2e}")
check(worst1 < 1e-12, "Z readout is exactly order-blind for diagonal priors with one input qubit")
check(np.median(multi) > 1e-4 and np.median(coh) > 1e-4, "...but not with two input qubits or a coherent prior")
out["zblind"] = dict(one_input_max=worst1, two_input_median=float(np.median(multi)), coherent_median=float(np.median(coh)))

# ---------------------------------------------------------------- (2) finite-shot order benchmark
X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1., -1.]).astype(complex)
N, tau, h, ua, ub = 4, 1.0, -Delta, 0.05, 0.15
S, ntrain, ntest = 100, 2000, 2000
rng = np.random.default_rng(41)


def lda_accuracy(F0tr, F1tr, F0te, F1te):
    m0, m1 = F0tr.mean(0), F1tr.mean(0)
    C = 0.5 * (np.cov(F0tr.T) + np.cov(F1tr.T)) + 1e-9 * np.eye(F0tr.shape[1])
    w = np.linalg.solve(np.atleast_2d(C), m1 - m0); c = w @ (m0 + m1) / 2
    return 0.5 * (np.mean(F0te @ w < c) + np.mean(F1te @ w >= c))


def sample_features(rho1, n, bases):
    feats = []
    for P in bases:
        pplus = np.clip(np.real(np.trace(rho1 @ (np.eye(2) + P) / 2)), 0, 1)
        feats.append(2 * rng.binomial(S, pplus, size=n) / S - 1)
    return np.stack(feats, 1)


bench = []
for K in np.concatenate([np.linspace(0.04, 0.7, 34), np.linspace(0.72, 2.2, 38)]):
    psi = np.zeros(2 ** N, complex); psi[0] = 1                        # |up up up up>
    Ua = expm(-1j * hamiltonian(N, K, {0: ua}, 1.0, Delta).toarray() * tau)
    Ub = expm(-1j * hamiltonian(N, K, {0: ub}, 1.0, Delta).toarray() * tau)
    s1, s2 = Ub @ Ua @ psi, Ua @ Ub @ psi                                 # "a then b", "b then a"
    red = lambda v: v.reshape(2, -1) @ v.reshape(2, -1).conj().T          # qubit 1 (others stay up)
    r1, r2 = red(s1), red(s2)
    T = tdist(np.outer(s1, s1.conj()), np.outer(s2, s2.conj()))
    x = abs(h + K * (N - 1)) * tau
    acc_xyz = lda_accuracy(sample_features(r1, ntrain, [X, Y, Z]), sample_features(r2, ntrain, [X, Y, Z]),
                           sample_features(r1, ntest, [X, Y, Z]), sample_features(r2, ntest, [X, Y, Z]))
    acc_z = lda_accuracy(sample_features(r1, ntrain, [Z]), sample_features(r2, ntrain, [Z]),
                         sample_features(r1, ntest, [Z]), sample_features(r2, ntest, [Z]))
    first = abs(ub - ua) * tau * 2 * np.sin(x) ** 2 / x
    bench.append(dict(K=K, x=x, T=T, first_order=first, acc_xyz=acc_xyz, acc_z=acc_z,
                      acc_pred=norm.cdf(np.sqrt(S) * first), helstrom=0.5 * (1 + T)))
xs = np.array([d["x"] for d in bench]); ax = np.array([d["acc_xyz"] for d in bench])
xpk = xs[np.argmax(ax)]
pred = np.array([d["acc_pred"] for d in bench])
print(f"(2) benchmark: accuracy peak at x = {xpk:.2f} (first-order prediction 1.166); max |acc - predicted| = "
      f"{np.max(np.abs(ax - pred)):.3f}; Z-only accuracy in [{min(d['acc_z'] for d in bench):.3f}, {max(d['acc_z'] for d in bench):.3f}]")
check(0.9 < xpk < 1.45, "benchmark accuracy peaks inside the predicted window")
check(np.max(np.abs(ax - pred)) < 0.04, "finite-shot accuracy follows the first-order prediction")
out["benchmark"] = dict(S=S, ntrain=ntrain, ntest=ntest, a=ua, b=ub, tau=tau, rows=bench, x_peak=xpk)

# ---------------------------------------------------------------- (3) |+>^N: Helstrom vs restricted readouts
K, tau, h, du = 8.0, 0.5, -0.1, 0.02
acc = []
for kind in ["in", "ord"]:
    for N in [10, 11, 100, 101, 1000, 1001, 4000, 4001, 40000, 40001]:
        b, P, m = uniform_blocks(N, K, h)
        q = exact_quantities(b, P, tau, du, kind)
        q.update(kind=kind, N=N)
        if kind == "in":
            Emu = float(np.sum(P * np.minimum(tau, 1 / np.maximum(np.abs(b), 1e-300))))
            n = N - 1; mstar = round(-h / K)
            logb = n ** -0.5 * (tau + 4 / K * (1 + np.log(n + abs(mstar))))
            q.update(prop_bound=3 * du * Emu, log_bound=3 * du * logb)
            check(q["Et"] <= 3 * du * Emu <= 3 * du * logb + 1e-15, f"spectator-readout bound, N={N}")
        check(abs(q["T"] ** 2 - q["Et2"] - q["varz"]) < 1e-12, f"T^2 = E t^2 + Var z ({kind}, N={N})")
        acc.append(q)
        print(f"(3) {kind:3s} N={N:6d}: T={q['T']:.3e}  E t={q['Et']:.3e} (ratio {q['Et']/q['T']:.3f})  "
              f"T_loc={q['Tloc']:.2e}  E t^2/T^2={q['Et2']/q['T']**2:.3f}")
out["access"] = acc
save("s06_readout.json", out)
