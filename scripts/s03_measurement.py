"""Proposition 5 (one Z-basis measurement between steps).
 (a) exact-T ratio after/before a Z measurement for 1200 random priors (one input qubit);
 (b) leading-term monotonicity L_in(P rho) <= L_in(rho) for one input qubit (Prop. 5(d));
 (c) with two or three input qubits the leading term and the exact response CAN increase
     (Prop. 5(e)), including the exact dark-state example.  -> results/s03_measurement.json"""
from _common import check, save
import numpy as np
from scipy.linalg import expm
from qtime.core import op_on, sx, fker, flip_freqs, hamiltonian, tdist, dephase

Delta = 0.1
out = {}

# ---------------------------------------------------------------- (a) exact ratios, 1200 priors
rng = np.random.default_rng(1); dI = 0.02; ratios = []
for N in [3, 4, 5, 6]:
    for K, dt in [(2.0, 0.5), (8.0, 0.5), (0.5, 0.5)]:
        Ua = expm(-1j * hamiltonian(N, K, {0: 0.0}, 1.0, Delta).toarray() * dt)
        Ub = expm(-1j * hamiltonian(N, K, {0: dI}, 1.0, Delta).toarray() * dt)
        for trial in range(100):
            if trial % 2 == 0:
                v = rng.normal(size=2 ** N) + 1j * rng.normal(size=2 ** N); v /= np.linalg.norm(v); rho = np.outer(v, v.conj())
            else:
                w = rng.normal(size=(2 ** N, 3)) + 1j * rng.normal(size=(2 ** N, 3)); rho = w @ w.conj().T; rho /= np.trace(rho)
            T0 = tdist(Ua @ rho @ Ua.conj().T, Ub @ rho @ Ub.conj().T)
            r2 = dephase(rho)
            T1 = tdist(Ua @ r2 @ Ua.conj().T, Ub @ r2 @ Ub.conj().T)
            ratios.append(T1 / T0)
r = np.array(ratios)
print(f"(a) {r.size} priors: median ratio {np.median(r):.3f}, max {r.max():.3f}")
check(r.max() < 1, "exact one-input response never increased by a Z measurement in the 1200 priors")
out["ratios"] = r.tolist()


# ---------------------------------------------------------------- (b), (c) leading-term (in)equality
def Aop(d, N, K, tau):
    A = np.zeros((2 ** N, 2 ** N), complex)
    for i, v in d.items():
        A += v * (op_on(sx, i, N).toarray() @ np.diag(fker(flip_freqs(N, K, i, 1.0, Delta), tau)))
    return A


def L(A, rho):
    return 0.5 * np.linalg.svd(A @ rho - rho @ A, compute_uv=False).sum()


def rand_rho(N, k):
    d = 2 ** N
    if k == 0:
        v = rng.normal(size=d) + 1j * rng.normal(size=d); v /= np.linalg.norm(v); return np.outer(v, v.conj())
    w = rng.normal(size=(d, k)) + 1j * rng.normal(size=(d, k)); x = w @ w.conj().T; return x / np.trace(x)


rng = np.random.default_rng(5)
stats = {}
for nin in [1, 2, 3]:
    worst, viol, tot = 0.0, 0, 0
    for N in [3, 4, 5]:
        for K in [0.3, 1.0, 2.0, 8.0]:
            for tau in [0.5, 1.0, 2.0]:
                for t in range(60):
                    d = {i: rng.normal() for i in range(nin)}
                    A = Aop(d, N, K, tau); rho = rand_rho(N, [0, 1, 2, 3, 8][t % 5])
                    q = L(A, dephase(rho)) / L(A, rho)
                    worst = max(worst, q); viol += q > 1 + 1e-10; tot += 1
    stats[nin] = dict(n=tot, worst=worst, violations=int(viol))
    print(f"(b/c) {nin} input qubit(s): {tot} priors, max L(P rho)/L(rho) = {worst:.3f}, increases: {viol}")
check(stats[1]["violations"] == 0, "Prop. 5(d): L_in(P rho) <= L_in(rho) for one input qubit")
check(stats[2]["violations"] + stats[3]["violations"] > 0, "Prop. 5(e): monotonicity fails for several input qubits")
out["leading_term"] = stats

# exact response with several inputs: random search
rng = np.random.default_rng(9); inc, tot, best = 0, 0, 1.0
for trial in range(4000):
    N = int(rng.choice([3, 4])); K = float(rng.choice([0.3, 1, 2, 8])); tau = float(rng.choice([0.5, 1, 2])); nin = int(rng.choice([2, 3]))
    x = rng.normal(size=nin); x = 0.02 * x / np.abs(x).sum(); d = {i: x[i] for i in range(nin)}
    rho = rand_rho(N, int(rng.choice([0, 1, 2, 3])))
    U0 = expm(-1j * hamiltonian(N, K, {}, 1.0, Delta).toarray() * tau); U1 = expm(-1j * hamiltonian(N, K, d, 1.0, Delta).toarray() * tau)
    Tf = lambda q: tdist(U0 @ q @ U0.conj().T, U1 @ q @ U1.conj().T)
    q = Tf(dephase(rho)) / Tf(rho); inc += q > 1; tot += 1; best = max(best, q)
print(f"(c) several inputs, exact response: increased in {inc}/{tot} random priors, max ratio {best:.2f}")
out["multi_exact"] = dict(increased=int(inc), n=tot, max_ratio=best)

# dark state: N=2, h=0, inputs du on both qubits, prior (|00>-|11>)/sqrt2
N, K, tau, dd = 2, 1.0, 1.0, 0.02
H0 = K * np.diag([1., -1., -1., 1.]).astype(complex)
X1 = np.kron(np.array([[0, 1], [1, 0]]), np.eye(2)); X2 = np.kron(np.eye(2), np.array([[0, 1], [1, 0]]))
psi = np.array([1, 0, 0, -1], complex) / np.sqrt(2); rho = np.outer(psi, psi.conj())
U0 = expm(-1j * H0 * tau); U1 = expm(-1j * (H0 + dd * (X1 + X2)) * tau)
T_pure = tdist(U0 @ rho @ U0.conj().T, U1 @ rho @ U1.conj().T)
rp = dephase(rho); T_meas = tdist(U0 @ rp @ U0.conj().T, U1 @ rp @ U1.conj().T)
print(f"(c) dark state: T_in(rho) = {T_pure:.2e}, T_in(P rho) = {T_meas:.4e}")
check(T_pure < 1e-12 and T_meas > 1e-3, "dark state: zero response before, nonzero after a Z measurement")
out["dark"] = dict(K=K, tau=tau, du=dd, T_pure=T_pure, T_meas=T_meas)
save("s03_measurement.json", out)
