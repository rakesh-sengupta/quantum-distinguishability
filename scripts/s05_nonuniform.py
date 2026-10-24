"""Beyond uniform couplings (Proposition 3): Fourier formula for L_in with arbitrary
couplings K_1j, checked against brute-force enumeration, and the N-dependence for
uniform and disordered couplings.  -> results/s05_nonuniform.json"""
from _common import check, save
import numpy as np
from scipy.integrate import quad
from qtime.blocks import L_fourier, L_bruteforce, uniform_blocks, L_first_order

h, tau, du = -0.1, 0.5, 0.02
out = dict(check=[], scaling={})

# ---------------------------------------------------------------- Fourier vs enumeration
rng = np.random.default_rng(21)
for n in [3, 6, 9, 12, 15]:
    for kind in ["uniform-disorder", "gaussian", "sparse"]:
        if kind == "uniform-disorder":
            Ks = 8 * (1 + 0.3 * rng.uniform(-1, 1, n))
        elif kind == "gaussian":
            Ks = rng.normal(2.0, 3.0, n)
        else:
            Ks = rng.choice([0.0, 5.0, -3.0], n)
        a, b = L_fourier(Ks, h, tau, du), L_bruteforce(Ks, h, tau, du)
        out["check"].append(dict(n=n, kind=kind, fourier=a, brute=b))
        check(abs(a - b) <= 1e-9 * b + 1e-14, f"Fourier formula = enumeration, N={n+1}, {kind}: {a:.6e}")


# ---------------------------------------------------------------- scaling with N
def L_fourier_peaky(Ks, h, tau, du):
    """Same as L_fourier, with break points at the possible resonances k = pi j / mean|K|."""
    Ks = np.asarray(Ks, float); Kbar = np.mean(np.abs(Ks))
    pts = [np.pi * j / Kbar for j in range(1, int(2 * tau * Kbar / np.pi) + 1)]
    rechi = lambda k: np.cos(k * h) * np.prod(np.cos(k * Ks))
    w = 1 / (Kbar * np.sqrt(len(Ks)))
    pts = sorted(set([x for p in [0.0] + pts for x in (p - 8 * w, p, p + 8 * w) if 0 < x < 2 * tau]))
    E2 = quad(lambda k: (tau - k / 2) * rechi(k), 0, 2 * tau, limit=20000, points=pts, epsabs=1e-14, epsrel=1e-10)[0]
    E1 = quad(lambda k: 0.5 * rechi(k), 0, 2 * tau, limit=20000, points=pts, epsabs=1e-14, epsrel=1e-10)[0]
    return abs(du) * np.sqrt(max(0.0, E2 - E1 ** 2))


Ns = sorted(set(list(np.unique(np.round(np.logspace(1, np.log10(4001), 22)).astype(int))) + [4000, 4001]))
xi = np.random.default_rng(2026).uniform(-1, 1, 5000)          # one quenched draw, nested in N
cases = {"uniform K=8": lambda n: np.full(n, 8.0),
         "disordered K=8(1+0.25 xi)": lambda n: 8.0 * (1 + 0.25 * xi[:n]),
         "uniform K=2": lambda n: np.full(n, 2.0)}
for name, Kf in cases.items():
    rows = []
    for N in Ns:
        Ks = Kf(N - 1)
        if name.startswith("uniform"):
            b, P, _ = uniform_blocks(N, Ks[0], h); L = L_first_order(b, P, tau, du)
            if N <= 200:
                check(abs(L_fourier_peaky(Ks, h, tau, du) - L) < 1e-6 * L, f"Fourier = block value, {name}, N={N}")
        else:
            L = L_fourier_peaky(Ks, h, tau, du)
        V = np.sum(Ks ** 2)
        cont = abs(du) * np.sqrt(np.pi * tau) * (2 * np.pi * V) ** -0.25 * np.exp(-h * h / (4 * V))
        rows.append(dict(N=int(N), L=L, continuum=cont))
    out["scaling"][name] = rows
    big = [r for r in rows if r["N"] >= 1000]
    dev = max(abs(r["L"] / r["continuum"] - 1) for r in big)
    print(f"{name}: max |L/continuum - 1| for N >= 1000: {dev:.3f}")
    if name != "uniform K=8":
        check(dev < 0.05, f"{name}: continuum law within 5% for N >= 1000")
save("s05_nonuniform.json", out)
