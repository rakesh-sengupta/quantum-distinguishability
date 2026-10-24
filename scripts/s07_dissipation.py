"""Dissipation numerics (Section on dissipation): controlled single-step test and the recurrent
protocol (figure and table).  Reconstructs and supersedes the original d4.json / out_d3a.txt /
out_d4b.txt / out_d4c.txt.  -> results/s07_dissipation.json"""
from _common import check, save
import numpy as np
from scipy.linalg import expm
from qtime.core import hamiltonian, liouvillian, tdist, fidelity, dephase, plus_state, zero_state

Delta = 0.1
out = dict(controlled={}, recurrent={}, table=[])

# ---------------------------------------------------------------- controlled single-step test
N, tau, du = 4, 4.0, 0.02
grid = np.logspace(-2, np.log10(200), 41)
psi = plus_state(N); rho0 = np.outer(psi, psi.conj()).reshape(-1)
for K in [1.0, 2.0, 4.0]:
    H0, H1 = hamiltonian(N, K, {0: 0.0}, 1.0, Delta), hamiltonian(N, K, {0: du}, 1.0, Delta)
    for chan in ["damping", "dephasing"]:
        Ts, TVs = [], []
        for g in grid:
            ga, gp = (g, 0.0) if chan == "damping" else (0.0, g)
            ra = (expm((liouvillian(H0, N, ga, gp) * tau).toarray()) @ rho0).reshape(16, 16)
            rb = (expm((liouvillian(H1, N, ga, gp) * tau).toarray()) @ rho0).reshape(16, 16)
            Ts.append(tdist(ra, rb)); TVs.append(0.5 * np.abs(np.real(np.diag(ra - rb))).sum())
        Ts, TVs = np.array(Ts), np.array(TVs)
        r = dict(grid=grid.tolist(), T=Ts.tolist(), TV=TVs.tolist(),
                 T_peak_rate=float(grid[Ts.argmax()]), T_gain=float(Ts.max() / Ts[0]),
                 TV_peak_rate=float(grid[TVs.argmax()]), TV_gain=float(TVs.max() / TVs[0]))
        out["controlled"][f"K={K},{chan}"] = r
        print(f"controlled K={K} {chan:9s}: full-state T max at rate {r['T_peak_rate']:.3g} (gain {r['T_gain']:.2f}x); "
              f"Z-readout TV max at {r['TV_peak_rate']:.3g} (gain {r['TV_gain']:.2f}x)")

# ---------------------------------------------------------------- recurrent protocol
N, tau, nsteps = 4, 0.5, 20
inputs = [{i: 0.5 * np.sin(np.pi * n / 2 + i) for i in range(N)} for n in range(1, nsteps + 1)]   # n = 1..20


def recurrent(K, gamma, prior, measure):
    psi = zero_state(N) if prior == "up" else plus_state(N)
    rho = np.outer(psi, psi.conj())
    E = [expm((liouvillian(hamiltonian(N, K, u, 1.0, Delta), N, gamma) * tau).toarray()) for u in inputs]
    D, T = [], []
    for n in range(nsteps - 1):                    # 19 comparisons: u^(n) vs u^(n+1) from the current state
        if measure:
            rho = dephase(rho)
        ra = (E[n] @ rho.reshape(-1)).reshape(16, 16); rb = (E[n + 1] @ rho.reshape(-1)).reshape(16, 16)
        D.append(1 - fidelity(ra, rb)); T.append(tdist(ra, rb))
        rho = ra                                   # the network actually receives u^(n)
    return float(np.mean(D)), float(np.mean(T))


gam = np.logspace(-2, 2, 29)
for K in [0.5, 2.0]:
    out["recurrent"][f"K={K}"] = dict(gamma=gam.tolist(), D=[recurrent(K, g, "up", False)[0] for g in gam])
d05 = out["recurrent"]["K=0.5"]["D"]
check(abs(d05[0] - 0.18735214) < 1e-6, "reproduces original d4.json value D(gamma=0.01, K=0.5)")
print(f"recurrent K=0.5: D(gamma=0.01) / D(gamma=0.15) = {d05[0] / recurrent(0.5, 0.15, 'up', False)[0]:.2f}")

for K in [0.5, 2.0]:
    for ratio in [0.3, 4.0, 12.0]:
        g = ratio * K
        row = dict(K=K, gamma=g, ratio=ratio)
        for prior in ["up", "plus"]:
            row[f"{prior}_D_noM"], row[f"{prior}_T_noM"] = recurrent(K, g, prior, False)
            row[f"{prior}_D_M"], row[f"{prior}_T_M"] = recurrent(K, g, prior, True)
        out["table"].append(row)
        print("table:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
check(abs(out["table"][0]["up_D_noM"] - 0.0493) < 5e-5 and abs(out["table"][0]["plus_D_M"] - 0.0329) < 5e-5,
      "reproduces original Table 3 entries")
save("s07_dissipation.json", out)
