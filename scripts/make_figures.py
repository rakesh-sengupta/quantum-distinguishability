"""All figures of the manuscript, written to figures/*.pdf.  Needs the results/*.json files
produced by s01-s07 (run_all.py runs everything in order)."""
from _common import FIGURES, load
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qtime.blocks import uniform_blocks, exact_quantities, asym

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.3, "figure.dpi": 150})
fig_path = lambda name: os.path.join(FIGURES, name)
h = -0.1


def T_exact(N, K, tau, du, kind="in"):
    b, P, _ = uniform_blocks(N, K, h)
    return exact_quantities(b, P, tau, du, kind)["T"]


def parity_Ns(Ns, par):
    return np.array(sorted(set(int(n) + (1 if int(n) % 2 != par else 0) for n in Ns)))


# ------------------------------------------------------------------ Fig. 1: scaling
fig, ax = plt.subplots(1, 2, figsize=(5.8, 2.55))
K, tau, du = 8.0, 0.5, 0.02
Ns = np.unique(np.round(np.logspace(np.log10(4), np.log10(4000), 26)).astype(int))
for par, c, lab in [(0, "C0", "even $N$"), (1, "C3", "odd $N$")]:
    NN = parity_Ns(Ns, par)
    ax[0].loglog(NN, [T_exact(n, K, tau, du) for n in NN], "o", ms=3.5, c=c, label=f"exact, {lab}")
    ax[0].loglog(NN, [asym(n, K, tau, h, du) for n in NN], "-", c=c, lw=1, label=f"Theorem 3, {lab}")
NN = np.array(sorted(Ns))
ax[0].loglog(NN, du * 2 / (K * NN), "k--", lw=1, label=r"collective-field guess $\propto 1/(KN)$")
ax[0].set(xlabel="$N$", ylabel=r"$\mathcal{T}_{\rm in}$ per step"); ax[0].legend(loc="lower left"); ax[0].set_title("(a)", loc="left")
Ks = np.logspace(0, 2, 30)
for N, c in [(100, "C0"), (101, "C3")]:
    ax[1].loglog(Ks, [T_exact(N, k, tau, du) for k in Ks], "o", ms=3, c=c, label=f"exact, $N={N}$")
    ax[1].loglog(Ks, [asym(N, k, tau, h, du) for k in Ks], "-", c=c, lw=1)
ax[1].axvline(np.pi / (2 * tau), c="gray", ls=":", lw=1)
ax[1].text(np.pi / (2 * tau) * 1.08, 6e-5, r"$K\tau=\pi/2$", fontsize=6.3, color="gray")
ax[1].set(xlabel=r"coupling $K$", ylabel=r"$\mathcal{T}_{\rm in}$ per step"); ax[1].legend(loc="lower left"); ax[1].set_title("(b)", loc="left")
plt.tight_layout(); plt.savefig(fig_path("fig_scaling.pdf")); plt.close()

# ------------------------------------------------------------------ Fig. 2: order
d = load("s04_order_encoding.json"); w = d["window"]
fig, ax = plt.subplots(1, 2, figsize=(5.8, 2.55))
x = np.array([e["x"] for e in w]); T = np.array([e["T"] for e in w]); xx = np.linspace(0.01, x.max(), 400)
ax[0].plot(xx, 2 * np.sin(xx) ** 2 / xx, "k-", lw=1, label=r"first order: $2\sin^2 x/x$")
ax[0].plot(x, T / (0.005 * 1.0), "o", ms=3, c="C2", label="exact (two-step)")
ax[0].set(xlabel=r"$x=|b|\tau$ (local field $\times$ step length)", ylabel=r"$\mathcal{T}_{\rm ord}/(|\delta u|\tau)$")
ax[0].legend(); ax[0].set_title("(a)", loc="left")
for par, c, lab in [(0, "C0", "even $N$"), (1, "C3", "odd $N$")]:
    NN = parity_Ns(Ns, par)
    ax[1].loglog(NN, [T_exact(n, K, tau, du, "ord") for n in NN], "s", ms=3.5, c=c, label=f"exact, {lab}")
    ax[1].loglog(NN, [asym(n, K, tau, h, du, "ord") for n in NN], "-", c=c, lw=1, label=f"Corollary 5, {lab}")
ax[1].set(xlabel="$N$", ylabel=r"$\mathcal{T}_{\rm ord}$"); ax[1].legend(loc="lower left"); ax[1].set_title("(b)", loc="left")
plt.tight_layout(); plt.savefig(fig_path("fig_order.pdf")); plt.close()

# ------------------------------------------------------------------ Fig. 3: measurement
r = np.array(load("s03_measurement.json")["ratios"]); z = d["zeno"]
fig, ax = plt.subplots(1, 2, figsize=(5.8, 2.55))
ax[0].hist(r, bins=30, color="C1", edgecolor="k", lw=0.3); ax[0].axvline(1, c="k", ls="--", lw=1)
ax[0].set(xlabel=r"$\mathcal{T}_{\rm in}[\mathcal{P}(\rho)]\,/\,\mathcal{T}_{\rm in}[\rho]$", ylabel="number of priors"); ax[0].set_title("(a)", loc="left")
for dd, c in [(0.2, "C0"), (0.5, "C3")]:
    M = np.array([e["M"] for e in z if e["du"] == dd]); TT = np.array([e["T"] for e in z if e["du"] == dd])
    B = np.array([e["bound"] for e in z if e["du"] == dd], float)
    ax[1].loglog(M, TT, "o-", ms=3, c=c, label=fr"exact, $\delta u={dd}$"); ax[1].loglog(M, B, "--", c=c, lw=1, label="Proposition 6 bound")
ax[1].set(xlabel=r"number of $Z$ measurements $M$ in one step", ylabel=r"$\mathcal{T}_{\rm in}$"); ax[1].legend(); ax[1].set_title("(b)", loc="left")
plt.tight_layout(); plt.savefig(fig_path("fig_measure.pdf")); plt.close()

# ------------------------------------------------------------------ Fig. 4: beyond uniform couplings
s5 = load("s05_nonuniform.json")["scaling"]
fig, ax = plt.subplots(1, 2, figsize=(5.8, 2.55))
styles = {"uniform K=8": ("C0", "o", r"uniform $K=8$ ($K\tau=4$)"),
          "disordered K=8(1+0.25 xi)": ("C2", "^", r"disordered $K_{1j}=8(1+0.25\,\xi_j)$"),
          "uniform K=2": ("C4", "s", r"uniform $K=2$ ($K\tau=1<\pi/2$)")}
for name, (c, mk, lab) in styles.items():
    rows = s5[name]; NN = np.array([q["N"] for q in rows]); L = np.array([q["L"] for q in rows]); C = np.array([q["continuum"] for q in rows])
    ax[0].loglog(NN, L, mk, ms=3, c=c, label=lab); ax[0].loglog(NN, C, "-", c=c, lw=0.8)
    ax[1].semilogx(NN, L / C, mk + "-", ms=3, lw=0.6, c=c, label=lab)
ax[0].set(xlabel="$N$", ylabel=r"$L_{\rm in}$ for $|+\rangle^{\otimes N}$", ylim=(1.2e-4, 9e-3)); ax[0].legend(loc="lower left", fontsize=5.6); ax[0].set_title("(a)", loc="left")
ax[1].axhline(1, c="k", lw=0.6, ls="--")
ax[1].set(xlabel="$N$", ylabel=r"$L_{\rm in}$ / continuum law"); ax[1].set_title("(b)", loc="left")
plt.tight_layout(); plt.savefig(fig_path("fig_nonuniform.pdf")); plt.close()

# ------------------------------------------------------------------ Fig. 5: readout
s6 = load("s06_readout.json"); bm = s6["benchmark"]["rows"]; S = s6["benchmark"]["S"]
fig, ax = plt.subplots(1, 2, figsize=(5.8, 2.55))
xb = np.array([q["x"] for q in bm]); o = np.argsort(xb)
ax[0].plot(xb[o], np.array([q["acc_pred"] for q in bm])[o], "k-", lw=1, label=r"prediction $\Phi(\sqrt{S}\,L_{\rm ord})$")
ax[0].plot(xb[o], np.array([q["acc_xyz"] for q in bm])[o], "o", ms=2.8, c="C2", label=r"input qubit, $X,Y,Z$ shots")
ax[0].plot(xb[o], np.array([q["acc_z"] for q in bm])[o], "x", ms=3, c="C3", label=r"$Z$ shots only")
ax[0].axvline(1.166, c="gray", ls=":", lw=1)
ax[0].set(xlabel=r"$x=|b|\tau$", ylabel="test accuracy, $ab$ vs $ba$", ylim=(0.45, 1.0)); ax[0].legend(loc="upper right"); ax[0].set_title("(a)", loc="left")
acc = [q for q in s6["access"] if q["kind"] == "in"]
for par, c, lab in [(0, "C0", "even"), (1, "C3", "odd")]:
    rows = [q for q in acc if q["N"] % 2 == par]; NN = np.array([q["N"] for q in rows])
    ax[1].loglog(NN, [q["T"] for q in rows], "o-", ms=3, lw=0.8, c=c, label=fr"Helstrom $\mathcal{{T}}_{{\rm in}}$, {lab} $N$")
    ax[1].loglog(NN, [q["Et"] for q in rows], "s--", ms=3, lw=0.8, c=c, label=fr"spectators in $Z$, {lab} $N$")
    ax[1].loglog(NN, [q["Tloc"] for q in rows], "^:", ms=3, lw=0.8, c=c, label=fr"input qubit only, {lab} $N$")
ax[1].set(xlabel="$N$", ylabel="single-shot distinguishability", ylim=(1e-7, 1.5e-2)); ax[1].legend(loc="lower left", fontsize=5.4, ncol=1); ax[1].set_title("(b)", loc="left")
plt.tight_layout(); plt.savefig(fig_path("fig_readout.pdf")); plt.close()

# ------------------------------------------------------------------ Fig. 6: dissipation
s7 = load("s07_dissipation.json")["recurrent"]
fig, ax = plt.subplots(figsize=(3.6, 2.7))
for k, c, b in [("K=0.5", "C0", 0.5), ("K=2.0", "C3", 2.0)]:
    ax.loglog(s7[k]["gamma"], s7[k]["D"], "-", c=c, label=fr"$K={b}$"); ax.axvline(b * 4, c=c, ls=":", lw=1)
ax.set(xlabel=r"damping rate $\gamma$", ylabel=r"$D_{\rm in}=1-F$ (mean over 19 steps)"); ax.legend()
plt.tight_layout(); plt.savefig(fig_path("fig_dissipation.pdf")); plt.close()
print("figures written to", FIGURES)
