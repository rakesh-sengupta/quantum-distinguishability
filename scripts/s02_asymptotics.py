"""Prior |+>^N, one driven qubit, uniform couplings: exact symmetry-reduced values up to
N = 40001, Theorem 3, Corollaries 2 and 5, Proposition 2 and Remark 5 (design rule).
-> results/s02_asymptotics.json"""
from _common import check, save
import numpy as np
from qtime.blocks import (uniform_blocks, exact_quantities, L_first_order, asym, S_poisson,
                          S_trunc, E_mu2)

h, du = -0.1, 0.02
out = dict(table=[], poisson=[], slopes={}, design=[])

# ------------------------------------------------ Corollary 2: Poisson closed form of S_n, S_n^ord
for K, tau in [(8.0, 0.5), (20.0, 1.0), (2.0, 0.5), (3.0, 0.5), (1.0, 1.0), (0.5, 1.5)]:
    for par in [0, 1]:
        for kind in ["in", "ord"]:
            a, b = S_trunc(K, tau, h, par, kind), S_poisson(K, tau, h, par, kind)
            out["poisson"].append(dict(K=K, tau=tau, parity=par, kind=kind, trunc=a, closed=b))
            check(abs(a - b) <= 1e-4 * b + 1e-9, f"Poisson form of S ({kind}) K={K} tau={tau} m-parity={par}: {a:.6f} vs {b:.6f}")
    # no parity effect when |K| tau <= pi/2 (input) or pi/4 (order)
    if K * tau <= np.pi / 2:
        check(abs(S_poisson(K, tau, h, 0) - np.pi * tau / (2 * K)) < 1e-12 and
              abs(S_poisson(K, tau, h, 1) - np.pi * tau / (2 * K)) < 1e-12,
              f"S_n = pi tau/(2K) for both parities when K tau = {K*tau:.2f} <= pi/2")

# ------------------------------------------------ Table 2 and Proposition 2
for K, tau in [(8.0, 0.5), (20.0, 1.0)]:
    for N in [10, 11, 100, 101, 1000, 1001, 4000, 4001, 40000, 40001]:
        b, P, _ = uniform_blocks(N, K, h)
        qi = exact_quantities(b, P, tau, du, "in")
        qo = exact_quantities(b, P, tau, du, "ord")
        Li, Lo = L_first_order(b, P, tau, du, "in"), L_first_order(b, P, tau, du, "ord")
        Ai, Ao = asym(N, K, tau, h, du, "in"), asym(N, K, tau, h, du, "ord")
        bound4 = 2 * du ** 2 * tau * np.sqrt(E_mu2(b, P, tau))                    # Proposition 2(iii)
        Stil = S_trunc(K, tau, h, (N - 1) % 2, "tilde")
        eps_bound = 2 * (np.pi / 2) ** 0.25 * du * tau * np.sqrt(Stil / S_poisson(K, tau, h, (N - 1) % 2))  # Prop. 2(iv)
        eps = (qi["T"] - Li) / Ai
        n = N - 1
        row = dict(K=K, tau=tau, N=N, T_in=qi["T"], L_in=Li, A_in=Ai, T_ord=qo["T"], L_ord=Lo, A_ord=Ao,
                   prop4_bound=bound4, eps=eps, eps_bound=eps_bound)
        out["table"].append(row)
        print(f"K={K:4.0f} tau={tau} N={N:6d}: T_in {qi['T']:.4e}  L_in {Li:.4e} ((T-L)/L {(qi['T']-Li)/Li:+.1e}; "
              f"Prop4 bound/L {bound4/Li:.3f})  asym {Ai:.4e} ({(Ai/qi['T']-1)*100:+.2f}%) | "
              f"T_ord {qo['T']:.4e} L_ord {Lo:.4e} asym {Ao:.4e} ({(Ao/qo['T']-1)*100:+.2f}%)")
        check(abs(qi["T"] - Li) <= bound4, f"Proposition 2(iii) bound holds at K={K}, N={N}")
        check(abs(eps) <= eps_bound, f"Proposition 2(iv): |eps_n| = {abs(eps):.1e} <= {eps_bound:.4f} at K={K}, N={N}")

# ------------------------------------------------ log-log slopes over 40 <= N <= 4001
for K, tau in [(8.0, 0.5), (20.0, 1.0)]:
    for par, lab in [(0, "even"), (1, "odd")]:
        Ns = np.unique(np.round(np.logspace(np.log10(40), np.log10(4001), 15)).astype(int))
        Ns = np.array([x + ((x % 2) != par) for x in Ns])
        Ti, To = [], []
        for N in Ns:
            b, P, _ = uniform_blocks(N, K, h)
            Ti.append(exact_quantities(b, P, tau, du, "in")["T"]); To.append(exact_quantities(b, P, tau, du, "ord")["T"])
        si = np.polyfit(np.log(Ns), np.log(Ti), 1)[0]; so = np.polyfit(np.log(Ns), np.log(To), 1)[0]
        out["slopes"][f"K={K},tau={tau},{lab}"] = dict(input=si, order=so)
        print(f"slopes K={K} tau={tau} {lab} N: input {si:.3f}, order {so:.3f}")

# ------------------------------------------------ Remark (design rule): K = kappa/(tau sqrt(n))
tau = 0.5
for N in [10, 100, 1000, 4000, 40000]:
    K = 1.0 / (tau * np.sqrt(N - 1))
    b, P, _ = uniform_blocks(N, K, h)
    L = L_first_order(b, P, tau, du, "in")
    out["design"].append(dict(N=N, ratio=L / (du * tau)))
    print(f"design rule kappa=1: N={N}: L/(du tau) = {L/(du*tau):.4f}")
save("s02_asymptotics.json", out)
