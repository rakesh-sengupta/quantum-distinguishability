# Code for "Input distinguishability and sequence-order sensitivity in Ising-coupled quantum networks"

R. Sengupta, Krea University. Every number, table and figure in the manuscript is produced by the
scripts below; every inequality or identity they test is an `assert`, so a violation stops the run.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py          # ~3 min; writes results/*.json and figures/*.pdf
```

`python run_all.py --quick` skips the dissipation sweep (s07, ~95 s) if its results already exist.

## Layout

| Path | Contents |
|---|---|
| `qtime/core.py` | Dense/sparse model for N <= 6: Ising H0, transverse/longitudinal inputs, Lindblad generator (amplitude damping, dephasing), trace distance, fidelity, kernels f. `bound_collective_guess` is the collective-field *heuristic* of the Introduction, kept only for comparison; it is not a result of the paper. |
| `qtime/blocks.py` | Exact symmetry-reduced calculations for one driven qubit and the prior \|+>^N (two-level blocks indexed by the spectator configuration); closed form of S_n (Poisson summation); Fourier formula for arbitrary couplings. |
| `scripts/s01_first_order.py` | Theorem 1 remainder (972 cases), Theorem 2 population bound (1296 cases), Theorem 4 remainder (972 cases). |
| `scripts/s02_asymptotics.py` | Table 2, Theorem 3, Corollaries 2 and 5, Proposition 2 (exact vs first order, uniform in N), log-log slopes, design rule (Remark 5). Up to N = 40001. |
| `scripts/s03_measurement.py` | Proposition 5: 1200 random priors (exact), monotonicity of the leading term for one input qubit, counterexamples with several input qubits, dark state. |
| `scripts/s04_order_encoding.py` | Proposition 1(b), Proposition 6 (Zeno), Theorem 4 window, Theorem 5 (longitudinal inputs with dissipation, local readout). |
| `scripts/s05_nonuniform.py` | Proposition 3 (Fourier formula) and Corollary 4 (continuum law) against brute-force enumeration; uniform vs disordered couplings up to N = 4001. |
| `scripts/s06_readout.py` | Proposition 7 (Z-readout order blindness), finite-shot order-discrimination benchmark (Sec. 8.2), Proposition 8 (Helstrom vs spectator-Z vs local readout) up to N = 40001. |
| `scripts/s07_dissipation.py` | Section 9: controlled dissipation test and the recurrent protocol (Fig. 6, Table 1). Reconstructs the dissipation data that were previously shipped only as JSON/text outputs. |
| `scripts/make_figures.py` | Figures 1-6 (fig_scaling, fig_nonuniform, fig_measure, fig_order, fig_readout, fig_dissipation). |

## Conventions

Basis bit 0 is spin up (s = +1); sigma^- maps s = +1 to s = -1. Qubit index 0 in the code is
"qubit 1" in the paper. Unless stated otherwise h_i = -0.1 and couplings are uniform.

"Exact" for N > 6 means exact within the symmetry reduction: with H0 diagonal and a single driven
qubit, the spectator configuration is conserved, so the dynamics splits into independent two-level
problems. No generic 2^N-dimensional simulation is performed or claimed.
