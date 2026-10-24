"""Reproduce every number, table and figure of the manuscript from a clean checkout.

    python run_all.py            # everything (about 3 minutes on a laptop)
    python run_all.py --quick    # skip the slow dissipation sweep (s07)

Each script asserts the inequalities and identities it checks; any failure stops the run
with a non-zero exit code. Outputs go to results/ (JSON) and figures/ (PDF).
"""
import os, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["s01_first_order.py", "s02_asymptotics.py", "s03_measurement.py", "s04_order_encoding.py",
           "s05_nonuniform.py", "s06_readout.py", "s07_dissipation.py", "make_figures.py"]

if __name__ == "__main__":
    quick = "--quick" in sys.argv
    for s in SCRIPTS:
        if quick and s == "s07_dissipation.py" and os.path.exists(os.path.join(ROOT, "results", "s07_dissipation.json")):
            print(f"== {s}: skipped (--quick, using existing results)"); continue
        print(f"== {s}", flush=True); t0 = time.time()
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", s)], cwd=os.path.join(ROOT, "scripts"))
        if r.returncode != 0:
            print(f"!! {s} FAILED (exit code {r.returncode})"); sys.exit(r.returncode)
        print(f"   done in {time.time() - t0:.1f} s", flush=True)
    print("All checks passed; results/ and figures/ are up to date.")
