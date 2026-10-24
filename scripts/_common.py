"""Shared helpers: output locations relative to the package root, and checks that fail loudly."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)


def check(cond, msg):
    """Raise (and so make run_all.py fail) if a claimed inequality or identity is violated."""
    if not bool(cond):
        raise AssertionError("CHECK FAILED: " + msg)
    print("  ok:", msg)


def save(name, obj):
    path = os.path.join(RESULTS, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
    print("  wrote", os.path.relpath(path, ROOT))


def load(name):
    with open(os.path.join(RESULTS, name)) as fh:
        return json.load(fh)
