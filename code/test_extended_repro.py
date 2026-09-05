"""Regression gate for the two run_extended.py experiments that were rebuilt onto
seeded implementations: hsj_extended (was raw ART HopSkipJump) and
adaptive_pgd_against_at (was an unseeded AdversarialTrainer).

Same standard applied to the primary pipeline earlier: identical inputs must give
identical scientific output in a FRESH PROCESS, not merely within one process, since
a single process can appear reproducible purely because the global RNG happens to be
in the same state. Volatile operational fields (elapsed_s and anything derived from
wall-clock) are excluded from comparison by name -- comparing whole result objects
including timings is exactly the mistake that produced a false non-reproducibility
finding earlier in this project.

Usage:  python Code/test_extended_repro.py
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PY = sys.executable

VOLATILE = {"elapsed_s", "adv_train_s", "retrain_s", "smooth_s_per_sample", "train_s"}

HSJ_SCRIPT = """
import sys, json; sys.path.insert(0, '.')
import run_experiments as base, run_extended as ext
base.set_seed(42)
X_tr, X_te, y_tr, y_te, feats, n_feats, extras = base.prepare_split('WUSTLEHMS2020', 42, full=True)
models, _ = base.train_baselines(X_tr, y_tr, 42)
rows = [ext.hsj_extended(models[m], m, X_te, y_te, 60, 42, dataset='WUSTLEHMS2020')
        for m in ['LR', 'SVM', 'RF']]
print('RESULT' + json.dumps(rows, sort_keys=True, default=str))
"""

ADAPTIVE_SCRIPT = """
import sys, json; sys.path.insert(0, '.')
import run_experiments as base, run_extended as ext
base.set_seed(42)
X_tr, X_te, y_tr, y_te, feats, n_feats, extras = base.prepare_split('WUSTLEHMS2020', 42, full=True)
rows = ext.adaptive_pgd_against_at(X_tr, y_tr, X_te, y_te, 42,
                                   dataset='WUSTLEHMS2020', eps_values=(0.30,))
print('RESULT' + json.dumps(rows, sort_keys=True, default=str))
"""


def run(script: str, timeout: int = 3600):
    r = subprocess.run([PY, "-c", script], capture_output=True, text=True,
                       cwd=str(CODE_DIR), timeout=timeout)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise RuntimeError(f"subprocess failed (exit {r.returncode})")
    for line in r.stdout.splitlines():
        if line.startswith("RESULT"):
            return json.loads(line[len("RESULT"):])
    raise RuntimeError("no RESULT line found")


def compare(rows_a, rows_b, label):
    """Field-by-field over scientific keys only; volatile timings excluded by name."""
    print(f"\n=== {label}: fresh-process reproducibility ===")
    if len(rows_a) != len(rows_b):
        print(f"  FAIL: row count {len(rows_a)} vs {len(rows_b)}")
        return False
    ok = True
    for i, (a, b) in enumerate(zip(rows_a, rows_b)):
        for k in sorted(set(a) | set(b)):
            if k in VOLATILE:
                continue
            if a.get(k) != b.get(k):
                print(f"  DIFFERS row{i}.{k}: {a.get(k)!r} vs {b.get(k)!r}")
                ok = False
    print("  PASS" if ok else "  FAIL")
    return ok


if __name__ == "__main__":
    results = {}

    print("running hsj_extended in two fresh processes...", flush=True)
    results["hsj_extended"] = compare(run(HSJ_SCRIPT), run(HSJ_SCRIPT), "hsj_extended")

    print("\nrunning adaptive_pgd_against_at in two fresh processes...", flush=True)
    results["adaptive_pgd"] = compare(run(ADAPTIVE_SCRIPT), run(ADAPTIVE_SCRIPT),
                                      "adaptive_pgd_against_at")

    print("\n" + "=" * 70)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 70)
    all_ok = all(results.values())
    print("GATE PASSED: extended experiments are reproducible, safe to rerun at scale"
          if all_ok else
          "GATE FAILED: do not launch the full rerun until this is understood")
    sys.exit(0 if all_ok else 1)
