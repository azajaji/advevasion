"""Recompute the primary pipeline's whitebox.csv (LR/SVM/RF direct HopSkipJump)
under ReproducibleHopSkipJump, across all dataset x seed combinations.

Superseded because ART's original HopSkipJump (a) is not reproducible run to
run for identical inputs, verified empirically, and (b) can hang indefinitely
against tree ensembles. Neither the split, model training, nor any other part
of the primary pipeline changes: only whitebox.csv is touched, and only the
attack implementation and the ASR denominator (now eligible-samples-only, not
all sampled) differ. Old files are backed up before being overwritten.

Usage:  python Code/recompute_whitebox.py
"""
from __future__ import annotations
import shutil
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
import run_experiments as base  # noqa: E402

DATASETS = ["WUSTLEHMS2020", "TONIOT", "CSECICIDS2018"]
SEEDS = [42, 7, 123, 31, 99]
OUT = CODE_DIR.parent / "results_v2"
BACKUP = OUT / "_superseded_pre_reproducible_hsj"


def main() -> None:
    BACKUP.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        for seed in SEEDS:
            target = OUT / ds / f"seed_{seed}" / "whitebox.csv"
            if not target.parent.exists():
                print(f"skip {ds} seed={seed}: cell directory missing")
                continue

            if target.exists():
                shutil.copy2(target, BACKUP / f"{ds}_seed{seed}_whitebox.csv")

            print(f"\n===== {ds} seed={seed} =====", flush=True)
            t0 = time.time()
            base.set_seed(seed)
            X_tr, X_te, y_tr, y_te, feats, n_feats, extras = base.prepare_split(
                ds, seed, full=True)
            models, _ = base.train_baselines(X_tr, y_tr, seed)

            rows = []
            for mname in ["RF", "SVM", "LR"]:
                r = base.whitebox_attack_decision_based(
                    models[mname], mname, X_te, y_te, seed=seed, dataset=ds)
                rows.append(r)
                print(f"  {mname}: asr={r['asr']:.3f} n_eligible={r['n_eligible']} "
                      f"nonconv={r['query_budget_exhausted']+r['numerical_stagnation']} "
                      f"med_q={r['median_queries_success']:.0f}", flush=True)

            import pandas as pd
            pd.DataFrame(rows).to_csv(target, index=False)
            print(f"  wrote {target}  ({time.time()-t0:.1f}s)", flush=True)

    print("\nAll whitebox.csv files recomputed.", flush=True)


if __name__ == "__main__":
    main()
