"""Retry a single adv_training.csv cell after a transient joblib
BrokenProcessPool crash during feature selection (unrelated to the AT
reproducibility fix). Same logic as recompute_adversarial_training.py,
scoped to one dataset/seed passed on the command line.

Usage:  python Code/retry_adv_training_cell.py <DATASET> <SEED>
"""
from __future__ import annotations
import shutil
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
import run_experiments as base  # noqa: E402

OUT = CODE_DIR.parent / "results_v2"
BACKUP = OUT / "_superseded_pre_seeded_adv_training"


def main() -> None:
    ds, seed = sys.argv[1], int(sys.argv[2])
    BACKUP.mkdir(parents=True, exist_ok=True)
    target = OUT / ds / f"seed_{seed}" / "adv_training.csv"

    if target.exists():
        shutil.copy2(target, BACKUP / f"{ds}_seed{seed}_adv_training.csv")

    print(f"===== {ds} seed={seed} =====", flush=True)
    t0 = time.time()
    base.set_seed(seed)
    X_tr, X_te, y_tr, y_te, feats, n_feats, extras = base.prepare_split(
        ds, seed, full=True)
    models, _ = base.train_baselines(X_tr, y_tr, seed)
    attacks = base.generate_attacks(models["MLP"], X_te)

    rows = base.adversarial_training_run(
        X_tr, y_tr, X_te, y_te, models, attacks, seed, dataset=ds)

    import pandas as pd
    pd.DataFrame(rows).to_csv(target, index=False)
    print(f"  wrote {target}  ({len(rows)} rows, {time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
