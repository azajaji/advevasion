"""Recompute the primary pipeline's adv_training.csv (AdversarialTrainer
defense, FGSM/PGD ratio sweep) under the per-cell deterministic seeding fix,
across all dataset x seed combinations.

Superseded because AdversarialTrainer.fit() draws on the bare global numpy
RNG (np.random.shuffle / np.random.choice) without an explicit random_state,
so repeated cells drifted run to run -- confirmed empirically via
test_adversarial_training_repro.py (field-by-field diff showed ~1.6pp
accuracy swings across 237/240 rows on a whole-object-hash sweep). The fix
(at_cell_seed + reset_all_rng immediately before building each fresh
model/attack/trainer) passed both the fresh-process-reproducibility and
config-order-invariance gates with exact hash matches -- ART itself is left
unmodified.

Scope, per the explicit decision: ONLY adv_training.csv is touched here.
Splits, preprocessing, feature selection, SMOTE, baseline training, attack
generation, and whitebox/latency/randomized-smoothing results are consumed
as-is (already verified reproducible / already fixed) to reconstruct the
inputs adversarial_training_run needs -- they are not being rerun as new
science, and their output files are not overwritten. Old adv_training.csv
files are backed up before being replaced.

Usage:  python Code/recompute_adversarial_training.py
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
BACKUP = OUT / "_superseded_pre_seeded_adv_training"


def main() -> None:
    BACKUP.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        for seed in SEEDS:
            target = OUT / ds / f"seed_{seed}" / "adv_training.csv"
            if not target.parent.exists():
                print(f"skip {ds} seed={seed}: cell directory missing")
                continue

            if target.exists():
                shutil.copy2(target, BACKUP / f"{ds}_seed{seed}_adv_training.csv")

            print(f"\n===== {ds} seed={seed} =====", flush=True)
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

    print("\nAll adv_training.csv files recomputed.", flush=True)


if __name__ == "__main__":
    main()
