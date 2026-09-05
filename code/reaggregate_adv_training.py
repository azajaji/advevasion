"""Regenerate results_v2/<DATASET>/aggregated/adv_training_{full,summary}.csv
from the freshly-recomputed per-seed adv_training.csv files.

Scoped identically to aggregate() in run_experiments.py but restricted to the
adv_training stage only -- the other aggregated files (baseline_clean,
attacks, whitebox, latency, randomized_smoothing) are untouched, since their
per-seed inputs did not change.

Usage:  python Code/reaggregate_adv_training.py
"""
from __future__ import annotations
import shutil
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
import numpy as np
import pandas as pd

OUT = CODE_DIR.parent / "results_v2"
BACKUP = OUT / "_superseded_pre_seeded_adv_training"
DATASETS = ["WUSTLEHMS2020", "TONIOT", "CSECICIDS2018"]
SEEDS = [42, 7, 123, 31, 99]


def main() -> None:
    BACKUP.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        agg_dir = OUT / ds / "aggregated"
        agg_dir.mkdir(parents=True, exist_ok=True)

        for suffix in ("_full.csv", "_summary.csv"):
            old = agg_dir / f"adv_training{suffix}"
            if old.exists():
                shutil.copy2(old, BACKUP / f"{ds}_adv_training{suffix}")

        dfs = []
        for s in SEEDS:
            f = OUT / ds / f"seed_{s}" / "adv_training.csv"
            if not f.exists():
                print(f"MISSING {f}")
                continue
            df = pd.read_csv(f)
            df["seed"] = s
            dfs.append(df)
        if not dfs:
            continue
        full = pd.concat(dfs, ignore_index=True)
        full.to_csv(agg_dir / "adv_training_full.csv", index=False)

        numeric = full.select_dtypes(include=[np.number]).columns.difference(["seed"])
        group_cols = [c for c in full.columns if c not in numeric and c != "seed"]
        if group_cols:
            stats = full.groupby(group_cols, dropna=False)[list(numeric)].agg(["mean", "std"])
            stats.columns = [f"{a}_{b}" for a, b in stats.columns]
            stats = stats.reset_index()
            stats.to_csv(agg_dir / "adv_training_summary.csv", index=False)
        print(f"{ds}: wrote adv_training_full.csv ({len(full)} rows) and adv_training_summary.csv")


if __name__ == "__main__":
    main()
