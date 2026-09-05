"""Re-aggregate per-seed CSVs over all available seeds (5-seed bump support).

Run after both `run_experiments.py --seeds 31 99` and the original 3-seed run
have finished, so that `results/<DATASET>/aggregated/` reflects the union of
{42, 7, 123, 31, 99} rather than the most recent --seeds value.

Same logic for extended results: aggregates `results/<DATASET>/extended/seed_*`
into `results/<DATASET>/extended/aggregated/`.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
DATASETS = ["CSECICIDS2018", "TONIOT", "WUSTLEHMS2020"]
ALL_SEEDS = [42, 7, 123, 31, 99]

MAIN_FILES = ["baseline_clean", "attacks", "whitebox", "latency",
              "adv_training", "randomized_smoothing"]
EXTENDED_FILES = ["hsj_n500", "constrained_attacks", "adaptive_pgd", "mi_clean"]


def aggregate_dir(base: Path, seed_glob: str, fnames: list[str], agg_subdir: str) -> None:
    agg_dir = base / agg_subdir
    agg_dir.mkdir(parents=True, exist_ok=True)
    discovered_seeds = sorted({int(d.name.split("_")[1])
                               for d in base.glob(seed_glob)
                               if d.is_dir() and d.name.startswith("seed_")})
    if not discovered_seeds:
        return
    for fname in fnames:
        dfs = []
        for s in discovered_seeds:
            f = base / f"seed_{s}" / f"{fname}.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f)
            df["seed"] = s
            dfs.append(df)
        if not dfs:
            continue
        full = pd.concat(dfs, ignore_index=True)
        full.to_csv(agg_dir / f"{fname}_full.csv", index=False)
        numeric = full.select_dtypes(include=[np.number]).columns.difference(["seed"])
        group_cols = [c for c in full.columns if c not in numeric and c != "seed"]
        if group_cols:
            stats = full.groupby(group_cols, dropna=False)[list(numeric)].agg(["mean", "std"])
            stats.columns = [f"{a}_{b}" for a, b in stats.columns]
            stats = stats.reset_index()
            stats.to_csv(agg_dir / f"{fname}_summary.csv", index=False)
    print(f"  aggregated {len(discovered_seeds)} seeds in {base}: {discovered_seeds}", flush=True)


def main():
    for ds in DATASETS:
        print(f"\n=== {ds} ===", flush=True)
        ds_dir = RES / ds
        if ds_dir.exists():
            aggregate_dir(ds_dir, "seed_*", MAIN_FILES, "aggregated")
        ext_dir = ds_dir / "extended"
        if ext_dir.exists():
            aggregate_dir(ext_dir, "seed_*", EXTENDED_FILES, "aggregated")
    print("\nDone.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
