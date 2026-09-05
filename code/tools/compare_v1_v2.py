"""Compare the corrected (leakage-controlled) results against the submitted ones.

Checks the claims the manuscript actually makes, not just raw metric drift:
  1. clean baseline accuracy per model
  2. classifier ordering under clean data and under attack
  3. the transfer-vs-direct gap (the headline finding)
  4. peak FNR under attack (the abstract's "above 0.78")
  5. adversarial training vs randomized smoothing
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OLD, NEW = ROOT / "results", ROOT / "results_v2"
DATASETS = ["CSECICIDS2018", "TONIOT", "WUSTLEHMS2020"]


def load(base, ds, name):
    p = base / ds / "aggregated" / f"{name}_full.csv"
    return pd.read_csv(p) if p.exists() else None


def section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


section("1. CLEAN BASELINE ACCURACY (mean over 5 seeds)")
for ds in DATASETS:
    o, n = load(OLD, ds, "baseline_clean"), load(NEW, ds, "baseline_clean")
    if o is None or n is None:
        continue
    om = o.groupby("model")["accuracy"].mean()
    nm = n.groupby("model")["accuracy"].mean()
    print(f"\n{ds}")
    print(f"  {'model':6} {'submitted':>10} {'corrected':>10} {'delta':>9}")
    for m in ["LR", "SVM", "RF", "MLP"]:
        if m in om and m in nm:
            print(f"  {m:6} {om[m]:10.4f} {nm[m]:10.4f} {nm[m]-om[m]:+9.4f}")
    print(f"  ordering submitted: {' > '.join(om.sort_values(ascending=False).index)}")
    print(f"  ordering corrected: {' > '.join(nm.sort_values(ascending=False).index)}")
    print(f"  ORDERING PRESERVED: "
          f"{list(om.sort_values(ascending=False).index) == list(nm.sort_values(ascending=False).index)}")

section("2. PEAK FNR UNDER ATTACK (abstract claims 'above 0.78')")
for ds in DATASETS:
    o, n = load(OLD, ds, "attacks"), load(NEW, ds, "attacks")
    if o is None or n is None:
        continue
    print(f"\n{ds}")
    for label, df in (("submitted", o), ("corrected", n)):
        g = df.groupby("model")["fnr"].max()
        print(f"  {label:10} max FNR per model: " +
              "  ".join(f"{m}={g[m]:.3f}" for m in g.index))
        print(f"  {label:10} #(model,attack) cells with FNR>0.78: "
              f"{int((df['fnr'] > 0.78).sum())} / {len(df)}")

section("3. TRANSFER vs DIRECT (headline claim)")
print("Transferred = gradient attacks crafted on the MLP, evaluated on LR/SVM/RF.")
print("Direct      = HopSkipJump queried against each model itself.")
for ds in DATASETS:
    print(f"\n{ds}")
    for label, base in (("submitted", OLD), ("corrected", NEW)):
        atk, wb = load(base, ds, "attacks"), load(base, ds, "whitebox")
        if atk is None or wb is None:
            continue
        tr = atk[atk["model"].isin(["LR", "SVM", "RF"])].groupby("model")["accuracy"].mean()
        di = wb.groupby("model")["adv_accuracy"].mean() if "adv_accuracy" in wb else None
        if di is None:
            col = [c for c in wb.columns if "acc" in c.lower()]
            di = wb.groupby("model")[col[0]].mean() if col else None
        if di is None:
            continue
        print(f"  {label}:")
        for m in ["LR", "SVM", "RF"]:
            if m in tr.index and m in di.index:
                print(f"    {m:4} transferred acc={tr[m]:.3f}  direct acc={di[m]:.3f}  "
                      f"gap={tr[m]-di[m]:+.3f}")

section("4. DEFENSES: adversarial training vs randomized smoothing")
for ds in DATASETS:
    print(f"\n{ds}")
    for label, base in (("submitted", OLD), ("corrected", NEW)):
        at, rs = load(base, ds, "adv_training"), load(base, ds, "randomized_smoothing")
        if at is None or rs is None:
            continue
        acol_at = "adv_accuracy" if "adv_accuracy" in at else "accuracy"
        acol_rs = "adv_accuracy" if "adv_accuracy" in rs else "accuracy"
        print(f"  {label:10} AT mean {acol_at}={at[acol_at].mean():.4f}   "
              f"RS mean {acol_rs}={rs[acol_rs].mean():.4f}   "
              f"AT-RS={at[acol_at].mean()-rs[acol_rs].mean():+.4f}")

section("5. FEATURE COUNTS SELECTED PER SEED")
import json
for ds in DATASETS:
    olds, news = [], []
    for s in [42, 7, 123, 31, 99]:
        for base, acc in ((OLD, olds), (NEW, news)):
            p = base / ds / f"seed_{s}" / "timings.json"
            if p.exists():
                acc.append(json.load(open(p)).get("n_features"))
    print(f"  {ds:16} submitted={olds}  corrected={news}")
