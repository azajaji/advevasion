"""
Compact "main takeaway" summary table:
  per (dataset, model): clean | undefended adv | AT-defended adv | RS-defended adv
all averaged across attacks (FGSM, PGD, C&W) and seeds.

Reads:  results/<DATASET>/aggregated/{baseline_clean,attacks,adv_training,randomized_smoothing}_summary.csv
Writes: paper/tables_out/summary_takeaway.tex
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results_v2"
OUT = ROOT / "paper" / "tables_out" / "summary_takeaway.tex"

DATASETS = [
    ("CSECICIDS2018", "CSE-CIC-IDS2018"),
    ("TONIOT",        "TON\\_IoT"),
    ("WUSTLEHMS2020", "WUSTL-EHMS-2020"),
]
MODELS = ["LR", "SVM", "RF", "MLP"]


def fmt(x):
    return "--" if pd.isna(x) else f"{x:.3f}"


def mean_over_attacks(df, model):
    sub = df[df["model"] == model]
    sub = sub[sub["family"].isin(["FGSM", "PGD", "CW"])]
    return sub["accuracy_mean"].mean() if len(sub) else float("nan")


def clean_acc(df, model):
    sub = df[df["model"] == model]
    return sub["accuracy_mean"].iloc[0] if len(sub) else float("nan")


def at_defended(df, model):
    # AT-trained MLP for MLP; surrogate-augmented retraining for others
    sub = df[(df["model"] == model) & (df["eval_attack"] != "Clean")]
    sub = sub[sub["family"].isin(["FGSM", "PGD", "CW"])]
    return sub["accuracy_mean"].mean() if len(sub) else float("nan")


def rs_defended(df, model, sigma=0.1):
    # df here is the FULL (per-seed) randomized_smoothing CSV
    sub = df[(df["model"] == model) & (df["eval_attack"] != "Clean")]
    sub = sub[(sub["sigma"] - sigma).abs() < 1e-6]
    sub = sub[sub["family"].isin(["FGSM", "PGD", "CW"])]
    return sub["accuracy"].mean() if len(sub) else float("nan")


def main():
    rows = []
    for ds_key, ds_lbl in DATASETS:
        agg = RES / ds_key / "aggregated"
        clean = pd.read_csv(agg / "baseline_clean_summary.csv")
        atk   = pd.read_csv(agg / "attacks_summary.csv")
        adv   = pd.read_csv(agg / "adv_training_summary.csv")
        rs    = pd.read_csv(agg / "randomized_smoothing_full.csv")
        for m in MODELS:
            rows.append((ds_lbl, m,
                         clean_acc(clean, m),
                         mean_over_attacks(atk, m),
                         at_defended(adv, m),
                         rs_defended(rs, m, sigma=0.1)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\\begin{tabular}{l l c c c c}\n\\toprule\n")
        f.write("\\textbf{Dataset} & \\textbf{Model} & "
                "\\textbf{Clean} & \\textbf{Undefended adv.} & "
                "\\textbf{AT / SAR adv.} & \\textbf{RS adv. ($\\sigma{=}0.1$)} \\\\\n\\midrule\n")
        prev_ds = None
        for ds, m, c, u, a, r in rows:
            ds_cell = ds if ds != prev_ds else ""
            prev_ds = ds
            f.write(f"{ds_cell} & {m} & {fmt(c)} & {fmt(u)} & {fmt(a)} & {fmt(r)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
