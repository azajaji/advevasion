"""Generate the supplementary tables that have no other producer.

The manuscript promises a supplementary document in four places (attack-parameter
settings, the full per-sigma randomized-smoothing grid, the N=500 HopSkipJump tables,
and the mutual-information ablation). The per-experiment fragments are produced by
build_tables.py and build_extended_tables.py; this script adds the two that were
promised but never had a producer at all:

  supp_hyperparams.tex  -- every model, attack, defense and pipeline setting, read
                           directly from the config dictionaries in run_experiments.py
                           so the table cannot drift from the code that ran.
  supp_hsj_budget.tex   -- the HopSkipJump query-budget ladder, which is the evidence
                           for the saturation claim in the robustness-checks section.

Usage:  python Code/build_supplementary_tables.py
"""
from __future__ import annotations
import glob
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

import pandas as pd

import run_experiments as base

OUT = CODE_DIR.parent / "paper" / "tables_out"
OUT.mkdir(parents=True, exist_ok=True)
RES = CODE_DIR.parent / "results_v2"

DATASET_LABELS = {
    "CSECICIDS2018": "CSE-CIC-IDS2018",
    "TONIOT": "TON\\_IoT",
    "WUSTLEHMS2020": "WUSTL-EHMS-2020",
}


def _row(f, group, param, value):
    f.write(f"{group} & {param} & {value} \\\\\n")


def write_hyperparams():
    a = base.ATTACK_CONFIGS
    d = base.DEFENSE_CONFIGS
    m = base.MLP_CONFIG
    ml = base.ML_MODELS_CONFIG

    out = OUT / "supp_hyperparams.tex"
    with open(out, "w") as f:
        f.write("\\begin{tabular}{@{}l l l@{}}\n\\toprule\n")
        f.write("\\textbf{Component} & \\textbf{Parameter} & \\textbf{Value} \\\\\n")
        f.write("\\midrule\n")

        _row(f, "Split", "Test fraction", f"{base.TEST_SIZE} (stratified)")
        _row(f, "", "Seeds", "42, 7, 123, 31, 99")
        _row(f, "Preprocessing", "Scaler", "StandardScaler (fitted on train fold)")
        _row(f, "", "Feature ranker", "RF impurity importance (default); "
                                      "mutual information (ablation)")
        _row(f, "", "Feature-count search",
             "$n \\in \\{5,10,\\dots,50\\}$, 3-fold CV, macro-F1")
        _row(f, "", "Class imbalance", "SMOTE on the training partition only")
        f.write("\\midrule\n")

        _row(f, "LR", "max\\_iter", ml["LR"]["max_iter"])
        _row(f, "Linear SVM", "max\\_iter", ml["SVM"]["max_iter"])
        _row(f, "RF", "n\\_estimators", ml["RF"]["n_estimators"])
        _row(f, "", "max\\_depth", "unbounded (grown to purity)")
        _row(f, "MLP", "Hidden layers", " , ".join(str(x) for x in m["hidden_layers"]))
        _row(f, "", "Learning rate", m["lr"])
        _row(f, "", "Batch size / epochs", f"{m['batch_size']} / {m['nb_epochs']}")
        f.write("\\midrule\n")

        _row(f, "FGSM", "$\\epsilon$",
             ", ".join(str(e) for e in a["FGSM"]["epsilon_values"]))
        _row(f, "PGD", "$\\epsilon$",
             ", ".join(str(e) for e in a["PGD"]["epsilon_values"]))
        _row(f, "", "max\\_iter",
             ", ".join(str(i) for i in a["PGD"]["max_iter_values"]))
        _row(f, "C\\&W ($\\ell_2$)", "Confidence",
             ", ".join(str(c) for c in a["CW"]["confidence_values"]))
        _row(f, "", "max\\_iter",
             ", ".join(str(i) for i in a["CW"]["max_iter_values"]))
        _row(f, "HopSkipJump", "max\\_iter / max\\_eval", "10 / 200")
        _row(f, "", "init\\_eval / init\\_size", "20 / 20")
        _row(f, "", "Norm / batch size", "$\\ell_2$ / 32")
        _row(f, "", "Evaluation sample", "$N{=}200$ (main), $N{=}500$ (appendix)")
        f.write("\\midrule\n")

        at, rs = d["adversarial_training"], d["randomized_smoothing"]
        _row(f, "Adversarial training", "Inner attacks",
             ", ".join(at["attack_types"]))
        _row(f, "", "Adversarial ratio",
             ", ".join(str(r) for r in at["ratio_values"]))
        _row(f, "", "Epochs", at["nb_epochs"])
        _row(f, "Randomized smoothing", "$\\sigma$",
             ", ".join(str(s) for s in rs["sigma_values"]))
        _row(f, "", "$n_0$ / $n$", f"{rs['n0']} / {rs['n']}")
        _row(f, "", "$\\alpha$", rs["alpha"])
        _row(f, "", "Evaluation subset", rs["test_subset_size"])

        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"wrote {out}")


def write_hsj_budget():
    """Query-budget ladder: the evidence behind the ASR-saturation claim."""
    files = sorted(glob.glob(str(RES / "*" / "reviewer" / "seed_*" / "hsj_budget.csv")))
    if not files:
        print("no hsj_budget.csv found, skipped")
        return
    frames = []
    for f in files:
        p = Path(f)
        df = pd.read_csv(f)
        df["dataset"] = p.parents[2].name
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)

    g = (full.groupby(["dataset", "model", "max_iter", "max_eval"])
              .agg(asr_mean=("asr", "mean"), asr_std=("asr", "std"),
                   exhausted=("query_budget_exhausted", "sum"))
              .reset_index())

    ladders = sorted({(int(r.max_iter), int(r.max_eval)) for r in g.itertuples()})
    out = OUT / "supp_hsj_budget.tex"
    with open(out, "w") as f:
        cols = "l l" + " c" * len(ladders)
        f.write(f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n")
        head = " & ".join(f"({mi},{me})" for mi, me in ladders)
        f.write(f"\\textbf{{Dataset}} & \\textbf{{Model}} & {head} \\\\\n")
        f.write("\\midrule\n")
        prev = None
        for ds in ["CSECICIDS2018", "TONIOT", "WUSTLEHMS2020"]:
            for m in ["LR", "SVM", "RF"]:
                sub = g[(g.dataset == ds) & (g.model == m)]
                if sub.empty:
                    continue
                cells = []
                for mi, me in ladders:
                    r = sub[(sub.max_iter == mi) & (sub.max_eval == me)]
                    cells.append(f"{r.asr_mean.iloc[0]:.3f}" if len(r) else "--")
                lbl = DATASET_LABELS.get(ds, ds)
                ds_cell = lbl if lbl != prev else ""
                prev = lbl
                f.write(f"{ds_cell} & {m} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"wrote {out}")
    tot = int(g["exhausted"].sum())
    print(f"  (query-budget exhaustion events across the whole ladder: {tot})")


if __name__ == "__main__":
    write_hyperparams()
    write_hsj_budget()
