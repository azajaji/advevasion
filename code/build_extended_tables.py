"""
Aggregate per-seed CSVs from run_extended.py into LaTeX tables for main.tex.

Reads from:  results_v2/<DATASET>/extended/seed_<S>/*.csv
Writes to:   paper/tables_out/{hsj_n500,constrained_attacks,adaptive_pgd,mi_clean,mi_attacks}.tex

Reads results_v2, not results: the latter is the pre-correction submitted snapshot,
and the extended experiments have been rerun under the corrected pipeline (seeded
HopSkipJump, seeded AdversarialTrainer, eligible-denominator ASR) so that these
appendix tables report the same quantities, on the same splits, as the main body.

Usage:
    python Code/build_extended_tables.py
"""
from __future__ import annotations
from pathlib import Path
from statistics import NormalDist
import math
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results_v2"
OUT = ROOT / "paper" / "tables_out"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["CSECICIDS2018", "TONIOT", "WUSTLEHMS2020"]
DATASET_LABELS = {
    "CSECICIDS2018": "CSE-CIC-IDS2018",
    "TONIOT": "TON\\_IoT",
    "WUSTLEHMS2020": "WUSTL-EHMS-2020",
}


def gather(filename: str) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        ds_dir = RES / ds / "extended"
        if not ds_dir.exists():
            continue
        for seed_dir in sorted(ds_dir.glob("seed_*")):
            f = seed_dir / filename
            if not f.exists():
                continue
            df = pd.read_csv(f)
            df["dataset"] = DATASET_LABELS[ds]
            df["seed"] = int(seed_dir.name.split("_")[1])
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def fmt(x, p=3):
    if pd.isna(x):
        return "--"
    return f"{x:.{p}f}"


def fmt_pm(mean, std, p=3):
    """Format mean +/- std, omitting the +/- when std is NaN (single seed)."""
    if pd.isna(std):
        return fmt(mean, p)
    return f"{fmt(mean, p)}\\,$\\pm$\\,{fmt(std, p)}"


def span(series) -> str:
    """Render an integer count that varies across seeds as a range.

    Collapses to a single number when every seed agrees, so a constant count is
    not dressed up as a range.
    """
    vals = sorted({int(v) for v in series.dropna()})
    if not vals:
        return "--"
    if len(vals) == 1:
        return str(vals[0])
    return f"{vals[0]}--{vals[-1]}"


def wilson_ci(k: int, n: int, conf: float = 0.95):
    """Wilson score interval for k successes out of n Bernoulli trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = NormalDist().inv_cdf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def write_hsj(df: pd.DataFrame):
    if df.empty:
        return
    grp = df.groupby(["dataset", "model"], sort=False)
    rows = []
    for (ds, m), sub in grp:
        # Two denominators, reported side by side rather than one at a time.
        #
        # The eligible-conditional rate divides by the samples the clean model
        # already classifies correctly, since only those can be evaded; it measures
        # attack potency. The full-test rate divides by every evaluated sample and
        # measures the attack's contribution to whole-sub-sample failure. They
        # diverge exactly when clean accuracy is low, which is the case for the
        # linear models on WUSTL-EHMS-2020: conditional ASR is 1.000 there while
        # roughly a quarter of the sub-sample was never eligible to begin with.
        # Printing only the conditional rate reads as total evasion and overstates
        # the operational effect, so both are printed and Eq. (4)'s estimand is
        # named in the caption.
        n = int(sub["n_samples"].iloc[0])
        elig = sub["n_eligible"] if "n_eligible" in sub else sub["n_samples"]
        # Exact successes recovered per seed, then pooled. The previous version
        # averaged the five per-seed Wilson bounds, which is neither a pooled
        # binomial interval nor a between-seed dispersion measure.
        succ = (sub["asr"] * elig).round().astype(int)
        k_pool = int(succ.sum())
        n_pool = int(elig.sum())
        lo, hi = wilson_ci(k_pool, n_pool)
        asr_mean = sub["asr"].mean()
        asr_sd = sub["asr"].std(ddof=1)
        full = succ / sub["n_samples"]
        full_mean = full.mean()
        full_sd = full.std(ddof=1)
        l2 = sub["mean_l2_perturbation"].mean()
        n_elig = int(round(elig.mean()))
        rows.append((ds, m, n, n_elig, asr_mean, asr_sd, lo, hi,
                     full_mean, full_sd, l2))
    out = OUT / "hsj_n500.tex"
    with open(out, "w") as f:
        # Under ieeeaccess.cls a \textbf whose ENTIRE argument is a math group
        # ("\textbf{$N$}") raises "Extra }, or forgotten $" and aborts the table.
        # Math embedded inside text ("\textbf{Mean $\ell_2$ norm}") is fine, so
        # the bare-math headers are written outside \textbf instead.
        f.write("\\begin{tabular}{l l c c c c c c}\n\\toprule\n")
        f.write("& & & & \\multicolumn{2}{c}{\\textbf{Eligible-conditional ASR}} "
                "& \\textbf{Full-test} & \\\\\n")
        f.write("\\cmidrule(lr){5-6}\n")
        f.write("\\textbf{Dataset} & \\textbf{Model} & $N$ & "
                "\\textbf{Eligible} & \\textbf{Mean} $\\pm$ \\textbf{SD} & "
                "\\textbf{Pooled 95\\% CI} & \\textbf{ASR} & "
                "\\textbf{Mean} $\\ell_2$ \\\\\n\\midrule\n")
        prev_ds = None
        for (ds, m, n, n_elig, asr, sd, lo, hi,
             full_mean, full_sd, l2) in rows:
            ds_cell = ds if ds != prev_ds else ""
            prev_ds = ds
            f.write(f"{ds_cell} & {m} & {n} & {n_elig} & {fmt_pm(asr, sd)} & "
                    f"[{fmt(lo)}, {fmt(hi)}] & {fmt_pm(full_mean, full_sd)} & "
                    f"{fmt(l2,2)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"wrote {out}")


def write_constrained(df: pd.DataFrame):
    if df.empty:
        return
    grp = df.groupby(["dataset", "attack", "epsilon"], sort=False)
    rows = []
    for (ds, atk, eps), sub in grp:
        asr_mean = sub["asr"].mean()
        asr_std = sub["asr"].std()
        acc_mean = sub["adv_accuracy"].mean()
        # The selector keeps a different number of features on each seed, so both
        # counts are ranges. Taking .iloc[0] printed whichever seed happened to
        # sort first as though the count were fixed, which disagreed with the
        # ranges the surrounding text reports.
        n_frozen = span(sub["n_features_frozen"])
        n_total = span(sub["n_features_total"])
        rows.append((ds, atk, eps, n_total, n_frozen, asr_mean, asr_std, acc_mean))
    out = OUT / "constrained_attacks.tex"
    with open(out, "w") as f:
        # Math is kept OUTSIDE \textbf here. Under ieeeaccess.cls a math shift
        # inside \textbf in a tabular header raises "Extra }, or forgotten $"
        # and aborts the table; the same header with the math outside compiles.
        f.write("\\begin{tabular}{l l c c c c c}\n\\toprule\n")
        f.write("\\textbf{Dataset} & \\textbf{Attack} & $\\epsilon$ & "
                "\\textbf{Total feats.} & \\textbf{Frozen} & "
                "\\textbf{Adv.\\ Acc.} & \\textbf{ASR} (mean\\,$\\pm$\\,std) \\\\\n\\midrule\n")
        prev_ds = None
        for ds, atk, eps, n_total, n_frozen, asr_m, asr_s, acc_m in rows:
            ds_cell = ds if ds != prev_ds else ""
            prev_ds = ds
            f.write(f"{ds_cell} & {atk.replace('_',' ')} & {eps} & "
                    f"{n_total} & {n_frozen} & "
                    f"{fmt(acc_m)} & {fmt_pm(asr_m, asr_s)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"wrote {out}")


def write_adaptive(df: pd.DataFrame):
    if df.empty:
        return
    grp = df.groupby(["dataset", "attack", "epsilon"], sort=False)
    rows = []
    for (ds, atk, eps), sub in grp:
        asr_mean = sub["asr"].mean()
        asr_std = sub["asr"].std()
        acc_mean = sub["adv_accuracy"].mean()
        rows.append((ds, atk, eps, asr_mean, asr_std, acc_mean))
    out = OUT / "adaptive_pgd.tex"
    with open(out, "w") as f:
        # Bare math kept out of \textbf; see the note in write_hsj().
        f.write("\\begin{tabular}{l l c c c}\n\\toprule\n")
        f.write("\\textbf{Dataset} & \\textbf{Attack} & $\\epsilon$ & "
                "\\textbf{Adv.\\ Acc.} & \\textbf{ASR} (mean\\,$\\pm$\\,std) \\\\\n\\midrule\n")
        prev_ds = None
        for ds, atk, eps, asr_m, asr_s, acc_m in rows:
            ds_cell = ds if ds != prev_ds else ""
            prev_ds = ds
            f.write(f"{ds_cell} & {atk.replace('_',' ')} & {eps} & "
                    f"{fmt(acc_m)} & {fmt_pm(asr_m, asr_s)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"wrote {out}")


def write_mi_clean(df: pd.DataFrame):
    if df.empty:
        return
    grp = df.groupby(["dataset", "model"], sort=False)
    rows = []
    for (ds, m), sub in grp:
        rows.append((ds, m, sub["accuracy"].mean(), sub["precision"].mean(),
                     sub["recall"].mean(), sub["f1"].mean()))
    out = OUT / "mi_clean.tex"
    with open(out, "w") as f:
        f.write("\\begin{tabular}{l l c c c c}\n\\toprule\n")
        f.write("\\textbf{Dataset} & \\textbf{Model} & \\textbf{Acc.} & "
                "\\textbf{Prec.} & \\textbf{Rec.} & \\textbf{F1} \\\\\n\\midrule\n")
        prev_ds = None
        for ds, m, a, p, r, f1 in rows:
            ds_cell = ds if ds != prev_ds else ""
            prev_ds = ds
            f.write(f"{ds_cell} & {m} & {fmt(a)} & {fmt(p)} & {fmt(r)} & {fmt(f1)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"wrote {out}")


def write_mi_attacks(df: pd.DataFrame):
    """Adversarial arm of the mutual-information selector ablation.

    The reviewer concern this ablation answers is whether the ADVERSARIAL-robustness
    ordering is an artifact of RF-importance feature selection, so the clean-accuracy
    table alone cannot settle it. Mean adversarial accuracy per (dataset, model,
    attack family) under MI-selected features, directly comparable to the main-body
    attacks_summary table computed under RF-importance selection.
    """
    if df.empty:
        return
    df = df.copy()
    df["fam"] = df["attack"].str.split("_").str[0]
    df = df[df["fam"].isin(["FGSM", "PGD", "CW"])]
    if df.empty:
        return
    piv = (df.groupby(["dataset", "model", "fam"])["accuracy"]
             .mean().reset_index())
    out = OUT / "mi_attacks.tex"
    fams = ["FGSM", "PGD", "CW"]
    with open(out, "w") as f:
        f.write("\\begin{tabular}{l l c c c}\n\\toprule\n")
        f.write("\\textbf{Dataset} & \\textbf{Model} & \\textbf{FGSM} & "
                "\\textbf{PGD} & \\textbf{C\\&W} \\\\\n\\midrule\n")
        prev_ds = None
        for ds in piv["dataset"].unique():
            for m in ["LR", "SVM", "RF", "MLP"]:
                sub = piv[(piv.dataset == ds) & (piv.model == m)]
                if sub.empty:
                    continue
                ds_cell = ds if ds != prev_ds else ""
                prev_ds = ds
                vals = [sub[sub.fam == fam]["accuracy"] for fam in fams]
                cells = [fmt(v.iloc[0]) if len(v) else "--" for v in vals]
                f.write(f"{ds_cell} & {m} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"wrote {out}")


def main():
    write_hsj(gather("hsj_n500.csv"))
    write_constrained(gather("constrained_attacks.csv"))
    write_adaptive(gather("adaptive_pgd.csv"))
    write_mi_clean(gather("mi_clean.csv"))
    write_mi_attacks(gather("mi_attacks.csv"))


if __name__ == "__main__":
    main()
