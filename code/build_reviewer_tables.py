"""Generate the LaTeX tables that fill the \\PENDING blocks in the resubmission draft.

Each table answers a specific reviewer comment and is built from experiment data that is
already on disk under results_v2/<DATASET>/reviewer/seed_<S>/. No model training happens
here; this is purely aggregation and formatting.

  linear_wb.tex    R2.20  direct FGSM/PGD on LR and SVM's own gradients, versus transfer
  gbm.tex          R2.17  XGBoost and LightGBM under the same protocol as the four baselines
  family.tex       R2.13  attack success per traffic category, not just binary
  sensitivity.tex  R2.12/2.16/2.18/2.19/2.21  one row per protocol factor varied independently

Usage:
    python Code/build_reviewer_tables.py                     # writes to paper/tables_out
    python Code/build_reviewer_tables.py --out <dir>         # e.g. the resubmission work dir
"""
from __future__ import annotations
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results_v2"
DATASETS = ["CSECICIDS2018", "TONIOT", "WUSTLEHMS2020"]
LABEL = {"CSECICIDS2018": "CSE-CIC-IDS2018", "TONIOT": "TON\\_IoT",
         "WUSTLEHMS2020": "WUSTL-EHMS-2020"}


def gather(exp: str) -> pd.DataFrame:
    """Concatenate one reviewer experiment across all datasets and seeds."""
    frames = []
    for ds in DATASETS:
        for f in sorted(glob.glob(str(RESULTS / ds / "reviewer" / "seed_*" / f"{exp}.csv"))):
            df = pd.read_csv(f)
            df["dataset"] = ds
            df["seed"] = int(Path(f).parent.name.split("_")[1])
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def gather_main(name: str) -> pd.DataFrame:
    """Concatenate a primary-pipeline aggregate across datasets."""
    frames = []
    for ds in DATASETS:
        p = RESULTS / ds / "aggregated" / f"{name}_full.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["dataset"] = ds
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def f3(x) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def seed_ms(frame, statfn=None):
    """Return (aggregate, SD across seeds of that same aggregate).

    The SD is this table's own statistic recomputed inside each seed, then the
    dispersion of those five values. It answers the question the reviewer asked:
    would the row move if the study were repeated with different seeds.

    It is deliberately not the standard deviation over every pooled
    (model, dataset, configuration) cell. That quantity is 2 to 14 times larger
    here because RF sits near 0.7 adversarial accuracy while the MLP sits near
    0.1, so it measures the spread BETWEEN classifiers, which Tables 9 and 12
    already report directly. Printing it in the +/- position would read as
    sampling error and make every arm overlap every other arm for a reason that
    has nothing to do with uncertainty.
    """
    if frame is None or len(frame) == 0:
        return (np.nan, np.nan)
    fn = statfn or (lambda d: float(d["accuracy"].mean()))
    per = []
    for s in sorted(frame["seed"].unique()):
        v = fn(frame[frame.seed == s])
        if v is not None and not np.isnan(v):
            per.append(float(v))
    sd = float(np.std(per, ddof=1)) if len(per) > 1 else np.nan
    return (fn(frame), sd)


def f3pm(pair) -> str:
    """Format a (mean, seed-SD) pair; fall back to the bare mean when SD is absent."""
    m, sd = pair
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "--"
    if sd is None or (isinstance(sd, float) and np.isnan(sd)):
        return f"{m:.3f}"
    return f"{m:.3f}\\,$\\pm$\\,{sd:.3f}"


def esc(s: str) -> str:
    return str(s).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def write(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
# R2.20 -- direct gradient attacks on the linear models
# --------------------------------------------------------------------------- #

def build_linear_wb(out: Path) -> None:
    df = gather("linear_wb")
    if df.empty:
        print("linear_wb: no data, skipped")
        return
    atk = gather_main("attacks")

    rows = []
    for ds in DATASETS:
        for m in ["LR", "SVM"]:
            d = df[(df.dataset == ds) & (df.model == m)]
            if d.empty:
                continue
            for eps in sorted(d.epsilon.dropna().unique()):
                sub = d[d.epsilon == eps]
                direct_acc = sub["accuracy"].mean()
                direct_asr = sub["asr"].mean()
                # matched transferred attack at the same epsilon, same model/dataset
                t = atk[(atk.dataset == ds) & (atk.model == m)] if not atk.empty else pd.DataFrame()
                if not t.empty and "attack" in t:
                    t = t[t["attack"].str.contains(f"eps{eps:g}", regex=False, na=False)]
                trans_acc = t["accuracy"].mean() if len(t) else np.nan
                trans_asr = t["asr"].mean() if len(t) else np.nan
                rows.append((ds, m, eps, trans_acc, direct_acc, trans_asr, direct_asr))

    lines = [r"\begin{tabular}{l l c c c c c}", r"\toprule"]
    lines.append(r"& & & \multicolumn{2}{c}{\textbf{Adversarial accuracy}} & "
                 r"\multicolumn{2}{c}{\textbf{ASR}} \\")
    lines.append(r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
    lines.append(r"\textbf{Dataset} & \textbf{Model} & $\epsilon$ & "
                 r"\textbf{Transferred} & \textbf{Direct} & "
                 r"\textbf{Transferred} & \textbf{Direct} \\")
    lines.append(r"\midrule")
    prev = None
    for ds, m, eps, ta, da, tr, dr in rows:
        cell = LABEL[ds] if ds != prev else ""
        prev = ds
        lines.append(f"{cell} & {m} & {eps:g} & {f3(ta)} & {f3(da)} & {f3(tr)} & {f3(dr)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out / "linear_wb.tex", lines)


# --------------------------------------------------------------------------- #
# R2.17 -- gradient-boosted targets
# --------------------------------------------------------------------------- #

def build_gbm(out: Path) -> None:
    df = gather("gbm")
    if df.empty:
        print("gbm: no data, skipped")
        return
    fams = ["FGSM", "PGD", "CW"]
    lines = [r"\begin{tabular}{l l c c c c c}", r"\toprule"]
    lines.append(r"\textbf{Dataset} & \textbf{Model} & \textbf{Clean} & "
                 r"\textbf{FGSM} & \textbf{PGD} & \textbf{C\&W} & "
                 r"\textbf{HSJ (direct)} \\")
    lines.append(r"\midrule")
    prev = None
    for ds in DATASETS:
        for m in ["XGB", "LGBM"]:
            d = df[(df.dataset == ds) & (df.model == m)]
            if d.empty:
                continue
            clean = d[d["attack"] == "clean"]["accuracy"].mean()
            cells = []
            for fam in fams:
                s = d[d.get("family").eq(fam)] if "family" in d else pd.DataFrame()
                cells.append(s["accuracy"].mean() if len(s) else np.nan)
            hsj = d[d["attack"] == "HopSkipJump_direct"]
            hsj_acc = hsj["accuracy"].mean() if len(hsj) else np.nan
            cell = LABEL[ds] if ds != prev else ""
            prev = ds
            name = "XGBoost" if m == "XGB" else "LightGBM"
            lines.append(f"{cell} & {name} & {f3(clean)} & " +
                         " & ".join(f3(c) for c in cells) + f" & {f3(hsj_acc)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out / "gbm.tex", lines)


# --------------------------------------------------------------------------- #
# R2.13 -- per-traffic-family vulnerability
# --------------------------------------------------------------------------- #

def build_family(out: Path) -> None:
    """Requires the traffic_family column added after the key-collision fix.

    Older family.csv files carry only the ATTACK family (FGSM/PGD/CW) because a
    dict-key collision overwrote the traffic category. Those files cannot answer
    the reviewer's question and are refused here rather than silently rendered.
    """
    df = gather("family")
    if df.empty:
        print("family: no data, skipped")
        return
    if "traffic_family" not in df.columns:
        print("family: SKIPPED -- traffic_family column absent (pre-fix data). "
              "Re-run: python Code/run_cells.py --only family --out results_v2")
        return

    df = df[df["n_detected_clean"] > 0]
    # Math stays outside \textbf: under ieeeaccess.cls a math shift inside
    # \textbf in a tabular header raises "Extra }, or forgotten $" and kills
    # the table (see the same fix in build_extended_tables.py).
    lines = [r"\begin{tabular}{l l r c c}", r"\toprule"]
    lines.append(r"\textbf{Dataset} & \textbf{Traffic family} & $n$ & "
                 r"\textbf{Mean ASR} & \textbf{Mean FNR under attack} \\")
    lines.append(r"\midrule")
    prev = None
    for ds in DATASETS:
        d = df[df.dataset == ds]
        if d.empty:
            continue
        g = (d.groupby("traffic_family")
               .agg(n=("n", "max"), asr=("asr", "mean"), fnr=("fnr_under_attack", "mean"))
               .reset_index().sort_values("asr", ascending=False))

        # The reviewer asked for attack-specific vulnerability, which the spread
        # across families answers; enumerating every rare "- Attempted" subclass
        # does not add to it. A subclass is collapsed only when it is small
        # (n < RARE_N) AND unremarkable, meaning its ASR sits inside the range
        # already spanned by the categories kept. The extremes are always kept,
        # and so is any rare category whose ASR is surprising, since that is
        # exactly the kind of case the breakdown exists to surface.
        RARE_N = 100
        keep_mask = g["n"] >= RARE_N
        if keep_mask.sum() >= 2:
            lo, hi = g.loc[keep_mask, "asr"].min(), g.loc[keep_mask, "asr"].max()
            # extremes of the whole dataset, and rare-but-anomalous categories
            keep_mask |= g["asr"].eq(g["asr"].max()) | g["asr"].eq(g["asr"].min())
            keep_mask |= (g["n"] < RARE_N) & ((g["asr"] > hi) | (g["asr"] < lo))
        else:
            keep_mask[:] = True
        keep, rare = g[keep_mask], g[~keep_mask]

        for _, r in keep.iterrows():
            cell = LABEL[ds] if ds != prev else ""
            prev = ds
            lines.append(f"{cell} & {esc(r['traffic_family'])} & {int(r['n'])} & "
                         f"{f3(r['asr'])} & {f3(r['fnr'])} \\\\")
        if len(rare) >= 2:
            cell = LABEL[ds] if ds != prev else ""
            prev = ds
            label = (f"Other rare subclasses ({len(rare)} categories, "
                     f"$n < {RARE_N}$; ASR {f3(rare['asr'].min())}--{f3(rare['asr'].max())})")
            lines.append(f"{cell} & {label} & {int(rare['n'].sum())} & "
                         f"{f3(rare['asr'].mean())} & {f3(rare['fnr'].mean())} \\\\")
            print(f"  {ds}: grouped {len(rare)} rare subclasses "
                  f"(ASR {rare['asr'].min():.3f}-{rare['asr'].max():.3f}, kept {len(keep)})")
        elif len(rare) == 1:
            for _, r in rare.iterrows():
                cell = LABEL[ds] if ds != prev else ""
                prev = ds
                lines.append(f"{cell} & {esc(r['traffic_family'])} & {int(r['n'])} & "
                             f"{f3(r['asr'])} & {f3(r['fnr'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out / "family.tex", lines)


# --------------------------------------------------------------------------- #
# Protocol sensitivity: one row per factor varied independently
# --------------------------------------------------------------------------- #

def gather_temporal() -> pd.DataFrame:
    """temporal lives under its own dataset key (the timestamp-retaining CSE
    variant), so it is not picked up by gather()'s DATASETS loop."""
    frames = []
    for f in sorted(glob.glob(str(RESULTS / "CSECICIDS2018_TEMPORAL" / "reviewer"
                                  / "seed_*" / "temporal.csv"))):
        df = pd.read_csv(f)
        df["seed"] = int(Path(f).parent.name.split("_")[1])
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_temporal(out: Path) -> None:
    """Random versus chronological splitting on CSE-CIC-IDS2018.

    Reported as its own table rather than a single sensitivity row because it is
    the one protocol factor that changes a conclusion: clean accuracy falls by
    roughly 20-28 points under chronological evaluation and the clean-data
    ranking of RF and the MLP swaps.
    """
    df = gather_temporal()
    if df.empty:
        print("temporal: no data, skipped")
        return
    clean = df[df["attack"] == "clean"]
    adv = df[df["attack"] != "clean"]

    lines = [r"\begin{tabular}{l c c c c c}", r"\toprule"]
    lines.append(r"& \multicolumn{2}{c}{\textbf{Clean accuracy}} & "
                 r"\multicolumn{2}{c}{\textbf{Adversarial accuracy}} & \\")
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    lines.append(r"\textbf{Model} & \textbf{Random} & \textbf{Chronological} & "
                 r"\textbf{Random} & \textbf{Chronological} & "
                 r"$\Delta$ \textbf{clean} \\")
    lines.append(r"\midrule")
    # Each cell here is a single model on a single dataset, so the dispersion
    # across seeds is an unambiguous seed-to-seed uncertainty with nothing else
    # pooled into it. That is not true of the sensitivity table, whose cells
    # average over four classifiers and three benchmarks as well.
    for m in ["LR", "SVM", "RF", "MLP"]:
        cr = seed_ms(clean[(clean.model == m) & (clean.split == "random")])
        ct = seed_ms(clean[(clean.model == m) & (clean.split == "temporal")])
        ar = seed_ms(adv[(adv.model == m) & (adv.split == "random")])
        at = seed_ms(adv[(adv.model == m) & (adv.split == "temporal")])
        d = ct[0] - cr[0]
        lines.append(f"{m} & {f3pm(cr)} & {f3pm(ct)} & {f3pm(ar)} & {f3pm(at)} & "
                     f"{d:+.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out / "temporal.tex", lines)

    n = df["seed"].nunique()
    print(f"  (temporal built from {n} seed(s))")


def build_tuning_grid(out: Path) -> None:
    """R2.18 / R3.5: the search space and what it actually selected.

    Reviewer 3 asked how the baseline arrived at 50 trees. Reporting the pooled
    accuracy change alone does not answer that; the candidate values and the
    settings the search returned do. Selections vary by seed and dataset, so the
    modal choice is reported with its frequency over the 15 (dataset, seed) runs
    rather than as a single fixed configuration.
    """
    import ast
    from collections import Counter

    tu = gather("tuning")
    if tu.empty or "best_params" not in tu.columns:
        print("tuning grid: no data, skipped")
        return
    tu = tu[tu.config == "tuned"]

    # Grid sizes matter: the search is capped at twelve evaluations, but the LR
    # and SVM grids are smaller than that cap, so scikit-learn reduces n_iter to
    # the grid size and searches them exhaustively. Only RF is actually sampled.
    SPACE = {
        "LR":  r"$C \in \{0.01, 0.1, 1, 10, 100\}$; class weight $\in$ \{None, balanced\}; "
               r"10 configurations, searched exhaustively",
        "SVM": r"$C \in \{0.01, 0.1, 1, 10\}$; class weight $\in$ \{None, balanced\}; "
               r"8 configurations, searched exhaustively",
        "RF":  r"trees $\in \{50, 100, 200, 400\}$; max depth $\in$ \{None, 10, 20, 40\}; "
               r"min leaf $\in \{1, 2, 5\}$; max features $\in$ \{sqrt, log2, None\}; "
               r"144 configurations, 12 sampled per run",
    }
    PRETTY = {"C": "$C$", "class_weight": "class weight", "n_estimators": "trees",
              "max_depth": "max depth", "min_samples_leaf": "min leaf",
              "max_features": "max features"}
    ORDER = {"LR": ["C", "class_weight"], "SVM": ["C", "class_weight"],
             "RF": ["n_estimators", "max_depth", "min_samples_leaf", "max_features"]}

    lines = [r"\begin{tabularx}{\linewidth}{@{}l X X@{}}", r"\toprule",
             r"\textbf{Model} & \textbf{Candidate values searched} & "
             r"\textbf{Most frequently selected} \\", r"\midrule"]
    for m in ["LR", "SVM", "RF"]:
        sub = tu[tu.model == m]
        if sub.empty:
            continue
        n = len(sub)
        picks = []
        for k in ORDER[m]:
            c = Counter(str(ast.literal_eval(s).get(k)) for s in sub.best_params)
            val, freq = c.most_common(1)[0]
            # "None" for max_features means every feature is considered, which
            # reads as its opposite if rendered as "none"; keep it explicit.
            val = {"None": "None", "True": "yes", "False": "no"}.get(val, val)
            if k == "max_features" and val == "None":
                val = "None (all features)"
            picks.append(f"{PRETTY[k]} = {esc(val)} ({freq}/{n})")
        lines.append(f"{m} & {SPACE[m]} & {'; '.join(picks)} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    write(out / "tuning_grid.tex", lines)


def build_sensitivity(out: Path) -> None:
    rows = []

    sel = gather("selection")
    if not sel.empty and "selector" in sel:
        adv = sel[sel["attack"] != "clean"]
        for s in ["rf", "mi", "none"]:
            d = adv[adv.selector == s]
            c = sel[(sel.selector == s) & (sel["attack"] == "clean")]
            if d.empty and c.empty:
                continue
            nf = sel[sel.selector == s]["n_features"].dropna()
            rows.append(("Feature selector", {"rf": "RF importance (as submitted)",
                                              "mi": "Mutual information",
                                              "none": "No selection"}[s],
                         f"{nf.mean():.0f}" if len(nf) else "--",
                         seed_ms(c), seed_ms(d)))

    res = gather("resampling")
    if not res.empty and "oversample" in res:
        adv = res[res["attack"] != "clean"]
        # Reported per dataset rather than pooled. The imbalance threshold fires
        # only on TON_IoT and WUSTL-EHMS-2020, whose adversarial accuracies sit at
        # different levels, so a pooled mean over the two hides the per-dataset
        # comparison the text actually makes and leaves those numbers with no
        # visible support.
        for ds in DATASETS:
            sub = res[res.dataset == ds]
            if sub.empty or not sub["resampled"].fillna(False).any():
                continue
            for o in sorted(sub.oversample.dropna().unique()):
                d = adv[(adv.oversample == o) & (adv.dataset == ds)]
                c = sub[(sub.oversample == o) & (sub["attack"] == "clean")]
                # FNR is carried alongside accuracy for this factor because the
                # text compares the schemes on missed attacks, and on WUSTL the
                # two metrics disagree: dropping oversampling raises clean
                # accuracy while also raising the clean false-negative rate.
                cf = c["fnr"].mean() if len(c) else np.nan
                df_ = d["fnr"].mean() if len(d) else np.nan
                rows.append(("Oversampling", f"{esc(o)} ({LABEL[ds]})",
                             f"{cf:.3f} / {df_:.3f}" if len(c) and len(d) else "--",
                             seed_ms(c), seed_ms(d)))

    sur = gather("surrogate")
    if not sur.empty and "surrogate" in sur:
        for s in sur.surrogate.dropna().unique():
            d = sur[sur.surrogate == s]
            sur_clean = (seed_ms(d, lambda x: float(x["surrogate_clean_acc"].mean()))
                         if "surrogate_clean_acc" in d else (np.nan, np.nan))
            rows.append(("Surrogate capacity", esc(s),
                         str(d["hidden"].iloc[0]) if "hidden" in d else "--",
                         sur_clean, seed_ms(d)))

    # Clean accuracy comes from the `tuning` cell; the adversarial column comes
    # from `tuning_adv`, which puts both configurations through the same
    # transferred attack grid the primary models face. Without it this row had a
    # dash under adversarial accuracy, so the arm spoke only to baseline strength
    # on clean data and not to whether tuning changes robustness.
    tun = gather("tuning")
    tadv = gather("tuning_adv")
    if not tadv.empty and "attack" in tadv:
        tadv = tadv.copy()
        tadv["fam"] = tadv["attack"].astype(str).str.split("_").str[0]
        tadv = tadv[tadv.fam.isin(["FGSM", "PGD", "CW"])]
    if not tun.empty and "config" in tun:
        # mean per attack family first, so the three families weigh equally
        # regardless of how many budget configurations each has
        famw = lambda x: float(
            x.groupby(["dataset", "model", "fam"])["accuracy"].mean().mean())
        for c in sorted(tun.config.dropna().unique()):
            d = tun[tun.config == c]
            adv = (np.nan, np.nan)
            if not tadv.empty:
                s = tadv[tadv.config == c]
                if not s.empty:
                    adv = seed_ms(s, famw)
            rows.append(("Hyperparameter tuning", esc(c), "--",
                         seed_ms(d), adv))

    # Data split. Scope differs from the other factors: chronological evaluation
    # needs capture timestamps, which only the CSE-CIC variant retains, so this
    # row is CSE-CIC-only while the rows above pool all three datasets. Flagged
    # in the setting label rather than left for the reader to infer.
    tmp = gather_temporal()
    if not tmp.empty:
        for split, label in (("random", "Random stratified (as submitted)"),
                             ("temporal", "Chronological")):
            c = tmp[(tmp.split == split) & (tmp["attack"] == "clean")]
            a = tmp[(tmp.split == split) & (tmp["attack"] != "clean")]
            rows.append(("Data split (CSE-CIC only)", label, "--",
                         seed_ms(c), seed_ms(a)))

    bud = gather("hsj_budget")
    if not bud.empty:
        for (mi, me), d in bud.groupby(["max_iter", "max_eval"]):
            rows.append(("HSJ query budget", f"({int(mi)}, {int(me)})", "--",
                         (np.nan, np.nan),
                         seed_ms(d, lambda x: float(x["adv_accuracy"].mean()))))

    lines = [r"\begin{tabular}{l l c c c}", r"\toprule"]
    lines.append(r"\textbf{Factor} & \textbf{Setting} & \textbf{Factor-specific value} & "
                 r"\textbf{Clean acc.} & \textbf{Adv.\ acc.} \\")
    lines.append(r"\midrule")
    prev = None
    for fac, setting, nf, ca, aa in rows:
        cell = fac if fac != prev else ""
        if fac != prev and prev is not None:
            lines.append(r"\midrule")
        prev = fac
        lines.append(f"{cell} & {setting} & {nf} & {f3pm(ca)} & {f3pm(aa)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out / "sensitivity.tex", lines)

    if not glob.glob(str(RESULTS / "*" / "reviewer" / "seed_*" / "temporal.csv")):
        print("  NOTE: no temporal.csv anywhere -- the chronological-split row of the "
              "sensitivity table has no data. Reviewer 2 asked for this explicitly.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "paper" / "tables_out"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    build_linear_wb(out)
    build_gbm(out)
    build_family(out)
    build_temporal(out)
    build_tuning_grid(out)
    build_sensitivity(out)


if __name__ == "__main__":
    main()
