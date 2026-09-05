"""Build LaTeX-ready tables from aggregated CSVs.

Outputs:
  paper/tables_out/baseline.tex          (Table 3: clean baseline)
  paper/tables_out/attacks_summary.tex   (Table 4: mean adversarial accuracy by family)
  paper/tables_out/attack_params.tex     (already in main.tex; just verify)
  paper/tables_out/fnr_under_attack.tex  (FNR table)
  paper/tables_out/cost.tex              (computational-cost table)
  paper/tables_out/rs_per_sigma.tex      (randomized smoothing per sigma)
  paper/tables_out/whitebox.tex          (HopSkipJump white-box accuracy)
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results_v2"
OUT = ROOT / "paper" / "tables_out"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["CSECICIDS2018", "TONIOT", "WUSTLEHMS2020"]
DATASET_LABELS = {
    "CSECICIDS2018": "CSE-CIC-IDS2018",
    "TONIOT": "TON\\_IoT",
    "WUSTLEHMS2020": "WUSTL-EHMS-2020",
}
MODELS = ["LR", "SVM", "RF", "MLP"]


def fmt(mean: float, std: float, dp: int = 3) -> str:
    if np.isnan(mean):
        return "--"
    if np.isnan(std) or std < 1e-6:
        return f"{mean:.{dp}f}"
    return f"{mean:.{dp}f}\\,$\\pm$\\,{std:.{dp}f}"


def safe_get(df: pd.DataFrame, model: str, col: str) -> tuple[float, float]:
    """Mean and std for column where model matches; assumes df has 'seed' column."""
    sub = df[df["model"] == model][col]
    if sub.empty:
        return float("nan"), float("nan")
    return float(sub.mean()), float(sub.std(ddof=0))


def load_full(ds: str, name: str) -> pd.DataFrame:
    p = RESULTS / ds / "aggregated" / f"{name}_full.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


# ---------------------------------------------------------------------------- #

def build_baseline_table():
    rows = []
    for ds in DATASETS:
        df = load_full(ds, "baseline_clean")
        for m in MODELS:
            sub = df[df["model"] == m]
            row = {"dataset": ds, "model": m}
            for col in ["accuracy", "precision", "recall", "f1"]:
                if sub.empty:
                    row[col] = ""
                else:
                    row[col] = fmt(sub[col].mean(), sub[col].std(ddof=0))
            rows.append(row)
    out = []
    out.append(r"\begin{tabular}{lcccccccccccc}")
    out.append(r"\toprule")
    out.append(
        " & " +
        " & ".join(f"\\multicolumn{{4}}{{c}}{{\\textbf{{{DATASET_LABELS[d]}}}}}"
                   for d in DATASETS) + r" \\")
    out.append(r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}")
    out.append(r"\textbf{Model} & "
               + " & ".join(["Acc.", "Prec.", "Rec.", "F1"] * 3) + r" \\")
    out.append(r"\midrule")
    for m in MODELS:
        cells = [m]
        for d in DATASETS:
            r = next(r for r in rows if r["dataset"] == d and r["model"] == m)
            cells += [r["accuracy"], r["precision"], r["recall"], r["f1"]]
        out.append(" & ".join(cells) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    (OUT / "baseline.tex").write_text("\n".join(out))
    print(f"wrote {OUT / 'baseline.tex'}")


def build_attacks_summary_table():
    rows = []
    for ds in DATASETS:
        df = load_full(ds, "attacks")
        if df.empty:
            continue
        df = df.copy()
        df["fam"] = df["attack"].str.split("_").str[0]
        for m in MODELS:
            d = {"dataset": ds, "model": m}
            for fam in ["FGSM", "PGD", "CW"]:
                sub = df[(df["model"] == m) & (df["fam"] == fam)]
                if sub.empty:
                    d[fam] = "--"
                else:
                    d[fam] = fmt(sub["accuracy"].mean(),
                                 sub["accuracy"].std(ddof=0))
            rows.append(d)
    out = [r"\begin{tabular}{lccccccccc}", r"\toprule"]
    out.append(
        " & " +
        " & ".join(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{DATASET_LABELS[d]}}}}}"
                   for d in DATASETS) + r" \\")
    out.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}")
    out.append(r"\textbf{Model} & "
               + " & ".join(["FGSM", "PGD", "C\\&W"] * 3) + r" \\")
    out.append(r"\midrule")
    for m in MODELS:
        cells = [m]
        for d in DATASETS:
            r = next(r for r in rows if r["dataset"] == d and r["model"] == m)
            cells += [r["FGSM"], r["PGD"], r["CW"]]
        out.append(" & ".join(cells) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    (OUT / "attacks_summary.tex").write_text("\n".join(out))
    print(f"wrote {OUT / 'attacks_summary.tex'}")


def build_fnr_under_attack_table():
    rows = []
    for ds in DATASETS:
        df_a = load_full(ds, "attacks")
        df_c = load_full(ds, "baseline_clean")
        if df_a.empty:
            continue
        df_a = df_a.copy()
        df_a["fam"] = df_a["attack"].str.split("_").str[0]
        for m in MODELS:
            d = {"dataset": ds, "model": m}
            sub = df_c[df_c["model"] == m]
            d["Clean"] = fmt(sub["fnr"].mean(), sub["fnr"].std(ddof=0)) if not sub.empty else "--"
            for fam in ["FGSM", "PGD", "CW"]:
                s = df_a[(df_a["model"] == m) & (df_a["fam"] == fam)]
                d[fam] = fmt(s["fnr"].mean(), s["fnr"].std(ddof=0)) if not s.empty else "--"
            rows.append(d)
    out = [r"\begin{tabular}{lcccccccccccc}", r"\toprule"]
    out.append(
        " & " +
        " & ".join(f"\\multicolumn{{4}}{{c}}{{\\textbf{{{DATASET_LABELS[d]}}}}}"
                   for d in DATASETS) + r" \\")
    out.append(r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}")
    out.append(r"\textbf{Model} & "
               + " & ".join(["Clean", "FGSM", "PGD", "C\\&W"] * 3) + r" \\")
    out.append(r"\midrule")
    for m in MODELS:
        cells = [m]
        for d in DATASETS:
            r = next(r for r in rows if r["dataset"] == d and r["model"] == m)
            cells += [r["Clean"], r["FGSM"], r["PGD"], r["CW"]]
        out.append(" & ".join(cells) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    (OUT / "fnr_under_attack.tex").write_text("\n".join(out))
    print(f"wrote {OUT / 'fnr_under_attack.tex'}")


def build_cost_table():
    rows = []
    for ds in DATASETS:
        # training time per model from timings.json averaged over seeds
        timings = []
        for seed_dir in (RESULTS / ds).glob("seed_*"):
            tj = seed_dir / "timings.json"
            if tj.exists():
                timings.append(json.loads(tj.read_text())["timings"])
        lat = load_full(ds, "latency")
        rs = load_full(ds, "randomized_smoothing")

        # Defended training time. The discussion claims adversarial training raises
        # training cost, so the number backing that claim belongs in this table
        # rather than only in the prose. For the MLP this is true adversarial
        # training; for LR/SVM/RF it is surrogate-augmented retraining, which is a
        # different (and sometimes cheaper) procedure -- see the caption.
        adv = load_full(ds, "adv_training")

        for m in MODELS:
            train_key = f"{m}_train_s"
            train_vals = [t[train_key] for t in timings if train_key in t]
            tr_mean = float(np.mean(train_vals)) if train_vals else float("nan")
            tr_std = float(np.std(train_vals, ddof=0)) if train_vals else float("nan")

            at_mean = at_std = float("nan")
            if not adv.empty and "model" in adv:
                sub = adv[adv["model"] == m]
                # Both timing columns exist for every row; only one is populated
                # per model (adv_train_s for the adversarially trained MLP,
                # retrain_s for the surrogate-augmented classical models).
                # Selecting on presence alone always picked adv_train_s and left
                # LR/SVM/RF showing a dash despite their times being recorded.
                col = next((c for c in ("adv_train_s", "retrain_s")
                            if c in sub and sub[c].notna().any()), None)
                if col and not sub.empty:
                    per_seed = sub.groupby("seed")[col].max()
                    at_mean = float(per_seed.mean())
                    at_std = float(per_seed.std(ddof=0))

            l = lat[lat["model"] == m]["per_sample_ms"]
            l_mean = float(l.mean()) if not l.empty else float("nan")
            l_std = float(l.std(ddof=0)) if not l.empty else float("nan")

            r = rs[(rs["model"] == m) & (rs["sigma"] == 0.1)
                    & (rs["eval_attack"] == "Clean")]["smooth_s_per_sample"]
            r_mean = float(r.mean()) * 1000.0 if not r.empty else float("nan")
            r_std = float(r.std(ddof=0)) * 1000.0 if not r.empty else float("nan")

            rows.append({
                "dataset": ds, "model": m,
                "train": fmt(tr_mean, tr_std, dp=2),
                "at_train": fmt(at_mean, at_std, dp=2),
                # Reported in microseconds: the linear models predict in well
                # under a microsecond, which renders as "0.000 +/- 0.000" ms and
                # reads as unmeasured rather than fast.
                "infer": fmt(l_mean * 1000.0, l_std * 1000.0, dp=1),
                "smooth": fmt(r_mean, r_std, dp=2),
            })

    out = [r"\begin{tabular}{l l c c c c}", r"\toprule"]
    out.append(r"\textbf{Dataset} & \textbf{Model} & \textbf{Train (s)} & "
               r"\textbf{Defended train (s)} & "
               r"\textbf{Infer ($\mu$s/sample)} & \textbf{Smooth $\sigma{=}0.1$ (ms/sample)} \\")
    out.append(r"\midrule")
    cur = None
    for r in rows:
        ds_label = DATASET_LABELS[r["dataset"]] if r["dataset"] != cur else ""
        cur = r["dataset"]
        out.append(f"{ds_label} & {r['model']} & {r['train']} & {r['at_train']} & "
                   f"{r['infer']} & {r['smooth']} \\\\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    (OUT / "cost.tex").write_text("\n".join(out))
    print(f"wrote {OUT / 'cost.tex'}")


def build_rs_per_sigma_table():
    rows = []
    for ds in DATASETS:
        df = load_full(ds, "randomized_smoothing")
        if df.empty:
            continue
        df = df.copy()
        df["fam"] = df["eval_attack"].apply(
            lambda s: "Clean" if s == "Clean" else s.split("_")[0]
        )
        sigmas = sorted(df["sigma"].unique())
        for m in MODELS:
            d = {"dataset": ds, "model": m}
            for sigma in sigmas:
                for fam in ["Clean", "FGSM", "PGD", "CW"]:
                    s = df[(df["model"] == m) & (df["sigma"] == sigma) & (df["fam"] == fam)]
                    d[(sigma, fam)] = fmt(s["accuracy"].mean(),
                                          s["accuracy"].std(ddof=0)) if not s.empty else "--"
            d["sigmas"] = sigmas
            rows.append(d)

    out = [r"% Randomized smoothing accuracy per sigma per attack family per dataset."]
    for ds in DATASETS:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        if not ds_rows:
            continue
        sigmas = ds_rows[0]["sigmas"]
        out.append(f"\\paragraph*{{{DATASET_LABELS[ds]}}}")
        # Each dataset panel is 1 + 4*len(sigmas) columns wide -- 13 at three
        # sigma values -- which overruns even the full two-column text width by
        # about 215pt, clipping the last sigma block off the page entirely. No
        # font step closes a gap that size, so each panel is scaled to the line
        # width individually. It has to be per-panel: the three tabulars are
        # separated by \paragraph* headings, and a single \resizebox around the
        # whole block cannot contain them.
        out.append(r"\resizebox{\linewidth}{!}{%")
        out.append(r"\begin{tabular}{l" + "c" * (4 * len(sigmas)) + "}")
        out.append(r"\toprule")
        head1 = "Model"
        for sigma in sigmas:
            head1 += f" & \\multicolumn{{4}}{{c}}{{$\\sigma={sigma}$}}"
        out.append(head1 + r" \\")
        for i, _ in enumerate(sigmas):
            out.append(f"\\cmidrule(lr){{{2 + 4*i}-{5 + 4*i}}}")
        head2 = "" + "".join([" & Clean & FGSM & PGD & C\\&W"] * len(sigmas))
        out.append(head2 + r" \\")
        out.append(r"\midrule")
        for r in ds_rows:
            cells = [r["model"]]
            for sigma in sigmas:
                for fam in ["Clean", "FGSM", "PGD", "CW"]:
                    cells.append(r[(sigma, fam)])
            out.append(" & ".join(cells) + r" \\")
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}")
        out.append(r"}")
        out.append(r"\vspace{4pt}")
    (OUT / "rs_per_sigma.tex").write_text("\n".join(out))
    print(f"wrote {OUT / 'rs_per_sigma.tex'}")


def build_whitebox_table():
    rows = []
    for ds in DATASETS:
        df = load_full(ds, "whitebox")
        if df.empty:
            continue
        for m in ["LR", "SVM", "RF"]:
            sub = df[df["model"] == m]
            d = {"dataset": ds, "model": m}
            # FNR is reported alongside accuracy and ASR here for the same reason it
            # is reported for the transferred attacks: for an intrusion detector the
            # operationally meaningful quantity is the fraction of attack traffic that
            # goes undetected, not overall accuracy. It was already computed for the
            # direct attack but was not previously surfaced in the table.
            for col in ["accuracy", "asr", "fnr"]:
                d[col] = (fmt(sub[col].mean(), sub[col].std(ddof=0))
                          if (not sub.empty and col in sub) else "--")
            rows.append(d)
    out = [r"\begin{tabular}{l l c c c}", r"\toprule"]
    out.append(r"\textbf{Dataset} & \textbf{Model} & \textbf{Accuracy under HSJ} & "
               r"\textbf{ASR (HSJ)} & \textbf{FNR under HSJ} \\")
    out.append(r"\midrule")
    cur = None
    for r in rows:
        ds_label = DATASET_LABELS[r["dataset"]] if r["dataset"] != cur else ""
        cur = r["dataset"]
        out.append(f"{ds_label} & {r['model']} & {r['accuracy']} & {r['asr']} & "
                   f"{r['fnr']} \\\\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    (OUT / "whitebox.tex").write_text("\n".join(out))
    print(f"wrote {OUT / 'whitebox.tex'}")


def build_atr_table():
    """Attack Transfer Rate per (dataset, target, family), plus the count of
    configurations exceeding ATR=1.

    ATR previously appeared only in figures and prose, so a reviewer could not check
    any of the transferability statements against a number in the PDF. The convention
    here is the family mean used by the transferability figures: mean target ASR over
    a family divided by mean source (MLP) ASR over the same family. The final column
    counts, out of the 14 attack configurations, how many have a seed-averaged ATR
    above 1 -- the above-source-transfer regime.
    """
    fams = ["FGSM", "PGD", "CW"]
    fam_labels = {"FGSM": "FGSM", "PGD": "PGD", "CW": "C\\&W"}
    rows = []
    for ds in DATASETS:
        df = load_full(ds, "attacks")
        if df.empty:
            continue
        df = df.copy()
        df["fam"] = df["attack"].str.split("_").str[0]
        src = df[df["model"] == "MLP"]
        for tgt in ["LR", "SVM", "RF"]:
            t = df[df["model"] == tgt]
            vals = []
            for fam in fams:
                s_mean = src[src["fam"] == fam]["asr"].mean()
                t_mean = t[t["fam"] == fam]["asr"].mean()
                vals.append(t_mean / s_mean if s_mean and s_mean > 0 else float("nan"))
            merged = t[["attack", "seed", "asr"]].merge(
                src[["attack", "seed", "asr"]], on=["attack", "seed"],
                suffixes=("_t", "_s"))
            merged["atr"] = merged["asr_t"] / merged["asr_s"].clip(lower=1e-9)
            per_cfg = merged.groupby("attack")["atr"].mean()
            n_over = int((per_cfg > 1).sum())
            n_cfg = int(len(per_cfg))
            rows.append({"dataset": ds, "target": tgt, "vals": vals,
                         "mean": float(np.nanmean(vals)),
                         "over": f"{n_over}/{n_cfg}"})

    out = [r"\begin{tabular}{l l c c c c c}", r"\toprule"]
    out.append(r"\textbf{Dataset} & \textbf{Target} & " +
               " & ".join(rf"\textbf{{{fam_labels[f]}}}" for f in fams) +
               r" & \textbf{Mean} & \textbf{Cfgs.\ $>1$} \\")
    out.append(r"\midrule")
    cur = None
    for r in rows:
        ds_label = DATASET_LABELS[r["dataset"]] if r["dataset"] != cur else ""
        cur = r["dataset"]
        cells = " & ".join("--" if np.isnan(v) else f"{v:.3f}" for v in r["vals"])
        out.append(f"{ds_label} & {r['target']} & {cells} & "
                   f"{r['mean']:.3f} & {r['over']} \\\\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    (OUT / "atr.tex").write_text("\n".join(out))
    print(f"wrote {OUT / 'atr.tex'}")


def main():
    build_baseline_table()
    build_attacks_summary_table()
    build_fnr_under_attack_table()
    build_cost_table()
    build_rs_per_sigma_table()
    build_whitebox_table()
    build_atr_table()


if __name__ == "__main__":
    main()
