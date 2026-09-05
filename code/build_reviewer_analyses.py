"""Analyses derived from the corrected run artifacts; no model training required.

  eps_units   R2.5   convert standardized attack budgets into native feature units
  base_rates  R2.10  operational metrics re-weighted to deployment prevalence
              R2.11
  hsj_comp    R2.22  realised composition of the HopSkipJump sub-sample

Writes LaTeX tables into tables_out/ and CSVs into results_v2/analysis/.
"""
from __future__ import annotations
from pathlib import Path
import json
import argparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results_v2"
TABLES = ROOT / "tables_out"
OUT = RESULTS / "analysis"
SEEDS = [42, 7, 123, 31, 99]
DATASETS = ["CSECICIDS2018", "TONIOT", "WUSTLEHMS2020"]

# Display names for the dataset keys. The tables built here previously printed
# the raw keys while those from build_reviewer_tables.py printed these, so the
# same benchmark appeared as both "CSECICIDS2018" and "CSE-CIC-IDS2018" in the
# same paper. Kept identical to LABEL in build_reviewer_tables.py.
LABEL = {"CSECICIDS2018": "CSE-CIC-IDS2018", "TONIOT": "TON\\_IoT",
         "WUSTLEHMS2020": "WUSTL-EHMS-2020"}


def ds_label(key: str) -> str:
    return LABEL.get(key, key)
EPS = [0.15, 0.30, 0.50, 0.75]

# Native units of the features surfaced in the epsilon table, for readability.
UNIT_HINTS = {
    "Byts": "bytes", "Bytes": "bytes", "Len": "bytes", "Pkts": "packets",
    "Pkt": "packets", "IAT": "microseconds", "Duration": "microseconds",
    "Dur": "seconds", "Rate": "per second", "Load": "bits/s",
    "Heart_rate": "beats/min", "Pulse_Rate": "beats/min", "Resp_Rate": "breaths/min",
    "SpO2": "percent", "SYS": "mmHg", "DIA": "mmHg", "Temp": "degrees C",
    "Jitter": "seconds", "Port": "port number", "duration": "seconds",
    "src_bytes": "bytes", "dst_bytes": "bytes",
}


def unit_for(feature: str) -> str:
    for k, v in UNIT_HINTS.items():
        if k.lower() in feature.lower():
            return v
    return "native units"


def esc(s: str) -> str:
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


# --------------------------------------------------------------------------- #
# R2.5 — epsilon in native units
# --------------------------------------------------------------------------- #

def eps_units(top_k: int = 6):
    rows = []
    for ds in DATASETS:
        scales, feats, cats = [], None, set()
        for s in SEEDS:
            p = RESULTS / ds / f"seed_{s}" / "timings.json"
            if not p.exists():
                continue
            j = json.load(open(p))
            cats |= set(j.get("categorical_selected", []))
            if feats is None:
                feats = j["selected_features"]
                scales.append(np.asarray(j["scaler_scale"], dtype=float))
            elif j["selected_features"] == feats:
                scales.append(np.asarray(j["scaler_scale"], dtype=float))
        if feats is None:
            continue
        scale = np.mean(scales, axis=0)
        # Label-encoded columns are excluded. Their standard deviation is a
        # property of the arbitrary integer codes LabelEncoder assigned, not of
        # any physical quantity, so converting a budget into "native units" for
        # them is meaningless. WUSTL's Sport is the case in point: it arrives as
        # text (it contains service names as well as numeric ports), so its
        # scale of ~4.7e3 is the spread of 16,314 lexicographically ordered
        # codes and not a count of port numbers.
        excluded = [i for i, f in enumerate(feats) if f in cats]
        order = [i for i in np.argsort(scale)[::-1] if i not in excluded][:top_k]
        if excluded:
            print(f"  {ds}: excluded label-encoded from native units: "
                  f"{[feats[i] for i in excluded]}")
        for i in order:
            row = {"dataset": ds, "feature": feats[i], "unit": unit_for(feats[i]),
                   "std_dev_native": scale[i]}
            for e in EPS:
                row[f"eps_{e}"] = e * scale[i]
            rows.append(row)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "eps_native_units.csv", index=False)

    lines = [r"\begin{tabular}{l l r r r r r}", r"\toprule",
             r"\textbf{Dataset} & \textbf{Feature} & \textbf{1 s.d.} & "
             r"\multicolumn{4}{c}{\textbf{Perturbation at $\epsilon$ (native units)}} \\",
             r"\cmidrule(lr){4-7}",
             r" & & & $0.15$ & $0.30$ & $0.50$ & $0.75$ \\", r"\midrule"]
    last = None
    for _, r in df.iterrows():
        ds_cell = ds_label(r["dataset"]) if r["dataset"] != last else ""
        last = r["dataset"]
        lines.append(
            f"{ds_cell} & \\texttt{{{esc(r['feature'])}}} ({r['unit']}) & "
            f"{r['std_dev_native']:.3g} & " +
            " & ".join(f"{r[f'eps_{e}']:.3g}" for e in EPS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "eps_native_units.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {TABLES / 'eps_native_units.tex'} ({len(df)} rows)")
    return df


# --------------------------------------------------------------------------- #
# R2.10 / R2.11 — deployment prevalence
# --------------------------------------------------------------------------- #

def base_rates(deployment_prevalence: dict[str, float]):
    """Re-weight measured TPR/FPR to the prevalence a deployment would see.

    Accuracy and precision measured on a class-rebalanced sample do not transfer
    to a link whose real attack rate is far lower. TPR and FPR are prevalence
    invariant, so they are measured once and re-weighted here.
    """
    rows = []
    for ds in DATASETS:
        pi_eval_path = RESULTS / ds / "aggregated" / "baseline_clean_full.csv"
        atk_path = RESULTS / ds / "aggregated" / "attacks_full.csv"
        if not pi_eval_path.exists():
            continue
        clean = pd.read_csv(pi_eval_path)
        pi = deployment_prevalence.get(ds)
        if pi is None:
            continue
        for cond, df in (("clean", clean),
                         ("under attack", pd.read_csv(atk_path) if atk_path.exists() else None)):
            if df is None:
                continue
            g = df.groupby("model")[["fpr", "fnr"]].mean()
            for m, r in g.iterrows():
                tpr, fpr = 1.0 - r["fnr"], r["fpr"]
                denom = pi * tpr + (1 - pi) * fpr
                ppv = (pi * tpr / denom) if denom > 0 else np.nan
                rows.append({
                    "dataset": ds, "model": m, "condition": cond,
                    "deployment_prevalence": pi, "tpr": tpr, "fpr": fpr,
                    "precision_at_deployment": ppv,
                    "alerts_per_10k_flows": 10_000 * denom,
                    "false_alerts_per_10k_flows": 10_000 * (1 - pi) * fpr,
                    "missed_attacks_per_10k_flows": 10_000 * pi * (1 - tpr),
                })
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "deployment_base_rates.csv", index=False)

    lines = [r"\begin{tabular}{l l l r r r r}", r"\toprule",
             r"\textbf{Dataset} & \textbf{Model} & \textbf{Condition} & \textbf{TPR} & "
             r"\textbf{FPR} & \textbf{Precision} & \textbf{False alerts} \\",
             r" & & & & & \textbf{at deploy.\ rate} & \textbf{per 10k flows} \\",
             r"\midrule"]
    last_ds = last_m = None
    for _, r in df.iterrows():
        ds_cell = ds_label(r["dataset"]) if r["dataset"] != last_ds else ""
        m_cell = esc(r["model"]) if (r["model"] != last_m or ds_cell) else ""
        last_ds, last_m = r["dataset"], r["model"]
        lines.append(
            f"{ds_cell} & {m_cell} & {r['condition']} & {r['tpr']:.3f} & {r['fpr']:.3f} & "
            f"{r['precision_at_deployment']:.3f} & {r['false_alerts_per_10k_flows']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "deployment_base_rates.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {TABLES / 'deployment_base_rates.tex'} ({len(df)} rows)")
    return df


# --------------------------------------------------------------------------- #
# R2.22 — HopSkipJump sub-sample composition
# --------------------------------------------------------------------------- #

def hsj_composition():
    rows = []
    for ds in DATASETS:
        for s in SEEDS:
            fam_p = RESULTS / ds / f"seed_{s}" / "test_families.csv"
            if not fam_p.exists():
                continue
            fam = pd.read_csv(fam_p)
            y = fam["y"].values
            # replicate the stratified sub-sampler used for the HSJ baseline
            rng = np.random.RandomState(s)
            idx = []
            for c in np.unique(y):
                cls = np.where(y == c)[0]
                idx.extend(rng.choice(cls, min(500 // 2, len(cls)), replace=False))
            idx = np.array(idx)
            sub = fam.iloc[idx]
            counts = sub["family"].value_counts()
            full_counts = fam["family"].value_counts(normalize=True)
            # Emit a row for every category in the full test partition, not only
            # those the sub-sample happened to draw. Otherwise a category present
            # in 1 of 5 seeds is later averaged over that one seed alone and
            # reports a mean of 1.0 instead of 0.2, and the column stops summing
            # to the sub-sample size. Absent categories contribute an explicit 0.
            for f in full_counts.index:
                c = int(counts.get(f, 0))
                rows.append({"dataset": ds, "seed": s, "family": f,
                             "n_in_subsample": c,
                             "share_in_subsample": c / len(sub),
                             "share_in_full_test": float(full_counts[f])})
    if not rows:
        print("no family metadata found; skipping HSJ composition")
        return None
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "hsj_subsample_composition.csv", index=False)

    g = df.groupby(["dataset", "family"]).agg(
        n=("n_in_subsample", "mean"),
        share_sub=("share_in_subsample", "mean"),
        share_full=("share_in_full_test", "mean")).reset_index()

    # The mean counts must still add up to the sub-sample size, and the shares
    # to one. If they do not, some (seed, category) pair is missing from the
    # grid above and the means are being taken over the wrong denominator.
    for ds_name, part in g.groupby("dataset"):
        n_sum, s_sum = part["n"].sum(), part["share_sub"].sum()
        if abs(s_sum - 1.0) > 1e-6:
            raise AssertionError(
                f"{ds_name}: composition shares sum to {s_sum:.4f}, not 1.0 "
                f"(mean n sums to {n_sum:.1f}); per-category means are being "
                f"averaged over inconsistent seed counts")
        print(f"  {ds_name}: mean n sums to {n_sum:.1f}, shares to {s_sum:.4f}")
    # Categories that contribute a handful of instances per seed cost a full row
    # each while carrying almost no weight in the pooled ASR. They are collapsed
    # into one grouped row per dataset, reported with their combined mean count
    # and combined shares so the aggregate representation is still visible and
    # the column totals still reconcile. Grouping needs at least two categories
    # to save anything, and the benign class is never grouped: it is half of
    # every sub-sample by construction and is the reference the rest is read
    # against.
    RARE_N = 8.0
    BENIGN = {"Benign", "normal", "BENIGN"}

    lines = [r"\begin{tabular}{l l r r r}", r"\toprule",
             r"\textbf{Dataset} & \textbf{Attack family} & \textbf{Mean $n$} & "
             r"\textbf{Share of sub-sample} & \textbf{Share of test set} \\", r"\midrule"]
    last = None
    for ds in g["dataset"].unique():
        sub = g[g["dataset"] == ds]
        rare_mask = (sub["n"] <= RARE_N) & (~sub["family"].isin(BENIGN))
        rare, keep = sub[rare_mask], sub[~rare_mask]
        if len(rare) < 2:
            rare, keep = sub.iloc[0:0], sub
        for _, r in keep.iterrows():
            ds_cell = ds_label(ds) if ds != last else ""
            last = ds
            lines.append(f"{ds_cell} & {esc(r['family'])} & {r['n']:.1f} & "
                         f"{r['share_sub']:.3f} & {r['share_full']:.3f} \\\\")
        if len(rare):
            ds_cell = ds_label(ds) if ds != last else ""
            last = ds
            label = (f"Other rare categories ({len(rare)} categories; "
                     f"each mean $n \\leq {RARE_N:.0f}$)")
            lines.append(f"{ds_cell} & {label} & {rare['n'].sum():.1f} & "
                         f"{rare['share_sub'].sum():.3f} & "
                         f"{rare['share_full'].sum():.3f} \\\\")
            print(f"  {ds}: grouped {len(rare)} rare categories "
                  f"(combined mean n={rare['n'].sum():.1f}, "
                  f"sub-sample share={rare['share_sub'].sum():.3f}, "
                  f"test share={rare['share_full'].sum():.3f})")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "hsj_subsample_composition.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {TABLES / 'hsj_subsample_composition.tex'} ({len(g)} rows)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cic-prevalence", type=float, required=True,
                    help="true attack rate of the full CSE-CIC-IDS2018 release")
    ap.add_argument("--ton-prevalence", type=float, default=None)
    ap.add_argument("--wustl-prevalence", type=float, default=None)
    args = ap.parse_args()

    TABLES.mkdir(parents=True, exist_ok=True)
    eps_units()
    prev = {"CSECICIDS2018": args.cic_prevalence}
    if args.ton_prevalence:
        prev["TONIOT"] = args.ton_prevalence
    if args.wustl_prevalence:
        prev["WUSTLEHMS2020"] = args.wustl_prevalence
    base_rates(prev)
    hsj_composition()


if __name__ == "__main__":
    main()
