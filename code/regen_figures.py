"""Regenerate figures from the aggregated CSVs produced by run_experiments.py.

Outputs PDF + PNG files into paper/figures_out/ using the new fig{N}_*.pdf naming.
No image-baked titles (titles belong in LaTeX captions per MDPI style).
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results_v2"
FIGS = ROOT / "paper" / "figures_out"
FIGS.mkdir(parents=True, exist_ok=True)

DATASETS = ["CSECICIDS2018", "TONIOT", "WUSTLEHMS2020"]
DATASET_LABELS = {
    "CSECICIDS2018": "CSE-CIC-IDS2018",
    "TONIOT": "TON_IoT",
    "WUSTLEHMS2020": "WUSTL-EHMS-2020",
}
ATTACK_FAMILIES = ["FGSM", "PGD", "CW"]
ATTACK_LABELS = {"FGSM": "FGSM", "PGD": "PGD", "CW": "C&W"}
MODELS = ["LR", "SVM", "RF", "MLP"]
COLORS = {"LR": "#4c78a8", "SVM": "#f58518", "RF": "#54a24b", "MLP": "#e45756"}
ATTACK_COLORS = {"FGSM": "#f58518", "PGD": "#54a24b", "CW": "#4c78a8"}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "axes.grid": True,
    "axes.axisbelow": True,
    # Matplotlib defaults to Type 3 fonts in PDF output, which IEEE's
    # production checks reject; 42 emits embedded TrueType instead. Affects
    # only how glyphs are stored, not the rendered figure.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_attacks_full(ds: str) -> pd.DataFrame:
    p = RESULTS / ds / "aggregated" / "attacks_full.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    return df


def load_smoothing_full(ds: str) -> pd.DataFrame:
    p = RESULTS / ds / "aggregated" / "randomized_smoothing_full.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_advtrain_full(ds: str) -> pd.DataFrame:
    p = RESULTS / ds / "aggregated" / "adv_training_full.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_baseline(ds: str) -> pd.DataFrame:
    p = RESULTS / ds / "aggregated" / "baseline_clean_full.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def save_pdf(fig, name: str):
    out_pdf = FIGS / f"{name}.pdf"
    out_png = FIGS / f"{name}.png"
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")


# ----------------------------------------------------------------------------- #
# Fig 1: Average ASR per model per attack family (across all datasets)
# ----------------------------------------------------------------------------- #
def fig_avg_asr():
    rows = []
    for ds in DATASETS:
        df = load_attacks_full(ds)
        if df.empty:
            continue
        df = df.copy()
        df["fam"] = df["attack"].str.split("_").str[0]
        rows.append(df)
    if not rows:
        print("fig1: no data, skipped")
        return
    full = pd.concat(rows, ignore_index=True)
    grp = full.groupby(["model", "fam"])["asr"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 4.6))
    width = 0.25
    x = np.arange(len(MODELS))
    for i, fam in enumerate(ATTACK_FAMILIES):
        sub = grp[grp["fam"] == fam].set_index("model").reindex(MODELS).reset_index()
        bars = ax.bar(x + (i - 1) * width, sub["asr"], width,
                      label=ATTACK_LABELS[fam], color=ATTACK_COLORS[fam])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("Average ASR")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", ncol=3)
    save_pdf(fig, "fig1_avg_asr")


# ----------------------------------------------------------------------------- #
# Fig 2: Mean adversarial accuracy per dataset, per attack family, per model
# ----------------------------------------------------------------------------- #
def fig_mean_adv_acc():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True,
                              constrained_layout=True)
    for ax, ds in zip(axes, DATASETS):
        df = load_attacks_full(ds)
        if df.empty:
            ax.set_visible(False)
            continue
        df = df.copy()
        df["fam"] = df["attack"].str.split("_").str[0]
        grp = df.groupby(["model", "fam"])["accuracy"].mean().reset_index()
        x = np.arange(len(MODELS))
        width = 0.25
        for i, fam in enumerate(ATTACK_FAMILIES):
            sub = grp[grp["fam"] == fam].set_index("model").reindex(MODELS).reset_index()
            ax.bar(x + (i - 1) * width, sub["accuracy"], width,
                   label=ATTACK_LABELS[fam], color=ATTACK_COLORS[fam])
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS)
        ax.set_ylim(0, 1.0)
        ax.set_title(DATASET_LABELS[ds])
        if ax is axes[0]:
            ax.set_ylabel("Mean adversarial accuracy")
    axes[-1].legend(loc="upper right")
    save_pdf(fig, "fig2_mean_adv_acc")


# ----------------------------------------------------------------------------- #
# Figs 3-5: Defense effectiveness per dataset
# ----------------------------------------------------------------------------- #
def fig_defense(ds: str, out_name: str):
    df_adv = load_advtrain_full(ds)
    df_clean = load_baseline(ds)
    if df_adv.empty or df_clean.empty:
        print(f"{out_name}: no data, skipped")
        return

    # Take the highest ratio for plotting
    max_r = df_adv["ratio"].max()
    df = df_adv[df_adv["ratio"] == max_r].copy()
    df["eval_fam"] = df["eval_attack"].apply(
        lambda s: "Clean" if s == "Clean" else s.split("_")[0]
    )
    eval_groups = ["Clean", "FGSM", "PGD", "CW"]
    means = df.groupby(["model", "eval_fam"])["accuracy"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 4.6))
    width = 0.2
    x = np.arange(len(eval_groups))
    for i, m in enumerate(MODELS):
        sub = means[means["model"] == m].set_index("eval_fam").reindex(eval_groups).reset_index()
        ax.bar(x + (i - 1.5) * width, sub["accuracy"], width, label=m, color=COLORS[m])
    ax.set_xticks(x)
    ax.set_xticklabels(["Clean acc.", "Defended (FGSM)", "Defended (PGD)", "Defended (C&W)"])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=4, loc="upper right")
    save_pdf(fig, out_name)


# ----------------------------------------------------------------------------- #
# Figs 6-8: Transferability per dataset (ASR per target model, attack family)
# ----------------------------------------------------------------------------- #
def fig_transfer(ds: str, out_name: str):
    df = load_attacks_full(ds)
    if df.empty:
        print(f"{out_name}: no data, skipped")
        return
    df = df.copy()
    df["fam"] = df["attack"].str.split("_").str[0]

    # ATR = ASR_target / ASR_source (source = MLP)
    src = df[df["model"] == "MLP"].groupby("fam")["asr"].mean()
    rows = []
    for tgt in ["LR", "SVM", "RF"]:
        tgt_asr = df[df["model"] == tgt].groupby("fam")["asr"].mean()
        for fam in ATTACK_FAMILIES:
            denom = max(src.get(fam, 0.0), 1e-9)
            rows.append({"target": tgt, "fam": fam,
                         "atr": tgt_asr.get(fam, 0.0) / denom})
    atr_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    targets = ["LR", "SVM", "RF"]
    width = 0.25
    x = np.arange(len(targets))
    for i, fam in enumerate(ATTACK_FAMILIES):
        sub = atr_df[atr_df["fam"] == fam].set_index("target").reindex(targets).reset_index()
        bars = ax.bar(x + (i - 1) * width, sub["atr"], width,
                       label=ATTACK_LABELS[fam], color=ATTACK_COLORS[fam])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(1.0, ls="--", lw=0.8, color="grey")
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel("Attack Transfer Rate (ATR)")
    ax.legend(loc="upper left", ncol=3)
    save_pdf(fig, out_name)


# ----------------------------------------------------------------------------- #
# Fig 9: Mean transfer rate across datasets (per target model)
# ----------------------------------------------------------------------------- #
def fig_transfer_overall():
    rows = []
    for ds in DATASETS:
        df = load_attacks_full(ds)
        if df.empty:
            continue
        df = df.copy()
        df["fam"] = df["attack"].str.split("_").str[0]
        src = df[df["model"] == "MLP"].groupby("fam")["asr"].mean()
        for tgt in ["LR", "SVM", "RF"]:
            tgt_asr = df[df["model"] == tgt].groupby("fam")["asr"].mean()
            for fam in ATTACK_FAMILIES:
                denom = max(src.get(fam, 0.0), 1e-9)
                rows.append({"dataset": ds, "target": tgt, "fam": fam,
                             "atr": tgt_asr.get(fam, 0.0) / denom})
    if not rows:
        print("fig9: no data, skipped")
        return
    full = pd.DataFrame(rows)
    avg = full.groupby(["dataset", "target"])["atr"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 4.4))
    targets = ["LR", "SVM", "RF"]
    width = 0.27
    x = np.arange(len(targets))
    for i, ds in enumerate(DATASETS):
        sub = avg[avg["dataset"] == ds].set_index("target").reindex(targets).reset_index()
        bars = ax.bar(x + (i - 1) * width, sub["atr"], width,
                       label=DATASET_LABELS[ds],
                       color=["#4c78a8", "#f58518", "#54a24b"][i])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(1.0, ls="--", lw=0.8, color="grey")
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel("Mean Attack Transfer Rate")
    ax.legend(loc="upper left")
    save_pdf(fig, "fig9_transfer_overall")


# ----------------------------------------------------------------------------- #
# Consolidated panels: 1x3 grid of per-dataset defense / transfer charts
# These replace the three separate per-dataset figures in main.tex.
# ----------------------------------------------------------------------------- #
def fig_defense_panels(out_name: str = "fig3_defense_all"):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), sharey=True,
                              constrained_layout=True)
    eval_groups = ["Clean", "FGSM", "PGD", "CW"]
    width = 0.2
    x = np.arange(len(eval_groups))
    for ax, ds in zip(axes, DATASETS):
        df_adv = load_advtrain_full(ds)
        df_clean = load_baseline(ds)
        if df_adv.empty or df_clean.empty:
            ax.set_title(f"{DATASET_LABELS[ds]} (no data)")
            continue
        # Average over both adversarial-sample ratios, which is the convention the
        # summary table uses. Plotting only the largest ratio made the bars differ
        # from the AT/SAR column of that table by up to 0.027 for the same
        # quantity, with neither caption saying which ratio it meant.
        df = df_adv.copy()
        df["eval_fam"] = df["eval_attack"].apply(
            lambda s: "Clean" if s == "Clean" else s.split("_")[0]
        )
        means = df.groupby(["model", "eval_fam"])["accuracy"].mean().reset_index()
        for i, m in enumerate(MODELS):
            sub = means[means["model"] == m].set_index("eval_fam").reindex(eval_groups).reset_index()
            ax.bar(x + (i - 1.5) * width, sub["accuracy"], width, label=m, color=COLORS[m])
        ax.set_xticks(x)
        ax.set_xticklabels(["Clean", "FGSM", "PGD", "C&W"], rotation=0)
        ax.set_title(DATASET_LABELS[ds])
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Accuracy")
    axes[1].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12),
                    frameon=False)
    save_pdf(fig, out_name)


def fig_transfer_panels(out_name: str = "fig6_transfer_all"):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True,
                              constrained_layout=True)
    targets = ["LR", "SVM", "RF"]
    width = 0.25
    x = np.arange(len(targets))
    for ax, ds in zip(axes, DATASETS):
        df = load_attacks_full(ds)
        if df.empty:
            ax.set_title(f"{DATASET_LABELS[ds]} (no data)")
            continue
        df = df.copy()
        df["fam"] = df["attack"].str.split("_").str[0]
        src = df[df["model"] == "MLP"].groupby("fam")["asr"].mean()
        rows = []
        for tgt in targets:
            tgt_asr = df[df["model"] == tgt].groupby("fam")["asr"].mean()
            for fam in ATTACK_FAMILIES:
                denom = max(src.get(fam, 0.0), 1e-9)
                rows.append({"target": tgt, "fam": fam,
                             "atr": tgt_asr.get(fam, 0.0) / denom})
        atr_df = pd.DataFrame(rows)
        for i, fam in enumerate(ATTACK_FAMILIES):
            sub = atr_df[atr_df["fam"] == fam].set_index("target").reindex(targets).reset_index()
            bars = ax.bar(x + (i - 1) * width, sub["atr"], width,
                           label=ATTACK_LABELS[fam], color=ATTACK_COLORS[fam])
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                        f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8)
        ax.axhline(1.0, ls="--", lw=0.8, color="grey")
        ax.set_xticks(x)
        ax.set_xticklabels(targets)
        ax.set_title(DATASET_LABELS[ds])
    axes[0].set_ylabel("Attack Transfer Rate (ATR)")
    axes[1].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.10),
                    frameon=False)
    save_pdf(fig, out_name)


def main():
    fig_avg_asr()
    fig_mean_adv_acc()
    # Consolidated 1x3 panels (replace per-dataset figures fig3_*..fig8_*)
    fig_defense_panels("fig3_defense_all")
    fig_transfer_panels("fig6_transfer_all")
    # Aggregated cross-dataset transfer summary (formerly fig9)
    fig_transfer_overall()


if __name__ == "__main__":
    main()
