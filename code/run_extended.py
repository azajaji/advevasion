"""
Tier-B extension experiments for the unified evaluation framework.

Reuses the preprocessing, splitting, and model-training code in run_experiments.py
and adds four new experiments requested by reviewers:

  1. HSJ-N=500: HopSkipJump white-box at increased sample size with binomial CI.
  2. Constrained-feature FGSM/PGD: gradient updates restricted to attacker-controllable
     and semi-controllable features only; protocol-constrained features are frozen.
  3. Adaptive PGD against the adversarially-trained MLP: PGD re-optimised against the
     defended decision boundary rather than the undefended surrogate.
  4. Alternative feature selector (mutual information): re-runs the full pipeline with
     mutual_info_classif in place of RF feature importance and reports both the clean
     baselines and the full attack grid under MI-selected features.

Each experiment writes its own CSV under <out>/<DATASET>/extended/seed_<S>/. Pass
--out results_v2 to write alongside the corrected primary results; the older results/
tree is the pre-correction submitted snapshot and should not be mixed with it.

Reproducibility note: experiments 1 and 3 previously ran on ART's raw HopSkipJump and
an unseeded AdversarialTrainer respectively, both of which draw from uncontrolled RNG
state and do not reproduce run to run. They now use the same seeded implementations as
the primary pipeline (Code/attacks/reproducible_hop_skip_jump.py, and
base.at_cell_seed/reset_all_rng). Experiment 1's ASR also now uses the eligible
denominator, matching the main-body table, rather than dividing by all sampled points.

Usage:
    python Code/run_extended.py --experiment hsj500 --datasets WUSTLEHMS2020 --seeds 42 --out results_v2
    python Code/run_extended.py --experiment all --datasets CSECICIDS2018 TONIOT WUSTLEHMS2020 --seeds 42 7 123 31 99 --out results_v2
"""
from __future__ import annotations
import argparse, json, math, re, sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the validated primary pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiments as base  # type: ignore


# ----------------------------------------------------------------------------- #
# 1) HSJ at higher sample size with binomial CI
# ----------------------------------------------------------------------------- #

def hsj_extended(model, model_name, X_te, y_te, n_samples: int, seed: int,
                 dataset: str = "unknown"):
    """HopSkipJump on a stratified sub-sample of size n_samples; report Wilson CI.

    Runs through ReproducibleHopSkipJump via base.run_reproducible_hsj, the same
    driver the primary whitebox.csv uses. Previously this called ART's HopSkipJump
    directly, which draws from an unseeded RandomState in _init_sample and the bare
    global np.random in _compute_update, so identical inputs did not reproduce; it
    also has an unbounded step-size search (see
    Code/attacks/reproducible_hop_skip_jump.py). This N=500 check is a robustness
    check on the N=200 main-body result, so it has to be measured with the same
    instrument, or the two sample sizes are not comparable.

    ASR is over the ELIGIBLE denominator (samples the clean model already classifies
    correctly), matching run_reproducible_hsj and the main-body Table. The previous
    `flipped.mean()` divided by all n sampled points instead, which is a different
    quantity: it silently scales with how many already-misclassified points happen to
    be drawn, and made this appendix appear to disagree with the main-body table when
    the underlying attack results actually agreed. The Wilson interval is likewise
    computed on the eligible count, since that is the binomial n for this proportion.
    """
    from art.estimators.classification import SklearnClassifier

    rng = np.random.RandomState(seed)
    # stratified sub-sample
    idx_pool = []
    for c in np.unique(y_te):
        cls_idx = np.where(y_te == c)[0]
        per_class = min(n_samples // 2, len(cls_idx))
        idx_pool.extend(rng.choice(cls_idx, per_class, replace=False))
    idx = np.array(idx_pool[:n_samples])
    X_sub = X_te[idx].astype(np.float32)
    y_sub = y_te[idx]

    X_adv, diag = base.run_reproducible_hsj(
        model, model_name, X_sub, y_sub, experiment_seed=seed, dataset=dataset,
        max_iter=10, max_eval=200, init_eval=20, init_size=20, batch_size=32,
        estimator_factory=lambda: SklearnClassifier(model=model,
                                                    clip_values=(-10.0, 10.0)))

    asr = diag["asr"]
    n_elig = int(diag["n_eligible"])
    # Wilson 95% CI for ASR (binomial proportion over the eligible denominator)
    z = 1.959963984540054
    phat = asr if not math.isnan(asr) else 0.0
    n = max(n_elig, 1)
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2*n)) / denom
    half = z * math.sqrt(phat*(1-phat)/n + z**2/(4*n*n)) / denom
    ci_lo = max(0.0, centre - half)
    ci_hi = min(1.0, centre + half)
    pert = float(np.linalg.norm(X_adv - X_sub, axis=1).mean())
    return {
        "model": model_name,
        "attack": "HopSkipJump_whitebox",
        "n_samples": int(diag["n_samples"]),
        "n_eligible": n_elig,
        "asr": asr,
        "asr_ci_lo": ci_lo,
        "asr_ci_hi": ci_hi,
        "mean_l2_perturbation": pert,
        "median_queries_success": diag["median_queries_success"],
        "query_budget_exhausted": diag["query_budget_exhausted"],
        "numerical_stagnation": diag["numerical_stagnation"],
        "elapsed_s": diag["elapsed_s"],
    }


# ----------------------------------------------------------------------------- #
# 2) Constrained-feature FGSM / PGD
# ----------------------------------------------------------------------------- #

# Feature-name patterns that we treat as protocol-constrained (frozen during attack).
# Conservative: any feature whose name suggests a protocol flag, TCP state, window
# size, or initial handshake counter is frozen. Volume/timing features remain
# attacker-controllable.
#
# Matched on WORD TOKENS, not as raw substrings. The TCP flag mnemonics are short
# and collide badly inside ordinary flow-statistic names: "ACK" occurs inside
# "P(ack)et", so substring matching froze "Total Length of Fwd Packets", "Average
# Packet Size", "Bwd Packets/s" and "Packet Length Std" -- precisely the volume
# features the manipulability table classifies as attacker-controllable. That
# inverts the experiment: it would report a constrained attack that is weaker than
# it should be, because the attacker was denied fields it can in fact manipulate.
PROTOCOL_CONSTRAINED_TOKENS = (
    "flag", "flags", "fin", "syn", "ack", "psh", "urg", "rst", "ece", "cwr",
    "conn_state", "state",
)

# Multi-word / underscored names matched as substrings, which is safe because they
# are long enough not to collide with unrelated feature names.
PROTOCOL_CONSTRAINED_SUBSTRINGS = (
    "init_win", "init fwd win", "init bwd win", "init_fwd_win", "init_bwd_win",
    "ssl_version", "ssl_resumed", "dns_aa", "dns_rd", "dns_ra", "dns_rejected",
    "weird", "http_status_code", "proto", "service",
)

# Port fields are identifiers drawn from a discrete space, not continuous
# quantities: a gradient step of a few hundred "port numbers" denotes a different
# service rather than a smaller perturbation of the same one. They are frozen for
# the same reason as the protocol-state counters. Matched exactly.
PORT_FEATURES = (
    "Dst Port", "Src Port", "src_port", "dst_port", "Sport", "Dport", "Port",
)

# Patient biometrics carried by WUSTL-EHMS-2020. These are bounded by physiological
# plausibility and by their covariance with one another, and are reachable only from
# a compromised gateway or monitoring link, so they are frozen alongside the
# protocol-constrained fields. Matched exactly (case-insensitive) rather than as
# substrings: "ST" would otherwise match many flow-statistic names.
PHYSIOLOGICALLY_CONSTRAINED_FEATURES = (
    "Heart_rate", "Pulse_Rate", "SpO2", "Resp_Rate", "SYS", "DIA", "Temp", "ST",
)


def _tokens(name: str) -> set[str]:
    """Split a feature name into lowercase word tokens.

    Splitting on non-alphanumeric characters means "ACK Flag Count" yields
    {"ack","flag","count"} while "Total Length of Fwd Packets" yields
    {"total","length","of","fwd","packets"} -- "packets" does not equal "ack",
    so the flag mnemonics no longer match inside ordinary volume-feature names.
    """
    return set(re.split(r"[^A-Za-z0-9]+", name.lower())) - {""}


def is_protocol_constrained(name: str) -> bool:
    toks = _tokens(name)
    if toks & set(PROTOCOL_CONSTRAINED_TOKENS):
        return True
    low = name.lower()
    return any(s in low for s in PROTOCOL_CONSTRAINED_SUBSTRINGS)


def feature_mask(feature_names):
    """Return 1.0 where feature is attacker-controllable, 0.0 where frozen."""
    mask = np.ones(len(feature_names), dtype=np.float32)
    exact_frozen = {p.lower() for p in
                    PHYSIOLOGICALLY_CONSTRAINED_FEATURES + PORT_FEATURES}
    for i, fn in enumerate(feature_names):
        if is_protocol_constrained(fn):
            mask[i] = 0.0
        elif fn.strip().lower() in exact_frozen:
            mask[i] = 0.0
    return mask


def frozen_breakdown(feature_names):
    """Which frozen features fall in which constraint category, for reporting."""
    physio = {p.lower() for p in PHYSIOLOGICALLY_CONSTRAINED_FEATURES}
    ports = {p.lower() for p in PORT_FEATURES}
    out = {"protocol": [], "port": [], "physiological": []}
    for fn in feature_names:
        low = fn.strip().lower()
        if is_protocol_constrained(fn):
            out["protocol"].append(fn)
        elif low in ports:
            out["port"].append(fn)
        elif low in physio:
            out["physiological"].append(fn)
    return out


def constrained_attack(mlp_classifier, X_te, y_te, feature_names, eps_values=(0.30,)):
    """FGSM/PGD with gradient mask: protocol-constrained features are not perturbed."""
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
    mask = feature_mask(feature_names)
    n_frozen = int((mask == 0).sum())

    rows = []
    for eps in eps_values:
        # FGSM constrained
        atk = FastGradientMethod(estimator=mlp_classifier, eps=eps, batch_size=256)
        X_adv = atk.generate(X_te.astype(np.float32))
        # Project: keep frozen features at their original value
        X_adv = X_te + (X_adv - X_te) * mask
        rows.append(_eval_constrained(mlp_classifier, X_te, X_adv, y_te,
                                      "FGSM_constrained", eps, n_frozen, len(mask)))

        atk = ProjectedGradientDescent(estimator=mlp_classifier, eps=eps,
                                       max_iter=10, batch_size=256)
        X_adv = atk.generate(X_te.astype(np.float32))
        X_adv = X_te + (X_adv - X_te) * mask
        rows.append(_eval_constrained(mlp_classifier, X_te, X_adv, y_te,
                                      "PGD_constrained", eps, n_frozen, len(mask)))
    return rows


def _eval_constrained(mlp_classifier, X_te, X_adv, y_te, atk_name, eps, n_frozen, n_total):
    y_clean = np.argmax(mlp_classifier.predict(X_te.astype(np.float32)), axis=1)
    y_pred = np.argmax(mlp_classifier.predict(X_adv.astype(np.float32)), axis=1)
    asr = float(np.mean((y_clean == y_te) & (y_pred != y_te)))
    acc = float(np.mean(y_pred == y_te))
    return {
        "attack": atk_name,
        "epsilon": eps,
        "n_features_total": n_total,
        "n_features_frozen": n_frozen,
        "n_features_perturbable": n_total - n_frozen,
        "adv_accuracy": acc,
        "asr": asr,
    }


# ----------------------------------------------------------------------------- #
# 3) Adaptive PGD against AT-trained MLP
# ----------------------------------------------------------------------------- #

def adaptive_pgd_against_at(X_tr, y_tr, X_te, y_te, seed, dataset="unknown",
                            eps_values=(0.30,)):
    """
    Adversarial-train an MLP with PGD, then craft new PGD adversarials directly
    against the defended MLP. Compare with non-adaptive (transfer-from-undefended) PGD.

    Both models are built under explicit per-cell seeding (base.at_cell_seed /
    base.reset_all_rng), the same pattern adversarial_training_run uses. ART's
    AdversarialTrainer.fit() shuffles and selects its adversarial subset from the bare
    global numpy RNG with no random_state of its own, so without an explicit reset
    immediately before construction the defended model is not reproducible run to run
    (measured at roughly 1.6 percentage points of accuracy drift on the primary
    adversarial-training grid). The two models are given distinct derived seeds so the
    undefended control and the defended model are independent but each reproducible.
    """
    from art.attacks.evasion import ProjectedGradientDescent
    from art.defences.trainer import AdversarialTrainer

    # Undefended MLP (used to generate non-adaptive PGD as control)
    undef_seed = base.at_cell_seed(seed, dataset, "PGD", 0.0)
    base.reset_all_rng(undef_seed)
    mlp_undef = base.build_mlp(X_tr.shape[1], undef_seed)
    mlp_undef.fit(X_tr, y_tr,
                  batch_size=base.MLP_CONFIG["batch_size"],
                  nb_epochs=base.MLP_CONFIG["nb_epochs"])

    # AT-trained MLP: fresh model, fresh attack, fresh trainer, RNG reset first
    at_seed = base.at_cell_seed(seed, dataset, "PGD", 0.5)
    base.reset_all_rng(at_seed)
    mlp_fresh = base.build_mlp(X_tr.shape[1], at_seed)
    mlp_fresh.fit(X_tr, y_tr,
                  batch_size=base.MLP_CONFIG["batch_size"],
                  nb_epochs=base.MLP_CONFIG["nb_epochs"])
    inner_atk = ProjectedGradientDescent(estimator=mlp_fresh, eps=0.3,
                                          max_iter=10, batch_size=256)
    trainer = AdversarialTrainer(mlp_fresh, attacks=inner_atk, ratio=0.5)
    trainer.fit(X_tr, y_tr, nb_epochs=base.DEFENSE_CONFIGS["adversarial_training"]["nb_epochs"])
    mlp_def = trainer.get_classifier()

    rows = []
    for eps in eps_values:
        # Non-adaptive: PGD on undefended, evaluated on defended
        atk_nonadapt = ProjectedGradientDescent(estimator=mlp_undef, eps=eps,
                                                 max_iter=10, batch_size=256)
        X_adv_nonadapt = atk_nonadapt.generate(X_te.astype(np.float32))
        y_pred = np.argmax(mlp_def.predict(X_adv_nonadapt.astype(np.float32)), axis=1)
        rows.append({
            "attack": "PGD_nonadaptive",
            "epsilon": eps,
            "adv_accuracy": float(np.mean(y_pred == y_te)),
            "asr": float(np.mean((np.argmax(mlp_def.predict(X_te.astype(np.float32)), axis=1) == y_te) &
                                 (y_pred != y_te))),
        })

        # Adaptive: PGD generated directly against the defended MLP
        atk_adapt = ProjectedGradientDescent(estimator=mlp_def, eps=eps,
                                              max_iter=10, batch_size=256)
        X_adv_adapt = atk_adapt.generate(X_te.astype(np.float32))
        y_pred = np.argmax(mlp_def.predict(X_adv_adapt.astype(np.float32)), axis=1)
        rows.append({
            "attack": "PGD_adaptive",
            "epsilon": eps,
            "adv_accuracy": float(np.mean(y_pred == y_te)),
            "asr": float(np.mean((np.argmax(mlp_def.predict(X_te.astype(np.float32)), axis=1) == y_te) &
                                 (y_pred != y_te))),
        })
    return rows


# ----------------------------------------------------------------------------- #
# 4) Alternative feature selector (mutual information)
# ----------------------------------------------------------------------------- #

# The mutual-information selector now lives in run_experiments.select_features()
# and is fitted inside the training fold; use prepare_split(selector="mi").
# The former stand-alone mutual_info_pipeline() fitted the scaler and the ranker
# on the full dataset and has been removed.


# ----------------------------------------------------------------------------- #
# Orchestration
# ----------------------------------------------------------------------------- #

def run_for(dataset, seed, experiment, out_dir, hsj_n=500, full=False):
    print(f"\n===== EXTENDED [{experiment}] {dataset} seed={seed} full={full} =====", flush=True)
    base.set_seed(seed)

    selector = "mi" if experiment == "mi_selector" else "rf"
    X_tr, X_te, y_tr, y_te, feats, n_feats, extras = base.prepare_split(
        dataset, seed, full=full, selector=selector)

    out = out_dir / dataset / "extended" / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)

    if experiment in ("hsj500", "all"):
        models, _ = base.train_baselines(X_tr, y_tr, seed)
        rows = []
        for mname in ["RF", "SVM", "LR"]:
            try:
                rows.append(hsj_extended(models[mname], mname, X_te, y_te, hsj_n, seed,
                                          dataset=dataset))
            except Exception as e:
                print(f"  HSJ {mname} failed: {e}", flush=True)
        pd.DataFrame(rows).to_csv(out / f"hsj_n{hsj_n}.csv", index=False)
        print(f"  wrote {out}/hsj_n{hsj_n}.csv", flush=True)

    if experiment in ("constrained", "all"):
        models, _ = base.train_baselines(X_tr, y_tr, seed)
        rows = constrained_attack(models["MLP"], X_te, y_te, feats,
                                   eps_values=(0.15, 0.30, 0.50, 0.75))
        # Evaluate each constrained adv set on all classifiers (re-run separately would re-generate;
        # we keep this experiment focused on MLP + record per-(eps, attack) ASR)
        pd.DataFrame(rows).to_csv(out / "constrained_attacks.csv", index=False)
        print(f"  wrote {out}/constrained_attacks.csv", flush=True)

    if experiment in ("adaptive", "all"):
        rows = adaptive_pgd_against_at(X_tr, y_tr, X_te, y_te, seed, dataset=dataset,
                                        eps_values=(0.15, 0.30, 0.50))
        pd.DataFrame(rows).to_csv(out / "adaptive_pgd.csv", index=False)
        print(f"  wrote {out}/adaptive_pgd.csv", flush=True)

    if experiment == "mi_selector":
        models, _ = base.train_baselines(X_tr, y_tr, seed)
        clean_rows = base.evaluate_baselines_on_clean(models, X_te, y_te)
        pd.DataFrame(clean_rows).to_csv(out / "mi_clean.csv", index=False)
        with open(out / "mi_features.json", "w") as f:
            json.dump({"selected_features": feats, "n": n_feats}, f, indent=2)

        # Adversarial arm. The reviewer concern this ablation answers is whether the
        # ADVERSARIAL-robustness ordering (not just the clean-accuracy ordering) is an
        # artifact of RF-importance feature selection, so the attack grid has to be run
        # under MI-selected features too. Previously only the clean baselines above were
        # recorded, which cannot support a robustness claim. Same attack grid and same
        # evaluation functions as the primary pipeline, so the two are directly comparable.
        attacks = base.generate_attacks(models["MLP"], X_te)
        atk_rows = base.evaluate_attacks_on_models(models, attacks, X_te, y_te)
        pd.DataFrame(atk_rows).to_csv(out / "mi_attacks.csv", index=False)
        print(f"  wrote {out}/mi_clean.csv, mi_attacks.csv and mi_features.json",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(base.DATASET_FILES.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    ap.add_argument("--experiment", choices=["hsj500", "constrained", "adaptive",
                                              "mi_selector", "all"], default="all")
    ap.add_argument("--hsj-n", type=int, default=500)
    ap.add_argument("--out", default=str(base.RESULTS_DIR))
    ap.add_argument("--full", action="store_true",
                    help="Use full datasets without per-class subsampling for "
                         "WUSTLEHMS2020 and TONIOT (CSE-CIC-IDS2018 unaffected).")
    args = ap.parse_args()

    out_dir = Path(args.out)
    for ds in args.datasets:
        for s in args.seeds:
            try:
                run_for(ds, s, args.experiment, out_dir, hsj_n=args.hsj_n, full=args.full)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"!! {ds} seed={s} extended failed: {e}", flush=True)
    print("\nExtended runs complete.", flush=True)


if __name__ == "__main__":
    main()
