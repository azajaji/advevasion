"""Experiments added in response to the IEEE Access review of Access-2026-25244.

Every experiment here runs on the leakage-controlled path in run_experiments.py
(`prepare_split`), which fits the scaler and the feature selector inside the
training fold only.

Experiments and the reviewer comment each answers:

  linear_wb    R2.20  direct white-box FGSM/PGD on LR and linear SVM, using each
                      model's own gradients instead of transfer from the MLP.
  tuning       R2.18  matched hyperparameter search across LR/SVM/RF under one
                      budget, so no model is a strawman. Also answers R3.5.
  surrogate    R2.19  surrogate-quality sweep: MLP capacity and epochs versus the
                      transferability of the attacks it generates.
  selection    R2.16  RF-importance vs mutual information vs no selection.
  resampling   R2.8   SMOTE vs categorical-aware SMOTE-NC vs no oversampling.
               R2.9
  temporal     R2.12  chronological split (train earlier, test later) versus the
                      random stratified split.
  family       R2.13  per-attack-family breakdown of adversarial vulnerability,
                      instead of a single binary collapse.
  gbm          R2.17  XGBoost and LightGBM as additional target classifiers.
  hsj_budget   R2.21  HopSkipJump query-budget ladder, to show whether the
                      reported ASR has saturated.

Each writes results/<DATASET>/reviewer/seed_<S>/<experiment>.csv.

Usage:
    python Code/run_reviewer_experiments.py --experiment linear_wb --datasets WUSTLEHMS2020 --seeds 42
    python Code/run_reviewer_experiments.py --experiment all --seeds 42 7 123 31 99 --out results_v2
"""
from __future__ import annotations
import argparse, os, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiments as base  # type: ignore

EPS_VALUES = (0.15, 0.30, 0.50, 0.75)

#: Fixed thread count for the boosted ensembles. Pinned rather than n_jobs=-1 so
#: that gradient reduction order, and therefore the trained model, does not depend
#: on the core count of the machine or on what else is running. Chosen so several
#: cells can run concurrently without oversubscribing a 20-core host.
GBM_THREADS = int(os.environ.get("ADVEVASION_GBM_THREADS", "6"))


# --------------------------------------------------------------------------- #
# R2.20 — direct white-box attacks on the linear models
# --------------------------------------------------------------------------- #

def experiment_linear_whitebox(X_tr, X_te, y_tr, y_te, seed, **_):
    """FGSM/PGD computed on LR and linear SVM directly, not transferred.

    The submitted manuscript attacked LR and SVM only by transfer from the MLP
    surrogate and by decision-based HopSkipJump. Both models are differentiable
    in their own right, so their own loss gradients give a strictly stronger
    attack and a fairer robustness estimate.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
    from art.estimators.classification.scikitlearn import (
        ScikitlearnLogisticRegression, ScikitlearnSVC)

    rows = []
    fitted = {
        "LR": (LogisticRegression(random_state=seed, **base.ML_MODELS_CONFIG["LR"]).fit(X_tr, y_tr),
               ScikitlearnLogisticRegression),
        "SVM": (LinearSVC(random_state=seed, **base.ML_MODELS_CONFIG["SVM"]).fit(X_tr, y_tr),
                ScikitlearnSVC),
    }

    for mname, (model, wrapper) in fitted.items():
        est = wrapper(model=model, clip_values=(-10.0, 10.0))
        y_clean = model.predict(X_te)
        for eps in EPS_VALUES:
            for atk_name, atk in (
                ("FGSM_direct", FastGradientMethod(estimator=est, eps=eps, batch_size=256)),
                ("PGD_direct", ProjectedGradientDescent(estimator=est, eps=eps,
                                                        max_iter=10, batch_size=256)),
            ):
                t0 = time.time()
                try:
                    X_adv = atk.generate(X_te.astype(np.float32))
                except Exception as e:  # estimator/attack incompatibility
                    print(f"    {mname}/{atk_name} eps={eps} failed: {e}", flush=True)
                    continue
                y_pred = model.predict(X_adv)
                row = base.metric_block(y_te, y_pred)
                row.update({
                    "model": mname, "attack": atk_name, "epsilon": eps,
                    "asr": float(np.mean((y_clean == y_te) & (y_pred != y_te))),
                    "mean_l2": float(np.linalg.norm(X_adv - X_te, axis=1).mean()),
                    "elapsed_s": time.time() - t0,
                })
                rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# R2.18 / R3.5 — matched hyperparameter tuning
# --------------------------------------------------------------------------- #

def experiment_tuning(X_tr, X_te, y_tr, y_te, seed, **_):
    """Randomised search under one identical budget for every classical model.

    Answers the objection that RF at 50 trees with defaults is a weak baseline
    while other models were left untuned: all three now get the same number of
    candidate fits and the same CV protocol.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

    n_iter, cv = 12, StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    grids = {
        "LR": (LogisticRegression(random_state=seed, max_iter=1000),
               {"C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "class_weight": [None, "balanced"]}),
        "SVM": (LinearSVC(random_state=seed, max_iter=2000),
                {"C": [0.01, 0.1, 1.0, 10.0],
                 "class_weight": [None, "balanced"]}),
        "RF": (RandomForestClassifier(random_state=seed, n_jobs=-1),
               {"n_estimators": [50, 100, 200, 400],
                "max_depth": [None, 10, 20, 40],
                "min_samples_leaf": [1, 2, 5],
                "max_features": ["sqrt", "log2", None]}),
    }

    rows = []
    for mname, (est, grid) in grids.items():
        t0 = time.time()
        search = RandomizedSearchCV(est, grid, n_iter=n_iter, cv=cv, scoring="f1_macro",
                                    random_state=seed, n_jobs=-1, refit=True)
        search.fit(X_tr, y_tr)
        fit_s = time.time() - t0
        tuned_row = base.metric_block(y_te, search.best_estimator_.predict(X_te))
        tuned_row.update({"model": mname, "config": "tuned", "search_budget": n_iter,
                          "best_params": str(search.best_params_),
                          "cv_f1_macro": float(search.best_score_), "fit_s": fit_s})
        rows.append(tuned_row)

        # the submitted configuration, evaluated identically for comparison
        t0 = time.time()
        if mname == "RF":
            default = RandomForestClassifier(random_state=seed, **base.ML_MODELS_CONFIG["RF"])
        elif mname == "SVM":
            default = LinearSVC(random_state=seed, **base.ML_MODELS_CONFIG["SVM"])
        else:
            default = LogisticRegression(random_state=seed, **base.ML_MODELS_CONFIG["LR"])
        default.fit(X_tr, y_tr)
        d_row = base.metric_block(y_te, default.predict(X_te))
        d_row.update({"model": mname, "config": "as_submitted", "search_budget": 0,
                      "best_params": str(base.ML_MODELS_CONFIG.get(mname, {})),
                      "cv_f1_macro": np.nan, "fit_s": time.time() - t0})
        rows.append(d_row)
    return rows


# --------------------------------------------------------------------------- #
# R2.19 — surrogate quality versus transferability
# --------------------------------------------------------------------------- #

def experiment_surrogate(X_tr, X_te, y_tr, y_te, seed, **_):
    """Vary MLP capacity and training length, then measure downstream transfer.

    If a bigger or better-trained surrogate transfers no better, the reported
    transfer numbers are not an artefact of an undertrained surrogate.
    """
    from art.attacks.evasion import ProjectedGradientDescent

    configs = [
        ("submitted", [32, 12], 12),
        ("longer", [32, 12], 50),
        ("wider", [128, 64], 12),
        ("wider_longer", [128, 64], 50),
        ("deep_long", [256, 128, 64], 50),
    ]
    targets, _ = base.train_baselines(X_tr, y_tr, seed)
    orig_cfg = dict(base.MLP_CONFIG)

    rows = []
    for cfg_name, hidden, epochs in configs:
        base.MLP_CONFIG["hidden_layers"] = hidden
        base.MLP_CONFIG["nb_epochs"] = epochs
        try:
            surrogate = base.build_mlp(X_tr.shape[1], seed)
            t0 = time.time()
            surrogate.fit(X_tr, y_tr, batch_size=base.MLP_CONFIG["batch_size"],
                          nb_epochs=epochs)
            train_s = time.time() - t0
            surr_pred = np.argmax(surrogate.predict(X_te.astype(np.float32)), axis=1)
            surr_acc = float(np.mean(surr_pred == y_te))

            for eps in (0.30, 0.50):
                atk = ProjectedGradientDescent(estimator=surrogate, eps=eps,
                                               max_iter=10, batch_size=256)
                X_adv = atk.generate(X_te.astype(np.float32))
                for tname, tmodel in targets.items():
                    if tname == "MLP":
                        continue  # transfer to the classical targets is the question
                    y_clean = base.predict(tmodel, X_te, tname)
                    y_pred = base.predict(tmodel, X_adv, tname)
                    row = base.metric_block(y_te, y_pred)
                    row.update({
                        "surrogate": cfg_name, "hidden": str(hidden), "epochs": epochs,
                        "surrogate_clean_acc": surr_acc, "surrogate_train_s": train_s,
                        "target": tname, "epsilon": eps,
                        "asr": float(np.mean((y_clean == y_te) & (y_pred != y_te))),
                    })
                    rows.append(row)
        finally:
            base.MLP_CONFIG.update(orig_cfg)
    return rows


# --------------------------------------------------------------------------- #
# R2.16 — feature-selection sensitivity
# --------------------------------------------------------------------------- #

def load_primary_arm(dataset, seed, out_dir, tag: dict):
    """Reuse the primary run as the baseline arm of an ablation.

    The primary pipeline is exactly `prepare_split` with selector="rf",
    oversample="smote" and a random split, so recomputing that arm would repeat
    work already on disk. Verified to reproduce the recomputed arm exactly.
    Returns None when the primary results are unavailable, so callers fall back
    to computing the arm.
    """
    base_dir = Path(out_dir) / dataset / f"seed_{seed}"
    clean_p, atk_p = base_dir / "baseline_clean.csv", base_dir / "attacks.csv"
    timings_p = base_dir / "timings.json"
    if not (clean_p.exists() and atk_p.exists() and timings_p.exists()):
        return None

    import json
    n_feats = json.load(open(timings_p))["n_features"]
    rows = []
    for r in pd.read_csv(clean_p).to_dict("records"):
        r.update({**tag, "n_features": n_feats, "attack": "clean", "asr": 0.0})
        rows.append(r)
    for r in pd.read_csv(atk_p).to_dict("records"):
        r.update({**tag, "n_features": n_feats})
        rows.append(r)
    return rows


def experiment_selection(dataset, seed, full, out_dir=None, **_):
    """RF-importance vs mutual information vs keeping every feature.

    Re-prepares the split under each selector, so the comparison includes the
    downstream effect on both clean accuracy and transferred-attack robustness.
    The RF arm is the primary configuration and is read back rather than re-run.
    """
    rows = []
    for selector in ("rf", "mi", "none"):
        if selector == "rf" and out_dir is not None:
            reused = load_primary_arm(dataset, seed, out_dir, {"selector": "rf"})
            if reused is not None:
                print("    reusing primary run for selector=rf", flush=True)
                rows.extend(reused)
                continue

        X_tr, X_te, y_tr, y_te, feats, n_feats, _ = base.prepare_split(
            dataset, seed, full=full, selector=selector)
        models, _ = base.train_baselines(X_tr, y_tr, seed)
        attacks = base.generate_attacks(models["MLP"], X_te)

        for r in base.evaluate_baselines_on_clean(models, X_te, y_te):
            r.update({"selector": selector, "n_features": n_feats,
                      "attack": "clean", "asr": 0.0})
            rows.append(r)
        for r in base.evaluate_attacks_on_models(models, attacks, X_te, y_te):
            r.update({"selector": selector, "n_features": n_feats})
            rows.append(r)
    return rows


# --------------------------------------------------------------------------- #
# R2.8 / R2.9 — oversampling scheme
# --------------------------------------------------------------------------- #

def experiment_resampling(dataset, seed, full, out_dir=None, **_):
    """SMOTE vs SMOTE-NC vs none.

    Plain SMOTE interpolates label-encoded categorical codes, which can produce
    values that correspond to no real category. SMOTE-NC resamples those columns
    by majority vote instead. The no-oversampling arm shows what the imbalance
    correction is worth in the first place.

    Oversampling is applied only when the minority-to-majority ratio falls below
    0.75. On a dataset that never crosses that threshold the three arms are the
    same computation, so the comparison is skipped and recorded as inapplicable
    rather than reported as three coincidentally identical columns.
    """
    # Whether oversampling triggers is a property of the split, already recorded by
    # the primary run. Read it back rather than rebuilding the split to find out.
    triggered = None
    if out_dir is not None:
        tp = Path(out_dir) / dataset / f"seed_{seed}" / "timings.json"
        if tp.exists():
            import json
            triggered = json.load(open(tp)).get("resampled")
    if triggered is None:
        triggered = base.prepare_split(dataset, seed, full=full,
                                       oversample="smote")[6]["resampled"]

    if not triggered:
        print(f"    {dataset}: oversampling never triggers; arms coincide by "
              f"construction, recording as not applicable", flush=True)
        return [{"dataset": dataset, "oversample": scheme,
                 "applicable": False,
                 "note": "minority-to-majority ratio above the 0.75 threshold; "
                         "no oversampling is applied under any scheme"}
                for scheme in ("smote", "smotenc", "none")]

    rows = []
    for scheme in ("smote", "smotenc", "none"):
        if scheme == "smote" and out_dir is not None:
            reused = load_primary_arm(dataset, seed, out_dir,
                                      {"oversample": "smote", "applicable": True})
            if reused is not None:
                print("    reusing primary run for oversample=smote", flush=True)
                rows.extend(reused)
                continue

        X_tr, X_te, y_tr, y_te, feats, n_feats, extras = base.prepare_split(
            dataset, seed, full=full, oversample=scheme)
        models, _ = base.train_baselines(X_tr, y_tr, seed)
        attacks = base.generate_attacks(models["MLP"], X_te)

        common = {"oversample": scheme, "applicable": True,
                  "n_train": int(len(y_tr)),
                  "n_categorical_selected": len(extras["categorical_selected"]),
                  "categorical_selected": ";".join(extras["categorical_selected"]),
                  "resampled": extras["resampled"]}
        for r in base.evaluate_baselines_on_clean(models, X_te, y_te):
            r.update({**common, "attack": "clean", "asr": 0.0})
            rows.append(r)
        for r in base.evaluate_attacks_on_models(models, attacks, X_te, y_te):
            r.update(common)
            rows.append(r)
    return rows


# --------------------------------------------------------------------------- #
# R2.12 — chronological split
# --------------------------------------------------------------------------- #

def experiment_temporal(dataset, seed, full, **_):
    """Train on earlier traffic, test on later traffic.

    Requires a dataset carrying capture timestamps; build the CSE-CIC variant with
        python Code/build_cic_sample.py --keep-timestamp --out Datasets/CSE_CIC_IDS2018_temporal.csv
    A random stratified split can place flows from the same session or time window
    on both sides, which flatters both clean and adversarial performance.
    """
    rows = []
    for split in ("random", "temporal"):
        X_tr, X_te, y_tr, y_te, feats, n_feats, _ = base.prepare_split(
            dataset, seed, full=full, split=split)
        models, _ = base.train_baselines(X_tr, y_tr, seed)
        attacks = base.generate_attacks(models["MLP"], X_te)

        common = {"split": split, "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
                  "test_attack_rate": float(np.mean(y_te))}
        for r in base.evaluate_baselines_on_clean(models, X_te, y_te):
            r.update({**common, "attack": "clean", "asr": 0.0})
            rows.append(r)
        for r in base.evaluate_attacks_on_models(models, attacks, X_te, y_te):
            r.update(common)
            rows.append(r)
    return rows


# --------------------------------------------------------------------------- #
# R2.13 — per-attack-family breakdown
# --------------------------------------------------------------------------- #

def experiment_family(X_tr, X_te, y_tr, y_te, seed, extras=None, **_):
    """Report attack success per original attack category, not just binary.

    Collapsing every non-benign flow into one class hides that some families are
    far easier to push across the boundary than others.
    """
    families = extras.get("family_test") if extras else None
    if not families:
        print("    no family metadata for this dataset; skipping", flush=True)
        return []
    families = np.asarray(families)

    models, _ = base.train_baselines(X_tr, y_tr, seed)
    attacks = base.generate_attacks(models["MLP"], X_te)

    rows = []
    for atk_name, (X_adv, params) in attacks.items():
        for mname, model in models.items():
            y_clean = base.predict(model, X_te, mname)
            y_pred = base.predict(model, X_adv, mname)
            for fam in np.unique(families):
                m = families == fam
                if m.sum() == 0:
                    continue
                # attack success is only meaningful on originally-detected attacks
                detected = m & (y_clean == y_te) & (y_te == 1)
                evaded = detected & (y_pred != y_te)
                rows.append({
                    "model": mname, "attack": atk_name,
                    "n": int(m.sum()), "n_detected_clean": int(detected.sum()),
                    "asr": float(evaded.sum() / detected.sum()) if detected.sum() else np.nan,
                    "fnr_under_attack": float(np.mean(y_pred[m & (y_te == 1)] != 1))
                                        if (m & (y_te == 1)).sum() else np.nan,
                    **params,
                    # Written AFTER **params deliberately. params carries its own
                    # "family" key (the attack family: FGSM/PGD/CW), which silently
                    # overwrote the traffic category when this row was built the other
                    # way round, leaving the per-category rows distinguishable only by
                    # their row count. The traffic category is the whole point of this
                    # experiment, so it is written last and under a distinct name.
                    "traffic_family": str(fam),
                })
    return rows


# --------------------------------------------------------------------------- #
# R2.17 — gradient-boosting targets
# --------------------------------------------------------------------------- #

def experiment_gbm(X_tr, X_te, y_tr, y_te, seed, dataset=None, **_):
    """Add XGBoost and LightGBM as targets under the same attack protocol.

    These are the tree ensembles most commonly deployed for tabular NIDS, and
    like RF they are non-differentiable, so they are evaluated under both
    transferred gradient attacks and direct decision-based HopSkipJump.
    """
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    # Thread count is pinned rather than left at n_jobs=-1. Boosted ensembles are
    # not thread-count invariant: the number of threads fixes the order in which
    # partial gradient sums are reduced, so n_jobs=-1 makes results depend on how
    # many cores the host happens to expose. A fixed value makes the numbers
    # reproducible on any machine and independent of what else is running. It is a
    # resource setting, not a model hyperparameter: depth, estimators, learning
    # rate, subsampling and the objective are unchanged.
    extra = {
        "XGB": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                             random_state=seed, n_jobs=GBM_THREADS,
                             eval_metric="logloss"),
        "LGBM": LGBMClassifier(n_estimators=200, max_depth=-1, learning_rate=0.1,
                               random_state=seed, n_jobs=GBM_THREADS, verbose=-1),
    }

    rows = []
    timings = {}
    for name, model in extra.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        timings[name] = time.time() - t0

    # surrogate for the transferred attacks is the same compact MLP as the paper
    mlp = base.build_mlp(X_tr.shape[1], seed)
    mlp.fit(X_tr, y_tr, batch_size=base.MLP_CONFIG["batch_size"],
            nb_epochs=base.MLP_CONFIG["nb_epochs"])
    attacks = base.generate_attacks(mlp, X_te)

    for name, model in extra.items():
        y_clean = model.predict(X_te)
        r = base.metric_block(y_te, y_clean)
        r.update({"model": name, "attack": "clean", "asr": 0.0,
                  "train_s": timings[name]})
        rows.append(r)
        for atk_name, (X_adv, params) in attacks.items():
            y_pred = model.predict(X_adv)
            r = base.metric_block(y_te, y_pred)
            r.update({"model": name, "attack": atk_name,
                      "asr": float(np.mean((y_clean == y_te) & (y_pred != y_te))),
                      "train_s": timings[name], **params})
            rows.append(r)

        # Direct decision-based attack, as for the other non-differentiable models.
        try:
            rows.append(hsj_blackbox(model, name, X_te, y_te, seed, dataset or "unknown"))
        except Exception as e:
            print(f"    HSJ on {name} failed: {e}", flush=True)
    return rows


def hsj_blackbox(model, name, X_te, y_te, seed, dataset, n_samples: int = 100):
    """HopSkipJump against any model exposing .predict (XGBoost, LightGBM).

    Runs through ReproducibleHopSkipJump via ART's BlackBoxClassifier, since
    neither library's sklearn wrapper satisfies ART's native SklearnClassifier
    interface. This is also where the runaway step-size search was originally
    found: 926,000+ single-row predict calls in 258 seconds with zero
    convergence, tracked down to ART's own unbounded `while not success` loop
    (see Code/attacks/reproducible_hop_skip_jump.py). Runs on a smaller
    sub-sample than the sklearn models, since black-box query overhead is
    higher per call for boosted ensembles.
    """
    from art.estimators.classification import BlackBoxClassifier

    # single-threaded prediction: HSJ issues many small batches, and thread
    # spin-up per call dominates the actual computation
    for attr, val in (("n_jobs", 1), ("nthread", 1)):
        try:
            model.set_params(**{attr: val})
        except (ValueError, AttributeError, TypeError):
            pass

    def predict_fn(x):
        preds = np.asarray(model.predict(x)).astype(int).ravel()
        onehot = np.zeros((len(preds), 2), dtype=np.float32)
        onehot[np.arange(len(preds)), preds] = 1.0
        return onehot

    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(y_te):
        cls = np.where(y_te == c)[0]
        idx.extend(rng.choice(cls, min(n_samples // 2, len(cls)), replace=False))
    idx = np.array(idx)
    X_sub, y_sub = X_te[idx].astype(np.float32), y_te[idx]

    X_adv, diag = base.run_reproducible_hsj(
        model, name, X_sub, y_sub, experiment_seed=seed, dataset=dataset,
        max_iter=10, max_eval=200, init_eval=20, init_size=20,
        estimator_factory=lambda: BlackBoxClassifier(
            predict_fn, input_shape=(X_te.shape[1],), nb_classes=2,
            clip_values=(-10.0, 10.0)))

    y_pred = model.predict(X_adv)
    row = base.metric_block(y_sub, y_pred)
    row.update({
        "model": name, "attack": "HopSkipJump_direct",
        "asr": diag["asr"], "n_samples": diag["n_samples"],
        "n_eligible": diag["n_eligible"],
        "mean_l2": float(np.linalg.norm(X_adv - X_sub, axis=1).mean()),
        "median_queries_success": diag["median_queries_success"],
        "query_budget_exhausted": diag["query_budget_exhausted"],
        "numerical_stagnation": diag["numerical_stagnation"],
        "elapsed_s": diag["elapsed_s"],
    })
    return row


# --------------------------------------------------------------------------- #
# R2.21 — HopSkipJump query budget
# --------------------------------------------------------------------------- #

def experiment_hsj_budget(X_tr, X_te, y_tr, y_te, seed, dataset=None, **_):
    """Ladder of HopSkipJump budgets, to show whether the reported ASR saturates.

    The submitted results used a single low budget (max_iter=10, max_eval=200).
    If ASR still climbs at the top of the ladder, the reported figures are a
    lower bound on what a patient attacker achieves, and should be stated as such.

    Runs through ReproducibleHopSkipJump (see run_experiments.run_reproducible_hsj
    and Code/attacks/reproducible_hop_skip_jump.py). ART's own HopSkipJump was
    the direct cause of a 14.6-hour hang on this exact experiment: its step-size
    search has no bound, and against RF's piecewise-constant decision surface it
    can genuinely never satisfy its own success condition. It is also not
    reproducible run-to-run (verified: identical inputs gave three different
    ASR values across three runs), because it draws from an unseeded internal
    RandomState. Both are fixed at the attack level, not by monkey-patching.
    """
    from art.estimators.classification import SklearnClassifier

    ladder = [(5, 100), (10, 200), (20, 500), (40, 1000)]
    n_sub = 200  # kept small: query cost dominates and grows with the ladder
    models, _ = base.train_baselines(X_tr, y_tr, seed)

    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(y_te):
        cls_idx = np.where(y_te == c)[0]
        idx.extend(rng.choice(cls_idx, min(n_sub // 2, len(cls_idx)), replace=False))
    idx = np.array(idx)
    X_sub, y_sub = X_te[idx].astype(np.float32), y_te[idx]

    rows = []
    for mname in ("LR", "SVM", "RF"):
        model = models[mname]
        for max_iter, max_eval in ladder:
            X_adv, diag = base.run_reproducible_hsj(
                model, mname, X_sub, y_sub, experiment_seed=seed,
                dataset=dataset or "unknown", max_iter=max_iter, max_eval=max_eval,
                init_eval=20, init_size=20, batch_size=32,
                estimator_factory=lambda: SklearnClassifier(model=model, clip_values=(-10.0, 10.0)))
            y_pred = model.predict(X_adv)
            rows.append({
                "model": mname, "max_iter": max_iter, "max_eval": max_eval,
                "approx_queries": max_iter * max_eval,
                "n_samples": diag["n_samples"], "n_eligible": diag["n_eligible"],
                "asr": diag["asr"],
                "adv_accuracy": float(np.mean(y_pred == y_sub)),
                "mean_l2": float(np.linalg.norm(X_adv - X_sub, axis=1).mean()),
                "median_queries_success": diag["median_queries_success"],
                "query_budget_exhausted": diag["query_budget_exhausted"],
                "numerical_stagnation": diag["numerical_stagnation"],
                "elapsed_s": diag["elapsed_s"],
            })
            print(f"    {mname} iter={max_iter} eval={max_eval} "
                  f"asr={rows[-1]['asr']:.3f} nonconv="
                  f"{diag['query_budget_exhausted']+diag['numerical_stagnation']} "
                  f"({diag['elapsed_s']:.0f}s)", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

# experiments that re-prepare the split themselves
SPLIT_OWNING = {"selection": experiment_selection,
                "resampling": experiment_resampling,
                "temporal": experiment_temporal}

# experiments that take the standard prepared split
def experiment_tuning_adv(X_tr, X_te, y_tr, y_te, seed, dataset=None, **_):
    """R2.18 follow-up: put the TUNED configurations through the attack grid.

    The `tuning` cell compares tuned against submitted configurations on clean
    data only, which answers whether the submitted baseline was weak but says
    nothing about whether tuning changes robustness. This runs both
    configurations through the same attacks the primary models face, so the two
    are compared on the axis the paper is actually about.

    The search here is deliberately identical to `experiment_tuning` -- same
    grids, same n_iter, same CV, same seed -- so the tuned models are the same
    ones that cell selected, and the clean numbers must reproduce it.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

    # Worker count is pinned rather than left at -1. With n_jobs=-1 on both the
    # search and the forest, joblib spawns one worker per logical core and each
    # takes its own copy of the training partition; on CSE-CIC that exceeded
    # available memory and the cell ran roughly fifteen times slower than the
    # equivalent search in the `tuning` cell. Neither RandomForest nor the search
    # depends on worker count for its result -- the trees are seeded from
    # random_state and the candidates are independent -- so this is a resource
    # setting, exactly as GBM_THREADS is for the boosted ensembles.
    SEARCH_JOBS = int(os.environ.get("ADVEVASION_SEARCH_JOBS", "4"))

    n_iter, cv = 12, StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    grids = {
        "LR": (LogisticRegression(random_state=seed, max_iter=1000),
               {"C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "class_weight": [None, "balanced"]}),
        "SVM": (LinearSVC(random_state=seed, max_iter=2000),
                {"C": [0.01, 0.1, 1.0, 10.0],
                 "class_weight": [None, "balanced"]}),
        "RF": (RandomForestClassifier(random_state=seed, n_jobs=1),
               {"n_estimators": [50, 100, 200, 400],
                "max_depth": [None, 10, 20, 40],
                "min_samples_leaf": [1, 2, 5],
                "max_features": ["sqrt", "log2", None]}),
    }
    submitted = {
        "LR": lambda: LogisticRegression(random_state=seed, **base.ML_MODELS_CONFIG["LR"]),
        "SVM": lambda: LinearSVC(random_state=seed, **base.ML_MODELS_CONFIG["SVM"]),
        "RF": lambda: RandomForestClassifier(random_state=seed, **base.ML_MODELS_CONFIG["RF"]),
    }

    fitted = {}
    for mname, (est, grid) in grids.items():
        t0 = time.time()
        search = RandomizedSearchCV(est, grid, n_iter=n_iter, cv=cv, scoring="f1_macro",
                                    random_state=seed, n_jobs=SEARCH_JOBS, refit=True)
        search.fit(X_tr, y_tr)
        print(f"    {mname} search took {time.time() - t0:.0f}s", flush=True)
        fitted[(mname, "tuned")] = (search.best_estimator_, str(search.best_params_))
        d = submitted[mname]()
        d.fit(X_tr, y_tr)
        fitted[(mname, "as_submitted")] = (d, str(base.ML_MODELS_CONFIG.get(mname, {})))
        print(f"    fitted {mname}: tuned={search.best_params_}", flush=True)

    # one surrogate, shared by both configurations, exactly as in the main grid
    mlp = base.build_mlp(X_tr.shape[1], seed)
    mlp.fit(X_tr, y_tr, batch_size=base.MLP_CONFIG["batch_size"],
            nb_epochs=base.MLP_CONFIG["nb_epochs"])
    attacks = base.generate_attacks(mlp, X_te)

    rows = []
    for (mname, config), (model, params) in fitted.items():
        y_clean = model.predict(X_te)
        r = base.metric_block(y_te, y_clean)
        r.update({"model": mname, "config": config, "attack": "clean",
                  "asr": 0.0, "best_params": params})
        rows.append(r)
        for atk_name, (X_adv, ap) in attacks.items():
            y_pred = model.predict(X_adv)
            r = base.metric_block(y_te, y_pred)
            r.update({"model": mname, "config": config, "attack": atk_name,
                      "asr": float(np.mean((y_clean == y_te) & (y_pred != y_te))),
                      "best_params": params, **ap})
            rows.append(r)
        try:
            h = hsj_blackbox(model, mname, X_te, y_te, seed, dataset or "unknown")
            h.update({"config": config, "best_params": params})
            rows.append(h)
        except Exception as e:
            print(f"    HSJ on {mname}/{config} failed: {e}", flush=True)
    return rows


SPLIT_TAKING = {"linear_wb": experiment_linear_whitebox,
                "tuning": experiment_tuning,
                "tuning_adv": experiment_tuning_adv,
                "surrogate": experiment_surrogate,
                "family": experiment_family,
                "gbm": experiment_gbm,
                "hsj_budget": experiment_hsj_budget}

ALL_EXPERIMENTS = list(SPLIT_TAKING) + list(SPLIT_OWNING)


def run_for(dataset, seed, experiment, out_dir, full=False, resume=True):
    out = Path(out_dir) / dataset / "reviewer" / f"seed_{seed}"
    target = out / f"{experiment}.csv"
    if resume and target.exists() and target.stat().st_size > 0:
        # A run interrupted mid-write can leave a truncated file that still has
        # non-zero size, so confirm the CSV actually parses before skipping it.
        try:
            if len(pd.read_csv(target)) > 0:
                print(f"----- SKIP [{experiment}] {dataset} seed={seed} (already done)",
                      flush=True)
                return
            raise ValueError("no rows")
        except Exception as e:
            print(f"----- REDO [{experiment}] {dataset} seed={seed} "
                  f"(existing output unreadable: {e})", flush=True)

    print(f"\n===== REVIEWER [{experiment}] {dataset} seed={seed} =====", flush=True)
    base.set_seed(seed)
    out.mkdir(parents=True, exist_ok=True)

    if experiment in SPLIT_OWNING:
        rows = SPLIT_OWNING[experiment](dataset=dataset, seed=seed, full=full,
                                        out_dir=out_dir)
    else:
        X_tr, X_te, y_tr, y_te, feats, n_feats, extras = base.prepare_split(
            dataset, seed, full=full)
        rows = SPLIT_TAKING[experiment](X_tr, X_te, y_tr, y_te, seed,
                                        extras=extras, feats=feats, dataset=dataset)
    if not rows:
        print(f"  {experiment}: no rows produced", flush=True)
        return
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{experiment}.csv", index=False)
    print(f"  wrote {out / (experiment + '.csv')}  ({len(df)} rows)", flush=True)


def aggregate(dataset, seeds, experiment, out_dir):
    """Mean/std across seeds, grouped by every non-numeric key of the experiment."""
    frames = []
    for s in seeds:
        p = Path(out_dir) / dataset / "reviewer" / f"seed_{s}" / f"{experiment}.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["seed"] = s
            frames.append(df)
    if not frames:
        return
    full = pd.concat(frames, ignore_index=True)
    agg_dir = Path(out_dir) / dataset / "reviewer" / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)
    full.to_csv(agg_dir / f"{experiment}_full.csv", index=False)

    keys = [c for c in full.columns
            if full[c].dtype == object or c in ("epsilon", "max_iter", "max_eval",
                                                "epochs", "n_features", "search_budget")]
    keys = [k for k in keys if k != "seed"]
    num = [c for c in full.columns
           if c not in keys + ["seed"] and pd.api.types.is_numeric_dtype(full[c])]
    if keys and num:
        summary = full.groupby(keys, dropna=False)[num].agg(["mean", "std"])
        summary.to_csv(agg_dir / f"{experiment}_summary.csv")
        print(f"  aggregated -> {agg_dir / (experiment + '_summary.csv')}", flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", nargs="+", default=["all"],
                    help=f"one or more of: {', '.join(ALL_EXPERIMENTS)}, or 'all'")
    ap.add_argument("--datasets", nargs="+", default=list(base.DATASET_FILES.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123, 31, 99])
    ap.add_argument("--out", default="results_v2")
    ap.add_argument("--full", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true",
                    help="recompute cells whose output CSV already exists")
    args = ap.parse_args()

    experiments = ALL_EXPERIMENTS if "all" in args.experiment else args.experiment
    resume = not args.no_resume
    todo = [(e, d, s) for e in experiments for d in args.datasets for s in args.seeds]
    print(f"plan: {len(todo)} cells ({len(experiments)} experiments x "
          f"{len(args.datasets)} datasets x {len(args.seeds)} seeds), "
          f"resume={'on' if resume else 'off'}", flush=True)

    failed = []
    for n, (exp, ds, s) in enumerate(todo, 1):
        if exp not in ALL_EXPERIMENTS:
            print(f"!! unknown experiment: {exp}", flush=True)
            continue
        print(f"[{n}/{len(todo)}]", end=" ", flush=True)
        try:
            run_for(ds, s, exp, args.out, full=args.full, resume=resume)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"!! {exp} {ds} seed={s} failed: {e}", flush=True)
            failed.append((exp, ds, s))

    for exp in experiments:
        for ds in args.datasets:
            try:
                aggregate(ds, args.seeds, exp, args.out)
            except Exception as e:
                print(f"!! aggregate {exp} {ds} failed: {e}", flush=True)

    if failed:
        print(f"\nCOMPLETED WITH {len(failed)} FAILED CELL(S):", flush=True)
        for exp, ds, s in failed:
            print(f"   {exp} {ds} seed={s}", flush=True)
        print("Re-run the same command to retry only these.", flush=True)
    else:
        print("\nAll reviewer experiments complete.", flush=True)


if __name__ == "__main__":
    main()
