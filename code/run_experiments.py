"""
Run adversarial-robustness experiments end-to-end on three NIDS datasets.

Differences from the original Colab notebook:
  - Local paths instead of Google Drive
  - Multi-seed loop (default 5 seeds) — every metric ends up as mean +/- std
  - Direct white-box baseline (gradient-free HopSkipJump on RF/SVM via ART)
  - Wall-clock timing (training + inference + smoothing)
  - Per-sigma columns retained in randomized-smoothing output
  - Aggregated CSVs written to ./results/<DATASET>/seed_<S>/ and ./results/<DATASET>/aggregated/
"""

from __future__ import annotations
import os, sys, time, warnings, argparse, json, random
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Datasets"
RESULTS_DIR = ROOT / "results"

DATASET_FILES = {
    "CSECICIDS2018": DATA_DIR / "CSE_CIC_IDS2018.csv",
    "TONIOT": DATA_DIR / "ton_iot_dataset.csv",
    "WUSTLEHMS2020": DATA_DIR / "wustl-ehms-2020_with_attacks_categories.csv",
}

# Variants used only by the reviewer-requested ablations; kept out of DATASET_FILES
# so they are never picked up as default targets of the primary pipeline.
VARIANT_DATASET_FILES = {
    # same reservoir sample, capture Timestamp retained for the temporal split
    "CSECICIDS2018_TEMPORAL": DATA_DIR / "CSE_CIC_IDS2018_temporal.csv",
}


def dataset_path(name: str) -> Path:
    if name in DATASET_FILES:
        return DATASET_FILES[name]
    if name in VARIANT_DATASET_FILES:
        return VARIANT_DATASET_FILES[name]
    raise KeyError(f"unknown dataset: {name}")

ATTACK_CONFIGS = {
    "FGSM": {"epsilon_values": [0.15, 0.30, 0.50, 0.75]},
    "PGD":  {"epsilon_values": [0.15, 0.30, 0.50, 0.75], "max_iter_values": [10, 25]},
    "CW":   {"confidence_values": [0.0, 0.5], "max_iter_values": [10]},
}

DEFENSE_CONFIGS = {
    "adversarial_training": {"ratio_values": [0.3, 0.5], "nb_epochs": 12,
                             "attack_types": ["FGSM", "PGD"]},
    "randomized_smoothing": {"sigma_values": [0.1, 0.25, 0.5],
                             "n0": 50, "n": 200, "alpha": 0.001,
                             "test_subset_size": 200},
}

MLP_CONFIG = {"hidden_layers": [32, 12], "nb_classes": 2, "lr": 1e-3,
              "batch_size": 64, "nb_epochs": 12}

ML_MODELS_CONFIG = {
    "RF":  {"n_estimators": 50},
    "SVM": {"max_iter": 500},
    "LR":  {"max_iter": 500},
}

TEST_SIZE = 0.2

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metric_block(y_true, y_pred):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix)
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return dict(accuracy=acc, precision=pre, recall=rec, f1=f1, fpr=fpr, fnr=fnr)


def load_raw(name: str, seed: int, full: bool = False):
    """Dataset-specific cleaning and categorical encoding only.

    Performs no scaling, no feature selection, and no resampling: every
    data-dependent transform is deferred to `prepare_split` so that it can be
    fitted on the training partition alone.

    Args:
        full: when True, do not apply per-class subsampling for WUSTLEHMS2020 and
              TONIOT (use the entire dataset, preserving the natural class
              imbalance). CSE-CIC-IDS2018 always uses the pre-built stratified
              sample on disk regardless of this flag.

    Returns:
        X: DataFrame with categorical columns label-encoded, unscaled.
        y: binary label array.
        meta: DataFrame of non-predictive metadata retained for analysis
              (`family` = original multi-class attack label where available).
        cat_cols: names of columns that were categorical before encoding.
    """
    from sklearn.preprocessing import LabelEncoder

    path = dataset_path(name)
    df = pd.read_csv(path)
    meta = pd.DataFrame(index=df.index)

    if name.startswith("CSECICIDS2018"):
        # keep the original multi-class attack name before binarising
        meta["family"] = df["Label"].astype(str).values
        # the temporal variant of the sample retains the capture timestamp; it is
        # metadata for ordering the split, never a predictor
        if "Timestamp" in df.columns:
            meta["timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce").values
            df = df.drop(columns=["Timestamp"])
        df["Label"] = df["Label"].apply(lambda x: 0 if x == "Benign" else 1)
        label_col = "Label"
    elif name == "WUSTLEHMS2020":
        if "Attack Category" in df.columns:
            meta["family"] = df["Attack Category"].astype(str).values
        drop_cols = ["Dir", "Flgs", "SrcAddr", "DstAddr", "Dport", "SrcMac",
                     "DstMac", "Packet_num", "Attack Category"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        if not full:
            rng = np.random.RandomState(seed)
            per_class = 2000
            idx = []
            for c in df["Label"].unique():
                sub = df.index[df["Label"] == c]
                n = min(per_class, len(sub))
                idx.extend(rng.choice(sub, n, replace=False))
            df = df.loc[idx].copy()
            meta = meta.loc[idx].copy()
        # Sport is dropped for the same reason src_ip/dst_ip are dropped from
        # TON_IoT: it is identifier-like, not a predictor. It holds 16,314
        # distinct values over 16,318 flows, 16,310 of them occurring once, so
        # label-encoding it produces a near-unique code per row and roughly a
        # fifth of those codes exist only in the test partition.
        if "Sport" in df.columns:
            df = df.drop(columns="Sport")
        label_col = "Label"
    else:  # TONIOT
        df = df.replace("-", np.nan)
        miss_pct = (df.isnull().sum() / len(df)) * 100
        df = df.drop(columns=miss_pct[miss_pct > 50].index)
        if not full:
            rng = np.random.RandomState(seed)
            per_class = 2000
            idx = []
            for c in df["label"].unique():
                sub = df.index[df["label"] == c]
                n = min(per_class, len(sub))
                idx.extend(rng.choice(sub, n, replace=False))
            df = df.loc[idx].copy()
            meta = meta.loc[idx].copy()
        if "type" in df.columns:
            meta["family"] = df["type"].astype(str).values
        for c in ["src_ip", "dst_ip", "type"]:
            if c in df.columns:
                df = df.drop(columns=c)
        label_col = "label"

    y = df[label_col].astype(int).values
    X = df.drop(columns=[label_col]).copy()

    cat_cols = list(X.select_dtypes(include="object").columns)
    for c in cat_cols:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))

    meta = meta.reset_index(drop=True)
    return X.reset_index(drop=True), y, meta, cat_cols


def select_features(X_tr, y_tr, seed, selector: str = "rf"):
    """Rank features and choose how many to keep, using training data only.

    Returns the column positions to keep. `selector="none"` keeps everything.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    if selector == "none":
        return np.arange(X_tr.shape[1])

    if selector == "rf":
        ranker = RandomForestClassifier(n_estimators=50, random_state=seed)
        ranker.fit(X_tr, y_tr)
        order = np.argsort(ranker.feature_importances_)[::-1]
    elif selector == "mi":
        from sklearn.feature_selection import mutual_info_classif
        mi = mutual_info_classif(np.asarray(X_tr), y_tr, random_state=seed)
        order = np.argsort(mi)[::-1]
    else:
        raise ValueError(f"unknown selector: {selector}")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    best_n, best_score = min(20, X_tr.shape[1]), -1.0
    for n_feat in range(5, min(50, X_tr.shape[1]) + 1, 5):
        sel = order[:n_feat]
        score = cross_val_score(
            RandomForestClassifier(n_estimators=20, random_state=seed),
            np.asarray(X_tr)[:, sel], y_tr, cv=cv, scoring="f1_macro", n_jobs=-1
        ).mean()
        if score > best_score:
            best_score, best_n = score, n_feat
    return order[:best_n]


def prepare_split(name: str, seed: int, full: bool = False, selector: str = "rf",
                  oversample: str = "smote", split: str = "random",
                  order_index=None):
    """Load, split, then fit every data-dependent transform on the training fold.

    Order of operations (this is the leakage-controlled path):
        clean/encode -> train/test split -> fit scaler on train -> fit feature
        selector on train -> resample train only.

    Args:
        selector: "rf" (RF impurity importance), "mi" (mutual information), or
                  "none" (keep all features).
        oversample: "smote", "smotenc" (categorical-aware), or "none".
        split: "random" (stratified) or "temporal" (ordered by `order_index`,
               earlier rows train / later rows test).
        order_index: array of sortable values defining time order; required when
                     split="temporal".

    Returns:
        X_tr, X_te, y_tr, y_te, feats, n_feats, extras (dict of split metadata).
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    X, y, meta, cat_cols = load_raw(name, seed, full=full)
    all_cols = list(X.columns)

    if split == "temporal":
        if order_index is None and "timestamp" in meta:
            order_index = meta["timestamp"].values
        if order_index is None:
            raise ValueError(
                "split='temporal' requires order_index, or a dataset carrying a "
                "timestamp column (see build_cic_sample.py --keep-timestamp)")
        order = np.argsort(np.asarray(order_index), kind="stable")
        cut = int(len(order) * (1.0 - TEST_SIZE))
        tr_idx, te_idx = order[:cut], order[cut:]
    else:
        tr_idx, te_idx = train_test_split(
            np.arange(len(y)), test_size=TEST_SIZE, random_state=seed, stratify=y
        )

    X_tr_df, X_te_df = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    # scaler fitted on the training partition only
    sc = StandardScaler()
    X_tr_s = pd.DataFrame(sc.fit_transform(X_tr_df), columns=all_cols)
    X_te_s = pd.DataFrame(sc.transform(X_te_df), columns=all_cols)
    X_tr_s = X_tr_s.astype(np.float32).fillna(0.0)
    X_te_s = X_te_s.astype(np.float32).fillna(0.0)

    # feature selection fitted on the training partition only
    sel = select_features(X_tr_s.values, y_tr, seed, selector=selector)
    feats = [all_cols[i] for i in sel]
    X_tr_s, X_te_s = X_tr_s.iloc[:, sel], X_te_s.iloc[:, sel]

    X_tr_a = np.asarray(X_tr_s, dtype=np.float32)
    X_te_a = np.asarray(X_te_s, dtype=np.float32)
    y_tr = np.asarray(y_tr, dtype=np.int64)
    y_te = np.asarray(y_te, dtype=np.int64)

    counts = pd.Series(y_tr).value_counts()
    resampled = False
    if oversample != "none" and counts.min() / counts.max() < 0.75:
        if oversample == "smotenc":
            from imblearn.over_sampling import SMOTENC
            cat_idx = [i for i, f in enumerate(feats) if f in cat_cols]
            if cat_idx:
                sampler = SMOTENC(categorical_features=cat_idx, random_state=seed)
            else:  # no categorical survived selection; SMOTENC is undefined
                from imblearn.over_sampling import SMOTE
                sampler = SMOTE(random_state=seed)
        else:
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(random_state=seed)
        X_tr_a, y_tr = sampler.fit_resample(X_tr_a, y_tr)
        X_tr_a = np.asarray(X_tr_a, dtype=np.float32)
        y_tr = np.asarray(y_tr, dtype=np.int64)
        resampled = True

    extras = {
        "scaler_mean": sc.mean_[sel].tolist(),
        "scaler_scale": sc.scale_[sel].tolist(),
        "categorical_selected": [f for f in feats if f in cat_cols],
        "resampled": resampled,
        "family_test": meta["family"].values[te_idx].tolist() if "family" in meta else None,
        "test_index": np.asarray(te_idx).tolist(),
    }
    return X_tr_a, X_te_a, y_tr, y_te, feats, len(feats), extras


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #

def build_mlp(input_dim, seed):
    import torch.nn as nn
    import torch.optim as optim
    from art.estimators.classification import PyTorchClassifier
    import torch

    torch.manual_seed(seed)
    layers = []
    prev = input_dim
    for h in MLP_CONFIG["hidden_layers"]:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, MLP_CONFIG["nb_classes"]))
    net = nn.Sequential(*layers)
    optimizer = optim.Adam(net.parameters(), lr=MLP_CONFIG["lr"])
    device_type = "gpu" if torch.cuda.is_available() else "cpu"
    cls = PyTorchClassifier(
        model=net,
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        input_shape=(input_dim,),
        nb_classes=MLP_CONFIG["nb_classes"],
        clip_values=(-10.0, 10.0),
        device_type=device_type,
    )
    return cls


def train_baselines(X_tr, y_tr, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier

    timings = {}
    models = {}
    t = time.time()
    models["LR"] = LogisticRegression(random_state=seed, **ML_MODELS_CONFIG["LR"]).fit(X_tr, y_tr)
    timings["LR_train_s"] = time.time() - t

    t = time.time()
    models["SVM"] = LinearSVC(random_state=seed, **ML_MODELS_CONFIG["SVM"]).fit(X_tr, y_tr)
    timings["SVM_train_s"] = time.time() - t

    t = time.time()
    models["RF"] = RandomForestClassifier(random_state=seed, **ML_MODELS_CONFIG["RF"]).fit(X_tr, y_tr)
    timings["RF_train_s"] = time.time() - t

    t = time.time()
    cls = build_mlp(X_tr.shape[1], seed)
    cls.fit(X_tr, y_tr, batch_size=MLP_CONFIG["batch_size"], nb_epochs=MLP_CONFIG["nb_epochs"])
    timings["MLP_train_s"] = time.time() - t
    models["MLP"] = cls
    return models, timings


def predict(model, X, name):
    if name == "MLP":
        return np.argmax(model.predict(X.astype(np.float32)), axis=1)
    return model.predict(X)


# --------------------------------------------------------------------------- #
# attacks
# --------------------------------------------------------------------------- #

def generate_attacks(mlp_classifier, X_te):
    from art.attacks.evasion import (FastGradientMethod, ProjectedGradientDescent,
                                     CarliniL2Method)
    out = {}
    for eps in ATTACK_CONFIGS["FGSM"]["epsilon_values"]:
        atk = FastGradientMethod(estimator=mlp_classifier, eps=eps, batch_size=256)
        out[f"FGSM_eps{eps}"] = (atk.generate(X_te.astype(np.float32)),
                                  {"family": "FGSM", "epsilon": eps, "iters": 1})
    for eps in ATTACK_CONFIGS["PGD"]["epsilon_values"]:
        for it in ATTACK_CONFIGS["PGD"]["max_iter_values"]:
            atk = ProjectedGradientDescent(estimator=mlp_classifier, eps=eps,
                                           max_iter=it, batch_size=256)
            out[f"PGD_eps{eps}_it{it}"] = (atk.generate(X_te.astype(np.float32)),
                                            {"family": "PGD", "epsilon": eps, "iters": it})
    for c in ATTACK_CONFIGS["CW"]["confidence_values"]:
        for it in ATTACK_CONFIGS["CW"]["max_iter_values"]:
            atk = CarliniL2Method(classifier=mlp_classifier, confidence=c,
                                  max_iter=it, batch_size=128)
            out[f"CW_c{c}_it{it}"] = (atk.generate(X_te.astype(np.float32)),
                                       {"family": "CW", "confidence": c, "iters": it})
    return out


# --------------------------------------------------------------------------- #
# reproducible HopSkipJump: per-sample deterministic seeding
# --------------------------------------------------------------------------- #

# Stable integer codes for SeedSequence input. Never use Python's hash() for
# this: str/bytes hashing is randomized per-process by default (PYTHONHASHSEED)
# and does not reproduce across runs, which is exactly the property this whole
# fix exists to guarantee.
_DATASET_CODE = {"CSECICIDS2018": 1, "TONIOT": 2, "WUSTLEHMS2020": 3,
                 "CSECICIDS2018_TEMPORAL": 4}
_MODEL_CODE = {"LR": 1, "SVM": 2, "RF": 3, "XGB": 4, "LGBM": 5}


def hsj_sample_seed(experiment_seed: int, dataset: str, model_name: str,
                    sample_id: int) -> int:
    """Deterministic per-sample seed for ReproducibleHopSkipJump.

    Derived from (experiment seed, dataset, model, sample_id) via
    numpy.random.SeedSequence, which is designed for exactly this: combining
    several integers into a well-distributed seed without the collision risk
    of naive arithmetic combination. sample_id must be a persistent row
    identifier (its position within the selected sub-sample array is stable
    for a fixed dataset+experiment_seed, since the sub-sampler itself is
    deterministic), not the loop index after any reordering.
    """
    ds_code = _DATASET_CODE.get(dataset, 0)
    mdl_code = _MODEL_CODE.get(model_name, 0)
    ss = np.random.SeedSequence([int(experiment_seed), ds_code, mdl_code, int(sample_id)])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def run_reproducible_hsj(model, model_name: str, X_sub, y_sub, experiment_seed: int,
                         dataset: str, max_iter: int, max_eval: int,
                         init_eval: int = 20, init_size: int = 20,
                         batch_size: int = 32, estimator_factory=None):
    """Shared driver for the reproducible, guarded HopSkipJump attack.

    Runs one sample at a time, each with its own deterministic seed from
    hsj_sample_seed(), through ReproducibleHopSkipJump. Used by every direct-
    HSJ call site (primary whitebox attack, the hsj_budget ladder, and the
    gbm black-box attack) so all five model families share one implementation
    and one accounting convention.

    ASR is computed over the ELIGIBLE denominator only: samples the clean
    model already classifies correctly. Samples that are misclassified before
    any attack begins were never eligible to be evaded and must not appear in
    either the numerator or the denominator (see the docstring correction:
    the original code computed ASR over all sampled points, which is not the
    standard conditional definition and silently changes with how many
    already-wrong samples happen to be drawn).

    Returns (X_adv, diagnostics_dict) where diagnostics_dict has: asr,
    n_samples, n_eligible, median_queries_success, query_budget_exhausted,
    numerical_stagnation, elapsed_s.
    """
    from art.estimators.classification import SklearnClassifier
    from attacks.reproducible_hop_skip_jump import ReproducibleHopSkipJump

    if estimator_factory is None:
        estimator_factory = lambda: SklearnClassifier(model=model, clip_values=(-10.0, 10.0))

    y_clean = model.predict(X_sub)
    eligible = y_clean == y_sub

    t0 = time.time()
    X_adv = np.empty_like(X_sub)
    n_queries, term_reasons = [], []
    for i in range(len(X_sub)):
        seed_i = hsj_sample_seed(experiment_seed, dataset, model_name, i)
        est = estimator_factory()
        atk = ReproducibleHopSkipJump(
            classifier=est, max_iter=max_iter, max_eval=max_eval,
            init_eval=init_eval, init_size=init_size, batch_size=batch_size,
            verbose=False, random_state=seed_i)
        X_adv[i] = atk.generate(X_sub[i:i + 1])[0]
        n_queries.append(atk.n_queries)
        term_reasons.append(atk.termination_reason)
    elapsed = time.time() - t0

    y_pred = model.predict(X_adv)
    successful = eligible & (y_pred != y_sub)
    n_queries = np.array(n_queries)
    successful_q = n_queries[successful]

    diagnostics = {
        "n_samples": len(X_sub),
        "n_eligible": int(eligible.sum()),
        "asr": (float(successful.sum() / eligible.sum())
               if eligible.sum() else float("nan")),
        "median_queries_success": (float(np.median(successful_q))
                                   if len(successful_q) else float("nan")),
        "query_budget_exhausted": sum(1 for r in term_reasons if r == "query_budget"),
        "numerical_stagnation": sum(1 for r in term_reasons
                                    if r in ("step_halving_cap", "epsilon_underflow")),
        "elapsed_s": elapsed,
    }
    return X_adv, diagnostics


def whitebox_attack_decision_based(model, model_name, X_te, y_te, n_samples=200,
                                   seed=42, dataset="unknown"):
    """Direct gradient-free whitebox attack on RF/SVM/LR using HopSkipJump.

    Runs through ReproducibleHopSkipJump (Code/attacks/reproducible_hop_skip_jump.py)
    rather than ART's HopSkipJump directly: ART's implementation draws from an
    unseeded RandomState in _init_sample and the bare global np.random in
    _compute_update, so identical inputs do not reproduce (verified: three runs
    of the unmodified code gave ASR 0.865/0.855/0.845 for the same model and
    data). It also has an unbounded step-size search that can loop forever
    against a tree ensemble's piecewise-constant decision surface (see that
    module's docstring for both, verified against the installed ART source).
    """
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_te), min(n_samples, len(X_te)), replace=False)
    X_sub = X_te[idx].astype(np.float32)
    y_sub = y_te[idx]

    from art.estimators.classification import SklearnClassifier
    X_adv, diag = run_reproducible_hsj(
        model, model_name, X_sub, y_sub, experiment_seed=seed, dataset=dataset,
        max_iter=10, max_eval=200, init_eval=20, init_size=20, batch_size=32,
        estimator_factory=lambda: SklearnClassifier(model=model, clip_values=(-10.0, 10.0)))

    y_pred = model.predict(X_adv)
    metrics = metric_block(y_sub, y_pred)
    metrics.update({
        "model": model_name, "attack": "HopSkipJump_whitebox",
        "asr": diag["asr"], "elapsed_s": diag["elapsed_s"],
        "n_samples": diag["n_samples"], "n_eligible": diag["n_eligible"],
        "median_queries_success": diag["median_queries_success"],
        "query_budget_exhausted": diag["query_budget_exhausted"],
        "numerical_stagnation": diag["numerical_stagnation"],
    })
    return metrics


# --------------------------------------------------------------------------- #
# defenses
# --------------------------------------------------------------------------- #

_AT_FAMILY_CODE = {"FGSM": 1, "PGD": 2}


def at_cell_seed(experiment_seed: int, dataset: str, family: str, ratio: float) -> int:
    """Deterministic seed for one adversarial-training configuration.

    Same construction as hsj_sample_seed: numpy.random.SeedSequence over the
    identifying integers, not Python's hash() (randomized per-process, does not
    reproduce). ratio is a float (0.3/0.5 here), folded in via its value scaled
    to an integer so it contributes to the seed without floating-point hashing
    concerns.
    """
    ds_code = _DATASET_CODE.get(dataset, 0)
    fam_code = _AT_FAMILY_CODE.get(family, 0)
    ratio_code = int(round(ratio * 1000))
    ss = np.random.SeedSequence([int(experiment_seed), ds_code, fam_code, ratio_code])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def reset_all_rng(cell_seed: int) -> None:
    """Reset every RNG this pipeline touches, immediately before building a
    fresh model/attack/trainer for one adversarial-training cell.

    ART's AdversarialTrainer.fit() (art/defences/trainer/adversarial_trainer.py,
    verified against the installed 1.20.1 source) calls the bare global
    np.random.shuffle/np.random.choice for epoch shuffling and adversarial-
    sample selection -- no local RandomState, no random_state constructor
    argument. That is reproducible whenever the global generator is reseeded
    immediately beforehand and nothing else consumes it in between, which is
    exactly what this function plus a fresh model/attack/trainer per cell
    guarantees. Verified empirically below (test_adversarial_training_repro.py)
    rather than assumed.
    """
    import torch
    random.seed(cell_seed)
    np.random.seed(cell_seed)
    torch.manual_seed(cell_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cell_seed)


def adversarial_training_run(X_tr, y_tr, X_te, y_te, models, attacks, seed, dataset="unknown"):
    """
    Adversarial training (MLP) and surrogate-augmented retraining (others).
    Output rows: model, defense_attack, ratio, eval_attack, metrics...

    Each (attack family, ratio) configuration is an independently seeded cell:
    every RNG is reset and a fresh model, optimizer, attack, and
    AdversarialTrainer are built before training, so ART's use of the global
    numpy random state (see reset_all_rng) reproduces exactly. No trainer,
    attack, model, or RNG state is reused across configurations.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from art.defences.trainer import AdversarialTrainer
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

    rows = []
    for fam, ratio in [(f, r) for f in DEFENSE_CONFIGS["adversarial_training"]["attack_types"]
                              for r in DEFENSE_CONFIGS["adversarial_training"]["ratio_values"]]:
        cell_seed = at_cell_seed(seed, dataset, fam, ratio)
        reset_all_rng(cell_seed)

        # Build an attack instance against a freshly-trained MLP and adversarially train
        mlp_fresh = build_mlp(X_tr.shape[1], cell_seed)
        mlp_fresh.fit(X_tr, y_tr, batch_size=MLP_CONFIG["batch_size"],
                      nb_epochs=MLP_CONFIG["nb_epochs"])
        if fam == "FGSM":
            atk = FastGradientMethod(estimator=mlp_fresh, eps=0.3, batch_size=256)
        else:
            atk = ProjectedGradientDescent(estimator=mlp_fresh, eps=0.3, max_iter=10, batch_size=256)

        t = time.time()
        trainer = AdversarialTrainer(mlp_fresh, attacks=atk, ratio=ratio)
        trainer.fit(X_tr, y_tr, nb_epochs=DEFENSE_CONFIGS["adversarial_training"]["nb_epochs"])
        mlp_def = trainer.get_classifier()
        adv_train_s = time.time() - t

        # Evaluate defended MLP on each attack's adversarial test set
        for name, (X_adv, params) in attacks.items():
            y_pred = np.argmax(mlp_def.predict(X_adv.astype(np.float32)), axis=1)
            row = metric_block(y_te, y_pred)
            row.update({"model": "MLP", "defense_attack": fam, "ratio": ratio,
                         "eval_attack": name, "adv_train_s": adv_train_s, **params})
            rows.append(row)
        # MLP on clean
        y_pred = np.argmax(mlp_def.predict(X_te.astype(np.float32)), axis=1)
        row = metric_block(y_te, y_pred)
        row.update({"model": "MLP", "defense_attack": fam, "ratio": ratio,
                     "eval_attack": "Clean", "adv_train_s": adv_train_s,
                     "family": "Clean", "epsilon": 0.0, "iters": 0})
        rows.append(row)

        # Surrogate-augmented retraining for the other classifiers
        X_adv_train = atk.generate(X_tr.astype(np.float32))
        n_aug = int(ratio * len(X_tr))
        rng = np.random.RandomState(seed)
        sel = rng.choice(len(X_adv_train), n_aug, replace=False)
        X_aug = np.vstack([X_tr, X_adv_train[sel]])
        y_aug = np.concatenate([y_tr, y_tr[sel]])

        for mname in ["LR", "SVM", "RF"]:
            t = time.time()
            if mname == "RF":
                m = RandomForestClassifier(random_state=seed, **ML_MODELS_CONFIG["RF"]).fit(X_aug, y_aug)
            elif mname == "SVM":
                m = LinearSVC(random_state=seed, **ML_MODELS_CONFIG["SVM"]).fit(X_aug, y_aug)
            else:
                m = LogisticRegression(random_state=seed, **ML_MODELS_CONFIG["LR"]).fit(X_aug, y_aug)
            retrain_s = time.time() - t

            # On clean
            y_pred = m.predict(X_te)
            row = metric_block(y_te, y_pred)
            row.update({"model": mname, "defense_attack": fam, "ratio": ratio,
                         "eval_attack": "Clean", "retrain_s": retrain_s,
                         "family": "Clean", "epsilon": 0.0, "iters": 0})
            rows.append(row)
            for name, (X_adv, params) in attacks.items():
                y_pred = m.predict(X_adv)
                row = metric_block(y_te, y_pred)
                row.update({"model": mname, "defense_attack": fam, "ratio": ratio,
                             "eval_attack": name, "retrain_s": retrain_s, **params})
                rows.append(row)
    return rows


def randomized_smoothing_run(models, X_te, y_te, attacks, seed):
    """Randomized smoothing per sigma; subsamples test set for speed."""
    from scipy import stats
    cfg = DEFENSE_CONFIGS["randomized_smoothing"]
    rng = np.random.RandomState(seed)
    n_sub = min(cfg["test_subset_size"], len(X_te))
    idx = rng.choice(len(X_te), n_sub, replace=False)
    X_sub = X_te[idx].astype(np.float32)
    y_sub = y_te[idx]

    rows = []
    for sigma in cfg["sigma_values"]:
        for mname, m in models.items():
            t0 = time.time()
            n0_preds = []
            n_preds = []
            for x in X_sub:
                noise0 = rng.randn(cfg["n0"], len(x)).astype(np.float32) * sigma
                noisen = rng.randn(cfg["n"], len(x)).astype(np.float32) * sigma
                p0 = predict(m, (x + noise0), mname)
                pn = predict(m, (x + noisen), mname)
                n0_preds.append(np.bincount(p0, minlength=2).argmax())
                n_preds.append(np.bincount(pn, minlength=2).argmax())
            elapsed = time.time() - t0
            y_clean = np.array(n_preds)
            row = metric_block(y_sub, y_clean)
            row.update({"model": mname, "sigma": sigma, "eval_attack": "Clean",
                         "smooth_s_per_sample": elapsed / n_sub,
                         "n0": cfg["n0"], "n": cfg["n"]})
            rows.append(row)

            # Adversarial
            for name, (X_adv, params) in attacks.items():
                X_adv_sub = X_adv[idx]
                preds = []
                for x in X_adv_sub:
                    noisen = rng.randn(cfg["n"], len(x)).astype(np.float32) * sigma
                    pn = predict(m, (x + noisen), mname)
                    preds.append(np.bincount(pn, minlength=2).argmax())
                row = metric_block(y_sub, np.array(preds))
                row.update({"model": mname, "sigma": sigma, "eval_attack": name,
                             "n0": cfg["n0"], "n": cfg["n"], **params})
                rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

def evaluate_attacks_on_models(models, attacks, X_te, y_te):
    rows = []
    for name, (X_adv, params) in attacks.items():
        for mname, m in models.items():
            y_pred = predict(m, X_adv, mname)
            y_clean = predict(m, X_te.astype(np.float32) if mname == "MLP" else X_te, mname)
            base = metric_block(y_te, y_pred)
            asr = float(np.mean((y_clean == y_te) & (y_pred != y_te)))
            base.update({"model": mname, "attack": name, "asr": asr, **params})
            rows.append(base)
    return rows


def evaluate_baselines_on_clean(models, X_te, y_te):
    rows = []
    for mname, m in models.items():
        y_pred = predict(m, X_te.astype(np.float32) if mname == "MLP" else X_te, mname)
        row = metric_block(y_te, y_pred)
        row["model"] = mname
        rows.append(row)
    return rows


def measure_inference_latency(models, X_te):
    rows = []
    n = min(500, len(X_te))
    X = X_te[:n]
    for mname, m in models.items():
        t = time.time()
        _ = predict(m, X.astype(np.float32) if mname == "MLP" else X, mname)
        elapsed = time.time() - t
        rows.append({"model": mname, "n_samples": n, "total_s": elapsed,
                     "per_sample_ms": 1000 * elapsed / n})
    return rows


def run_one(dataset, seed, out_dir, full=False):
    print(f"\n===== {dataset} seed={seed} full={full} =====", flush=True)
    set_seed(seed)
    X_tr, X_te, y_tr, y_te, feats, n_feats, extras = prepare_split(
        dataset, seed, full=full)
    print(f"  selected_features={n_feats}", flush=True)
    print(f"  train={X_tr.shape}, test={X_te.shape}", flush=True)

    models, timings = train_baselines(X_tr, y_tr, seed)

    print("  generating attacks...", flush=True)
    t = time.time()
    attacks = generate_attacks(models["MLP"], X_te)
    timings["attack_gen_s"] = time.time() - t

    print("  evaluating clean baselines...", flush=True)
    base_rows = evaluate_baselines_on_clean(models, X_te, y_te)
    print("  evaluating attacks on baselines (transfer for non-MLP)...", flush=True)
    atk_rows = evaluate_attacks_on_models(models, attacks, X_te, y_te)

    print("  white-box gradient-free attacks on RF/SVM/LR...", flush=True)
    wb_rows = []
    for mname in ["RF", "SVM", "LR"]:
        try:
            wb_rows.append(whitebox_attack_decision_based(
                models[mname], mname, X_te, y_te, seed=seed, dataset=dataset))
        except Exception as e:
            print(f"    {mname} whitebox failed: {e}", flush=True)

    print("  inference latency...", flush=True)
    lat_rows = measure_inference_latency(models, X_te)

    print("  adversarial training / surrogate-augmented retraining...", flush=True)
    adv_rows = adversarial_training_run(X_tr, y_tr, X_te, y_te, models, attacks, seed, dataset=dataset)

    print("  randomized smoothing...", flush=True)
    rs_rows = randomized_smoothing_run(models, X_te, y_te, attacks, seed)

    out = out_dir / dataset / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(base_rows).to_csv(out / "baseline_clean.csv", index=False)
    pd.DataFrame(atk_rows).to_csv(out / "attacks.csv", index=False)
    pd.DataFrame(wb_rows).to_csv(out / "whitebox.csv", index=False)
    pd.DataFrame(lat_rows).to_csv(out / "latency.csv", index=False)
    pd.DataFrame(adv_rows).to_csv(out / "adv_training.csv", index=False)
    pd.DataFrame(rs_rows).to_csv(out / "randomized_smoothing.csv", index=False)
    with open(out / "timings.json", "w") as f:
        json.dump({"timings": timings, "n_features": n_feats,
                   "selected_features": feats,
                   "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
                   # retained for the epsilon-to-raw-units mapping and the
                   # per-attack-family breakdown
                   "scaler_scale": extras["scaler_scale"],
                   "scaler_mean": extras["scaler_mean"],
                   "categorical_selected": extras["categorical_selected"],
                   "resampled": extras["resampled"]}, f, indent=2)
    if extras.get("family_test") is not None:
        pd.DataFrame({"family": extras["family_test"], "y": y_te}).to_csv(
            out / "test_families.csv", index=False)
    print(f"  wrote {out}", flush=True)


def aggregate(dataset, seeds, out_dir):
    base = out_dir / dataset
    agg_dir = base / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["baseline_clean", "attacks", "whitebox", "latency",
                  "adv_training", "randomized_smoothing"]:
        dfs = []
        for s in seeds:
            f = base / f"seed_{s}" / f"{fname}.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f)
            df["seed"] = s
            dfs.append(df)
        if not dfs:
            continue
        full = pd.concat(dfs, ignore_index=True)
        full.to_csv(agg_dir / f"{fname}_full.csv", index=False)

        numeric = full.select_dtypes(include=[np.number]).columns.difference(["seed"])
        group_cols = [c for c in full.columns if c not in numeric and c != "seed"]
        if group_cols:
            stats = full.groupby(group_cols, dropna=False)[list(numeric)].agg(["mean", "std"])
            stats.columns = [f"{a}_{b}" for a, b in stats.columns]
            stats = stats.reset_index()
            stats.to_csv(agg_dir / f"{fname}_summary.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=list(DATASET_FILES.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    ap.add_argument("--out", default=str(RESULTS_DIR))
    ap.add_argument("--full", action="store_true",
                    help="Use full datasets without per-class subsampling for "
                         "WUSTLEHMS2020 and TONIOT (CSE-CIC-IDS2018 is unaffected).")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        for s in args.seeds:
            try:
                run_one(ds, s, out_dir, full=args.full)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"!! {ds} seed={s} failed: {e}", flush=True)
        aggregate(ds, args.seeds, out_dir)

    print("\nAll runs complete.", flush=True)


if __name__ == "__main__":
    main()
