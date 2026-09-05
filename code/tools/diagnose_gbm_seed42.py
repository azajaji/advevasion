"""Instrumented diagnostic for the CSE-CIC `gbm` cell timeout (seeds 42, 7,
123, 31 killed at the 45-minute cap; seed 99 finished at 44.7 min -- no
safety margin either way).

Investigates WHERE the time goes for XGBoost and LightGBM separately, on the
exact seed=42 cell, using the exact same code path, hyperparameters, sample
selection, and HSJ budget as the real `gbm` cell (run_reviewer_experiments.
experiment_gbm / hsj_blackbox). Nothing about the attack protocol is changed:
same max_iter, init_eval, max_eval, init_size, n_samples, query budget,
stopping criteria, or seed derivation. The only addition is per-sample,
per-phase instrumentation (Code/attacks/instrumented_hop_skip_jump.py, whose
equivalence to the unmodified attack was verified separately: bit-identical
outputs, matching query counts, matching termination reasons, exact query
accounting across 15 test samples including an init-failure case).

Also benchmarks batched vs. sequential single-row prediction for both models,
to separate algorithmic query cost from Python/model-wrapper call overhead
(the repeated "Failed to draw a random image" warning only proves frequent
init failure, not where the 45 minutes actually goes).

Outputs (all under results_v2/CSECICIDS2018/_diagnostics/):
  gbm_seed42_persample.csv   -- one row per (model, sample): full phase
                                 breakdown, persistent sample id, status
  gbm_seed42_report.txt      -- aggregate summary + decision-rule evaluation
  gbm_seed42_predict_bench.csv -- batched vs sequential prediction timing

Usage:  python Code/diagnose_gbm_seed42.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

import numpy as np
import pandas as pd
from art.estimators.classification import BlackBoxClassifier

import run_experiments as base
import run_reviewer_experiments as rev
from attacks.instrumented_hop_skip_jump import InstrumentedHopSkipJump

DATASET = "CSECICIDS2018"
SEED = 42
N_SAMPLES = 100  # matches hsj_blackbox's default, unchanged
OUT_DIR = CODE_DIR.parent / "results_v2" / DATASET / "_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_models(X_tr, y_tr):
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    models = {
        "XGB": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                             random_state=SEED, n_jobs=rev.GBM_THREADS,
                             eval_metric="logloss"),
        "LGBM": LGBMClassifier(n_estimators=200, max_depth=-1, learning_rate=0.1,
                               random_state=SEED, n_jobs=rev.GBM_THREADS, verbose=-1),
    }
    timings = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        timings[name] = time.time() - t0
        print(f"  trained {name} in {timings[name]:.1f}s", flush=True)
    return models, timings


def pin_single_thread(model):
    for attr, val in (("n_jobs", 1), ("nthread", 1)):
        try:
            model.set_params(**{attr: val})
        except (ValueError, AttributeError, TypeError):
            pass


def benchmark_predict(model, name, X_pool):
    """One batched predict over 10,000 rows vs. 10,000 sequential single-row
    predicts, on the SAME trained model, post-thread-pinning (matching what
    hsj_blackbox actually runs under)."""
    n = min(10_000, len(X_pool))
    X_bench = X_pool[:n]

    t0 = time.perf_counter()
    _ = model.predict(X_bench)
    batched_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(n):
        _ = model.predict(X_bench[i:i + 1])
    sequential_s = time.perf_counter() - t0

    print(f"  [{name}] batched predict ({n} rows): {batched_s:.3f}s "
          f"({batched_s/n*1000:.3f} ms/row)", flush=True)
    print(f"  [{name}] sequential predict ({n} x 1 row): {sequential_s:.3f}s "
          f"({sequential_s/n*1000:.3f} ms/row)  "
          f"[{sequential_s/max(batched_s,1e-9):.1f}x batched]", flush=True)

    return {
        "model": name, "n_rows": n,
        "batched_s": batched_s, "batched_ms_per_row": batched_s / n * 1000,
        "sequential_s": sequential_s, "sequential_ms_per_row": sequential_s / n * 1000,
        "sequential_over_batched_ratio": sequential_s / max(batched_s, 1e-9),
    }


def run_instrumented_hsj(model, name, X_te, y_te):
    """Mirrors hsj_blackbox's sample selection and run_reproducible_hsj's
    per-sample loop exactly (same RNG, same seed derivation, same HSJ
    params), swapping only ReproducibleHopSkipJump for InstrumentedHopSkipJump
    so every sample's phase breakdown is recorded."""
    pin_single_thread(model)

    def predict_fn(x):
        preds = np.asarray(model.predict(x)).astype(int).ravel()
        onehot = np.zeros((len(preds), 2), dtype=np.float32)
        onehot[np.arange(len(preds)), preds] = 1.0
        return onehot

    rng = np.random.RandomState(SEED)
    idx = []
    for c in np.unique(y_te):
        cls = np.where(y_te == c)[0]
        idx.extend(rng.choice(cls, min(N_SAMPLES // 2, len(cls)), replace=False))
    idx = np.array(idx)
    X_sub, y_sub = X_te[idx].astype(np.float32), y_te[idx]

    y_clean = model.predict(X_sub)
    eligible = y_clean == y_sub

    rows = []
    t_cell0 = time.time()
    for i in range(len(X_sub)):
        seed_i = base.hsj_sample_seed(SEED, DATASET, name, i)
        est = BlackBoxClassifier(predict_fn, input_shape=(X_te.shape[1],),
                                 nb_classes=2, clip_values=(-10.0, 10.0))
        atk = InstrumentedHopSkipJump(
            classifier=est, max_iter=10, max_eval=200, init_eval=20, init_size=20,
            batch_size=32, verbose=False, random_state=seed_i)
        t0 = time.perf_counter()
        atk.generate(X_sub[i:i + 1])
        wall = time.perf_counter() - t0
        diag = atk.rich_diagnostics[0]

        check_sum = (diag["init_queries"] + diag["bsearch_queries"]
                     + diag["gradest_queries"] + diag["stepsearch_queries"])
        assert check_sum == diag["total_queries"], (
            f"query accounting mismatch: {check_sum} != {diag['total_queries']}")

        rows.append({
            "model": name,
            "sample_pos": i,               # position in X_sub -- the seed basis
            "orig_test_idx": int(idx[i]),  # persistent id: row in X_te
            "clean_correct": bool(y_clean[i] == y_sub[i]),
            "eligible": bool(eligible[i]),
            "wall_s_measured": wall,        # independent of internal timers, sanity cross-check
            **diag,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(X_sub):
            elapsed = time.time() - t_cell0
            rate = elapsed / (i + 1)
            projected = rate * len(X_sub)
            print(f"  [{name}] {i+1}/{len(X_sub)} done, {elapsed:.1f}s elapsed, "
                  f"{rate:.2f}s/sample, projected {projected/60:.1f} min for this model's "
                  f"{len(X_sub)} samples", flush=True)

    cell_elapsed = time.time() - t_cell0
    print(f"  [{name}] all {len(X_sub)} samples: {cell_elapsed:.1f}s total "
          f"({cell_elapsed/60:.2f} min)", flush=True)
    return rows, cell_elapsed


def main():
    print(f"===== diagnostic: {DATASET} seed={SEED}, gbm cell (XGB, LGBM separately) =====",
          flush=True)
    base.set_seed(SEED)
    X_tr, X_te, y_tr, y_te, feats, n_feats, extras = base.prepare_split(
        DATASET, SEED, full=True)
    print(f"  train={X_tr.shape} test={X_te.shape}", flush=True)

    print("\n--- training models (same hyperparameters as experiment_gbm) ---", flush=True)
    models, train_timings = build_models(X_tr, y_tr)

    print("\n--- prediction benchmark: batched vs. sequential (post thread-pin) ---", flush=True)
    bench_rows = []
    for name, model in models.items():
        pin_single_thread(model)
        bench_rows.append(benchmark_predict(model, name, X_te.astype(np.float32)))
    bench_df = pd.DataFrame(bench_rows)
    bench_path = OUT_DIR / "gbm_seed42_predict_bench.csv"
    bench_df.to_csv(bench_path, index=False)
    print(f"  wrote {bench_path}", flush=True)

    print("\n--- instrumented HopSkipJump, XGBoost and LightGBM separately ---", flush=True)
    all_rows = []
    cell_elapsed = {}
    for name, model in models.items():
        print(f"\n[{name}]", flush=True)
        rows, elapsed = run_instrumented_hsj(model, name, X_te, y_te)
        all_rows.extend(rows)
        cell_elapsed[name] = elapsed

    per_sample = pd.DataFrame(all_rows)
    persample_path = OUT_DIR / "gbm_seed42_persample.csv"
    per_sample.to_csv(persample_path, index=False)
    print(f"\nwrote {persample_path}  ({len(per_sample)} rows)", flush=True)

    # ---- aggregate report ----
    lines = []
    lines.append("=" * 78)
    lines.append(f"GBM CSE-CIC seed=42 timeout diagnostic")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Training time:")
    for name, t in train_timings.items():
        lines.append(f"  {name}: {t:.1f}s")
    lines.append("")
    lines.append("Prediction benchmark (10,000 rows, post single-thread pin):")
    for _, r in bench_df.iterrows():
        lines.append(f"  {r['model']}: batched={r['batched_ms_per_row']:.4f} ms/row  "
                     f"sequential={r['sequential_ms_per_row']:.4f} ms/row  "
                     f"({r['sequential_over_batched_ratio']:.1f}x overhead)")
    lines.append("")

    for name in models:
        sub = per_sample[per_sample["model"] == name]
        n = len(sub)
        n_init_fail = int((sub["status"] == "init_failure").sum())
        n_qbudget = int((sub["status"] == "query_budget").sum())
        n_stagnant = int((sub["status"].isin(["step_halving_cap", "epsilon_underflow"])).sum())
        n_success = int((sub["status"] == "success").sum())
        totals = sub[["init_time_s", "bsearch_time_s", "gradest_time_s",
                      "stepsearch_time_s", "total_time_s"]].sum()
        max_row = sub.loc[sub["total_time_s"].idxmax()]
        top5 = sub.nlargest(5, "total_time_s")[
            ["orig_test_idx", "status", "total_time_s", "total_queries"]]

        lines.append(f"--- {name} ({n} samples, cell wall time {cell_elapsed[name]/60:.2f} min) ---")
        lines.append(f"  status counts: success={n_success} init_failure={n_init_fail} "
                     f"query_budget={n_qbudget} stagnation={n_stagnant}")
        lines.append(f"  time by phase (sum over {n} samples):")
        lines.append(f"    init:        {totals['init_time_s']:8.1f}s "
                     f"({100*totals['init_time_s']/totals['total_time_s']:5.1f}%)")
        lines.append(f"    binary-search:{totals['bsearch_time_s']:8.1f}s "
                     f"({100*totals['bsearch_time_s']/totals['total_time_s']:5.1f}%)")
        lines.append(f"    grad-est:    {totals['gradest_time_s']:8.1f}s "
                     f"({100*totals['gradest_time_s']/totals['total_time_s']:5.1f}%)")
        lines.append(f"    step-search: {totals['stepsearch_time_s']:8.1f}s "
                     f"({100*totals['stepsearch_time_s']/totals['total_time_s']:5.1f}%)")
        lines.append(f"    TOTAL (instrumented, per-sample sum): {totals['total_time_s']:8.1f}s")
        lines.append(f"  single slowest sample: orig_test_idx={int(max_row['orig_test_idx'])} "
                     f"status={max_row['status']} time={max_row['total_time_s']:.1f}s "
                     f"queries={int(max_row['total_queries'])}")
        lines.append(f"  slowest sample as % of cell time: "
                     f"{100*max_row['total_time_s']/totals['total_time_s']:.1f}%")
        lines.append(f"  top 5 slowest samples:")
        for _, r in top5.iterrows():
            lines.append(f"    idx={int(r['orig_test_idx']):5d} status={r['status']:14s} "
                         f"time={r['total_time_s']:7.1f}s  queries={int(r['total_queries'])}")
        lines.append("")

    total_train = sum(train_timings.values())
    total_hsj = sum(cell_elapsed.values())
    projected_cell_min = (total_train + total_hsj) / 60
    lines.append(f"Projected full cell time (train + both models' HSJ, excluding "
                 f"transferred-attack eval which is fast and unrelated): "
                 f"{projected_cell_min:.1f} min")
    lines.append("")
    lines.append("Decision-rule inputs (see instruction): does any single sample dominate?")
    for name in models:
        sub = per_sample[per_sample["model"] == name]
        share = sub["total_time_s"].max() / sub["total_time_s"].sum()
        lines.append(f"  {name}: slowest single sample = {share*100:.1f}% of that model's time")
    lines.append("")
    lines.append("Does sequential prediction overhead dominate over batched?")
    for _, r in bench_df.iterrows():
        lines.append(f"  {r['model']}: {r['sequential_over_batched_ratio']:.1f}x "
                     f"(sequential ms/row {r['sequential_ms_per_row']:.4f} vs "
                     f"batched ms/row {r['batched_ms_per_row']:.4f})")

    report = "\n".join(lines)
    report_path = OUT_DIR / "gbm_seed42_report.txt"
    report_path.write_text(report)
    print("\n" + report)
    print(f"\nwrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
