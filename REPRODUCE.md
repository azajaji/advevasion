# Reproduction guide

Accompanies *"Feature-Space Adversarial Robustness Evaluation of Lightweight
ML-Based Intrusion Detection for IoT and Healthcare-Sensing Networks"*
(IEEE Access, 2026).

This reproduces the per-seed CSVs in `results/`, the 19 LaTeX tables in
`tables/`, and the figures used in Section IV, starting from the three public
benchmarks. No proprietary data and no private model checkpoints are involved.

---

## 1. Environment

The reported runs used a Windows 11 workstation, 16 GB RAM, NVIDIA RTX 5070
(CUDA 12.8), Python 3.13. The MLP surrogate, the gradient attacks (FGSM, PGD,
C&W), and adversarial training run on the GPU; LR, linear SVM, RF, and
HopSkipJump run on CPU. The pipeline runs CPU-only as well, just slower.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# for a CUDA GPU, replace the CPU wheel:
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## 2. Data

The benchmarks are not redistributed here. Download them from their providers
and place them under `Datasets/` at the repository root:

| Dataset | Source | Expected file |
|---|---|---|
| CSE-CIC-IDS2018, Improved release | https://www.unb.ca/cic/datasets/ids-2018.html | `Datasets/CSECICIDS2018_improved/*.csv` (10 daily files) |
| TON_IoT | https://research.unsw.edu.au/projects/toniot-datasets | `Datasets/ton_iot_dataset.csv` |
| WUSTL-EHMS-2020 | https://www.cse.wustl.edu/~jain/ehms/index.html | `Datasets/wustl-ehms-2020_with_attacks_categories.csv` |

TON_IoT and WUSTL-EHMS-2020 are used in full after cleaning. CSE-CIC-IDS2018 is
evaluated on a 551,512-flow stratified reservoir sample (300,000 Benign plus up
to 30,000 per attack category) drawn from all ten daily files:

```powershell
python code/build_cic_sample.py
```

This streams each daily file once, so memory is bounded by the output size rather
than by the ~6.3e7-flow input. About 19 minutes on the reference workstation.

## 3. Run the experiments

Five seeds are used throughout: **42, 7, 123, 31, 99**. All reported values are
mean +/- standard deviation across them.

```powershell
# Primary: clean baselines, transferred FGSM/PGD/C&W, HopSkipJump,
# adversarial training and surrogate-augmented retraining, randomized
# smoothing, and the cost accounting.
python code/run_experiments.py --datasets WUSTLEHMS2020 TONIOT CSECICIDS2018 `
    --seeds 42 7 123 31 99 --full

# Extended: HopSkipJump at N=500 with Wilson intervals, adaptive PGD against
# the AT-defended MLP, constrained generation, feature-selector ablation.
python code/run_extended.py --datasets WUSTLEHMS2020 TONIOT CSECICIDS2018 `
    --seeds 42 7 123 31 99 --full --experiment all --hsj-n 500

# Protocol-sensitivity and supplementary targets: direct gradient attacks on
# the linear models, XGBoost/LightGBM, per-family vulnerability, matched
# hyperparameter tuning, chronological splitting.
python code/run_reviewer_experiments.py --datasets WUSTLEHMS2020 TONIOT CSECICIDS2018 `
    --seeds 42 7 123 31 99 --full --out results
```

Per-seed CSVs land under `results/<DATASET>/seed_<S>/`, extended results under
`results/<DATASET>/extended/`, and protocol-sensitivity results under
`results/<DATASET>/reviewer/`. The chronological-split arm writes to
`results/CSECICIDS2018_TEMPORAL/`. The third script defaults its `--out` to
`results_v2`, so `--out results` above keeps all three writing to one tree.

On Windows, disable sleep for the multi-hour CSE-CIC-IDS2018 seeds:

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```

## 4. Rebuild the tables and figures

Each of the 19 tables printed in the paper comes from one of five builders:

```powershell
python code/build_tables.py             # baseline, attacks_summary, fnr_under_attack,
                                        # atr, rs_per_sigma, cost
python code/build_summary_table.py      # summary_takeaway
python code/build_extended_tables.py    # hsj_n500, adaptive_pgd, constrained_attacks
python code/build_reviewer_tables.py    # linear_wb, gbm, family, tuning_grid,
                                        # sensitivity, temporal
python code/build_reviewer_analyses.py  # deployment_base_rates, eps_native_units,
                                        # hsj_subsample_composition
python code/regen_figures.py            # the plotted figures (PDF + PNG)
```

`code/framework_fig.tex` is the standalone source for the architecture figure.
The builders are idempotent and overwrite their outputs in place.

## 5. Regression tests

Two tests guard the seeded stochastic paths. Both run the routine in separate
processes and compare, so they catch a global-random-state dependency that a
single-process check would miss.

```powershell
python code/test_extended_repro.py
python code/test_adversarial_training_repro.py
```

## 6. Wall time

About 14 hours for the primary pipeline and 5 hours for the extended pipeline on
the reference workstation, dominated by the CSE-CIC-IDS2018 seeds. TON_IoT and
WUSTL-EHMS-2020 each finish in well under an hour.
