# AdvEvasion — reproducibility archive

Code, per-seed results, and generated result tables for:

> A. AlAjaji, R. Alharbi, M. M. Hassan, and M. Almukaynizi,
> "Feature-Space Adversarial Robustness Evaluation of Lightweight ML-Based
> Intrusion Detection for IoT and Healthcare-Sensing Networks,"
> *IEEE Access*, 2026.

The paper is an evaluation-protocol study. It proposes no new attack and no new
defense. It fixes one shared representation, one attack grid, and one cost
accounting across four lightweight classifiers (logistic regression, linear SVM,
random forest, a compact MLP) and two supplementary gradient-boosted targets
(XGBoost, LightGBM), on three public benchmarks: CSE-CIC-IDS2018 (Improved
release), TON_IoT, and WUSTL-EHMS-2020.

## What is in here

| Path | Contents |
|---|---|
| `code/` | The experiment pipeline, table builders, figure builder, and regression tests |
| `code/attacks/` | Seeded and instrumented HopSkipJump wrappers over the ART implementation |
| `code/tools/` | Run schedulers and diagnostics used to drive the multi-hour CSE-CIC-IDS2018 seeds |
| `code/notebooks/` | The original exploratory notebooks the scripted pipeline replaced |
| `results/` | Per-seed CSV metrics, aggregated mean/std summaries, and the JSON run record for each (dataset, seed) |
| `tables/` | The 19 LaTeX tables printed in the paper, plus three analysis CSVs |

`REPRODUCE.md` is the step-by-step guide.

## What is not in here

The three benchmarks are **not** redistributed. They are public and must be
obtained from their original providers under their own terms; `REPRODUCE.md`
gives the links and the expected file layout. Nothing in `results/` contains raw
flow records or biometric values: the per-sample files hold only a traffic-family
label and the binary label.

The manuscript source is not included. The published article is open access at
IEEE Access.

## Reproducibility notes

Three properties of this pipeline are worth stating, because they are what the
reported numbers depend on:

1. **Every data-dependent transform is fitted inside the training partition.**
   The stratified split precedes standardization, feature selection, and
   oversampling. This matters more than usual here: perturbation budgets are
   expressed in standardized units, so scaling statistics estimated with test
   data present would change what a given epsilon physically means.
2. **The seed loop encloses the split.** The split, the scaler, the selector, the
   resampler, and every model are re-derived per seed, so the reported mean and
   standard deviation over the five seeds (42, 7, 123, 31, 99) capture pipeline
   variability, not just model initialization.
3. **The stochastic attack paths are seeded.** HopSkipJump and the
   adversarial-training routine each draw from an explicit seeded generator
   rather than a global or internal random state. `code/test_extended_repro.py`
   and `code/test_adversarial_training_repro.py` check this across processes.

Each run writes a `timings.json` per (dataset, seed) holding the per-stage
wall-clock cost, the number and names of the selected features, the partition
sizes, whether the imbalance threshold triggered oversampling, and the
per-feature standardization mean and scale. The scale vector is what allows an
attack budget in standardized units to be converted back into packets, bytes, or
beats per minute.

## Requirements

Python 3.13. See `requirements.txt`. The MLP surrogate and the gradient attacks
use PyTorch and benefit from a GPU; the classical classifiers and HopSkipJump run
on CPU.

## License

MIT, see `LICENSE`. The datasets carry their own licenses from their providers.
