# Tier-B extended experiments

These four experiments address common reviewer concerns and feed the
**Extended Robustness Analyses** subsection (§4.6) of `main.tex`.

| Experiment | Concern addressed | Script flag | Output CSV |
|---|---|---|---|
| HSJ at N=500 with Wilson 95% CI | "HSJ underpowered at N=200" | `--experiment hsj500` | `hsj_n500.csv` |
| Constrained-feature FGSM/PGD | "feature-space attacks not realistic" | `--experiment constrained` | `constrained_attacks.csv` |
| Adaptive PGD vs AT-MLP | "defense lacks adaptive evaluation" | `--experiment adaptive` | `adaptive_pgd.csv` |
| Mutual-information selector | "RF-feature-importance bias" | `--experiment mi_selector` | `mi_clean.csv` + `mi_features.json` |

## Run all four on all datasets, all seeds

macOS / Linux:

```bash
cd "$HOME/AdvEvasion Paper"
source .venv/bin/activate

python Code/run_extended.py \
    --experiment all \
    --datasets CSECICIDS2018 TONIOT WUSTLEHMS2020 \
    --seeds 42 7 123 \
    --hsj-n 500
```

Windows (PowerShell):

```powershell
cd "$HOME\Documents\CodingWorkSpaces\AdvEvasion Paper"
.\.venv\Scripts\Activate.ps1

python Code\run_extended.py `
    --experiment all `
    --datasets CSECICIDS2018 TONIOT WUSTLEHMS2020 `
    --seeds 42 7 123 `
    --hsj-n 500
```

Full run is ≈ 4–8 hours wall clock on CPU.

Per-seed CSVs land under `results/<DATASET>/extended/seed_<S>/`.

## Build the LaTeX tables

After the runs finish, aggregate into the four LaTeX tables `main.tex` already references via `\IfFileExists`:

```bash
python Code/build_extended_tables.py
```

This writes:

- `paper/tables_out/hsj_n500.tex`        → `\ref{tab:hsj_n500}`
- `paper/tables_out/constrained_attacks.tex` → `\ref{tab:constrained}`
- `paper/tables_out/adaptive_pgd.tex`    → `\ref{tab:adaptive}`
- `paper/tables_out/mi_clean.tex`        → `\ref{tab:mi_selector}`

Until those `.tex` files exist, the `\IfFileExists` placeholders in `main.tex`
fall back to italic notes naming the script flag, so the document still
compiles cleanly without the extended results.

## Selective runs (cheap pilots)

```bash
# Only HSJ-500 on the smallest dataset, single seed (≈ 5 minutes)
python Code/run_extended.py --experiment hsj500 --datasets WUSTLEHMS2020 --seeds 42

# Only constrained attacks (depends on which features are masked; see PROTOCOL_CONSTRAINED_PATTERNS in run_extended.py)
python Code/run_extended.py --experiment constrained --datasets WUSTLEHMS2020 --seeds 42 7 123

# Only adaptive PGD (re-trains AT-MLP per seed; ≈ 10–30 minutes per seed per dataset)
python Code/run_extended.py --experiment adaptive --datasets TONIOT --seeds 42

# Only the MI selector ablation (clean baselines under alternative ranker)
python Code/run_extended.py --experiment mi_selector --datasets CSECICIDS2018 TONIOT WUSTLEHMS2020 --seeds 42
```

## Notes

- The constrained-attack mask in `run_extended.py` is conservative: any feature
  whose name contains one of `PROTOCOL_CONSTRAINED_PATTERNS` (TCP flags,
  Init_Win, ssl_*, dns_*, http_status_code, proto, service, weird_*) is frozen
  during gradient updates. The list reflects the categorisation in
  Table 7 of the manuscript and can be widened or narrowed per dataset.
- HSJ at N=500 reports Wilson 95% binomial CI explicitly; the per-(dataset, model, seed)
  cost scales roughly linearly with N at fixed `max_iter=10, max_eval=200`.
- Adaptive PGD trains a fresh AT-MLP per seed (does not reuse the AT model from
  `run_experiments.py`), so seeds are independent.
- MI selector replaces only the feature-ranking step; everything downstream
  (preprocessing, splitting, training, attacks) is identical to the primary
  pipeline.
