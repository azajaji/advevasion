# Cell scheduler — operations guide

Parallel execution for the reviewer experiments. Scientific definitions are
untouched: `run_experiments.py` and `run_reviewer_experiments.py` are imported
as-is and this layer only decides when and where a cell runs.

## Files

| File | Role |
|---|---|
| `Code/cell_scheduler.py` | cell model, config hashing, atomic writes, claims, completion markers |
| `Code/run_cells.py` | CLI entry point and queue orchestration |
| `Code/bench_scheduler.py` | throughput benchmark and reproducibility gate |
| `run_scheduler.ps1` | Windows launch scripts |

Nothing in `run_experiments.py` or `run_reviewer_experiments.py` was modified for
parallelism.

## Why there is no separate GPU queue

Measured on this host: the GPU sits at 8–15% utilisation and about 3 GB of
12.8 GB. Only MLP training and gradient-attack generation touch CUDA, and both
are short relative to the sklearn work around them. A dedicated GPU queue would
serialise cells for no gain. `--gpu-workers` therefore caps how many CUDA-using
cells run at once for memory safety, rather than forming a separate pipeline.

## Numerical safety

* scikit-learn estimators are deterministic regardless of `n_jobs`, so running
  cells concurrently cannot change their results. Verified by the gate below.
* XGBoost and LightGBM are **not** thread-count invariant: thread count changes
  float summation order. `gbm` cells are therefore classified `exclusive`, run
  one at a time, with their thread settings left alone.
* No thread-limiting environment variables are injected, so the numerics of the
  parallel path are identical to the sequential path rather than merely close.

## Completion contract

A cell is skipped only when all of these hold:

1. its result CSV exists;
2. the CSV parses and has at least one row;
3. a marker exists at `results_v2/_cells/<cell-id>.done`;
4. the marker's `config_hash` matches the current configuration.

The hash covers the dataset, seed, attack/defence/model configuration, and the
source of the experiment function, `prepare_split`, `select_features` and
`load_raw`. Editing any of those invalidates the affected cells automatically.

Results produced before this scheduler existed have no marker. They are
**adopted** on first sight and recorded as `adopted_pre_scheduler: true` rather
than recomputed, so no completed work is thrown away.

## Concurrency safety

* One CSV per cell; no shared mutable result file.
* Writes go to a temporary file, are flushed and `fsync`ed, parsed back for
  validation, then renamed atomically. A reader never sees a partial file.
* An `O_EXCL` claim file prevents two runners from executing the same cell.
  Claims older than six hours are treated as abandoned.
* A worker crash affects only its own cell; the run continues and the failure is
  listed at the end.
* Aggregation happens after cells finish, never concurrently with them.

## Usage

```powershell
.\run_scheduler.ps1 status                    # what is pending
.\run_scheduler.ps1 dryrun                    # plan, hashes, output paths
.\run_scheduler.ps1 gate                      # prove equality with sequential
.\run_scheduler.ps1 combined                  # recommended: run everything
.\run_scheduler.ps1 aggregate                 # rebuild summaries only
```

Direct CLI equivalents:

```
python Code/run_cells.py --list-pending
python Code/run_cells.py --dry-run --cpu-workers 3
python Code/run_cells.py --resume --cpu-workers 3 --gpu-workers 2
python Code/run_cells.py --only resampling,gbm --datasets toniot,cse_cic --seeds 42,123
python Code/run_cells.py --aggregate-only
python Code/bench_scheduler.py            # benchmark
python Code/bench_scheduler.py --gate     # reproducibility gate
```

## Transition from the sequential runner

The sequential runner does not take claims, so the two must not overlap.

1. Wait for the sequential job to finish its current cell, or stop it. Because
   every cell writes one file atomically, stopping mid-cell loses only that cell.
2. Confirm nothing is half-written:
   ```
   python Code/run_cells.py --list-pending
   ```
   Any interrupted cell shows as pending with reason `unreadable` or `empty file`.
3. Start the scheduler:
   ```
   .\run_scheduler.ps1 combined
   ```
   Completed cells are adopted, pending cells run in parallel.

## Rollback

The scheduler adds files; it does not alter the result layout. To revert:

1. Stop the scheduler.
2. Resume the original runner, which ignores markers entirely:
   ```
   python Code/run_reviewer_experiments.py --experiment selection resampling gbm hsj_budget `
       --datasets WUSTLEHMS2020 TONIOT CSECICIDS2018 --seeds 42 7 123 31 99 --out results_v2
   ```
3. Optionally remove the scheduler bookkeeping, which no other code reads:
   ```
   Remove-Item -Recurse -Force results_v2\_cells, logs\cells, results_bench
   ```

Result CSVs written by the scheduler are byte-compatible with the sequential
runner, so a rollback needs no data migration.
