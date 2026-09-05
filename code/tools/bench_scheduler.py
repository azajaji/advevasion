"""Benchmark and reproducibility gate for the cell scheduler.

Two jobs:

  run_benchmark      time representative cells under several worker counts,
                     recording wall clock, peak RAM and peak GPU memory.

  reproducibility    re-run cells that already exist from the sequential runner
                     and require the new path to reproduce them exactly.

Benchmarks and gate runs write into a throwaway output directory so they can
never collide with, or overwrite, real results.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import cell_scheduler as cs  # noqa: E402

ROOT = CODE_DIR.parent
BENCH_OUT = ROOT / "results_bench"

#: metric columns compared by the reproducibility gate
METRIC_COLS = ["accuracy", "precision", "recall", "f1", "fpr", "fnr", "asr",
               "adv_accuracy", "mean_l2", "cv_f1_macro"]


def gpu_memory_mb() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return float(out[0])
    except Exception:
        return float("nan")


def ram_used_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().used / 1e9
    except Exception:
        return float("nan")


def frame_fingerprint(path: Path) -> str:
    """Order-insensitive fingerprint of the numeric content of a result file."""
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c in METRIC_COLS]
    key = [c for c in ("model", "attack", "selector", "oversample", "split",
                       "surrogate", "target", "family", "config", "epsilon",
                       "max_iter", "max_eval", "epochs") if c in df.columns]
    if key:
        df = df.sort_values(key, kind="stable").reset_index(drop=True)
    blob = df[cols].round(12).to_csv(index=False).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def compare_results(a: Path, b: Path) -> tuple[bool, str]:
    if not a.exists() or not b.exists():
        return False, "one side missing"
    da, db = pd.read_csv(a), pd.read_csv(b)
    if len(da) != len(db):
        return False, f"row count {len(da)} vs {len(db)}"
    cols = [c for c in da.columns if c in METRIC_COLS and c in db.columns]
    if not cols:
        return False, "no comparable metric columns"
    key = [c for c in ("model", "attack", "selector", "oversample", "split",
                       "surrogate", "target", "family", "config", "epsilon",
                       "max_iter", "max_eval", "epochs")
           if c in da.columns and c in db.columns]
    if key:
        da = da.sort_values(key, kind="stable").reset_index(drop=True)
        db = db.sort_values(key, kind="stable").reset_index(drop=True)
    import numpy as np
    diff = np.abs(da[cols].to_numpy(dtype=float) - db[cols].to_numpy(dtype=float))
    mx = float(np.nanmax(diff)) if diff.size else 0.0
    return mx == 0.0, f"max|diff|={mx:.3e}"


# --------------------------------------------------------------------------- #
# reproducibility gate
# --------------------------------------------------------------------------- #

GATE_CELLS = [("linear_wb", "WUSTLEHMS2020", 42),
              ("family", "WUSTLEHMS2020", 42),
              ("tuning", "WUSTLEHMS2020", 42)]


def reproducibility_gate(reference_out: str = "results_v2",
                         workers: int = 3) -> int:
    """Re-run already-computed cells through the scheduler and require equality.

    Takes the RunLock on the *reference* out_dir, not just on BENCH_OUT: the
    point is to guarantee no production scheduler is active on results_v2 while
    this reads from it and writes GPU-touching cells of its own.
    """
    with cs.RunLock(reference_out, label="bench_scheduler --gate"):
        return _reproducibility_gate_body(reference_out, workers)


def _reproducibility_gate_body(reference_out: str, workers: int) -> int:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if BENCH_OUT.exists():
        shutil.rmtree(BENCH_OUT, ignore_errors=True)

    specs = [cs.CellSpec(analysis=a, dataset=d, seed=s, full=True,
                         out_dir=str(BENCH_OUT),
                         config_hash=cs.compute_config_hash(a, d, s, True))
             for a, d, s in GATE_CELLS]

    print(f"reproducibility gate: {len(specs)} cell(s) through {workers} worker(s), "
          f"timeout={cs.DEFAULT_CELL_TIMEOUT_S/60:.0f} min/cell")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(cs.execute_cell_with_timeout, dict(
            analysis=s.analysis, dataset=s.dataset, seed=s.seed, full=s.full,
            out_dir=s.out_dir, config_hash=s.config_hash)) for s in specs]
        for f in as_completed(futs):
            r = f.result()
            print(f"  {r['status']:8} {r['cell_id']:38} {r['duration_s']:6.1f}s")
    print(f"  elapsed {time.time()-t0:.1f}s\n")

    all_ok = True
    print(f"{'cell':40} {'equal':6} detail")
    print("-" * 78)
    for a, d, s in GATE_CELLS:
        ref = Path(reference_out) / d / "reviewer" / f"seed_{s}" / f"{a}.csv"
        new = BENCH_OUT / d / "reviewer" / f"seed_{s}" / f"{a}.csv"
        ok, detail = compare_results(ref, new)
        fp_ref = frame_fingerprint(ref) if ref.exists() else "n/a"
        fp_new = frame_fingerprint(new) if new.exists() else "n/a"
        print(f"{a+'/'+d+'/'+str(s):40} {'YES' if ok else 'NO ':6} {detail}"
              f"   fp {fp_ref} vs {fp_new}")
        all_ok &= ok

    print("-" * 78)
    print("GATE PASSED: scheduler reproduces the sequential results exactly"
          if all_ok else "GATE FAILED: divergence found, do not use the scheduler")
    return 0 if all_ok else 1


# --------------------------------------------------------------------------- #
# throughput benchmark
# --------------------------------------------------------------------------- #

#: cheap but representative: one cell of each class on the smallest dataset
BENCH_CELLS = [("family", "WUSTLEHMS2020", 42),
               ("family", "WUSTLEHMS2020", 7),
               ("selection", "WUSTLEHMS2020", 123),
               ("linear_wb", "WUSTLEHMS2020", 31),
               ("tuning", "WUSTLEHMS2020", 99),
               ("surrogate", "WUSTLEHMS2020", 42)]


def _run_config(specs, workers: int) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    if BENCH_OUT.exists():
        shutil.rmtree(BENCH_OUT, ignore_errors=True)

    ram0, gpu0 = ram_used_gb(), gpu_memory_mb()
    peak_ram, peak_gpu = ram0, gpu0
    t0 = time.time()
    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(cs.execute_cell_with_timeout, dict(
            analysis=s.analysis, dataset=s.dataset, seed=s.seed, full=s.full,
            out_dir=s.out_dir, config_hash=s.config_hash)) for s in specs]
        while futs:
            peak_ram = max(peak_ram, ram_used_gb())
            peak_gpu = max(peak_gpu, gpu_memory_mb())
            done = [f for f in futs if f.done()]
            for f in done:
                if f.result()["status"] not in ("ok", "skipped"):
                    failures += 1
                futs.remove(f)
            time.sleep(1.0)
    elapsed = time.time() - t0

    fps = {}
    for s in specs:
        p = s.result_path
        if p.exists():
            fps[s.cell_id] = frame_fingerprint(p)
    return {"workers": workers, "elapsed_s": elapsed, "peak_ram_gb": peak_ram,
            "peak_gpu_mb": peak_gpu, "failures": failures, "fingerprints": fps}


def run_benchmark(args) -> int:
    reference_out = getattr(args, "out", "results_v2")
    with cs.RunLock(reference_out, label="bench_scheduler --benchmark-scheduler"):
        return _run_benchmark_body()


def _run_benchmark_body() -> int:
    specs = [cs.CellSpec(analysis=a, dataset=d, seed=s, full=True,
                         out_dir=str(BENCH_OUT),
                         config_hash=cs.compute_config_hash(a, d, s, True))
             for a, d, s in BENCH_CELLS]

    print(f"benchmark: {len(specs)} representative cells\n")
    configs = [1, 2, 3]
    runs = []
    for w in configs:
        print(f"--- {w} worker(s) ---", flush=True)
        r = _run_config(specs, w)
        runs.append(r)
        print(f"    wall {r['elapsed_s']:7.1f}s   peak RAM {r['peak_ram_gb']:5.1f} GB"
              f"   peak GPU {r['peak_gpu_mb']:7.0f} MB   failures {r['failures']}",
              flush=True)

    base_t = runs[0]["elapsed_s"]
    print(f"\n{'workers':>8} {'wall (s)':>10} {'speedup':>9} {'peak RAM':>10} "
          f"{'peak GPU':>10} {'fails':>6}")
    for r in runs:
        print(f"{r['workers']:>8} {r['elapsed_s']:>10.1f} "
              f"{base_t / r['elapsed_s']:>8.2f}x {r['peak_ram_gb']:>9.1f}G "
              f"{r['peak_gpu_mb']:>9.0f}M {r['failures']:>6}")

    ref = runs[0]["fingerprints"]
    identical = all(r["fingerprints"] == ref for r in runs[1:])
    print(f"\nresult fingerprints identical across worker counts: "
          f"{'YES' if identical else 'NO'}")
    if not identical:
        print("  -> parallel execution changed results; do not raise worker count")

    out = ROOT / "logs" / "scheduler_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(runs, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0 if identical else 1


if __name__ == "__main__":
    if "--gate" in sys.argv:
        sys.exit(reproducibility_gate())
    class _A:
        pass
    sys.exit(run_benchmark(_A()))
