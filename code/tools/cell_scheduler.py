"""Resource-aware scheduler for the reviewer-experiment cells.

Execution only. The scientific definitions live in run_experiments.py and
run_reviewer_experiments.py and are imported unchanged: this module decides
*when* and *in which process* a cell runs, never *what* it computes.

Design notes
------------
Measured on this host (20 physical cores, 68 GB RAM, RTX 5070 12.8 GB):

  * cells are independent and already write one CSV each, so the existing output
    layout is concurrency-safe by construction;
  * the GPU sits at 8-15% utilisation and ~3 GB, so CUDA is not the bottleneck.
    A separate GPU queue would serialise work for no gain; what is needed is a
    cap on how many CUDA-using cells run at once, for memory safety;
  * dataset parsing costs at most 2.8 s (CSE-CIC), so caching loads is noise.

Numerical safety
----------------
scikit-learn estimators are deterministic irrespective of ``n_jobs``, so running
cells concurrently cannot change their results. XGBoost and LightGBM are *not*:
thread count changes float summation order. Those cells are therefore given the
``exclusive`` resource class and run alone with their thread settings untouched,
so their numbers stay comparable with results already produced.

Completion contract
-------------------
A cell counts as done only when its CSV exists, parses, has rows, and a sidecar
marker records the configuration hash. Results produced before this module
existed have no marker; they are adopted on first sight and recorded as such
rather than recomputed.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


# --------------------------------------------------------------------------- #
# resource classification (measured, not inferred from names)
# --------------------------------------------------------------------------- #

#: Cells whose numerics depend on native thread count and cannot be pinned.
#: `gbm` used to live here: boosted ensembles are not thread-count invariant, so
#: with n_jobs=-1 their results depended on the host core count. Rather than
#: serialise them, run_reviewer_experiments now pins GBM_THREADS to a fixed value,
#: which makes them both reproducible across machines and safe to run in parallel.
#: Verified by running the same cell twice concurrently and comparing exactly.
EXCLUSIVE_ANALYSES: set[str] = set()

#: cells that train several MLPs and so hold non-trivial CUDA memory
GPU_HEAVY_ANALYSES = {"surrogate"}

#: every other analysis is CPU-dominated with brief CUDA bursts
def resource_class(analysis: str) -> str:
    if analysis in EXCLUSIVE_ANALYSES:
        return "exclusive"
    if analysis in GPU_HEAVY_ANALYSES:
        return "gpu"
    return "cpu"


# --------------------------------------------------------------------------- #
# cell specification
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CellSpec:
    analysis: str
    dataset: str
    seed: int
    full: bool
    out_dir: str
    config_hash: str = field(default="")

    @property
    def cell_id(self) -> str:
        return f"{self.analysis}__{self.dataset}__seed{self.seed}"

    @property
    def result_path(self) -> Path:
        # unchanged from the original layout, so existing results stay valid
        return (Path(self.out_dir) / self.dataset / "reviewer"
                / f"seed_{self.seed}" / f"{self.analysis}.csv")

    @property
    def marker_path(self) -> Path:
        return Path(self.out_dir) / "_cells" / f"{self.cell_id}.done"

    @property
    def claim_path(self) -> Path:
        return Path(self.out_dir) / "_cells" / f"{self.cell_id}.claim"

    @property
    def log_path(self) -> Path:
        return CODE_DIR.parent / "logs" / "cells" / f"{self.cell_id}.log"

    @property
    def resource(self) -> str:
        return resource_class(self.analysis)


def compute_config_hash(analysis: str, dataset: str, seed: int, full: bool) -> str:
    """Hash every input that can change a cell's result, including its own code."""
    import run_experiments as base
    import run_reviewer_experiments as rex

    fn = rex.SPLIT_OWNING.get(analysis) or rex.SPLIT_TAKING.get(analysis)
    try:
        fn_src = inspect.getsource(fn) if fn else ""
    except (OSError, TypeError):
        fn_src = ""

    payload = {
        "analysis": analysis, "dataset": dataset, "seed": seed, "full": full,
        "attack_configs": base.ATTACK_CONFIGS,
        "defense_configs": base.DEFENSE_CONFIGS,
        "mlp_config": base.MLP_CONFIG,
        "ml_models_config": base.ML_MODELS_CONFIG,
        "test_size": base.TEST_SIZE,
        "eps_values": list(rex.EPS_VALUES),
        # code of the experiment plus the shared split builder
        "fn_src": fn_src,
        "prepare_split_src": inspect.getsource(base.prepare_split),
        "select_features_src": inspect.getsource(base.select_features),
        "load_raw_src": inspect.getsource(base.load_raw),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# completion / validation
# --------------------------------------------------------------------------- #

def validate_result(path: Path) -> tuple[bool, str]:
    """A result is valid when it exists, parses, and holds at least one row."""
    if not path.exists():
        return False, "missing"
    if path.stat().st_size == 0:
        return False, "empty file"
    try:
        n = len(pd.read_csv(path))
    except Exception as e:  # truncated or corrupt
        return False, f"unreadable: {type(e).__name__}"
    return (n > 0, "ok" if n > 0 else "no rows")


def cell_status(spec: CellSpec) -> tuple[str, str]:
    """Return (status, detail) with status in {done, adopt, pending}."""
    valid, detail = validate_result(spec.result_path)
    if not valid:
        return "pending", detail

    if not spec.marker_path.exists():
        # produced before this scheduler existed; keep it rather than recompute
        return "adopt", "result predates the scheduler"

    try:
        marker = json.loads(spec.marker_path.read_text(encoding="utf-8"))
    except Exception:
        return "pending", "marker unreadable"

    if marker.get("config_hash") != spec.config_hash:
        return "pending", (f"config changed "
                           f"({marker.get('config_hash')} -> {spec.config_hash})")
    return "done", "ok"


def write_marker(spec: CellSpec, duration_s: float, adopted: bool = False) -> None:
    spec.marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cell_id": spec.cell_id,
        "analysis": spec.analysis,
        "dataset": spec.dataset,
        "seed": spec.seed,
        "config_hash": spec.config_hash,
        "result_sha256": sha256_file(spec.result_path),
        "n_rows": int(len(pd.read_csv(spec.result_path))),
        "duration_s": round(duration_s, 2),
        "adopted_pre_scheduler": adopted,
        "completed_unix": int(time.time()),
    }
    atomic_write_text(spec.marker_path, json.dumps(payload, indent=2))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    """Write, flush, validate, then rename, so no reader can see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)
        f.flush()
        os.fsync(f.fileno())
    # validate the temporary file before it becomes the real result
    check = pd.read_csv(tmp)
    if len(check) != len(df):
        tmp.unlink(missing_ok=True)
        raise IOError(f"validation failed for {path}: "
                      f"wrote {len(df)} rows, read back {len(check)}")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# exclusive claims, so two runners cannot execute the same cell
# --------------------------------------------------------------------------- #

class RunLock:
    """Cross-process guard against two independent scheduler processes at once.

    Root cause of a real incident: several "cpu"-class cells (hsj_budget,
    tuning, family, selection, resampling, linear_wb) call train_baselines(),
    which trains an MLP on the GPU as one of the four baseline models even when
    the experiment never attacks it, so the "cpu" queue was never GPU-free.
    That alone is not the problem: a benchmark with up to 3 concurrent workers
    under ONE process tree, several of them training that MLP, completed
    cleanly with no hangs and a GPU peak of ~3 GB. The problem was launching a
    SECOND, independent process tree (a verification script with its own
    ProcessPoolExecutor) while the scheduler's pool was already running. Two
    uncoordinated pools touching CUDA at once deadlocked three cells at 0% CPU
    for hours.

    The fix is therefore scoped to orchestration, not to individual cells:
    refuse to start a second scheduler/benchmark run while one is already
    active. This preserves the intra-pool concurrency already proven safe by
    the benchmark and the reproducibility gate; it does not throttle any cell.
    """

    STALE_AFTER_S = 8 * 3600  # longer than any observed real run

    def __init__(self, out_dir: str, label: str):
        self.path = Path(out_dir) / "_cells" / "_run.lock"
        self.label = label
        self.acquired = False

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            age = time.time() - self.path.stat().st_mtime
            if age > self.STALE_AFTER_S:
                self.path.unlink(missing_ok=True)
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {self.label}".encode())
            os.close(fd)
            self.acquired = True
        except FileExistsError:
            holder = self.path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(
                f"another scheduler run is already active ({holder}); "
                f"do not launch a second process against the same out_dir. "
                f"If that run has genuinely finished or crashed, remove "
                f"{self.path} and retry.")
        return self

    def __exit__(self, *exc) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)


class CellClaim:
    """Best-effort exclusive claim via atomic O_EXCL creation."""

    STALE_AFTER_S = 6 * 3600

    def __init__(self, spec: CellSpec):
        self.spec = spec
        self.acquired = False

    def __enter__(self) -> "CellClaim":
        p = self.spec.claim_path
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            age = time.time() - p.stat().st_mtime
            if age > self.STALE_AFTER_S:
                p.unlink(missing_ok=True)     # abandoned by a dead worker
        try:
            fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                json.dump({"pid": os.getpid(), "host": os.environ.get("COMPUTERNAME", "?"),
                           "started_unix": int(time.time())}, f)
            self.acquired = True
        except FileExistsError:
            self.acquired = False
        return self

    def __exit__(self, *exc) -> None:
        if self.acquired:
            self.spec.claim_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# cell execution (module-level so it is picklable on Windows spawn)
# --------------------------------------------------------------------------- #

#: Hard wall-clock cap per cell. Added after a real incident: hsj_budget cells
#: against RandomForest hung at 0% CPU for 14.6 hours with no other process
#: contending for the GPU or CPU, so the failure is not fully explained by the
#: earlier two-process-tree diagnosis and may be an ART/HopSkipJump boundary-
#: search edge case against RF's piecewise-constant decision surface. Rather
#: than depend on identifying the exact mechanism, every cell now runs under a
#: hard timeout, several times the slowest cell observed to complete normally
#: (worst measured: ~10.5 min), so a hang is caught in minutes, not hours.
DEFAULT_CELL_TIMEOUT_S = int(os.environ.get("ADVEVASION_CELL_TIMEOUT_S", str(45 * 60)))


def _execute_cell_target(spec_dict: dict, q) -> None:
    """multiprocessing.Process target; puts the result dict onto q."""
    try:
        q.put(execute_cell(spec_dict))
    except Exception as e:  # keep the parent from blocking on a dead child
        q.put({"cell_id": spec_dict.get("analysis", "?"), "status": "failed",
              "detail": f"worker exception: {type(e).__name__}: {e}",
              "duration_s": 0.0})


def execute_cell_with_timeout(spec_dict: dict, timeout_s: int | None = None) -> dict:
    """Run one cell in its own process; kill and report timeout if it exceeds timeout_s.

    A dedicated process per cell (rather than a shared pool) is what makes a
    hard kill possible: killing one hung task must not affect any other cell in
    flight, and a shared ProcessPoolExecutor worker can be running more than
    one task over its lifetime, so terminating it is not safe in general.
    """
    import multiprocessing as mp

    timeout_s = timeout_s or DEFAULT_CELL_TIMEOUT_S
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_execute_cell_target, args=(spec_dict, q))
    t0 = time.time()
    p.start()
    p.join(timeout_s)

    if p.is_alive():
        p.terminate()
        p.join(10)
        if p.is_alive():
            p.kill()
            p.join(5)
        spec = CellSpec(**spec_dict)
        # release the claim so the cell is retried, not left permanently locked
        spec.claim_path.unlink(missing_ok=True)
        return {"cell_id": spec.cell_id, "status": "timeout",
                "detail": f"exceeded {timeout_s}s wall-clock limit, process killed",
                "duration_s": time.time() - t0}

    if not q.empty():
        return q.get()
    spec = CellSpec(**spec_dict)
    spec.claim_path.unlink(missing_ok=True)
    return {"cell_id": spec.cell_id, "status": "failed",
            "detail": "worker exited without a result (crash?)",
            "duration_s": time.time() - t0}


def execute_cell(spec_dict: dict) -> dict:
    """Run one cell in this process. Returns a small serialisable summary."""
    spec = CellSpec(**spec_dict)
    t0 = time.time()

    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(spec.log_path, "w", encoding="utf-8", errors="replace")
    real_stdout = sys.stdout
    sys.stdout = log

    try:
        with CellClaim(spec) as claim:
            if not claim.acquired:
                return {"cell_id": spec.cell_id, "status": "skipped",
                        "detail": "claimed by another worker", "duration_s": 0.0}

            status, detail = cell_status(spec)
            if status in ("done", "adopt"):
                if status == "adopt":
                    write_marker(spec, 0.0, adopted=True)
                return {"cell_id": spec.cell_id, "status": "skipped",
                        "detail": detail, "duration_s": 0.0}

            import run_experiments as base
            import run_reviewer_experiments as rex

            base.set_seed(spec.seed)
            if spec.analysis in rex.SPLIT_OWNING:
                rows = rex.SPLIT_OWNING[spec.analysis](
                    dataset=spec.dataset, seed=spec.seed, full=spec.full,
                    out_dir=spec.out_dir)
            else:
                X_tr, X_te, y_tr, y_te, feats, n_feats, extras = base.prepare_split(
                    spec.dataset, spec.seed, full=spec.full)
                rows = rex.SPLIT_TAKING[spec.analysis](
                    X_tr, X_te, y_tr, y_te, spec.seed, extras=extras, feats=feats,
                    dataset=spec.dataset)

            if not rows:
                return {"cell_id": spec.cell_id, "status": "empty",
                        "detail": "experiment produced no rows",
                        "duration_s": time.time() - t0}

            atomic_write_csv(spec.result_path, pd.DataFrame(rows))
            dur = time.time() - t0
            write_marker(spec, dur)
            return {"cell_id": spec.cell_id, "status": "ok", "detail": "",
                    "duration_s": dur, "n_rows": len(rows)}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"cell_id": spec.cell_id, "status": "failed",
                "detail": f"{type(e).__name__}: {e}", "duration_s": time.time() - t0}
    finally:
        sys.stdout = real_stdout
        log.close()


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #

def build_plan(analyses: Iterable[str], datasets: Iterable[str],
               seeds: Iterable[int], out_dir: str, full: bool = True) -> list[CellSpec]:
    specs = []
    for a in analyses:
        for d in datasets:
            for s in seeds:
                specs.append(CellSpec(analysis=a, dataset=d, seed=int(s), full=full,
                                      out_dir=out_dir,
                                      config_hash=compute_config_hash(a, d, int(s), full)))
    return specs


def partition(specs: list[CellSpec]) -> dict[str, list[CellSpec]]:
    out: dict[str, list[CellSpec]] = {"cpu": [], "gpu": [], "exclusive": []}
    for s in specs:
        out[s.resource].append(s)
    return out
