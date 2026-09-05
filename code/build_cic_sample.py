"""Build a stratified CSE-CIC-IDS-2018 sample from the Improved release.

Source: Datasets/CSECICIDS2018_improved/  (Liu & Lashkari 2024 corrections to the
original Sharafaldin/Lashkari 2018 release; addresses data-quality issues raised
by Engelen et al. 2021). 10 daily files, 91 columns, ~63.3M flows total.

Approach: stream every daily file once, maintain a per-label reservoir
(Algorithm R) so memory is bounded by the size of the desired output sample.
Output is a binary-labeled CSV in the original Kaggle column shape that
run_experiments.py already consumes — column names normalized so the existing
preprocessing pipeline works without modification.

Usage:
    python Code/build_cic_sample.py                       # ~580k flows, all 10 days
    python Code/build_cic_sample.py --target-benign 50000 --target-per-attack 5000
"""
from __future__ import annotations
from pathlib import Path
import argparse
import gc
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "Datasets" / "CSECICIDS2018_improved"
OUT = ROOT / "Datasets" / "CSE_CIC_IDS2018.csv"

ALL_DAYS = [
    "Wednesday-14-02-2018.csv",
    "Thursday-15-02-2018.csv",
    "Friday-16-02-2018.csv",
    "Tuesday-20-02-2018.csv",
    "Wednesday-21-02-2018.csv",
    "Thursday-22-02-2018.csv",
    "Friday-23-02-2018.csv",
    "Wednesday-28-02-2018.csv",
    "Thursday-01-03-2018.csv",
    "Friday-02-03-2018.csv",
]

# Identifier and metadata columns dropped before any sampling.
DROP_COLS = (
    "id", "Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp",
    "Attempted Category",
)
LABEL_COL_CANDIDATES = ("Label", "label")

CHUNK = 250_000
SEED = 42

# After dropping ID columns the improved release has 84 feature columns + 1 label.
# Standardize column names to the spelling used by run_experiments.py and the
# original Kaggle release wherever the two diverge.
IMPROVED_TO_KAGGLE = {
    "Total Fwd Packet": "Tot Fwd Pkts",
    "Total Bwd packets": "Tot Bwd Pkts",
    "Total Length of Fwd Packet": "TotLen Fwd Pkts",
    "Total Length of Bwd Packet": "TotLen Bwd Pkts",
    "Fwd Packet Length Max": "Fwd Pkt Len Max",
    "Fwd Packet Length Min": "Fwd Pkt Len Min",
    "Fwd Packet Length Mean": "Fwd Pkt Len Mean",
    "Fwd Packet Length Std": "Fwd Pkt Len Std",
    "Bwd Packet Length Max": "Bwd Pkt Len Max",
    "Bwd Packet Length Min": "Bwd Pkt Len Min",
    "Bwd Packet Length Mean": "Bwd Pkt Len Mean",
    "Bwd Packet Length Std": "Bwd Pkt Len Std",
    "Fwd Packets/s": "Fwd Pkts/s",
    "Bwd Packets/s": "Bwd Pkts/s",
    "Packet Length Min": "Pkt Len Min",
    "Packet Length Max": "Pkt Len Max",
    "Packet Length Mean": "Pkt Len Mean",
    "Packet Length Std": "Pkt Len Std",
    "Packet Length Variance": "Pkt Len Var",
    "Fwd Header Length": "Fwd Header Len",
    "Bwd Header Length": "Bwd Header Len",
    "Average Packet Size": "Pkt Size Avg",
    "Fwd Segment Size Avg": "Fwd Seg Size Avg",
    "Bwd Segment Size Avg": "Bwd Seg Size Avg",
    "Fwd Bytes/Bulk Avg": "Fwd Byts/b Avg",
    "Fwd Packet/Bulk Avg": "Fwd Pkts/b Avg",
    "Fwd Bulk Rate Avg": "Fwd Blk Rate Avg",
    "Bwd Bytes/Bulk Avg": "Bwd Byts/b Avg",
    "Bwd Packet/Bulk Avg": "Bwd Pkts/b Avg",
    "Bwd Bulk Rate Avg": "Bwd Blk Rate Avg",
    "Subflow Fwd Packets": "Subflow Fwd Pkts",
    "Subflow Bwd Packets": "Subflow Bwd Pkts",
    "FWD Init Win Bytes": "Init Fwd Win Byts",
    "Bwd Init Win Bytes": "Init Bwd Win Byts",
    "Fwd Act Data Pkts": "Fwd Act Data Pkts",
    "Fwd Seg Size Min": "Fwd Seg Size Min",
}


class Reservoir:
    """Algorithm R reservoir sampler keeping `k` items uniformly from an unknown stream."""

    def __init__(self, k: int, rng: np.random.RandomState):
        self.k = k
        self.rng = rng
        self.buf: list = []
        self.seen = 0

    def offer_batch(self, rows: list) -> None:
        for row in rows:
            self.seen += 1
            if len(self.buf) < self.k:
                self.buf.append(row)
            else:
                j = self.rng.randint(0, self.seen)
                if j < self.k:
                    self.buf[j] = row


def normalize_chunk(chunk: pd.DataFrame, label_col: str,
                    keep_timestamp: bool = False) -> pd.DataFrame:
    """Drop ID/metadata cols, repeated header rows, NaN labels; rename to Kaggle naming."""
    drop_cols = tuple(c for c in DROP_COLS if not (keep_timestamp and c == "Timestamp"))
    drop_present = [c for c in chunk.columns if c.startswith("Unnamed") or c in drop_cols]
    if drop_present:
        chunk = chunk.drop(columns=drop_present)
    chunk = chunk[chunk[label_col].astype(str) != "Label"]
    chunk = chunk.dropna(subset=[label_col])
    if label_col != "Label":
        chunk = chunk.rename(columns={label_col: "Label"})
    rename_map = {k: v for k, v in IMPROVED_TO_KAGGLE.items() if k in chunk.columns}
    if rename_map:
        chunk = chunk.rename(columns=rename_map)
    return chunk


def stream_file(path: Path, keep_timestamp: bool = False):
    label_col = None
    for chunk in pd.read_csv(path, chunksize=CHUNK, low_memory=False, encoding="utf-8"):
        if label_col is None:
            label_col = next((c for c in LABEL_COL_CANDIDATES if c in chunk.columns), None)
            if label_col is None:
                print(f"  !! {path.name}: no label column found; skipping", flush=True)
                return
        yield normalize_chunk(chunk, label_col, keep_timestamp=keep_timestamp)


def build_sample(target_benign: int, target_per_attack: int, days, out_path: Path,
                 keep_timestamp: bool = False) -> None:
    rng = np.random.RandomState(SEED)
    reservoirs: dict[str, Reservoir] = {}
    label_counts: dict[str, int] = {}
    feature_columns: list[str] | None = None

    for fname in days:
        path = SRC_DIR / fname
        if not path.exists():
            print(f"  !! missing: {path}", flush=True)
            continue
        print(f"\nstreaming {fname}…", flush=True)
        rows_seen = 0
        for chunk in stream_file(path, keep_timestamp=keep_timestamp):
            if feature_columns is None:
                feature_columns = [c for c in chunk.columns if c != "Label"]
            for c in feature_columns:
                if c not in chunk.columns:
                    chunk[c] = np.nan
            chunk = chunk[["Label"] + feature_columns]
            for label, group in chunk.groupby("Label", sort=False):
                upper = str(label).upper()
                k = target_benign if upper == "BENIGN" else target_per_attack
                key = "BENIGN" if upper == "BENIGN" else label
                if key not in reservoirs:
                    reservoirs[key] = Reservoir(k=k, rng=rng)
                reservoirs[key].offer_batch(group.values.tolist())
                label_counts[key] = label_counts.get(key, 0) + len(group)
            rows_seen += len(chunk)
            print(
                f"  {fname}: rows_seen={rows_seen:>10,} "
                f"reservoir_total={sum(len(r.buf) for r in reservoirs.values()):>9,}",
                flush=True,
            )
        gc.collect()

    print("\n=== Per-label totals seen vs reservoir size ===")
    for label in sorted(label_counts.keys(), key=lambda k: -label_counts[k]):
        print(f"  {label:48} seen={label_counts[label]:>11,}  kept={len(reservoirs[label].buf):>9,}")

    pieces = []
    for label, reservoir in reservoirs.items():
        if not reservoir.buf:
            continue
        df = pd.DataFrame(reservoir.buf, columns=["Label"] + feature_columns)
        pieces.append(df)
    out = pd.concat(pieces, ignore_index=True)
    out = out.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Map all "* - Attempted" labels (failed-attempt traffic captured by the
    # Improved release) into the same fine-grained class as the successful
    # attempt; run_experiments.py reduces everything non-BENIGN to attack=1.
    out["Label"] = out["Label"].astype(str)
    # Normalize "BENIGN" -> "Benign" so existing run_experiments.py case check matches.
    out.loc[out["Label"].str.upper() == "BENIGN", "Label"] = "Benign"

    out = out.replace([np.inf, -np.inf], np.nan)
    n0 = len(out)
    for c in out.columns:
        if c in ("Label", "Timestamp"):
            continue
        if out[c].dtype == object:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna()
    print(f"\ndropped {n0 - len(out):,} rows with NaN/Inf features")

    print(f"final shape: {out.shape}")
    print("final label distribution:")
    print(out["Label"].value_counts())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-benign", type=int, default=300_000)
    ap.add_argument("--target-per-attack", type=int, default=30_000)
    ap.add_argument("--days", nargs="+", default=ALL_DAYS)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--keep-timestamp", action="store_true",
                    help="Retain the capture Timestamp as a non-predictive column, "
                         "for the temporal train/test split ablation.")
    args = ap.parse_args()
    build_sample(args.target_benign, args.target_per_attack, args.days, Path(args.out),
                 keep_timestamp=args.keep_timestamp)


if __name__ == "__main__":
    main()
