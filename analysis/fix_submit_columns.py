#!/usr/bin/env python3
"""
Fix bas_submit.tsv: add missing columns (cleaned_sequence, precursor_mz,
precursor_charge, q_value) by rebuilding from the individual TSV files
(which contain both target & decoy rows) and joining against source parquets.

Usage:
    python fix_submit_columns.py \
        --tsv_dir /path/to/tsv_files \
        --parquet_dir /home/yiwen/AIPC/test_data/bas_test_dataset \
        --output /path/to/bas_submit_fixed.tsv
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm


def clean_sequence(seq):
    if pd.isna(seq):
        return ""
    s = str(seq)
    s = re.sub(r"\[[^\]]*\]", "", s)
    s = s.replace("n", "")
    return s


def calculate_q_values(scores, labels):
    """Compute per-PSM q-value via target-decoy competition (input sorted by score desc)."""
    targets = np.array(labels, dtype=float)
    cum_targets = np.cumsum(targets)
    cum_decoys = np.cumsum(1.0 - targets)
    fdr = np.where(cum_targets > 0, cum_decoys / cum_targets, 0.0)
    return np.minimum.accumulate(fdr[::-1])[::-1]


def load_parquet_columns(parquet_dir: str, fname: str) -> pd.DataFrame:
    """Load scan_number, modified_sequence, precursor_mz, charge from one parquet."""
    path = os.path.join(parquet_dir, fname + "_benchmark.parquet")
    df = pd.read_parquet(path, columns=[
        "scan_number", "precursor_mz", "charge", "precursor_sequence",
    ])
    df = df.rename(columns={
        "charge": "precursor_charge",
        "precursor_sequence": "modified_sequence",
    })
    # dedup: one row per (scan_number, modified_sequence)
    df = df.drop_duplicates(subset=["scan_number", "modified_sequence"])
    df["cleaned_sequence"] = df["modified_sequence"].map(clean_sequence)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Fix submit TSV: rebuild with all columns from TSV + parquet"
    )
    parser.add_argument(
        "--tsv_dir",
        required=True,
        help="Directory containing individual TSV files (with target+decoy)",
    )
    parser.add_argument(
        "--parquet_dir",
        default="/home/yiwen/AIPC/test_data/bas_test_dataset",
        help="Directory containing source parquet files",
    )
    parser.add_argument("--output", required=True, help="Output path for fixed submit TSV")
    args = parser.parse_args()

    tsv_files = sorted(glob.glob(os.path.join(args.tsv_dir, "bas_*_benchmark.tsv")))
    if len(tsv_files) != 60:
        print(f"Expected 60 TSV files, found {len(tsv_files)}", file=sys.stderr)
    print(f"Found {len(tsv_files)} TSV files")

    # 1) Load all TSV, join with parquet, compute q_value per file then merge globally
    all_frames = []
    for tsv_path in tqdm(tsv_files, desc="Processing"):
        stem = os.path.splitext(os.path.basename(tsv_path))[0]
        # stem = e.g. bas_a_testdata_0_benchmark
        fname = stem[:-len("_benchmark")]  # bas_a_testdata_0

        # Load TSV
        tsv_df = pd.read_csv(tsv_path, sep="\t")

        # Join with parquet for missing columns
        pq = load_parquet_columns(args.parquet_dir, fname)
        tsv_df = tsv_df.merge(
            pq[["scan_number", "modified_sequence", "cleaned_sequence",
                "precursor_mz", "precursor_charge"]],
            on=["scan_number", "modified_sequence"],
            how="left",
        )

        tsv_df["filename"] = fname
        all_frames.append(tsv_df)

    merged = pd.concat(all_frames, ignore_index=True)
    print(f"\nTotal PSMs (target+decoy): {len(merged):,}")

    # 2) Compute global q_value (sort all by score descending, compute FDR across all)
    print("Computing global q-values...")
    merged = merged.sort_values("score", ascending=False).reset_index(drop=True)
    merged["q_value"] = calculate_q_values(
        merged["score"].to_numpy(), merged["label"].to_numpy()
    )

    # 3) Filter to target only
    targets = merged[merged["label"] == 1].copy()
    print(f"Target PSMs: {len(targets):,}")

    # 4) Dedup by (filename, scan_number), keeping highest score
    targets = targets.sort_values("score", ascending=False).drop_duplicates(
        subset=["filename", "scan_number"]
    )
    print(f"After dedup by scan_number: {len(targets):,}")

    # 5) Reorder and save
    target_order = [
        "cleaned_sequence",
        "precursor_mz",
        "precursor_charge",
        "modified_sequence",
        "label",
        "score",
        "q_value",
        "scan_number",
        "filename",
    ]
    targets = targets[[c for c in target_order if c in targets.columns]]
    targets.to_csv(args.output, sep="\t", index=False)

    n_1fdr = (targets["q_value"] <= 0.01).sum()
    print(f"\nq_value <= 0.01: {n_1fdr:,} / {len(targets):,}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
