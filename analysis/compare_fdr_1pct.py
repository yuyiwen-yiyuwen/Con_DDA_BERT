#!/usr/bin/env python3
"""
Compare 1% FDR total identification counts (target+decoy) between two
DDA/BERT benchmark directories.

Usage:
    python compare_fdr_1pct.py <dir1> <dir2> [-o output.csv]

Output CSV columns: file, <dir1_basename>, <dir2_basename>
"""

import argparse
import csv
import os
import sys
from pathlib import Path


def list_tsv(directory: str) -> list[Path]:
    return sorted(
        f for f in Path(directory).glob("*.tsv") if f.name != "summary.tsv"
    )


def read_summary_csv(directory: str) -> dict[str, int]:
    """Return {file_stem: total_psms_1fdr} from summary.csv, or None."""
    p = Path(directory) / "summary.csv"
    if not p.exists():
        return None
    out = {}
    with open(p, newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["file_stem"]] = int(row.get("total_psms_1fdr", 0))
    return out


def has_column(tsv_path: Path, col: str) -> bool:
    with open(tsv_path) as fh:
        return col in fh.readline().rstrip("\n").split("\t")


def count_qvalue(tsv_path: Path) -> int:
    """Count all PSMs (target+decoy) at 1% FDR via q_value <= 0.01."""
    n = 0
    with open(tsv_path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            try:
                if float(row["q_value"]) <= 0.01:
                    n += 1
            except (ValueError, KeyError):
                continue
    return n


def count_dir(directory: str) -> dict[str, int]:
    """Auto-detect and return {file_stem: total_psms_1fdr}."""
    dname = os.path.basename(directory.rstrip("/"))

    summary = read_summary_csv(directory)
    if summary is not None:
        print(f"[{dname}] 来源: summary.csv (total_psms_1fdr)")
        return summary

    tsvs = list_tsv(directory)
    if not tsvs:
        print(f"[{dname}] 警告: 未找到 .tsv 文件", file=sys.stderr)
        return {}

    use_q = has_column(tsvs[0], "q_value")
    print(f"[{dname}] 来源: {'q_value <= 0.01' if use_q else 'label == 1 (无FDR过滤)'}")

    results = {}
    for p in tsvs:
        if use_q:
            results[p.stem] = count_qvalue(p)
        else:
            # fallback: count all rows
            with open(p) as fh:
                results[p.stem] = sum(1 for _ in fh) - 1  # minus header
    return results


def main():
    parser = argparse.ArgumentParser(
        description="对比两个 benchmark 目录在 1% FDR 下的鉴定数目 (含 decoy)"
    )
    parser.add_argument("dir1")
    parser.add_argument("dir2")
    parser.add_argument("-o", "--output", default="comparison_1fdr.csv",
                        help="输出 CSV 路径 (默认: comparison_1fdr.csv)")
    args = parser.parse_args()

    n1 = os.path.basename(args.dir1.rstrip("/"))
    n2 = os.path.basename(args.dir2.rstrip("/"))

    c1 = count_dir(args.dir1)
    c2 = count_dir(args.dir2)

    all_stems = sorted(set(c1) | set(c2))

    with open(args.output, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", n1, n2])
        for stem in all_stems:
            w.writerow([stem, c1.get(stem, 0), c2.get(stem, 0)])

    print(f"共 {len(all_stems)} 个文件，结果已保存至: {args.output}")


if __name__ == "__main__":
    main()
