import argparse
import os
import re

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="将 tims 单文件 parquet 转为训练用逐PSM parquet")
    parser.add_argument("--in_parquet", type=str, required=True, help="输入 tims parquet")
    parser.add_argument("--out_parquet", type=str, required=True, help="输出 parquet")
    parser.add_argument(
        "--q_value_max",
        type=float,
        default=0.2,
        help="Sage target 的 q-value 上限，默认 0.2",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="随机种子，控制 decoy 采样可复现",
    )
    return parser.parse_args()


def _to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    if pd.isna(x):
        return []
    return [x]


def _clean_sequence(seq: str) -> str:
    if seq is None:
        return ""
    s = str(seq)
    s = s.replace("n[42]", "")
    s = s.replace("N[.98]", "N").replace("Q[.98]", "Q")
    s = s.replace("M[15.99]", "M").replace("C[57.02]", "C")
    s = re.sub(r"\[[^\]]+\]", "", s)
    return s


def _iter_candidate_records(df: pd.DataFrame):
    base_cols = ["scan", "precursor_mz", "rt", "mz_array", "intensity_array"]
    cand_cols = [
        "charge",
        "peptide",
        "label",
        "predicted_rt",
        "spectrum_q",
        "sage_discriminant_score",
    ]

    missing = [c for c in (base_cols + cand_cols) if c not in df.columns]
    if missing:
        raise ValueError(f"输入文件缺少必要列: {missing}")

    for row in df.itertuples(index=False):
        base = {
            "scan": int(getattr(row, "scan")),
            "precursor_mz": getattr(row, "precursor_mz"),
            "rt": getattr(row, "rt"),
            "mz_array": getattr(row, "mz_array"),
            "intensity_array": getattr(row, "intensity_array"),
        }

        charges = _to_list(getattr(row, "charge"))
        peptides = _to_list(getattr(row, "peptide"))
        labels = _to_list(getattr(row, "label"))
        predicted_rts = _to_list(getattr(row, "predicted_rt"))
        spectrum_qs = _to_list(getattr(row, "spectrum_q"))
        scores = _to_list(getattr(row, "sage_discriminant_score"))

        cand_len = min(
            len(charges),
            len(peptides),
            len(labels),
            len(predicted_rts),
            len(spectrum_qs),
            len(scores),
        )
        if cand_len == 0:
            continue

        for i in range(cand_len):
            pep = peptides[i]
            if pep is None or (isinstance(pep, float) and np.isnan(pep)):
                continue

            q_val = pd.to_numeric(spectrum_qs[i], errors="coerce")
            score = pd.to_numeric(scores[i], errors="coerce")
            chg = pd.to_numeric(charges[i], errors="coerce")
            prt = pd.to_numeric(predicted_rts[i], errors="coerce")
            if pd.isna(q_val) or pd.isna(score) or pd.isna(chg):
                continue

            # 关键修复：delta_rt 由 scan级 rt 与候选 predicted_rt 现场重算，
            # 不再信任源文件中的 delta_rt 列（该列在 tims 源数据里常与 predicted_rt 重复）。
            scan_rt = pd.to_numeric(base["rt"], errors="coerce")
            if pd.isna(scan_rt) or pd.isna(prt):
                drt = 0.0
            else:
                drt = float(scan_rt) - float(prt)

            rec = {
                "precursor_sequence": str(pep),
                "charge": float(chg),
                "predicted_rt": float(prt) if not pd.isna(prt) else 0.0,
                "delta_rt": drt,
                "spectrum_q": float(q_val),
                "sage_discriminant_score": float(score),
                "is_target": bool(labels[i]),
            }
            rec.update(base)
            yield rec


def process_single_tims_file(in_parquet, out_parquet):
    df = pd.read_parquet(in_parquet)
    all_records = []

    for rec in _iter_candidate_records(df):
        all_records.append(rec)

    if not all_records:
        # 保存一个带有正确列的空 DataFrame
        pd.DataFrame(columns=[
            "scan", "precursor_mz", "rt", "mz_array", "intensity_array",
            "precursor_sequence", "charge", "predicted_rt", "delta_rt",
            "spectrum_q", "sage_discriminant_score", "label", "weight", "unmask"
        ]).to_parquet(out_parquet, index=False)
        return

    combined = pd.DataFrame(all_records)
    
    # 设置 label: is_target 为 True 标为 1，False 标为 0
    combined["label"] = combined["is_target"].map({True: 1, False: 0}).astype(int)
    combined["weight"] = 1.0
    combined["unmask"] = 0

    combined["cleaned_sequence"] = combined["precursor_sequence"].apply(_clean_sequence)
    combined["sequence_len"] = combined["cleaned_sequence"].astype(str).apply(len)

    # 保留长度和电荷的基础物理性质过滤，去掉 q_value 过滤
    combined = combined[(combined["sequence_len"] >= 7) & (combined["sequence_len"] <= 50)]
    combined = combined[(combined["charge"] >= 2) & (combined["charge"] <= 5)]

    cols_to_keep = [
        "scan",
        "precursor_mz",
        "charge",
        "rt",
        "mz_array",
        "intensity_array",
        "precursor_sequence",
        "label",
        "weight",
        "unmask",
        "predicted_rt",
        "delta_rt",
        "sage_discriminant_score",
        "spectrum_q",
    ]

    final_df = combined[[c for c in cols_to_keep if c in combined.columns]].copy()
    final_df.to_parquet(out_parquet, index=False)


def main():
    args = parse_args()

    in_parquet = os.path.abspath(args.in_parquet)
    out_parquet = os.path.abspath(args.out_parquet)
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)

    process_single_tims_file(
        in_parquet=in_parquet,
        out_parquet=out_parquet,
    )


if __name__ == "__main__":
    main()
