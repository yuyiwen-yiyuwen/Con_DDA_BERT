#!/usr/bin/env python3
"""
从单个模型预测结果直接合成最终加权平均 all_pred.tsv
公式:
  mzml/wiff: [A]1/2 + [B]1/4 + [D]1/8 + [E]1/8
  tims:      [A]1/2 + [C]1/2
"""

import pandas as pd
import os

# ---- 各模型的预测文件目录 ----
A_dir = "/home/yiwen/AIPC/scripts/organized_attantion/data/results/basic_all_mzml_tims_wiff_epochX/epoch11"
BC_dir = "/home/yiwen/AIPC/scripts/organized_attantion/data/results/new_test_mzml+tims_epochX/new_test_mzml+tims_epoch12"
# [B] = BC_dir 中的 mzml_*/wiff_* 文件  (withoutRT_with_val epoch10)
# [C] = BC_dir 中的 tims_* 文件           (tims_select epoch12)
D_dir = "/home/yiwen/AIPC/scripts/organized_attantion/data/results/mzml_finetune_mzml+tims+wiff_hard_decoy_epochX/epoch2"
E_dir = "/home/yiwen/AIPC/scripts/organized_attantion/data/results/checkpoints_hard_decoy_adv_mzml_hard_run_20%30%_easy_run_20%66.7%_epochX/epoch2"

out_dir = "/home/yiwen/AIPC/scripts/organized_attantion/data/results/merged/basic_all_mzml_tims_wiff_epoch11_merged_hard_decoy_and_adv_hard_run_hard_decoy_epoch2_avg_epoch12_v2"

# ---- 主逻辑 ----
os.makedirs(out_dir, exist_ok=True)
files = sorted([f for f in os.listdir(A_dir) if f.endswith("_pred.csv")])
print(f"Total files: {len(files)}")

dfs = []
for fname in files:
    prefix = fname.split("_bas_")[0]

    dA = pd.read_csv(os.path.join(A_dir, fname), header=None, names=["index", "score"])

    if prefix in ("mzml", "wiff"):
        dB = pd.read_csv(os.path.join(BC_dir, fname), header=None, names=["index", "score"])
        dD = pd.read_csv(os.path.join(D_dir, fname), header=None, names=["index", "score"])
        dE = pd.read_csv(os.path.join(E_dir, fname), header=None, names=["index", "score"])
        merged = dA[["index"]].copy()
        merged["score"] = 0.5 * dA["score"] + 0.25 * dB["score"] + 0.125 * dD["score"] + 0.125 * dE["score"]
    else:  # tims
        dC = pd.read_csv(os.path.join(BC_dir, fname), header=None, names=["index", "score"])
        merged = dA[["index"]].copy()
        merged["score"] = 0.5 * dA["score"] + 0.5 * dC["score"]

    dfs.append(merged)

all_pred = pd.concat(dfs, ignore_index=True)
all_pred.to_csv(os.path.join(out_dir, "all_pred.tsv"), sep="\t", index=False)
print(f"Done: {out_dir}/all_pred.tsv ({len(all_pred):,} rows)")
print("  mzml/wiff: [A]1/2 + [B]1/4 + [D]1/8 + [E]1/8")
print("  tims:      [A]1/2 + [C]1/2")
