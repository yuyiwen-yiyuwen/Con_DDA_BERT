"""
功能：从adv hard run zip文件读取parquet，Sage-FP取交集+1% q-value过滤，target:decoy=1:1，全部作为train输出pkl.gz
输入：
    --zip_root /home/yiwen/AIPC/scripts/organized_attantion/data/dataset/hard_adv/adv_hard_run_20%30%_easy_run_20%66.7%
    --work_dir /home/yiwen/AIPC/scripts/attantion
输出：
    data/dataset/adv_hard_decoy/mzml_adv_hard_run_20%30%_easy_run_20%66.7%/train/*.pkl.gz
"""

import argparse
import concurrent.futures
import glob
import gzip
import io
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import zipfile

ZIP_ROOT = "/home/yiwen/AIPC/scripts/organized_attantion/data/dataset/hard_adv/adv_hard_run_20%30%_easy_run_20%66.7%"
OUT_BASE = "/home/yiwen/AIPC/scripts/organized_attantion/data/dataset/adv_hard_decoy/mzml_adv_hard_run_20%30%_easy_run_20%66.7%"


def parse_args():
    parser = argparse.ArgumentParser(description="从 zip 文件提取 PSM → 8:2 切分 → pkl 编码")
    parser.add_argument("--zip_root", type=str, default=ZIP_ROOT)
    parser.add_argument("--work_dir", type=str, default="/home/yiwen/AIPC/scripts/attantion")
    parser.add_argument("--psm_out_dir", type=str, default=os.path.join(OUT_BASE, "_psm_tmp"))
    parser.add_argument("--split_out_dir", type=str, default=os.path.join(OUT_BASE, "_split_tmp"))
    parser.add_argument("--pkl_out_dir", type=str, default=OUT_BASE)
    parser.add_argument("--config", type=str, default="/home/yiwen/AIPC/scripts/attantion/model.yaml")
    parser.add_argument("--rows_per_pkl", type=int, default=1000000)
    parser.add_argument("--ncores", type=int, default=8)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--q_thresh", type=float, default=0.01)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--skip_pkl", action="store_true")
    parser.add_argument("--keep_intermediate", action="store_true")
    return parser.parse_args()


def run_cmd(cmd, cwd=None):
    print("执行命令:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def extract_zip_to_temp(zip_path: str, tmp_root: str) -> str:
    """解压单个 zip 到临时子目录，返回样本目录路径。"""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = zf.namelist()
        # zip 内结构: AIPC_data_XX/AIPC_data_XX_{sage,fp,raw}.parquet
        sample_name = names[0].split('/')[0]
        out_dir = os.path.join(tmp_root, sample_name)
        os.makedirs(out_dir, exist_ok=True)
        zf.extractall(tmp_root)
    return out_dir


def collect_triplets_from_zips(zip_root: str, tmp_root: str):
    """解压所有 zip，返回 (sample_dir, raw_path, sage_path, fp_path) 列表。"""
    zip_files = sorted(glob.glob(os.path.join(zip_root, "*.zip")))
    if not zip_files:
        raise RuntimeError(f"未在 {zip_root} 下找到 zip 文件")

    triplets = []
    for zf_path in zip_files:
        try:
            sample_dir = extract_zip_to_temp(zf_path, tmp_root)
        except Exception as e:
            print(f"  跳过 {os.path.basename(zf_path)}: {e}")
            continue

        sample_name = os.path.basename(sample_dir)
        sage = os.path.join(sample_dir, f"{sample_name}_sage.parquet")
        fp = os.path.join(sample_dir, f"{sample_name}_fp.parquet")
        raw = os.path.join(sample_dir, f"{sample_name}_rawspectrum.parquet")

        if os.path.exists(sage) and os.path.exists(fp) and os.path.exists(raw):
            triplets.append((sample_name, raw, sage, fp))
        else:
            for p in [sage, fp, raw]:
                if not os.path.exists(p):
                    print(f"  警告: 缺少文件 {p}")

    return triplets


def process_single_sample(raw_path, sage_path, fp_path, out_parquet, q_thresh=0.01):
    """
    与原 split_data_mzml.py 逻辑一致，唯一差异：
    启用 Sage-FP 交集后的 1% q-value 过滤（原脚本中此处被注释掉）。
    """
    # 1. 加载数据
    raw_df = pd.read_parquet(raw_path)
    raw_df['scan'] = raw_df['scan'].astype(int)

    sage_df = pd.read_parquet(sage_path)
    sage_df['scan'] = sage_df['scan'].astype(int)
    sage_df['psm_id'] = sage_df['scan'].astype(str) + '_' + sage_df['precursor_sequence']

    fp_df = pd.read_parquet(fp_path)
    if 'scan' in fp_df.columns:
        fp_df['psm_id'] = fp_df['scan'].astype(int).astype(str) + '_' + fp_df['detect_sequence']
        fp_df = fp_df.drop(columns=['scan'])
    else:
        if 'psm_id' not in fp_df.columns:
            raise ValueError(f"FP parquet missing 'scan' or 'psm_id' columns: {fp_path}")

    # 2. Target: Sage(label=1, spectrum_q<=1%) ∩ FP(q-value<=1%)
    fp_df_target = fp_df[fp_df['q-value'] <= q_thresh]
    sage_df_target = sage_df[(sage_df['label'] == 1) & (sage_df['spectrum_q'] <= q_thresh)]
    targets = sage_df_target.merge(fp_df_target, on='psm_id', how='inner', suffixes=('', '_fp'))

    # 3. Decoy: label=0, 一半高分 + 一半随机, 与 target 1:1
    sage_decoy = sage_df[sage_df['label'] == 0]
    decoy_num = len(targets)
    decoy_sorted = sage_decoy.sort_values(by='sage_discriminant_score', ascending=False).reset_index(drop=True)

    if len(decoy_sorted) <= decoy_num:
        decoys = decoy_sorted
    else:
        half = int(decoy_num / 2)
        decoy_high = decoy_sorted.iloc[:half]
        decoy_low = decoy_sorted.iloc[half:].sample(n=half, random_state=42)
        decoys = pd.concat([decoy_high, decoy_low], ignore_index=True)

    # 4. Label / Weight / Unmask
    targets['label'] = 1
    targets['weight'] = 1.0
    decoys['label'] = 0
    decoys['weight'] = 1.0
    targets['unmask'] = 0
    decoys['unmask'] = 0

    # 5. 合并 & 关联 raw spectrum
    combined = pd.concat([targets, decoys], ignore_index=True)
    final_df = combined.merge(raw_df, on='scan', how='inner')

    # 6. 序列清洗
    final_df['cleaned_sequence'] = (
        final_df['precursor_sequence'].astype(str)
        .str.replace('n[42]', '', regex=False)
        .str.replace('N[.98]', 'N', regex=False)
        .str.replace('Q[.98]', 'Q', regex=False)
        .str.replace('M[15.99]', 'M', regex=False)
        .str.replace('C[57.02]', 'C', regex=False)
    )
    final_df['sequence_len'] = final_df['cleaned_sequence'].apply(len)

    # 7. 过滤 & 选列
    final_df = final_df[(final_df['sequence_len'] >= 7) & (final_df['sequence_len'] <= 50)]
    final_df = final_df[(final_df['charge'] >= 2) & (final_df['charge'] <= 5)]

    cols_to_keep = [
        'scan', 'precursor_mz', 'charge', 'rt', 'mz_array', 'intensity_array',
        'precursor_sequence', 'label', 'weight', 'unmask',
        'predicted_rt', 'delta_rt', 'sage_discriminant_score', 'spectrum_q'
    ]
    existing_cols = [c for c in cols_to_keep if c in final_df.columns]
    final_df = final_df[existing_cols]
    for rt_col in ("predicted_rt", "delta_rt", "delta_rt_model"):
        if rt_col in final_df.columns:
            final_df[rt_col] = 0.0

    final_df.to_parquet(out_parquet, index=False)


def run_gen_one(gen_script, raw, sage, fp, out_parquet, q_thresh=0.01):
    try:
        process_single_sample(raw, sage, fp, out_parquet, q_thresh)
    except Exception as e:
        print(f"Error processing {raw}: {e}")
        raise e


class ParquetChunkWriter:
    def __init__(self, out_dir: str, prefix: str, chunk_rows: int):
        self.out_dir = out_dir
        self.prefix = prefix
        self.chunk_rows = chunk_rows
        self.buffers = []
        self.buffer_rows = 0
        self.file_idx = 0
        os.makedirs(out_dir, exist_ok=True)

    def _flush_if_needed(self):
        if self.buffer_rows < self.chunk_rows:
            return
        merged = pd.concat(self.buffers, ignore_index=True)
        while len(merged) >= self.chunk_rows:
            chunk = merged.iloc[:self.chunk_rows]
            out_file = os.path.join(self.out_dir, f"{self.prefix}.{self.file_idx:05d}.parquet")
            chunk.to_parquet(out_file, index=False)
            self.file_idx += 1
            merged = merged.iloc[self.chunk_rows:].reset_index(drop=True)
        self.buffers = [merged] if len(merged) > 0 else []
        self.buffer_rows = len(merged)

    def add(self, df: pd.DataFrame):
        if len(df) == 0:
            return
        self.buffers.append(df.reset_index(drop=True))
        self.buffer_rows += len(df)
        self._flush_if_needed()

    def finalize(self):
        if self.buffer_rows == 0:
            return
        merged = pd.concat(self.buffers, ignore_index=True)
        out_file = os.path.join(self.out_dir, f"{self.prefix}.{self.file_idx:05d}.parquet")
        merged.to_parquet(out_file, index=False)
        self.file_idx += 1
        self.buffers = []
        self.buffer_rows = 0


def normalize_columns_for_convert(parquet_dir: str):
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not files:
        return 0
    fixed = 0
    for fp in files:
        df = pd.read_parquet(fp)
        changed = False
        if "delta_rt_model" not in df.columns:
            df["delta_rt_model"] = 0.0
            changed = True
        elif (df["delta_rt_model"] != 0).any():
            df["delta_rt_model"] = 0.0
            changed = True
        if "delta_rt" in df.columns and (df["delta_rt"] != 0).any():
            df["delta_rt"] = 0.0
            changed = True
        if "predicted_rt" not in df.columns:
            df["predicted_rt"] = 0.0
            changed = True
        elif (df["predicted_rt"] != 0).any():
            df["predicted_rt"] = 0.0
            changed = True
        if "weight" not in df.columns:
            df["weight"] = 1.0
            changed = True
        if changed:
            df.to_parquet(fp, index=False)
            fixed += 1
    return fixed


def validate_stage12_outputs(psm_out_dir: str, expected_case_num: int):
    psm_files = sorted(glob.glob(os.path.join(psm_out_dir, "*.parquet")))
    if len(psm_files) != expected_case_num:
        raise RuntimeError(f"步骤1.2输出文件数异常: 期望={expected_case_num}, 实际={len(psm_files)}")
    total_rows = 0
    empty_files = []
    for fp in psm_files:
        n_rows = len(pd.read_parquet(fp, columns=["scan"]))
        total_rows += n_rows
        if n_rows == 0:
            empty_files.append(os.path.basename(fp))
    if empty_files:
        raise RuntimeError("步骤1.2存在空parquet文件: " + ", ".join(empty_files[:20]))
    print(f"1.2 总数验证通过: 文件数={len(psm_files)}, 总行数={total_rows}")
    return total_rows


def main():
    args = parse_args()

    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(args.psm_out_dir, exist_ok=True)
    os.makedirs(args.split_out_dir, exist_ok=True)
    os.makedirs(args.pkl_out_dir, exist_ok=True)

    gen_script = os.path.join(work_dir, "1_gen_parquet.py")
    convert_script = os.path.join(work_dir, "3_convert_parquet2pkl.py")
    for p in [gen_script, convert_script]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"脚本不存在: {p}")

    # 创建临时目录用于解压 zip
    tmp_root = tempfile.mkdtemp(prefix="mzml_adv_hard_run_")
    print(f"临时解压目录: {tmp_root}")

    try:
        # =========================================================================
        print("步骤1.2: 解压 zip → 生成 PSM parquet...")
        # =========================================================================
        triplets = collect_triplets_from_zips(args.zip_root, tmp_root)
        if not triplets:
            raise RuntimeError("未找到可用的样本三元组")
        print(f"发现可处理样本数: {len(triplets)}")

        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            for case_dir, raw, sage, fp in triplets:
                out_file = os.path.join(args.psm_out_dir, f"{case_dir}.parquet")
                futures.append(executor.submit(
                    run_gen_one, gen_script, raw, sage, fp, out_file, args.q_thresh))
            for i, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
                fut.result()
                if i % 10 == 0 or i == len(futures):
                    print(f"1.2 已完成: {i}/{len(futures)}")

        stage12_total_rows = validate_stage12_outputs(args.psm_out_dir, len(triplets))

        # =========================================================================
        print("步骤1.3: 合并所有 PSM 并重组为大 parquet（全部作为 train）...")
        # =========================================================================
        psm_files = sorted(glob.glob(os.path.join(args.psm_out_dir, "*.parquet")))
        if not psm_files:
            raise RuntimeError("1.2 未生成任何 parquet 文件")

        if os.path.exists(args.split_out_dir):
            shutil.rmtree(args.split_out_dir)
        os.makedirs(args.split_out_dir, exist_ok=True)

        train_dir = os.path.join(args.split_out_dir, "train")
        os.makedirs(train_dir, exist_ok=True)

        train_writer = ParquetChunkWriter(train_dir, "train", args.rows_per_pkl)

        for fp in psm_files:
            df = pd.read_parquet(fp)
            df = df.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)
            train_writer.add(df)

        train_writer.finalize()
        print(f"1.3 完成: train={len(glob.glob(os.path.join(train_dir, '*.parquet')))} 文件")

        stage13_train_rows = sum(
            len(pd.read_parquet(fp, columns=["scan"]))
            for fp in glob.glob(os.path.join(train_dir, "*.parquet")))
        print(f"1.3 总数: train={stage13_train_rows}")
        if stage13_train_rows != stage12_total_rows:
            raise RuntimeError(f"行数不一致: stage12={stage12_total_rows}, stage13={stage13_train_rows}")

        if args.skip_pkl:
            print("已跳过 pkl 编码。")
            return

        # =========================================================================
        print("步骤1.4: 编码为 pkl.gz...")
        # =========================================================================
        train_fixed = normalize_columns_for_convert(train_dir)
        print(f"1.4 字段标准化: train修复={train_fixed}")

        run_cmd([sys.executable, convert_script,
                 "--file_dir", train_dir, "--config", args.config,
                 "--task_name", "train", "--ncores", str(args.ncores),
                 "--save_dir", os.path.join(args.pkl_out_dir, "train")], cwd=work_dir)

        print("全流程完成: zip解压 → PSM提取(q≤1%) → 合并为train → pkl.gz")

    finally:
        # 清理临时解压目录
        if os.path.exists(tmp_root):
            shutil.rmtree(tmp_root, ignore_errors=True)
            print(f"已清理临时目录: {tmp_root}")

        # 清理中间 parquet 目录（除非指定保留）
        if not args.keep_intermediate:
            for d in [args.psm_out_dir, args.split_out_dir]:
                if os.path.exists(d):
                    shutil.rmtree(d, ignore_errors=True)
                    print(f"已清理中间目录: {d}")


if __name__ == "__main__":
    main()
