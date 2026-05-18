"""
wiff 数据全流程管线：gen_parquet → assign global PSM ID → split → convert → pkl.gz

适配自 split_data_tims.py。与 tims 版本的区别：
  - 源数据: /home/yiwen/AIPC/database/wiff
  - 输出 pkl.gz: /home/yiwen/AIPC/scripts/organized_attantion/data/dataset/wiff_all
  - 过程中确定性构建 PSM 索引（非后匹配），输出到 wiff_pkl_parquet_index/
  - 中间文件一律写入 /tmp，完成后自动清理

索引构建流程（嵌入管线）：
  1.2 gen_parquet: 每个输出行带 orig_row_idx / orig_candidate_idx
  1.25 assign IDs: 为全体 PSM parquet 分配全局递增 global_psm_id，建 id→来源 映射
  1.3 split: global_psm_id 列随数据 shuffle 后进入 train/val parquet
  1.35 extract index: 从 split parquet 读取 global_psm_id 列，联合映射写出索引
  1.4/1.5 convert: split parquet 名与 pkl.gz 名一一对应，索引自动成立
"""

import argparse
import concurrent.futures
import gc
import glob
import gzip
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

WIFF_ROOT = "/home/yiwen/AIPC/database/wiff"
WORK_DIR = "/home/yiwen/AIPC/scripts/attantion"
PROTON_MASS_AMU = 1.007276


def parse_args():
    parser = argparse.ArgumentParser(
        description="wiff 全流程管线：展平PSM → 8:2切分 → 编码 pkl.gz（过程中建索引）"
    )
    parser.add_argument("--wiff_root", type=str, default=WIFF_ROOT)
    parser.add_argument("--work_dir", type=str, default=WORK_DIR)
    parser.add_argument("--pkl_out_dir", type=str,
                        default="/home/yiwen/AIPC/scripts/organized_attantion/data/dataset/wiff_all")
    parser.add_argument("--index_dir", type=str,
                        default="/home/yiwen/AIPC/scripts/organized_attantion/data/dataset/wiff_pkl_parquet_index")
    parser.add_argument("--config", type=str,
                        default="/home/yiwen/AIPC/scripts/attantion/model.yaml")
    parser.add_argument("--tmp_root", type=str, default="")
    parser.add_argument("--rows_per_pkl", type=int, default=1000000)
    parser.add_argument("--ncores", type=int, default=8)
    parser.add_argument("--pkl_workers", type=int, default=4)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--only_case", type=str, default="")
    parser.add_argument("--skip_pkl", action="store_true")
    parser.add_argument("--start_from", type=str, default="original_parquet",
                        choices=["original_parquet", "assign_id", "split", "pkl"])
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--keep_tmp", action="store_true")
    parser.add_argument("--convert_timeout", type=int, default=900,
                        help="单文件 convert 超时秒数 (默认 900)")
    parser.add_argument("--split_dir", type=str, default="",
                        help="resume from pkl: path to existing split/ dir (containing train/ val/)")
    return parser.parse_args()


# ──────────── 工具函数 ────────────

def run_cmd(cmd, cwd=None):
    print("执行命令:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def collect_wiff_files(wiff_root: str, only_case: str):
    files = sorted(glob.glob(os.path.join(wiff_root, "*.parquet")))
    pairs = []
    for fp in files:
        case_name = os.path.splitext(os.path.basename(fp))[0]
        if only_case and case_name != only_case:
            continue
        pairs.append((case_name, fp))
    return pairs


def list_parquet_files(parquet_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))


def count_scan_rows(parquet_files: Sequence[str]) -> int:
    total = 0
    for fp in parquet_files:
        df = pd.read_parquet(fp, columns=["scan"])
        total += len(df)
        del df
    return total


def run_gen_one(gen_script: str, in_parquet: str, out_parquet: str, random_state: int):
    cmd = [
        sys.executable, gen_script,
        "--in_parquet", in_parquet,
        "--out_parquet", out_parquet,
        "--random_state", str(random_state),
    ]
    run_cmd(cmd)


def wait_pkl_growth_with_progress(proc: subprocess.Popen, pkl_path: str, desc: str,
                                   timeout: int = 900) -> None:
    last_size = 0
    last_growth_ts = time.time()
    start_ts = last_growth_ts
    last_heartbeat_ts = start_ts
    with tqdm(total=0, desc=desc, unit="B", unit_scale=True, leave=False) as pbar:
        while True:
            size_now = os.path.getsize(pkl_path) if os.path.exists(pkl_path) else 0
            if size_now > pbar.total:
                pbar.total = size_now
                pbar.refresh()
            if size_now > last_size:
                pbar.update(size_now - last_size)
                last_size = size_now
                last_growth_ts = time.time()
            ret = proc.poll()
            if ret is not None:
                final_size = os.path.getsize(pkl_path) if os.path.exists(pkl_path) else 0
                if final_size > pbar.total:
                    pbar.total = final_size
                if final_size > last_size:
                    pbar.update(final_size - last_size)
                pbar.refresh()
                if ret != 0:
                    raise subprocess.CalledProcessError(ret, proc.args)
                break
            # timeout: no file growth for too long
            now_ts = time.time()
            if now_ts - last_growth_ts > timeout:
                proc.kill()
                raise RuntimeError(
                    f"{desc} 超时: {timeout}s 内文件无增长 (当前大小={size_now} bytes)")
            if now_ts - last_heartbeat_ts >= 30:
                elapsed = int(now_ts - start_ts)
                idle = int(now_ts - last_growth_ts)
                print(f"{desc} 心跳: 已运行 {elapsed}s, pkl大小={size_now} bytes, 空闲{idle}s")
                last_heartbeat_ts = now_ts
            time.sleep(0.2)


def gzip_file_with_progress(src_pkl: str, dst_gz: str, desc: str,
                            chunk_size: int = 8 * 1024 * 1024) -> None:
    total_size = os.path.getsize(src_pkl)
    with open(src_pkl, "rb") as f_in, gzip.open(dst_gz, "wb", compresslevel=6) as f_out:
        with tqdm(total=total_size, desc=desc, unit="B", unit_scale=True, leave=False) as pbar:
            while True:
                chunk = f_in.read(chunk_size)
                if not chunk:
                    break
                f_out.write(chunk)
                pbar.update(len(chunk))


def normalize_columns_for_one_parquet(parquet_fp: str) -> bool:
    df = pd.read_parquet(parquet_fp)
    changed = False
    for col, default in [("delta_rt_model", 0.0), ("delta_rt", 0.0),
                          ("predicted_rt", 0.0)]:
        if col not in df.columns:
            df[col] = default
            changed = True
        elif (df[col] != default).any():
            df[col] = default
            changed = True
    if "weight" not in df.columns:
        df["weight"] = 1.0
        changed = True
    if changed:
        df.to_parquet(parquet_fp, index=False)
    return changed


def convert_and_compress_one(parquet_fp, convert_script, config, task_name,
                             save_dir, work_dir, convert_timeout=900):
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(parquet_fp))[0]
    expected_pkl = os.path.join(save_dir, f"{base_name}_{task_name}.pkl")
    expected_gz = f"{expected_pkl}.gz"
    convert_desc = f"1.4 {base_name} pkl"
    gzip_desc = f"1.5 {base_name} gzip"

    if not os.path.exists(expected_gz):
        normalize_columns_for_one_parquet(parquet_fp)

        if not os.path.exists(expected_pkl):
            with tempfile.TemporaryDirectory(prefix=f"cv_one_{task_name}_") as tmp_dir:
                link_fp = os.path.join(tmp_dir, os.path.basename(parquet_fp))
                os.symlink(parquet_fp, link_fp)
                cmd = [
                    sys.executable, convert_script,
                    "--file_dir", tmp_dir,
                    "--config", config,
                    "--task_name", task_name,
                    "--ncores", "1",
                    "--save_dir", save_dir,
                ]
                stderr_fd, stderr_path = tempfile.mkstemp(
                    prefix=f"convert_{task_name}_", suffix=".log")
                proc = subprocess.Popen(cmd, cwd=work_dir,
                                        stdout=subprocess.DEVNULL, stderr=stderr_fd)
                os.close(stderr_fd)  # 关闭父进程的 fd，子进程仍持有
                try:
                    wait_pkl_growth_with_progress(proc, expected_pkl, convert_desc,
                                                  timeout=convert_timeout)
                except Exception:
                    if os.path.exists(stderr_path) and os.path.getsize(stderr_path) > 0:
                        with open(stderr_path) as ef:
                            print(f"[{convert_desc}] 子进程错误输出:\n{ef.read()[-4000:]}")
                    raise
                finally:
                    if os.path.exists(stderr_path):
                        os.remove(stderr_path)

        if not os.path.exists(expected_pkl):
            raise RuntimeError(f"1.4 转换失败，未找到输出 pkl: {expected_pkl}")

        if os.path.exists(expected_gz):
            os.remove(expected_gz)
        gzip_file_with_progress(expected_pkl, expected_gz, gzip_desc)

    if os.path.exists(expected_pkl):
        os.remove(expected_pkl)

    return expected_gz


def convert_and_compress_parquets_parallel(parquet_files, convert_script, config,
                                           task_name, save_dir, work_dir, workers,
                                           convert_timeout=900):
    if not parquet_files:
        return []
    expected_gz_files = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(convert_and_compress_one, fp, convert_script, config,
                            task_name, save_dir, work_dir, convert_timeout)
            for fp in parquet_files
        ]
        with tqdm(total=len(futures), desc=f"1.4/1.5 {task_name}", unit="file") as pbar:
            for fut in concurrent.futures.as_completed(futures):
                gz_file = fut.result()
                expected_gz_files.append(gz_file)
                pbar.update(1)
    return expected_gz_files


def validate_stage12_outputs(psm_files, expected_case_num):
    if len(psm_files) != expected_case_num:
        raise RuntimeError(f"1.2 输出文件数异常: 期望={expected_case_num}, 实际={len(psm_files)}")
    total_rows = 0
    empty_files = []
    for fp in psm_files:
        n_rows = len(pd.read_parquet(fp, columns=["scan"]))
        total_rows += n_rows
        if n_rows == 0:
            empty_files.append(os.path.basename(fp))
    if empty_files:
        raise RuntimeError("1.2 存在空 parquet: " + ", ".join(empty_files[:20])
                           + (" ..." if len(empty_files) > 20 else ""))
    print(f"1.2 验证通过: 文件数={len(psm_files)}, 总行数={total_rows}")
    return total_rows


def validate_stage13_matches_stage12(train_dir, val_dir, stage12_total_rows):
    if stage12_total_rows is None:
        return
    stage13_train_rows = count_scan_rows(list_parquet_files(train_dir))
    stage13_val_rows = count_scan_rows(list_parquet_files(val_dir))
    stage13_total_rows = stage13_train_rows + stage13_val_rows
    if stage13_total_rows != stage12_total_rows:
        raise RuntimeError(
            f"1.2->1.3 总数不一致: stage12={stage12_total_rows}, stage13={stage13_total_rows}")


# ──────────── 阶段 1.25: 分配全局 PSM ID ────────────

def assign_global_psm_ids(psm_files: List[str]) -> None:
    """
    读所有 PSM parquet，分配全局递增 global_psm_id，并写入 orig_file 列。
    全部向量化操作，无逐行循环。

    不再构建 id_map dict — orig_file / sage_discriminant_score 随 parquet 列
    传递到 split 阶段，后续 extract_and_write_index 直接从 split parquet 读取。
    """
    print("步骤1.25: 分配全局 PSM ID...")
    next_id = 0

    for fp in tqdm(psm_files, desc="分配 global_psm_id"):
        df = pd.read_parquet(fp)
        fname = os.path.basename(fp)
        n = len(df)

        df["global_psm_id"] = np.arange(next_id, next_id + n, dtype=np.int64)
        df["orig_file"] = fname  # 向量化广播，瞬间完成

        # 补齐可能缺失的列
        if "sage_discriminant_score" not in df.columns:
            df["sage_discriminant_score"] = -1.0

        df.to_parquet(fp, index=False)
        next_id += n
        del df

    print(f"  分配 {next_id} 个 global_psm_id")


# ──────────── 阶段 1.3: split（带 global_psm_id 列） ────────────

def split_stage13_and_save_per_batch(psm_files, train_dir, val_dir, batch_rows, random_state):
    rng = np.random.default_rng(random_state)
    train_file_idx = 0
    val_file_idx = 0
    train_rows_written = 0
    val_rows_written = 0
    buffered_parts = []
    buffered_rows = 0
    global_flush_count = 0
    global_rows_read = 0

    def flush_buffer(reason: str):
        nonlocal buffered_parts, buffered_rows, train_file_idx, val_file_idx
        nonlocal global_flush_count, train_rows_written, val_rows_written
        if buffered_rows == 0:
            return False

        merged = pd.concat(buffered_parts, ignore_index=True)
        merged = merged.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
        split_idx = int(len(merged) * 0.8)

        train_df = merged.iloc[:split_idx].copy()
        val_df = merged.iloc[split_idx:].copy()
        for rt_col in ("predicted_rt", "delta_rt", "delta_rt_model"):
            if rt_col in train_df.columns:
                train_df.loc[:, rt_col] = 0.0
            if rt_col in val_df.columns:
                val_df.loc[:, rt_col] = 0.0

        # 写出 train
        if len(train_df) > 0:
            out_path = os.path.join(train_dir, f"train.{train_file_idx:05d}.parquet")
            train_df.to_parquet(out_path, index=False)
            train_file_idx += 1
            train_rows_written += len(train_df)

        # 写出 val
        if len(val_df) > 0:
            out_path = os.path.join(val_dir, f"val.{val_file_idx:05d}.parquet")
            val_df.to_parquet(out_path, index=False)
            val_file_idx += 1
            val_rows_written += len(val_df)

        global_flush_count += 1
        print(f"1.3 flush#{global_flush_count} ({reason}): "
              f"total={len(merged)}, train={len(train_df)}, val={len(val_df)}, "
              f"累计train={train_rows_written}, val={val_rows_written}")

        del train_df, val_df, merged
        buffered_parts = []
        buffered_rows = 0
        gc.collect()
        return False

    with tqdm(total=len(psm_files), desc="1.3 全局", unit="file") as global_file_pbar:
        for file_idx, fp in enumerate(psm_files, start=1):
            pq_file = pq.ParquetFile(fp)
            total_rows = pq_file.metadata.num_rows
            fname = os.path.basename(fp)

            with tqdm(total=total_rows, desc=f"1.3 {fname}", unit="row", leave=False) as file_row_pbar:
                for record_batch in pq_file.iter_batches(batch_size=max(1, batch_rows)):
                    df = record_batch.to_pandas()
                    if len(df) == 0:
                        continue

                    global_rows_read += len(df)
                    file_row_pbar.update(len(df))

                    start = 0
                    n = len(df)
                    while start < n:
                        need = batch_rows - buffered_rows
                        take = min(need, n - start)
                        buffered_parts.append(df.iloc[start:start + take].reset_index(drop=True))
                        buffered_rows += take
                        start += take
                        if buffered_rows >= batch_rows:
                            if flush_buffer("batch-full"):
                                break

                    del df
                    gc.collect()

            global_file_pbar.update(1)
            if file_idx % 10 == 0 or file_idx == len(psm_files):
                print(f"1.3 进度: {file_idx}/{len(psm_files)}, 已读{global_rows_read}行, "
                      f"flush#{global_flush_count}, 缓冲{buffered_rows}行, "
                      f"累计train={train_rows_written}, val={val_rows_written}")

    if buffered_rows > 0:
        flush_buffer("final")

    return train_file_idx, val_file_idx, train_rows_written, val_rows_written


# ──────────── 阶段 1.35: 从 split parquet 提取索引 ────────────

def _clean_seq(s: str) -> str:
    """与 1_gen_parquet_tims.py 一致的清洗逻辑，用于计算 precursor_mass。"""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("n[42]", "")
    s = s.replace("N[.98]", "N").replace("Q[.98]", "Q")
    s = s.replace("M[15.99]", "M").replace("C[57.02]", "C")
    s = re.sub(r"\[[^\]]+\]", "", s)
    return s


def _build_records_df(df: pd.DataFrame, pklgz_rel: str, is_val: bool) -> pd.DataFrame:
    """对单个 split parquet 构建索引 DataFrame（向量化，无逐行循环）。"""
    precursor_mz = df["precursor_mz"].astype(np.float32)
    charge = df["charge"].astype(np.float32)
    mass = ((precursor_mz - PROTON_MASS_AMU) * charge).round(3)
    cleaned = df["precursor_sequence"].astype(str).map(_clean_seq)
    score = df.get("sage_discriminant_score", pd.Series(-1.0, index=df.index))
    score = pd.to_numeric(score, errors="coerce").fillna(-1.0)

    return pd.DataFrame({
        "global_psm_id": df["global_psm_id"].astype("int64"),
        "orig_file": df["orig_file"].astype(str),
        "orig_row_idx": df["orig_row_idx"].astype("int64"),
        "orig_candidate_idx": df["orig_candidate_idx"].astype("int64"),
        "cleaned_peptide": cleaned,
        "precursor_mass": mass,
        "charge": charge.astype("int32"),
        "label": df["label"].astype("int32"),
        "sage_discriminant_score": score.astype(np.float32),
        "pklgz_file": pklgz_rel,
    })


def extract_and_write_index(train_dir: str, val_dir: str, index_dir: str):
    """
    从 split parquet 逐文件提取索引，向量化构建，pyarrow 追加写入。
    不累积内存，可处理任意数据量。
    """
    print("步骤1.35: 从 split parquet 提取 PSM 索引...")
    os.makedirs(index_dir, exist_ok=True)

    columns = ["global_psm_id", "precursor_sequence", "precursor_mz",
               "charge", "label", "orig_row_idx", "orig_candidate_idx",
               "orig_file", "sage_discriminant_score"]

    # 清除旧索引文件
    for old in glob.glob(os.path.join(index_dir, "*_index.parquet")):
        os.remove(old)

    for tag, parquet_dir in [("train", train_dir), ("val", val_dir)]:
        parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
        global_index_path = os.path.join(index_dir, f"{tag}_index.parquet")
        global_writer: Optional[pa.parquet.ParquetWriter] = None
        # per-file writers: orig_file_stem -> ParquetWriter
        per_file_writers: Dict[str, pa.parquet.ParquetWriter] = {}
        per_file_schemas: Dict[str, pa.Schema] = {}

        for fp in tqdm(parquet_files, desc=f"提取 {tag} 索引"):
            df = pd.read_parquet(fp, columns=columns)
            base = os.path.splitext(os.path.basename(fp))[0]
            pklgz_rel = f"{tag}/{base}_{tag}.pkl.gz"

            rec_df = _build_records_df(df, pklgz_rel, is_val=(tag == "val"))
            del df

            # 写入全局索引
            table = pa.Table.from_pandas(rec_df)
            if global_writer is None:
                global_writer = pq.ParquetWriter(global_index_path, table.schema)
            global_writer.write_table(table)

            # 按 orig_file 分组写入 per-file 索引
            for orig_file, group_df in rec_df.groupby("orig_file", sort=False):
                if not orig_file or orig_file == "":
                    continue
                stem = os.path.splitext(str(orig_file))[0]
                sub_df = group_df.drop(columns=["orig_file"])
                sub_table = pa.Table.from_pandas(sub_df)
                if stem not in per_file_writers:
                    per_path = os.path.join(index_dir, f"{stem}_index.parquet")
                    per_file_writers[stem] = pq.ParquetWriter(per_path, sub_table.schema)
                per_file_writers[stem].write_table(sub_table)

            del rec_df, table

        if global_writer is not None:
            global_writer.close()
        for w in per_file_writers.values():
            w.close()

        total_rows = len(pd.read_parquet(global_index_path, columns=["global_psm_id"]))
        print(f"  {tag}_index.parquet: {total_rows:,} rows")

    file_count = len(glob.glob(os.path.join(index_dir, "*_index.parquet"))) - 2
    print(f"  per-file 索引: {file_count} 个")


# ──────────── 主流程 ────────────

def main():
    args = parse_args()

    args.wiff_root = os.path.abspath(args.wiff_root)
    work_dir = os.path.abspath(args.work_dir)

    gen_script = os.path.join(work_dir, "1_gen_parquet_tims.py")
    convert_script = os.path.join(work_dir, "3_convert_parquet2pkl_tims.py")
    for p in [gen_script, convert_script]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"脚本不存在: {p}")

    os.makedirs(args.pkl_out_dir, exist_ok=True)
    os.makedirs(args.index_dir, exist_ok=True)

    # 中间文件全放 /tmp
    tmp_base = args.tmp_root or tempfile.gettempdir()
    tmp_root = tempfile.mkdtemp(prefix="wiff_pipeline_", dir=tmp_base)
    psm_out_dir = os.path.join(tmp_root, "psm_parquet")
    split_out_dir = os.path.join(tmp_root, "split")
    os.makedirs(psm_out_dir, exist_ok=True)
    os.makedirs(split_out_dir, exist_ok=True)

    print(f"中间文件目录: {tmp_root}")

    try:
        # ── 阶段 1.2: gen_parquet ──
        print("步骤1.2: 展平 wiff → PSM parquet...")
        wiff_files = collect_wiff_files(args.wiff_root, args.only_case)
        if not wiff_files:
            raise RuntimeError("未找到可用的 wiff parquet 文件")

        current_psm_files = [os.path.join(psm_out_dir, f"{case_name}.parquet")
                             for case_name, _ in wiff_files]

        if args.start_from in ["original_parquet", "assign_id"]:
            missing = [fp for fp in current_psm_files if not os.path.exists(fp)]
            if missing:
                print(f"  生成 {len(missing)} 个 PSM parquet (共 {len(current_psm_files)} 个)...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
                    futures = []
                    for case_name, in_file in wiff_files:
                        out_file = os.path.join(psm_out_dir, f"{case_name}.parquet")
                        if os.path.exists(out_file):
                            continue
                        futures.append(executor.submit(
                            run_gen_one, gen_script, in_file, out_file, args.random_state))
                    for i, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
                        fut.result()
                        if i % 20 == 0 or i == len(futures):
                            print(f"  1.2 进度: {i}/{len(futures)}")
            else:
                print("  已全部存在，跳过。")
            stage12_total_rows = validate_stage12_outputs(current_psm_files, len(wiff_files))
        else:
            print(f"  根据 --start_from {args.start_from}，跳过 1.2。")
            stage12_total_rows = None

        # ── 阶段 1.25: 分配全局 PSM ID ──
        if args.start_from in ["original_parquet", "assign_id"]:
            assign_global_psm_ids(current_psm_files)

        # ── 阶段 1.3: split ──
        print("步骤1.3: 8:2 切分 train/val...")
        train_dir = os.path.join(split_out_dir, "train")
        val_dir = os.path.join(split_out_dir, "val")

        if args.start_from in ["original_parquet", "assign_id", "split"]:
            psm_files = current_psm_files
            if not psm_files:
                raise RuntimeError("1.2 未生成任何 parquet 文件")
            train_dir = os.path.join(split_out_dir, "train")
            val_dir = os.path.join(split_out_dir, "val")
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)

            _, _, train_rows, val_rows = split_stage13_and_save_per_batch(
                psm_files=psm_files, train_dir=train_dir, val_dir=val_dir,
                batch_rows=args.rows_per_pkl, random_state=args.random_state,
            )
            print(f"1.3 完成: train parquet={len(glob.glob(os.path.join(train_dir, '*.parquet')))}, "
                  f"val parquet={len(glob.glob(os.path.join(val_dir, '*.parquet')))}")
            validate_stage13_matches_stage12(train_dir, val_dir, stage12_total_rows)

            # 清理阶段 1.2 中间文件
            print("  清理 1.2 中间文件...")
            shutil.rmtree(psm_out_dir, ignore_errors=True)
        elif args.start_from == "pkl":
            if args.split_dir:
                train_dir = os.path.join(args.split_dir, "train")
                val_dir = os.path.join(args.split_dir, "val")
            else:
                raise RuntimeError("--start_from pkl 需提供 --split_dir <现有split目录>")
            print(f"  使用现有 split 目录: train={train_dir}, val={val_dir}")
        else:
            print(f"  根据 --start_from {args.start_from}，跳过 1.3。")

        # ── 阶段 1.35: 从 split parquet 提取索引 ──
        if args.start_from in ["original_parquet", "assign_id", "split"]:
            extract_and_write_index(train_dir, val_dir, args.index_dir)

        if args.skip_pkl:
            print("已跳过 1.4 编码。")
            return

        # ── 阶段 1.4/1.5: convert + compress ──
        print(f"步骤1.4/1.5: parquet→pkl→pkl.gz ({max(1, args.pkl_workers)} 进程)...")
        train_parquets = list_parquet_files(train_dir)
        val_parquets = list_parquet_files(val_dir)
        total_parquets = len(train_parquets) + len(val_parquets)
        if total_parquets == 0:
            raise RuntimeError("1.4 没有可转换的 parquet 文件")

        expected_gz_files = []
        for tag, parquets in [("train", train_parquets), ("val", val_parquets)]:
            gz_files = convert_and_compress_parquets_parallel(
                parquet_files=parquets, convert_script=convert_script, config=args.config,
                task_name=tag, save_dir=os.path.join(args.pkl_out_dir, tag),
                work_dir=work_dir, workers=args.pkl_workers,
                convert_timeout=args.convert_timeout,
            )
            expected_gz_files.extend(gz_files)

        missing_gz = [fp for fp in expected_gz_files if not os.path.exists(fp)]
        if missing_gz:
            raise RuntimeError("1.5 压缩后缺失: " + ", ".join(missing_gz[:10]))

        train_gz = len(glob.glob(os.path.join(args.pkl_out_dir, "train", "*.pkl.gz")))
        val_gz = len(glob.glob(os.path.join(args.pkl_out_dir, "val", "*.pkl.gz")))
        print(f"1.4/1.5 完成: train={train_gz}, val={val_gz} pkl.gz 文件")

        print("\n全流程完成。")

    finally:
        if not args.keep_tmp:
            print(f"清理中间目录: {tmp_root}")
            shutil.rmtree(tmp_root, ignore_errors=True)
        else:
            print(f"保留中间目录: {tmp_root}")


if __name__ == "__main__":
    main()
