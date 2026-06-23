"""
功能：timsTOF数据全流程：gen parquet→split train/val→convert pkl.gz（支持--start_from断点续跑）
输入：
    --tims_root /home/yiwen/AIPC/database/tims
    --work_dir /home/yiwen/AIPC/scripts/attantion
输出：
    data/dataset/tims_all/train/{base}_train.pkl.gz
    data/dataset/tims_all/val/{base}_val.pkl.gz
"""

import argparse
import concurrent.futures
import gc
import glob
import gzip
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

TEST_ONLY_ROOT = "/home/yiwen/AIPC/database/tims"


def parse_args():
    parser = argparse.ArgumentParser(
        description="分步执行 tims 数据处理：1.2生成PSM parquet、1.3按8:2切分、1.4转pkl"
    )
    parser.add_argument(
        "--tims_root",
        type=str,
        default=TEST_ONLY_ROOT,
        help="tims 数据根目录（每个样本一个 parquet）",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion",
        help="脚本所在工作目录",
    )
    parser.add_argument(
        "--psm_out_dir",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/processed_parquet_tims",
        help="第一阶段输出目录",
    )
    parser.add_argument(
        "--split_out_dir",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/pkl_dataset_tims/tims_split_all",
        help="第二阶段输出目录",
    )
    parser.add_argument(
        "--pkl_out_dir",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/pkl_dataset_tims/tims_pkl_all",
        help="第三阶段输出目录",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/model.yaml",
        help="model.yaml 配置文件",
    )
    parser.add_argument(
        "--rows_per_pkl",
        type=int,
        default=1000000,
        help="每个 parquet/pkl 文件最大行数",
    )
    parser.add_argument(
        "--ncores",
        type=int,
        default=8,
        help="pkl 转换阶段 CPU 并行核心数",
    )
    parser.add_argument(
        "--pkl_workers",
        type=int,
        default=4,
        help="parquet->pkl->压缩阶段并行进程数（每进程处理一个文件）",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="1.2 阶段并行进程数",
    )
    parser.add_argument(
        "--only_case",
        type=str,
        default="",
        help="可选：仅处理指定 parquet 文件名（不带 .parquet）",
    )
    parser.add_argument(
        "--skip_pkl",
        action="store_true",
        help="如果设置，则只进行 1.2 和 1.3，不进行 1.4",
    )
    parser.add_argument(
        "--start_from",
        type=str,
        default="original_parquet",
        choices=["original_parquet", "parquet", "pkl"],
        help="设置从哪一步开始执行：original_parquet(全流程), parquet(从 1.3 开始), pkl(从 1.4 开始)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="随机种子，保证可复现",
    )
    parser.add_argument(
        "--q_value_max",
        type=float,
        default=0.01,
        help="Sage target q-value 阈值（默认 0.2）",
    )
    return parser.parse_args()


def run_cmd(cmd, cwd=None):
    print("执行命令:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def collect_tims_files(tims_root: str, only_case: str):
    files = sorted(glob.glob(os.path.join(tims_root, "*.parquet")))
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
    return sum(len(pd.read_parquet(fp, columns=["scan"])) for fp in parquet_files)


def run_gen_one(gen_script: str, in_parquet: str, out_parquet: str, random_state: int):
    cmd = [
        sys.executable,
        gen_script,
        "--in_parquet",
        in_parquet,
        "--out_parquet",
        out_parquet,
        "--random_state",
        str(random_state),
    ]
    run_cmd(cmd)


def wait_pkl_growth_with_progress(proc: subprocess.Popen, pkl_path: str, desc: str) -> None:
    last_size = 0
    start_ts = time.time()
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

            now_ts = time.time()
            if now_ts - last_heartbeat_ts >= 30:
                elapsed = int(now_ts - start_ts)
                print(f"{desc} 心跳: 已运行 {elapsed}s, 当前pkl大小={size_now} bytes")
                last_heartbeat_ts = now_ts

            time.sleep(0.2)


def gzip_file_with_progress(src_pkl: str, dst_gz: str, desc: str, chunk_size: int = 8 * 1024 * 1024) -> None:
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
        df.to_parquet(parquet_fp, index=False)

    return changed


def convert_and_compress_one(
    parquet_fp: str,
    convert_script: str,
    config: str,
    task_name: str,
    save_dir: str,
    work_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(parquet_fp))[0]
    expected_pkl = os.path.join(save_dir, f"{base_name}_{task_name}.pkl")
    expected_gz = f"{expected_pkl}.gz"
    convert_desc = f"1.4 {base_name} pkl大小"
    gzip_desc = f"1.5 {base_name} gzip"
    normalized_changed = False

    if not os.path.exists(expected_gz):
        normalized_changed = normalize_columns_for_one_parquet(parquet_fp)

        if not os.path.exists(expected_pkl):
            with tempfile.TemporaryDirectory(prefix=f"convert_one_{task_name}_") as tmp_dir:
                link_fp = os.path.join(tmp_dir, os.path.basename(parquet_fp))
                os.symlink(parquet_fp, link_fp)
                cmd = [
                    sys.executable,
                    convert_script,
                    "--file_dir",
                    tmp_dir,
                    "--config",
                    config,
                    "--task_name",
                    task_name,
                    "--ncores",
                    "1",
                    "--save_dir",
                    save_dir,
                ]
                proc = subprocess.Popen(
                    cmd,
                    cwd=work_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                wait_pkl_growth_with_progress(proc, expected_pkl, convert_desc)

        if not os.path.exists(expected_pkl):
            raise RuntimeError(f"1.4 转换失败，未找到输出 pkl: {expected_pkl}")

        if os.path.exists(expected_gz):
            os.remove(expected_gz)
        gzip_file_with_progress(expected_pkl, expected_gz, gzip_desc)

    if os.path.exists(expected_pkl):
        os.remove(expected_pkl)
    if os.path.exists(parquet_fp):
        os.remove(parquet_fp)

    return expected_gz, normalized_changed


def convert_and_compress_parquets_parallel(
    parquet_files,
    convert_script: str,
    config: str,
    task_name: str,
    save_dir: str,
    work_dir: str,
    workers: int,
):
    if not parquet_files:
        return [], 0

    expected_gz_files = []
    normalized_changed_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(
                convert_and_compress_one,
                parquet_fp,
                convert_script,
                config,
                task_name,
                save_dir,
                work_dir,
            )
            for parquet_fp in parquet_files
        ]

        with tqdm(total=len(futures), desc=f"1.4/1.5 {task_name} 全局", unit="file") as pbar:
            for fut in concurrent.futures.as_completed(futures):
                gz_file, normalized_changed = fut.result()
                expected_gz_files.append(gz_file)
                if normalized_changed:
                    normalized_changed_count += 1
                pbar.update(1)

    return expected_gz_files, normalized_changed_count


def validate_stage12_outputs(psm_files, expected_case_num: int):
    if len(psm_files) != expected_case_num:
        raise RuntimeError(
            f"步骤1.2输出文件数异常: 期望={expected_case_num}, 实际={len(psm_files)}"
        )

    total_rows = 0
    empty_files = []
    for fp in psm_files:
        n_rows = len(pd.read_parquet(fp, columns=["scan"]))
        total_rows += n_rows
        if n_rows == 0:
            empty_files.append(os.path.basename(fp))

    if empty_files:
        raise RuntimeError(
            "步骤1.2存在空parquet文件: " + ", ".join(empty_files[:20])
            + (" ..." if len(empty_files) > 20 else "")
        )

    print(f"1.2 总数验证通过: 文件数={len(psm_files)}, 总行数={total_rows}")
    return total_rows


def validate_stage13_matches_stage12(
    train_dir: str,
    val_dir: str,
    stage12_total_rows: Optional[int],
) -> None:
    if stage12_total_rows is None:
        return

    stage13_train_rows = count_scan_rows(list_parquet_files(train_dir))
    stage13_val_rows = count_scan_rows(list_parquet_files(val_dir))
    stage13_total_rows = stage13_train_rows + stage13_val_rows
    if stage13_total_rows != stage12_total_rows:
        raise RuntimeError(
            f"1.2->1.3 总数不一致: stage12={stage12_total_rows}, stage13={stage13_total_rows}"
        )


def iter_parquet_in_batches(parquet_path: str, batch_rows: int):
    """按批读取 parquet，避免整文件进内存。"""
    pq_file = pq.ParquetFile(parquet_path)
    for record_batch in pq_file.iter_batches(batch_size=batch_rows):
        yield record_batch.to_pandas()


def split_stage13_and_save_per_batch(
    psm_files,
    train_dir: str,
    val_dir: str,
    batch_rows: int,
    random_state: int,
):
    """累计读取到 batch_rows 后再切分并落盘，随后释放内存。"""
    rng = np.random.default_rng(random_state)
    train_file_idx = 0
    val_file_idx = 0
    train_rows_written = 0
    val_rows_written = 0
    buffered_parts = []
    buffered_rows = 0
    global_flush_count = 0
    global_rows_read = 0

    def flush_buffer(flush_reason: str):
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

        if len(train_df) > 0:
            train_out = os.path.join(train_dir, f"train.{train_file_idx:05d}.parquet")
            train_df.to_parquet(train_out, index=False)
            train_file_idx += 1
            train_rows_written += len(train_df)

        if len(val_df) > 0:
            val_out = os.path.join(val_dir, f"val.{val_file_idx:05d}.parquet")
            val_df.to_parquet(val_out, index=False)
            val_file_idx += 1
            val_rows_written += len(val_df)

        global_flush_count += 1
        print(
            f"1.3 flush#{global_flush_count} ({flush_reason}): "
            f"total={len(merged)}, train={len(train_df)}, val={len(val_df)}, "
            f"累计train={train_rows_written}, 累计val={val_rows_written}"
        )

        del train_df
        del val_df
        del merged
        buffered_parts = []
        buffered_rows = 0
        gc.collect()
        return False

    with tqdm(total=len(psm_files), desc="1.3 全局", unit="file") as global_file_pbar:
        for file_idx, fp in enumerate(psm_files, start=1):
            pq_file = pq.ParquetFile(fp)
            total_rows_this_file = pq_file.metadata.num_rows
            file_name = os.path.basename(fp)

            with tqdm(
                total=total_rows_this_file,
                desc=f"1.3 {file_name}",
                unit="row",
                leave=False,
            ) as file_row_pbar:
                for record_batch in pq_file.iter_batches(batch_size=max(1, batch_rows)):
                    df = record_batch.to_pandas()
                    if len(df) == 0:
                        continue

                    global_rows_read += len(df)
                    file_row_pbar.update(len(df))

                    start = 0
                    total_len = len(df)
                    while start < total_len:
                        need_rows = batch_rows - buffered_rows
                        take = min(need_rows, total_len - start)
                        part = df.iloc[start : start + take]
                        buffered_parts.append(part.reset_index(drop=True))
                        buffered_rows += take
                        start += take

                        if buffered_rows >= batch_rows:
                            if flush_buffer(flush_reason="batch-full"):
                                break

                    del df
                    gc.collect()

            global_file_pbar.update(1)
            if file_idx % 10 == 0 or file_idx == len(psm_files):
                print(
                    f"1.3 进度: 样本={file_idx}/{len(psm_files)}, 已读取行数={global_rows_read}, "
                    f"已flush次数={global_flush_count}, 当前缓冲行数={buffered_rows}, "
                    f"累计train={train_rows_written}, 累计val={val_rows_written}"
                )

    # 收尾：最后不足 batch_rows 的残留也需要按 8:2 落盘
    if buffered_rows > 0:
        flush_buffer(flush_reason="final-remainder")

    return train_file_idx, val_file_idx, train_rows_written, val_rows_written


def main():
    args = parse_args()

    args.tims_root = os.path.abspath(args.tims_root)
    if args.tims_root != TEST_ONLY_ROOT:
        raise ValueError(f"当前脚本仅允许测试目录: {TEST_ONLY_ROOT}")

    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(args.psm_out_dir, exist_ok=True)
    os.makedirs(args.split_out_dir, exist_ok=True)
    os.makedirs(args.pkl_out_dir, exist_ok=True)

    gen_script = os.path.join(work_dir, "1_gen_parquet_tims.py")
    convert_script = os.path.join(work_dir, "3_convert_parquet2pkl_tims.py")

    for p in [gen_script, convert_script]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"脚本不存在: {p}")

    print("步骤1.2: 检查 1_gen_parquet_tims.py 输出...")
    tims_files = collect_tims_files(args.tims_root, args.only_case)
    if not tims_files:
        raise RuntimeError("未找到可用的 tims parquet 文件")

    current_psm_files = [os.path.join(args.psm_out_dir, f"{case_name}.parquet") for case_name, _ in tims_files]
    
    if args.start_from == "original_parquet":
        # 检查是否所有文件都已存在
        missing_outputs = [fp for fp in current_psm_files if not os.path.exists(fp)]
        
        if missing_outputs:
            print(f"发现 {len(missing_outputs)} 个缺失文件，开始执行 1.2 阶段生成...")
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
                for case_name, in_file in tims_files:
                    out_file = os.path.join(args.psm_out_dir, f"{case_name}.parquet")
                    if os.path.exists(out_file):
                        continue
                    futures.append(
                        executor.submit(
                            run_gen_one,
                            gen_script,
                            in_file,
                            out_file,
                            args.random_state,
                        )
                    )

                for i, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
                    fut.result()
                    if i % 10 == 0 or i == len(futures):
                        print(f"1.2 增量完成: {i}/{len(futures)}")
        else:
            print("所有 1.2 阶段输出文件已存在，跳过生成步骤。")
        
        stage12_total_rows = validate_stage12_outputs(current_psm_files, len(tims_files))
    else:
        print(f"根据 --start_from {args.start_from}，跳过步骤 1.2 (original_parquet) 及其校验。")
        stage12_total_rows = None

    print("步骤1.3: 按 8:2 划分训练/验证，并重组为大 parquet...")
    train_dir = os.path.join(args.split_out_dir, "train")
    val_dir = os.path.join(args.split_out_dir, "val")
    
    if args.start_from in ["original_parquet", "parquet"]:
        psm_files = current_psm_files
        if not psm_files:
            raise RuntimeError("1.2 未生成任何 parquet 文件")

        existing_train = glob.glob(os.path.join(train_dir, "*.parquet"))
        if existing_train and args.start_from != "parquet":
            print(f"检测到 {args.split_out_dir} 已有结果，跳过 1.3 划分阶段。")
        else:
            if os.path.exists(args.split_out_dir):
                shutil.rmtree(args.split_out_dir)
            os.makedirs(args.split_out_dir, exist_ok=True)
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)

            _, _, train_rows_written, val_rows_written = split_stage13_and_save_per_batch(
                psm_files=psm_files,
                train_dir=train_dir,
                val_dir=val_dir,
                batch_rows=args.rows_per_pkl,
                random_state=args.random_state,
            )

            print(
                f"1.3 完成: train parquet数={len(glob.glob(os.path.join(train_dir, '*.parquet')))}, "
                f"val parquet数={len(glob.glob(os.path.join(val_dir, '*.parquet')))}"
            )

        validate_stage13_matches_stage12(train_dir, val_dir, stage12_total_rows)
    else:
        print(f"根据 --start_from {args.start_from}，跳过步骤 1.3 (parquet)。")

    if args.skip_pkl:
        print("已按要求完成 1.2 和 1.3，跳过 1.4 编码。")
        return

    print(f"步骤1.4/1.5: 使用 {max(1, args.pkl_workers)} 个进程并行处理 parquet->pkl->pkl.gz（每进程一个文件）...")
    print("1.4 将采用单文件流水线: 标准化->转换->压缩->清理")

    train_parquets = list_parquet_files(train_dir)
    val_parquets = list_parquet_files(val_dir)
    total_parquets_before_convert = len(train_parquets) + len(val_parquets)
    if total_parquets_before_convert == 0:
        raise RuntimeError("1.4 没有可转换的 parquet 文件")

    expected_gz_files = []
    train_gz_files, train_fixed = convert_and_compress_parquets_parallel(
            parquet_files=train_parquets,
            convert_script=convert_script,
            config=args.config,
            task_name="train",
            save_dir=os.path.join(args.pkl_out_dir, "train"),
            work_dir=work_dir,
            workers=args.pkl_workers,
        )
    expected_gz_files.extend(train_gz_files)

    val_gz_files, val_fixed = convert_and_compress_parquets_parallel(
            parquet_files=val_parquets,
            convert_script=convert_script,
            config=args.config,
            task_name="val",
            save_dir=os.path.join(args.pkl_out_dir, "val"),
            work_dir=work_dir,
            workers=args.pkl_workers,
        )
    expected_gz_files.extend(val_gz_files)
    print(f"1.4 字段标准化完成: train 修复文件={train_fixed}, val 修复文件={val_fixed}")

    remaining_train_parquet = glob.glob(os.path.join(train_dir, "*.parquet"))
    remaining_val_parquet = glob.glob(os.path.join(val_dir, "*.parquet"))
    if remaining_train_parquet or remaining_val_parquet:
        raise RuntimeError(
            f"1.4 转换后仍存在 parquet: train={len(remaining_train_parquet)}, val={len(remaining_val_parquet)}"
        )

    print("步骤1.5: 并行压缩已在每个文件转换后即时完成。")

    remaining_raw_pkl = glob.glob(os.path.join(args.pkl_out_dir, "**", "*.pkl"), recursive=True)
    if remaining_raw_pkl:
        raise RuntimeError(
            "1.5 压缩后仍存在未删除的原始 pkl: "
            + ", ".join(remaining_raw_pkl[:10])
            + (" ..." if len(remaining_raw_pkl) > 10 else "")
        )

    missing_gz = [fp for fp in expected_gz_files if not os.path.exists(fp)]
    if missing_gz:
        raise RuntimeError(
            "1.5 压缩后缺失目标文件: "
            + ", ".join(missing_gz[:10])
            + (" ..." if len(missing_gz) > 10 else "")
        )

    if len(expected_gz_files) != total_parquets_before_convert:
        raise RuntimeError(
            f"1.5 文件数校验失败: parquet={total_parquets_before_convert}, gz={len(expected_gz_files)}"
        )

    print(
        "全流程完成: 1.2(tims) -> 1.3(每文件8:2) -> 1.4(逐个转pkl并删源parquet) "
        f"-> 1.5(逐个压缩pkl)，parquet数={total_parquets_before_convert}，压缩文件数={len(expected_gz_files)}"
    )


if __name__ == "__main__":
    main()
