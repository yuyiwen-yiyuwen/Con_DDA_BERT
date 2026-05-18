import argparse
import concurrent.futures
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 基于 split_data_mzml.py 修改：
#   - target 只保留 sage spectrum_q <= 0.01（1% FDR）
#   - 全部数据作为训练集（不拆 val）


def parse_args():
    """按 sage spectrum_q<=0.01 筛选 mzmL PSM，全部作为训练集。"""
    parser = argparse.ArgumentParser(description="sage 1%FDR 筛选 mzmL 数据 → 全部 train")
    parser.add_argument("--mzml_root", type=str,
                        default="/home/yiwen/AIPC/database/mzml")
    parser.add_argument("--work_dir", type=str,
                        default="/home/yiwen/AIPC/scripts/organized_attantion")
    parser.add_argument("--psm_out_dir", type=str,
                        default="/home/yiwen/AIPC/scripts/organized_attantion/data/tmp_mzml_sage_parquet")
    parser.add_argument("--pkl_out_dir", type=str,
                        default="/home/yiwen/AIPC/scripts/organized_attantion/data/dataset/mzml_sage_select")
    parser.add_argument("--config", type=str,
                        default="/home/yiwen/AIPC/scripts/organized_attantion/config/model.yaml")
    parser.add_argument("--rows_per_pkl", type=int, default=1000000)
    parser.add_argument("--ncores", type=int, default=8)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--only_case", type=str, default="")
    parser.add_argument("--skip_pkl", action="store_true")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--keep_intermediate", action="store_true")
    parser.add_argument("--q_threshold", type=float, default=0.01,
                        help="sage spectrum_q 阈值（默认 0.01 = 1% FDR）")
    return parser.parse_args()


def run_cmd(cmd, cwd=None):
    """执行外部 Shell 命令并打印。"""
    # 把 cmd 这个列表里的所有元素（如命令和参数），用空格 " " 连接成一行字符串
    print("执行命令:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def collect_triplets(mzml_root: str, only_case: str):
    """
    遍历目录，寻找配对的三个文件 (sage, fp, raw)。
    返回一个包含 (样本名, 原始谱图路径, Sage结果路径, FP结果路径) 的列表。
    """
    sage_files = sorted(glob.glob(os.path.join(mzml_root, "**", "*_sage.parquet"), recursive=True))

    triplets = []
    for sage in sage_files:
        case_dir = os.path.basename(os.path.dirname(sage))
        if only_case and case_dir != only_case:
            continue

        fp = sage.replace("_sage.parquet", "_fp.parquet")
        raw = sage.replace("_sage.parquet", "_rawspectrum.parquet")
        # 必须三个文件全部存在才加入处理队列
        if os.path.exists(fp) and os.path.exists(raw):
            triplets.append((case_dir, raw, sage, fp))

    return triplets


def process_single_sample(raw_path, sage_path, fp_path, out_parquet):
    # ---------------------------
    # 1. 加载数据 (Load DataFrames)
    # ---------------------------
    raw_df = pd.read_parquet(raw_path)
    raw_df['scan'] = raw_df['scan'].astype(int)

    sage_df = pd.read_parquet(sage_path)
    sage_df['scan'] = sage_df['scan'].astype(int)
    sage_df['psm_id'] = sage_df['scan'].astype(str) + '_' + sage_df['precursor_sequence']

    fp_df = pd.read_parquet(fp_path)
    # 确保 FP 数据的 psm_id 生成正确
    if 'scan' in fp_df.columns:
        fp_df['psm_id'] = fp_df['scan'].astype(int).astype(str) + '_' + fp_df['detect_sequence']
        fp_df = fp_df.drop(columns=['scan'])
    else:
        # 如果 scan 已经被删除但 psm_id 缺失 (根据上下文不太可能，但为了安全起见)
        if 'psm_id' not in fp_df.columns:
            raise ValueError(f"FP parquet missing 'scan' or 'psm_id' columns: {fp_path}")

    # ---------------------------
    # 2. 识别 Target, False Target, Decoy
    # ---------------------------

    # A. 真实 Targets: Sage label=1 AND spectrum_q <= q_threshold (1% FDR)
    fp_df_target = fp_df
    sage_df_target = sage_df[(sage_df['label'] == 1) & (sage_df['spectrum_q'] <= 0.01)]
    targets = sage_df_target.merge(fp_df_target, on='psm_id', how='inner', suffixes=('', '_fp'))
    
    """
    # B. False Targets (困难负样本)
    # 逻辑来源: gen_dataset_ipc.py
    # 定义: 谱图匹配得分较低(非Rank1)被丢弃，但其肽段序列在样本中是可信的。
    fp_df_false_target = fp_df[fp_df['q-value'] <= 0.1]
    sage_df_false_target = sage_df[(sage_df['label'] == 1) & (sage_df['spectrum_q'] <= 0.1)]
    # 1. 识别已鉴定的集合: FragPipe (q<=10%) 和 Sage(q<=10%) 的并集
    identified_psm_ids = set(fp_df_false_target['psm_id']) | set(sage_df_false_target['psm_id'])
    
    # 2. 找出每张谱图得分第一的 PSM (Rank 1): 认为是当前谱图的最佳解释
    sage_sorted = sage_df.sort_values(by=['scan', 'sage_discriminant_score'], ascending=[True, False])
    top1_psm_ids = set(sage_sorted.drop_duplicates(subset='scan')['psm_id'])
    
    # 3. 候选集生成:
    #    (1) 排除 Rank 1 的 PSM (即选 Rank 2, 3...) -> 通常是错误匹配
    candidates = sage_df[~sage_df['psm_id'].isin(top1_psm_ids)]
    
    #    (2) 排除在"已鉴定集合"中的 PSM -> 确保不是漏掉的正确匹配
    candidates = candidates[~candidates['psm_id'].isin(identified_psm_ids)]
    
    #    (3) 序列过滤: 只保留那些"肽段序列"在 High Confidence 列表(FP或Sage Target)中出现过的 PSM
    #        这意味着: 虽然这个PSM对这张谱图是错的，但这肽段在样品里是存在的。
    confirmed_sequences = set(fp_df_target['detect_sequence']) | set(sage_df_target['precursor_sequence'])
    candidates = candidates[candidates['precursor_sequence'].isin(confirmed_sequences)]
    
    # 4. 去重: 对于每个唯一的序列，只保留 Sage 分数最高的一个 False Target
    candidates = candidates.sort_values(by=['precursor_sequence', 'sage_discriminant_score'], ascending=[True, False])
    false_targets = candidates.drop_duplicates(subset='precursor_sequence').reset_index(drop=True)
    """
    # C. Decoys (诱饵序列)
    # 逻辑: 标准 Decoy 选取，数量尽量与 Targets 1:1
    # 额外补充: 排除已经在 False Targets 中选中的 psm_id
    sage_decoy = sage_df[sage_df['label'] == 0]
    # sage_decoy = sage_decoy[~sage_decoy['psm_id'].isin(false_targets['psm_id'])]
    
    decoy_num = len(targets)
    decoy_sorted = sage_decoy.sort_values(by='sage_discriminant_score', ascending=False).reset_index(drop=True)
    
    if len(decoy_sorted) <= decoy_num:
        # Decoy 不够，全用
        decoys = decoy_sorted
    else:
        # Decoy 充足，一半取高分(Hard Decoy)，一半随机取(Random Decoy)
        half = int(decoy_num / 2)
        decoy_high = decoy_sorted.iloc[:half]
        decoy_low = decoy_sorted.iloc[half:].sample(n=half, random_state=42)
        decoys = pd.concat([decoy_high, decoy_low], ignore_index=True)

    # ---------------------------
    # 3. 属性赋值 (Label, Weight, Unmask)
    # ---------------------------
    
    # Targets: Label=1, Weight=1.0
    targets['label'] = 1
    targets['weight'] = 1.0
    
    # False Targets: Label=0, Weight=0.3 (作为负样本但权重较低)
    # false_targets['label'] = 0
    # false_targets['weight'] = 0.3
    
    # Decoys: Label=0, Weight=1.0
    decoys['label'] = 0
    decoys['weight'] = 1.0

    # Unmask 逻辑:
    # 找出 Target 和 False Target 共有的序列
    # 对于这些序列，在 Target 和 False Target 中都将 unmask 设为 1
    # 其他情况(包括 Decoy) unmask 为 0
    # common_seqs = set(targets['precursor_sequence']) & set(false_targets['precursor_sequence'])
    targets['unmask'] = 0
    # targets['unmask'] = np.where(targets['precursor_sequence'].isin(common_seqs), 1, 0)
    # false_targets['unmask'] = np.where(false_targets['precursor_sequence'].isin(common_seqs), 1, 0)
    decoys['unmask'] = 0

    # ---------------------------
    # 4. 合并与过滤 (Merge and Filter)
    # ---------------------------
    # 合并所有类型的 PSM
    # combined = pd.concat([targets, false_targets, decoys], ignore_index=True)
    combined = pd.concat([targets, decoys], ignore_index=True)
    
    # 关联光谱数据 (Raw Spectrum)
    final_df = combined.merge(raw_df, on='scan', how='inner')
    
    # 序列清洗: 去除修饰标记，只保留氨基酸序列
    final_df['cleaned_sequence'] = final_df['precursor_sequence'].astype(str).str.replace('n[42]', '').str.replace('N[.98]', 'N').str.replace('Q[.98]', 'Q').str.replace('M[15.99]', 'M').str.replace('C[57.02]', 'C')
    final_df['sequence_len'] = final_df['cleaned_sequence'].apply(len)
    
    # 过滤条件: 
    # 1. 肽段长度 7-50
    # 2. 电荷状态 2-5
    final_df = final_df[(final_df['sequence_len'] <= 50) & (final_df['sequence_len'] >= 7)]
    final_df = final_df[(final_df['charge'] <= 5) & (final_df['charge'] >= 2)]
    
    # 最终列选择
    cols_to_keep = [
        'scan', 'precursor_mz', 'charge', 'rt', 'mz_array', 'intensity_array',
        'precursor_sequence', 'label', 'weight', 'unmask', 
        'predicted_rt', 'delta_rt', 'sage_discriminant_score', 'spectrum_q'
    ]
    
    # 选择存在的列
    existing_cols = [c for c in cols_to_keep if c in final_df.columns]
    final_df = final_df[existing_cols]
    for rt_col in ("predicted_rt", "delta_rt", "delta_rt_model"):
        if rt_col in final_df.columns:
            final_df[rt_col] = 0.0

    # 保存为 parquet
    final_df.to_parquet(out_parquet, index=False)


def run_gen_one(gen_script: str, raw: str, sage: str, fp: str, out_parquet: str):
    """调用 1_gen_parquet.py 对单样本进行merge处理"""
    # We now call the internal function directly to support the complex logic
    try:
        process_single_sample(raw, sage, fp, out_parquet)
    except Exception as e:
        print(f"Error processing {raw}: {e}")
        raise e


class ParquetChunkWriter:
    """
    高效写入器：
    将多个 DataFrame 累加，直到达到 chunk_rows 后，
    才将其写入一个新的 Parquet 文件，解决物理碎文件过多的问题。
    """

    def __init__(self, out_dir: str, prefix: str, chunk_rows: int):
        self.out_dir = out_dir
        self.prefix = prefix
        self.chunk_rows = chunk_rows
        self.buffers = []   # 用于存放待写入的 DataFrame 列表
        self.buffer_rows = 0
        self.file_idx = 0
        os.makedirs(out_dir, exist_ok=True)

    def _flush_if_needed(self):
        """核心逻辑：当 buffer 超过设定行数时，切分并刷入硬盘。"""
        if self.buffer_rows < self.chunk_rows:
            return

        merged = pd.concat(self.buffers, ignore_index=True)
        # 循环取出固定行数的块
        while len(merged) >= self.chunk_rows:
            chunk = merged.iloc[: self.chunk_rows]
            out_file = os.path.join(self.out_dir, f"{self.prefix}.{self.file_idx:05d}.parquet")
            chunk.to_parquet(out_file, index=False)
            self.file_idx += 1
            merged = merged.iloc[self.chunk_rows :].reset_index(drop=True)

        # 剩下不足一块的数据存回 buffer
        self.buffers = [merged] if len(merged) > 0 else []
        self.buffer_rows = len(merged)

    def add(self, df: pd.DataFrame):
        """向写入器添加新数据。"""
        if len(df) == 0:
            return
        self.buffers.append(df.reset_index(drop=True))
        self.buffer_rows += len(df)
        self._flush_if_needed()

    def finalize(self):
        """处理结束时，将最后 buffer 中残留的数据强行落盘。"""
        if self.buffer_rows == 0:
            return
        merged = pd.concat(self.buffers, ignore_index=True)
        out_file = os.path.join(self.out_dir, f"{self.prefix}.{self.file_idx:05d}.parquet")
        merged.to_parquet(out_file, index=False)
        self.file_idx += 1
        self.buffers = []
        self.buffer_rows = 0


def normalize_columns_for_convert(parquet_dir: str):
    """
    字段标准化修复：
    在执行 3_convert_parquet2pkl.py 之前，确保所有 Parquet 文件包含
    delta_rt_model, predicted_rt, weight 等必要字段，防止后续报错。
    """
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not files:
        return 0

    fixed = 0
    for fp in files:
        df = pd.read_parquet(fp)
        changed = False

        # 标准化并屏蔽 RT 相关字段
        if "delta_rt_model" not in df.columns:
            df["delta_rt_model"] = 0.0
            changed = True
        elif (df["delta_rt_model"] != 0).any():
            df["delta_rt_model"] = 0.0
            changed = True
        if "delta_rt" in df.columns and (df["delta_rt"] != 0).any():
            df["delta_rt"] = 0.0
            changed = True

        # 补齐缺省值字段
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
    """
    步骤1.2总数验证：
    1) 校验生成文件数量是否与样本数一致；
    2) 校验是否存在空文件；
    3) 统计并打印总行数，便于和历史运行对账。
    """
    psm_files = sorted(glob.glob(os.path.join(psm_out_dir, "*.parquet")))

    if len(psm_files) != expected_case_num:
        raise RuntimeError(
            f"步骤1.2输出文件数异常: 期望={expected_case_num}, 实际={len(psm_files)}"
        )

    total_rows = 0
    empty_files = []

    for fp in psm_files:
        # 只读取一列即可计数，避免无谓的内存开销。
        n_rows = len(pd.read_parquet(fp, columns=["scan"]))
        total_rows += n_rows
        if n_rows == 0:
            empty_files.append(os.path.basename(fp))

    if empty_files:
        raise RuntimeError(
            "步骤1.2存在空parquet文件: " + ", ".join(empty_files[:20])
            + (" ..." if len(empty_files) > 20 else "")
        )

    print(
        f"1.2 总数验证通过: 文件数={len(psm_files)}, 总行数={total_rows}"
    )
    return total_rows


def main():
    args = parse_args()

    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(args.psm_out_dir, exist_ok=True)
    os.makedirs(args.pkl_out_dir, exist_ok=True)

    convert_script = os.path.join(work_dir, "pipeline", "2_convert_pkl", "3_convert_parquet2pkl.py")
    if not os.path.exists(convert_script):
        raise FileNotFoundError(f"转换脚本不存在: {convert_script}")

    print(f"q-value 阈值: spectrum_q <= {args.q_threshold}")

    # ===== 阶段1: 按样本生成 PSM parquet（sage 1% FDR 筛选） =====
    print("=" * 60)
    print(f"阶段1: 按样本生成 PSM parquet (sage spectrum_q <= {args.q_threshold})")
    print("=" * 60)

    triplets = collect_triplets(args.mzml_root, args.only_case)
    if not triplets:
        raise RuntimeError("未找到可用的 *_sage/_fp/_rawspectrum 三元组文件")
    print(f"发现可处理样本数: {len(triplets)}")

    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        for case_dir, raw, sage, fp in triplets:
            out_file = os.path.join(args.psm_out_dir, f"{case_dir}.parquet")
            futures.append(executor.submit(run_gen_one, "", raw, sage, fp, out_file))

        for i, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
            fut.result()
            if i % 10 == 0 or i == len(futures):
                print(f"  阶段1 已完成: {i}/{len(futures)}")

    stage1_total_rows = validate_stage12_outputs(args.psm_out_dir, len(triplets))

    # ===== 阶段2: 全部合并为 train（无验证集） =====
    print("=" * 60)
    print("阶段2: 全部数据合并为 train（无 val 拆分）")
    print("=" * 60)

    psm_files = sorted(glob.glob(os.path.join(args.psm_out_dir, "*.parquet")))
    if not psm_files:
        raise RuntimeError("阶段1 未生成任何 parquet 文件")

    train_parquet_dir = os.path.join(args.psm_out_dir, "train_merged")
    if os.path.exists(train_parquet_dir):
        shutil.rmtree(train_parquet_dir)

    train_writer = ParquetChunkWriter(train_parquet_dir, "train", args.rows_per_pkl)

    for fp in psm_files:
        df = pd.read_parquet(fp)
        if len(df) == 0:
            continue
        df = df.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)
        train_writer.add(df)

    train_writer.finalize()

    train_files = sorted(glob.glob(os.path.join(train_parquet_dir, "*.parquet")))
    train_rows = sum(len(pd.read_parquet(fp, columns=["scan"])) for fp in train_files)
    print(f"train parquet 文件数: {len(train_files)}, 总行数={train_rows:,}")
    if train_rows != stage1_total_rows:
        raise RuntimeError(f"总数不一致: stage1={stage1_total_rows}, train={train_rows}")

    if args.skip_pkl:
        print("已跳过 pkl 编码。")
        return

    # ===== 阶段3: 转换为 pkl.gz =====
    print("=" * 60)
    print("阶段3: 转换为 pkl.gz")
    print("=" * 60)

    fixed = normalize_columns_for_convert(train_parquet_dir)
    print(f"字段标准化: 修复 {fixed} 个文件")

    train_pkl_dir = os.path.join(args.pkl_out_dir, "train")
    os.makedirs(train_pkl_dir, exist_ok=True)

    run_cmd(
        [
            sys.executable, convert_script,
            "--file_dir", train_parquet_dir,
            "--config", args.config,
            "--task_name", "train",
            "--ncores", "4",
            "--save_dir", train_pkl_dir,
        ],
        cwd=work_dir,
    )

    print(f"train pkl.gz 文件数: {len(glob.glob(os.path.join(train_pkl_dir, '*.pkl.gz')))}")

    if not args.keep_intermediate:
        psm_dir = os.path.abspath(args.psm_out_dir)
        if os.path.exists(psm_dir):
            shutil.rmtree(psm_dir, ignore_errors=True)
            print(f"已清理中间目录: {psm_dir}")

    print("全部完成: sage 1%FDR 筛选 -> 全量 train pkl.gz")


if __name__ == "__main__":
    main()
