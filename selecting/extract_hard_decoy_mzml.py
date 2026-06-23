"""
功能：从mzml数据库提取hard decoy+target+regular decoy，target:decoy=1:1
输入：
    --sample_root /home/yiwen/AIPC/database/mzml
    --config /home/yiwen/AIPC/scripts/organized_attantion/config/model.yaml
    --output_dir /home/yiwen/AIPC/scripts/organized_attantion/data/dataset/hard_decoy/mzml
输出：
    data/dataset/hard_decoy/mzml/hard_decoy.XXXXX.pkl.gz（7文件，~139万target+139万decoy 1:1）
    临时parquet目录自动删除
"""

import argparse
import glob
import gzip
import math
import os
import pickle
import shutil
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, "/home/yiwen/AIPC/scripts/organized_attantion")
from model.transformer.dataset import SpectrumDataset, collate_batch_weight_deltaRT

N_PEAKS = 200


def parse_args():
    p = argparse.ArgumentParser(description="Extract hard decoy dataset from mzml database")
    p.add_argument("--mzml_root", type=str,
                   default="/home/yiwen/AIPC/database/mzml")
    p.add_argument("--config", type=str,
                   default="/home/yiwen/AIPC/scripts/organized_attantion/config/model.yaml")
    p.add_argument("--output_dir", type=str,
                   default="/home/yiwen/AIPC/scripts/organized_attantion/data/hard_decoy/mzml")
    p.add_argument("--q_thresh", type=float, default=0.01)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--rows_per_chunk", type=int, default=500000)
    p.add_argument("--max_samples", type=int, default=0)
    return p.parse_args()


def make_output_df(df):
    """标准化 DataFrame 输出列"""
    df['cleaned_sequence'] = (
        df['precursor_sequence'].astype(str)
        .str.replace('n[42]', '', regex=False)
        .str.replace('N[.98]', 'N', regex=False)
        .str.replace('Q[.98]', 'Q', regex=False)
        .str.replace('M[15.99]', 'M', regex=False)
        .str.replace('C[57.02]', 'C', regex=False)
    )
    df['sequence_len'] = df['cleaned_sequence'].apply(len)
    df = df[(df['sequence_len'] >= 7) & (df['sequence_len'] <= 50)]
    df = df[(df['charge'] >= 2) & (df['charge'] <= 5)]
    df['predicted_rt'] = 0.0
    df['delta_rt'] = 0.0
    df['unmask'] = 0
    cols = ['scan', 'precursor_mz', 'charge', 'rt', 'mz_array', 'intensity_array',
            'precursor_sequence', 'label', 'weight', 'unmask',
            'predicted_rt', 'delta_rt', 'sage_discriminant_score', 'spectrum_q']
    return df[[c for c in cols if c in df.columns]]


def process_sample(sage_path, fp_path, raw_path, q_thresh):
    """
    处理单个样本.
    返回 (hard_decoy_df, corr_target_df, regular_decoy_df) 或 (None, None, None)
    """
    sage = pd.read_parquet(sage_path)
    fp = pd.read_parquet(fp_path)
    raw = pd.read_parquet(raw_path)

    sage['psm_id'] = sage['scan'].astype(str) + '_' + sage['precursor_sequence']
    fp['psm_id'] = fp['scan'].astype(int).astype(str) + '_' + fp['detect_sequence']

    # --- 高可信 PSM: FP ∩ Sage ---
    fp_high = fp[fp['q-value'] <= q_thresh]
    sage_high = sage[(sage['label'] == 1) & (sage['spectrum_q'] <= q_thresh)]
    both_ids = set(fp_high['psm_id']) & set(sage_high['psm_id'])

    if len(both_ids) == 0:
        return None, None, None

    fp_both = fp_high[fp_high['psm_id'].isin(both_ids)]
    sage_both = sage_high[sage_high['psm_id'].isin(both_ids)]
    confirmed_seqs = set(fp_both['detect_sequence']) | set(sage_both['precursor_sequence'])

    # --- 排除集合 (并集) ---
    identified = set(fp_high['psm_id']) | set(sage_high['psm_id'])

    # --- top-1 per scan ---
    sage_sorted = sage.sort_values(by=['scan', 'sage_discriminant_score'], ascending=[True, False])
    top1 = set(sage_sorted.drop_duplicates(subset='scan')['psm_id'])

    # --- hard decoy ---
    candidates = sage[~sage['psm_id'].isin(top1)]
    candidates = candidates[~candidates['psm_id'].isin(identified)]
    hd = candidates[candidates['precursor_sequence'].isin(confirmed_seqs)]
    if len(hd) == 0:
        return None, None, None

    hd = hd.sort_values(by=['precursor_sequence', 'sage_discriminant_score'], ascending=[True, False])
    hd = hd.drop_duplicates(subset='precursor_sequence')

    # --- corresponding targets (from both-identified) ---
    hd_seqs = set(hd['precursor_sequence'])
    ct = sage_both[sage_both['precursor_sequence'].isin(hd_seqs)]

    if len(ct) == 0:
        return None, None, None

    # --- regular decoys (Sage label=0, not in identified) ---
    rd = sage[(sage['label'] == 0) & (~sage['psm_id'].isin(identified))]
    if len(rd) == 0:
        # Fallback: use all label=0 not in identified (shouldn't happen)
        rd = sage[(sage['label'] == 0) & (~sage['psm_id'].isin(identified))]

    # --- 合并 raw spectrum ---
    raw['scan'] = raw['scan'].astype(int)
    raw_cols = ['scan', 'precursor_mz', 'mz_array', 'intensity_array', 'rt']

    hd = hd.merge(raw[raw_cols], on='scan', how='inner')
    ct = ct.merge(raw[raw_cols], on='scan', how='inner')
    rd = rd.merge(raw[raw_cols], on='scan', how='inner')

    if len(hd) == 0 or len(ct) == 0:
        return None, None, None

    hd['label'] = 0; hd['weight'] = 1.0
    ct['label'] = 1; ct['weight'] = 1.0
    rd['label'] = 0; rd['weight'] = 1.0

    hd_out = make_output_df(hd)
    ct_out = make_output_df(ct)
    rd_out = make_output_df(rd) if len(rd) > 0 else None

    return hd_out, ct_out, rd_out


class ParquetChunkWriter:
    def __init__(self, out_dir, prefix, chunk_rows):
        self.out_dir = out_dir
        self.prefix = prefix
        self.chunk_rows = chunk_rows
        self.buffer = []
        self.buffer_rows = 0
        self.file_idx = 0
        os.makedirs(out_dir, exist_ok=True)

    def add(self, df):
        if len(df) == 0:
            return
        self.buffer.append(df.reset_index(drop=True))
        self.buffer_rows += len(df)
        self._flush()

    def _flush(self):
        if self.buffer_rows < self.chunk_rows:
            return
        merged = pd.concat(self.buffer, ignore_index=True)
        while len(merged) >= self.chunk_rows:
            chunk = merged.iloc[:self.chunk_rows]
            path = os.path.join(self.out_dir, f"{self.prefix}.{self.file_idx:05d}.parquet")
            chunk.to_parquet(path, index=False)
            self.file_idx += 1
            merged = merged.iloc[self.chunk_rows:]
        self.buffer = [merged] if len(merged) > 0 else []
        self.buffer_rows = len(merged)

    def finalize(self):
        if self.buffer_rows == 0:
            return
        merged = pd.concat(self.buffer, ignore_index=True)
        path = os.path.join(self.out_dir, f"{self.prefix}.{self.file_idx:05d}.parquet")
        merged.to_parquet(path, index=False)
        self.file_idx += 1
        self.buffer = []
        self.buffer_rows = 0


def parquet_to_pklgz(parquet_path, output_path, s2i, n_peaks):
    df = pd.read_parquet(parquet_path)
    if len(df) == 0:
        return 0

    ds = SpectrumDataset(df, s2i, n_peaks, need_label=True, need_weight=True,
                         need_deltaRT=True, need_unmask=True)
    result = collate_batch_weight_deltaRT(ds)
    if len(result) == 8:
        spectra, spectra_mask, precursors, tokens, peptides, label, weight, unmask = result
    else:
        spectra, spectra_mask, precursors, tokens, peptides, label, weight = result
        unmask = torch.zeros_like(label)

    if precursors.shape[-1] > 2:
        precursors = precursors.clone()
        precursors[..., 2:] = 0

    out = {
        'spectra': spectra.numpy(),
        'spectra_mask': spectra_mask.numpy(),
        'precursors': precursors.numpy(),
        'tokens': tokens.numpy(),
        'peptides': peptides,
        'label': label.numpy(),
        'weight': weight.numpy(),
        'unmask': unmask.numpy(),
    }
    with gzip.open(output_path, 'wb') as f:
        pickle.dump(out, f, protocol=4)
    return len(df)


def discover_sample_dirs(mzml_root):
    dirs = []
    for d in sorted(os.listdir(mzml_root)):
        dp = os.path.join(mzml_root, d)
        if not os.path.isdir(dp):
            continue
        sage = os.path.join(dp, f"{d}_sage.parquet")
        fp = os.path.join(dp, f"{d}_fp.parquet")
        raw = os.path.join(dp, f"{d}_rawspectrum.parquet")
        if os.path.exists(sage) and os.path.exists(fp) and os.path.exists(raw):
            dirs.append((d, dp, sage, fp, raw))
    return dirs


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    vocab = ['<pad>', '<mask>'] + list(config["residues"].keys()) + ['<unk>']
    s2i = {v: i for i, v in enumerate(vocab)}
    n_peaks = int(config.get("n_peaks", N_PEAKS))

    sample_dirs = discover_sample_dirs(args.mzml_root)
    print(f"发现 {len(sample_dirs)} 个样本目录")

    if args.max_samples > 0:
        sample_dirs = sample_dirs[:args.max_samples]
        print(f"限制处理前 {args.max_samples} 个样本")

    # --- 第一步: 提取 HD, CT, RD ---
    parquet_tmp = os.path.join(args.output_dir, "_parquet_tmp")
    os.makedirs(parquet_tmp, exist_ok=True)

    writer = ParquetChunkWriter(parquet_tmp, "hard_decoy", args.rows_per_chunk)
    total_hd, total_ct, total_rd_available = 0, 0, 0
    all_rd_dfs = []

    for d, dp, sage_p, fp_p, raw_p in tqdm(sample_dirs, desc="提取 HD+CT+RD"):
        try:
            hd, ct, rd = process_sample(sage_p, fp_p, raw_p, args.q_thresh)
        except Exception as e:
            print(f"  {d}: 错误 - {e}")
            continue

        if hd is None:
            continue

        # 先把 HD + CT 写入 parquet (这些是必须保留的)
        combined = pd.concat([hd, ct], ignore_index=True)
        writer.add(combined)
        total_hd += len(hd)
        total_ct += len(ct)

        # 收集 regular decoys 备用
        if rd is not None and len(rd) > 0:
            all_rd_dfs.append(rd)
            total_rd_available += len(rd)

    writer.finalize()

    print(f"\n提取完成:")
    print(f"  HD:    {total_hd:,}")
    print(f"  CT:    {total_ct:,}")
    print(f"  RD可用: {total_rd_available:,}")

    # --- 第二步: 用一半一半原则选 regular decoy ---
    rd_needed = total_ct - total_hd
    if rd_needed > 0 and len(all_rd_dfs) > 0:
        all_rd = pd.concat(all_rd_dfs, ignore_index=True)
        print(f"  需要常规 decoy: {rd_needed:,}, 可用: {len(all_rd):,}")

        if len(all_rd) >= rd_needed:
            all_rd = all_rd.sort_values(by='sage_discriminant_score', ascending=False).reset_index(drop=True)
            half = rd_needed // 2

            rd_high = all_rd.iloc[:half]
            rest = all_rd.iloc[half:]
            rd_random = rest.sample(n=rd_needed - half, random_state=args.random_state)

            rd_selected = pd.concat([rd_high, rd_random], ignore_index=True)
            print(f"  一半高分: {len(rd_high):,}  一半随机: {len(rd_random):,}")
        else:
            print(f"  警告: 可用 decoy 不足, 使用全部 {len(all_rd):,} 条")
            rd_selected = all_rd
    elif rd_needed > 0:
        print(f"  警告: 需要 {rd_needed:,} 条 decoy 但无可用来源")
        rd_selected = None
    else:
        print(f"  HD >= CT, 无需补充 decoy")
        rd_selected = None

    # --- 第三步: 把选中的 RD 也写入 parquet ---
    if rd_selected is not None and len(rd_selected) > 0:
        writer.add(rd_selected)

    writer.finalize()

    # --- 第四步: parquet → pkl.gz ---
    parquet_files = sorted(glob.glob(os.path.join(parquet_tmp, "*.parquet")))
    print(f"\nParquet: {len(parquet_files)} 文件")

    os.makedirs(args.output_dir, exist_ok=True)
    total_rows = 0
    for fp in tqdm(parquet_files, desc="转换 parquet→pkl.gz"):
        basename = os.path.basename(fp).replace(".parquet", "")
        out_path = os.path.join(args.output_dir, f"{basename}.pkl.gz")
        n = parquet_to_pklgz(fp, out_path, s2i, n_peaks)
        total_rows += n
    print(f"pkl.gz: {total_rows} 条 → {len(parquet_files)} 个文件")

    # 统计
    import gzip as _gz
    labels_all = []
    for fp in sorted(glob.glob(os.path.join(args.output_dir, "*.pkl.gz"))):
        with _gz.open(fp, 'rb') as f:
            d = pickle.load(f)
        labels_all.append(d['label'])
    labels_all = np.concatenate(labels_all).flatten()
    n_t, n_d = int((labels_all > 0.5).sum()), int((labels_all <= 0.5).sum())
    print(f"最终: target={n_t:,}, decoy={n_d:,}, 比例={n_t/max(n_d,1):.2f}:1")

    shutil.rmtree(parquet_tmp)
    print(f"\n完成! 输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
