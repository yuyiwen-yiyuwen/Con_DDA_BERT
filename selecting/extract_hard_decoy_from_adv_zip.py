"""
功能：从adv zip提取hard decoy+target（不含regular decoy），处理完自动删源zip；每15分钟cron自动运行
输入：
    --zip_root /home/yiwen/AIPC/database/adv
    --output_dir /home/yiwen/AIPC/scripts/organized_attantion/data/dataset/adv_hard_decoy
    --q_thresh 0.01
输出：
    data/dataset/adv_hard_decoy/batch_*.pkl.gz（940个文件，4134万行，target:decoy≈2:1）
    data/dataset/adv_hard_decoy/mzml/adv_hard_decoy_XXX.pkl.gz（合并后40个chunk，每100万行）
    data/dataset/adv_hard_decoy/mzml_random/adv_hard_decoy_balanced_XXX.pkl.gz（随机抽样平衡，5个chunk，250万+250万 1:1）
    日志 >> /tmp/adv_hard_decoy_cron.log
"""

import argparse
import glob
import gzip
import io
import os
import pickle
import shutil
import sys
import zipfile
import gc
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, "/home/yiwen/AIPC/scripts/organized_attantion")
from model.transformer.dataset import SpectrumDataset, collate_batch_weight_deltaRT

# ========== 从 config 读取 ==========
CONFIG_PATH = "/home/yiwen/AIPC/scripts/organized_attantion/config/model.yaml"
with open(CONFIG_PATH) as f:
    _config = yaml.safe_load(f)
_residues = _config.get("residues", {})
VOCAB = ['<pad>', '<mask>'] + list(_residues.keys()) + ['<unk>']
S2I = {v: i for i, v in enumerate(VOCAB)}
N_PEAKS = int(_config.get("n_peaks", 200))
MAX_LENGTH = int(_config.get("max_length", 50))
PROTON_MASS_AMU = 1.007276
del _config, _residues


def parse_args():
    p = argparse.ArgumentParser(description="从 adv zip 提取 hard_decoy + target（不含 RD，输出 pkl.gz）")
    p.add_argument("--zip_root", type=str,
                   default="/home/yiwen/AIPC/database/adv",
                   help="包含 adv zip 文件的目录")
    p.add_argument("--output_dir", type=str,
                   default="/home/yiwen/AIPC/scripts/organized_attantion/data/dataset/adv_hard_decoy",
                   help="输出目录")
    p.add_argument("--q_thresh", type=float, default=0.01)
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()


def df_to_pklgz_dict(df):
    """使用官方 SpectrumDataset + collate_batch_weight_deltaRT 转换, RT 置零"""
    if len(df) == 0:
        return None

    ds = SpectrumDataset(df, S2I, N_PEAKS, MAX_LENGTH,
                         need_label=True, need_weight=True,
                         need_deltaRT=True, need_unmask=True)
    r = collate_batch_weight_deltaRT(ds)
    if len(r) == 8:
        spectra, spectra_mask, precursors, tokens, peptides, label, weight, unmask = r
    else:
        spectra, spectra_mask, precursors, tokens, peptides, label, weight = r
        unmask = torch.zeros_like(label)

    # RT 特征置零，与训练/验证代码保持一致
    if precursors.shape[-1] > 2:
        precursors = precursors.clone()
        precursors[..., 2:] = 0

    return {
        'spectra': spectra.numpy(),
        'spectra_mask': spectra_mask.numpy(),
        'precursors': precursors.numpy(),
        'tokens': tokens.numpy(),
        'peptides': peptides,
        'label': label.numpy(),
        'weight': weight.numpy(),
        'unmask': unmask.numpy(),
    }


# ========== 从 zip 读 parquet ==========
def read_parquet_from_zip(zf, name):
    data = zf.read(name)
    return pd.read_parquet(io.BytesIO(data))


# ========== 单样本处理 ==========
def process_sample(zf, sample_name, q_thresh):
    base = sample_name + "/" + sample_name
    try:
        sage = read_parquet_from_zip(zf, base + "_sage.parquet")
        fp = read_parquet_from_zip(zf, base + "_fp.parquet")
        raw = read_parquet_from_zip(zf, base + "_rawspectrum.parquet")
    except Exception:
        return None, None, None

    sage['psm_id'] = sage['scan'].astype(str) + '_' + sage['precursor_sequence']
    fp['psm_id'] = fp['scan'].astype(int).astype(str) + '_' + fp['detect_sequence']

    fp_high = fp[fp['q-value'] <= q_thresh]
    sage_high = sage[(sage['label'] == 1) & (sage['spectrum_q'] <= q_thresh)]
    both_ids = set(fp_high['psm_id']) & set(sage_high['psm_id'])
    if len(both_ids) == 0:
        return None, None, None

    fp_both = fp_high[fp_high['psm_id'].isin(both_ids)]
    sage_both = sage_high[sage_high['psm_id'].isin(both_ids)]
    confirmed_seqs = set(fp_both['detect_sequence']) | set(sage_both['precursor_sequence'])
    identified = set(fp_high['psm_id']) | set(sage_high['psm_id'])

    sage_sorted = sage.sort_values(by=['scan', 'sage_discriminant_score'],
                                   ascending=[True, False])
    top1 = set(sage_sorted.drop_duplicates(subset='scan')['psm_id'])

    candidates = sage[~sage['psm_id'].isin(top1)]
    candidates = candidates[~candidates['psm_id'].isin(identified)]
    hd = candidates[candidates['precursor_sequence'].isin(confirmed_seqs)]
    if len(hd) == 0:
        return None, None, None

    hd = hd.sort_values(by=['precursor_sequence', 'sage_discriminant_score'],
                         ascending=[True, False])
    hd = hd.drop_duplicates(subset='precursor_sequence')

    hd_seqs = set(hd['precursor_sequence'])
    ct = sage_both[sage_both['precursor_sequence'].isin(hd_seqs)]
    if len(ct) == 0:
        return None, None, None

    rd = sage[(sage['label'] == 0) & (~sage['psm_id'].isin(identified))]

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

    return make_output_df(hd), make_output_df(ct), make_output_df(rd)


def make_output_df(df):
    df = df.copy()
    seq = df['precursor_sequence'].astype(str)
    cleaned = (
        seq.str.replace('n[42]', '', regex=False)
        .str.replace('N[.98]', 'N', regex=False)
        .str.replace('Q[.98]', 'Q', regex=False)
        .str.replace('M[15.99]', 'M', regex=False)
        .str.replace('C[57.02]', 'C', regex=False)
    )
    df['cleaned_sequence'] = cleaned
    df['sequence_len'] = cleaned.apply(len)
    df = df[(df['sequence_len'] >= 7) & (df['sequence_len'] <= 50)]
    df = df[(df['charge'] >= 2) & (df['charge'] <= 5)]
    df['predicted_rt'] = 0.0
    df['delta_rt'] = 0.0
    df['unmask'] = 0
    cols = ['scan', 'precursor_mz', 'charge', 'rt', 'mz_array', 'intensity_array',
            'precursor_sequence', 'label', 'weight', 'unmask',
            'predicted_rt', 'delta_rt', 'sage_discriminant_score', 'spectrum_q']
    return df[[c for c in cols if c in df.columns]].reset_index(drop=True)


def list_sample_names_in_zip(zf):
    names = set()
    for f in zf.infolist():
        parts = f.filename.split('/')
        if parts[0]:
            names.add(parts[0])
    return sorted(names)


# ========== 主流程 ==========
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 文件锁：防止并发运行
    lock_path = os.path.join(args.output_dir, ".lock")
    if os.path.exists(lock_path):
        print(f"[{pd.Timestamp.now()}] 上次运行未完成，跳过本次")
        return
    with open(lock_path, 'w') as lf:
        lf.write(str(os.getpid()))

    try:
        _main_impl(args)
    finally:
        if os.path.exists(lock_path):
            os.remove(lock_path)


def _main_impl(args):
    # 跳过未上传完成的 .filepart 文件
    all_files = sorted(glob.glob(os.path.join(args.zip_root, "*.zip")))
    zip_files = [f for f in all_files if not f.endswith('.filepart')]
    print(f"发现 {len(zip_files)} 个 zip 文件 (跳过 {len(all_files) - len(zip_files)} 个 .filepart)")

    total_hd, total_ct = 0, 0
    pkl_count, skip_count = 0, 0

    for zi, zip_path in enumerate(tqdm(zip_files, desc="提取 HD+CT → pkl.gz")):
        zip_name = os.path.basename(zip_path)
        zip_stem = os.path.splitext(zip_name)[0]
        out_path = os.path.join(args.output_dir, f"{zip_stem}.pkl.gz")

        # 已处理过则跳过并删除源 zip
        if os.path.exists(out_path):
            skip_count += 1
            try:
                os.remove(zip_path)
            except OSError:
                pass
            continue
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                sample_names = list_sample_names_in_zip(zf)
                zip_hd_dfs, zip_ct_dfs = [], []

                for sn in sample_names:
                    hd, ct, _ = process_sample(zf, sn, args.q_thresh)
                    if hd is None:
                        continue
                    zip_hd_dfs.append(hd)
                    zip_ct_dfs.append(ct)
                    total_hd += len(hd)
                    total_ct += len(ct)

                if zip_hd_dfs:
                    combined = pd.concat(zip_hd_dfs + zip_ct_dfs, ignore_index=True)
                    d = df_to_pklgz_dict(combined)
                    if d is not None:
                        with gzip.open(out_path, 'wb') as f:
                            pickle.dump(d, f, protocol=4)
                        pkl_count += 1
                        del d
                    del combined
                gc.collect()

                # 处理成功，删除源 zip 释放空间
                try:
                    os.remove(zip_path)
                except OSError:
                    pass

        except Exception as e:
            print(f"\n  处理 {zip_name} 时出错: {e}")
            continue

    print(f"\n===== 完成 =====")
    print(f"Hard Decoy:    {total_hd:,}")
    print(f"Corr. Target:  {total_ct:,}")
    print(f"新生成 {pkl_count} 个 pkl.gz, 跳过 {skip_count} 个已有 → {args.output_dir}")


if __name__ == "__main__":
    main()
