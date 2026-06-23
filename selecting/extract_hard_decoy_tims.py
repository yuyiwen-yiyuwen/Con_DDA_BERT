"""
功能：从timsTOF分组parquet提取hard decoy+target+regular decoy，target:decoy=1:1
输入：
    --db_dir /home/yiwen/AIPC/database/tims
    --config /home/yiwen/AIPC/scripts/organized_attantion/config/model.yaml
    --output_dir /home/yiwen/AIPC/scripts/organized_attantion/data/dataset/hard_decoy/tims
输出：
    data/dataset/hard_decoy/tims/*.pkl.gz（8文件，~183万target+183万decoy 1:1）
    日志 > /tmp/hard_decoy_tims.log
"""

import argparse, glob, gzip, os, pickle, shutil, sys
import numpy as np
import pandas as pd
import torch, yaml
from tqdm import tqdm

sys.path.insert(0, "/home/yiwen/AIPC/scripts/organized_attantion")
from model.transformer.dataset import SpectrumDataset, collate_batch_weight_deltaRT

N_PEAKS = 200
Q_THRESH = 0.01


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db_dir", type=str, default="/home/yiwen/AIPC/database/tims")
    p.add_argument("--config", type=str, default="/home/yiwen/AIPC/scripts/organized_attantion/config/model.yaml")
    p.add_argument("--output_dir", type=str, default="/home/yiwen/AIPC/scripts/organized_attantion/data/dataset/hard_decoy/tims")
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--rows_per_chunk", type=int, default=500000)
    return p.parse_args()


def normalize_precursor_sequence_column(df):
    """与 3_convert_parquet2pkl_tims.py 完全一致的序列标准化."""
    if "precursor_sequence" not in df.columns:
        return df
    seq = df["precursor_sequence"].astype(str)
    seq = seq.str.replace("[+42]-", "n[42]", regex=False)
    seq = seq.str.replace("C[+57.0216]", "C[57.02]", regex=False)
    seq = seq.str.replace("M[+15.9949]", "M[15.99]", regex=False)
    seq = seq.str.replace("N[+0.9840]", "N[.98]", regex=False)
    seq = seq.str.replace("Q[+0.9840]", "Q[.98]", regex=False)
    seq = seq.str.replace("cC", "C[57.02]", regex=False)
    seq = seq.str.replace("oxM", "M[15.99]", regex=False)
    seq = seq.str.replace("M(ox)", "M[15.99]", regex=False)
    seq = seq.str.replace("deamN", "N[.98]", regex=False)
    seq = seq.str.replace("deamQ", "Q[.98]", regex=False)
    seq = seq.str.replace("a", "X", regex=False)
    out = df.copy()
    out["precursor_sequence"] = seq
    return out


def _to_list(x):
    """与 1_gen_parquet_tims.py 一致的 _to_list."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    if pd.isna(x):
        return []
    return [x]


def unnest_sample(df):
    """展开 grouped 格式, 与 1_gen_parquet_tims.py 的 _iter_candidate_records 逻辑一致."""
    rows = []
    for row in df.itertuples(index=False):
        base = {
            'scan': int(row.scan),
            'precursor_mz': float(row.precursor_mz),
            'mz_array': row.mz_array,
            'intensity_array': row.intensity_array,
            'rt': float(row.rt),
        }
        charges = _to_list(row.charge)
        peptides = _to_list(row.peptide)
        labels = _to_list(row.label)
        scores = _to_list(row.sage_discriminant_score)
        spectrum_qs = _to_list(row.spectrum_q)

        cand_len = min(len(charges), len(peptides), len(labels), len(scores), len(spectrum_qs))
        if cand_len == 0:
            continue

        for i in range(cand_len):
            pep = peptides[i]
            if pep is None or (isinstance(pep, float) and np.isnan(pep)):
                continue
            q_val = pd.to_numeric(spectrum_qs[i], errors='coerce')
            score = pd.to_numeric(scores[i], errors='coerce')
            chg = pd.to_numeric(charges[i], errors='coerce')
            if pd.isna(q_val) or pd.isna(score) or pd.isna(chg):
                continue

            rows.append({**base,
                'charge': int(chg),
                'peptide': str(pep),
                'label': bool(labels[i]),
                'spectrum_q': float(q_val),
                'sage_discriminant_score': float(score),
            })
    return pd.DataFrame(rows)


def process_sample(parquet_path):
    """处理单个 tims 样本, 返回 (hd_df, ct_df, rd_df)"""
    df = pd.read_parquet(parquet_path)
    df = unnest_sample(df)
    df['psm_id'] = df['scan'].astype(str) + '_' + df['peptide']

    # 高可信 PSM: Sage label=True & q<=0.01
    high = df[(df['label'] == True) & (df['spectrum_q'] <= Q_THRESH)]
    if len(high) == 0:
        return None, None, None
    confirmed_seqs = set(high['peptide'])
    identified = set(high['psm_id'])

    # top-1 per scan
    df_sorted = df.sort_values(by=['scan', 'sage_discriminant_score'], ascending=[True, False])
    top1 = set(df_sorted.drop_duplicates(subset='scan')['psm_id'])

    # hard decoy
    candidates = df[~df['psm_id'].isin(top1)]
    candidates = candidates[~candidates['psm_id'].isin(identified)]
    hd = candidates[candidates['peptide'].isin(confirmed_seqs)]
    if len(hd) == 0:
        return None, None, None
    hd = hd.sort_values(by=['peptide', 'sage_discriminant_score'], ascending=[True, False])
    hd = hd.drop_duplicates(subset='peptide')

    # corresponding targets
    hd_seqs = set(hd['peptide'])
    ct = high[high['peptide'].isin(hd_seqs)]
    if len(ct) == 0:
        return None, None, None

    # regular decoys
    rd = df[(df['label'] == False) & (~df['psm_id'].isin(identified))]

    def _make_df(d, label_val):
        d = d.copy()
        d = d[(d['charge'] >= 2) & (d['charge'] <= 5)]
        d['precursor_sequence'] = d['peptide']
        d['label'] = label_val
        d['weight'] = 1.0
        d['unmask'] = 0
        d['predicted_rt'] = 0.0
        d['delta_rt'] = 0.0
        cols = ['scan', 'precursor_mz', 'charge', 'rt', 'mz_array', 'intensity_array',
                'precursor_sequence', 'label', 'weight', 'unmask',
                'predicted_rt', 'delta_rt', 'sage_discriminant_score', 'spectrum_q']
        return d[cols]

    return _make_df(hd, 0), _make_df(ct, 1), _make_df(rd, 0) if len(rd) > 0 else None


class ParquetChunkWriter:
    def __init__(self, out_dir, prefix, chunk_rows):
        self.out_dir = out_dir; self.prefix = prefix; self.chunk_rows = chunk_rows
        self.buffer = []; self.buffer_rows = 0; self.file_idx = 0
        os.makedirs(out_dir, exist_ok=True)
    def add(self, df):
        if len(df) == 0: return
        self.buffer.append(df.reset_index(drop=True)); self.buffer_rows += len(df); self._flush()
    def _flush(self):
        if self.buffer_rows < self.chunk_rows: return
        m = pd.concat(self.buffer, ignore_index=True)
        while len(m) >= self.chunk_rows:
            c = m.iloc[:self.chunk_rows]
            c.to_parquet(os.path.join(self.out_dir, f"{self.prefix}.{self.file_idx:05d}.parquet"), index=False)
            self.file_idx += 1; m = m.iloc[self.chunk_rows:]
        self.buffer = [m] if len(m) > 0 else []; self.buffer_rows = len(m)
    def finalize(self):
        if self.buffer_rows == 0: return
        m = pd.concat(self.buffer, ignore_index=True)
        m.to_parquet(os.path.join(self.out_dir, f"{self.prefix}.{self.file_idx:05d}.parquet"), index=False)
        self.file_idx += 1; self.buffer = []; self.buffer_rows = 0


def parquet_to_pklgz(pp, op, s2i, npk):
    df = pd.read_parquet(pp)
    if len(df) == 0: return 0
    df = normalize_precursor_sequence_column(df)
    ds = SpectrumDataset(df, s2i, npk, need_label=True, need_weight=True, need_deltaRT=True, need_unmask=True)
    r = collate_batch_weight_deltaRT(ds)
    if len(r) == 8: sp, sm, pr, tk, pep, lb, wt, um = r
    else: sp, sm, pr, tk, pep, lb, wt = r; um = torch.zeros_like(lb)
    if pr.shape[-1] > 2: pr = pr.clone(); pr[..., 2:] = 0
    out = {'spectra': sp.numpy(), 'spectra_mask': sm.numpy(), 'precursors': pr.numpy(),
           'tokens': tk.numpy(), 'peptides': pep, 'label': lb.numpy(), 'weight': wt.numpy(), 'unmask': um.numpy()}
    with gzip.open(op, 'wb') as f: pickle.dump(out, f, protocol=4)
    return len(df)


def main():
    args = parse_args()
    with open(args.config) as f: config = yaml.safe_load(f)
    vocab = ['<pad>', '<mask>'] + list(config["residues"].keys()) + ['<unk>']
    s2i = {v: i for i, v in enumerate(vocab)}
    npk = int(config.get("n_peaks", N_PEAKS))

    files = sorted(glob.glob(os.path.join(args.db_dir, "*.parquet")))
    print(f"发现 {len(files)} 个文件")

    tmp = os.path.join(args.output_dir, "_parquet_tmp")
    writer = ParquetChunkWriter(tmp, "hard_decoy", args.rows_per_chunk)
    total_hd, total_ct, total_rd_avail = 0, 0, 0
    all_rd = []

    for fp in tqdm(files, desc="提取 HD+CT+RD"):
        try: hd, ct, rd = process_sample(fp)
        except Exception as e: print(f"  {os.path.basename(fp)}: {e}"); continue
        if hd is None: continue
        writer.add(pd.concat([hd, ct], ignore_index=True))
        total_hd += len(hd); total_ct += len(ct)
        if rd is not None and len(rd) > 0: all_rd.append(rd); total_rd_avail += len(rd)

    writer.finalize()
    print(f"\nHD={total_hd:,}  CT={total_ct:,}  RD可用={total_rd_avail:,}")

    rd_needed = total_ct - total_hd
    if rd_needed > 0 and all_rd:
        all_rd_df = pd.concat(all_rd, ignore_index=True)
        if len(all_rd_df) >= rd_needed:
            all_rd_df = all_rd_df.sort_values('sage_discriminant_score', ascending=False).reset_index(drop=True)
            half = rd_needed // 2
            rd_sel = pd.concat([all_rd_df.iloc[:half], all_rd_df.iloc[half:].sample(n=rd_needed - half, random_state=args.random_state)], ignore_index=True)
            print(f"一半高分:{half:,}  一半随机:{rd_needed-half:,}")
        else:
            rd_sel = all_rd_df
            print(f"警告: 可用不足, 全用 {len(all_rd_df):,}")
        writer.add(rd_sel)
    writer.finalize()

    pf = sorted(glob.glob(os.path.join(tmp, "*.parquet")))
    print(f"Parquet: {len(pf)} 文件")
    os.makedirs(args.output_dir, exist_ok=True)
    total_rows = 0
    for fp in tqdm(pf, desc="parquet→pkl.gz"):
        bn = os.path.basename(fp).replace(".parquet", "")
        total_rows += parquet_to_pklgz(fp, os.path.join(args.output_dir, f"{bn}.pkl.gz"), s2i, npk)

    labels_all = []
    for fp in sorted(glob.glob(os.path.join(args.output_dir, "*.pkl.gz"))):
        with gzip.open(fp, 'rb') as f: labels_all.append(pickle.load(f)['label'])
    labels_all = np.concatenate(labels_all).flatten()
    print(f"最终: target={int((labels_all>0.5).sum()):,}  decoy={int((labels_all<=0.5).sum()):,}  比例={((labels_all>0.5).sum()/max((labels_all<=0.5).sum(),1)):.2f}:1")
    shutil.rmtree(tmp); print(f"完成: {args.output_dir}")


if __name__ == "__main__":
    main()
