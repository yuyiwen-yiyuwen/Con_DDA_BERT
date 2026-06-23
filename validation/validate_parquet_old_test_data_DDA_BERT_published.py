"""
功能：使用 DDA-BERT 发行版权重对 parquet 测试数据打分（无 label，含 delta_rt/predicted_rt）。
      输出 {basename}_pred.csv + all_pred.tsv。

与 validate_parquet_old_test_data.py 的区别：
    - 使用 DDA-BERT 发行版 checkpoint (mp_rank_00_model_states.pt)
    - 使用 dda_bert.transformer.model.DDA_BERT 而非自定义 MSGPT
    - delta_rt 和 predicted_rt 使用实际值（非零填充）
    - 不需要 label 列，输出仅 index + score

输入：
    --val_parquet_dir /home/yiwen/AIPC/test_data/bas_test_dataset
    --output_dir /home/yiwen/AIPC/scripts/organized_attantion/data/results_old_test_data/DDA-BERT-Pubilished
    --device cuda:0 --num_files 999 --batch_size 1024

输出：
    --output_dir 下含 {basename}_pred.csv + all_pred.tsv
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import polars as pl
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# 全部使用 DDA-BERT 发行版代码
from dda_bert.transformer.dataset import SpectrumDataset, collate_batch_weight_deltaRT_index
from dda_bert.transformer.model import DDA_BERT

# DDA-BERT 发行版硬编码的 vocab
DDA_BERT_VOCAB = [
    '<pad>', '<mask>',
    'A', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'Y',
    'C[57.02]', 'M[15.99]', 'N[.98]', 'Q[.98]', 'X', '<unk>',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="DDA-BERT Published: parquet -> _pred.csv -> all_pred.tsv"
    )
    parser.add_argument("--val_parquet_dir", type=str,
                        default="/home/yiwen/AIPC/test_data/bas_test_dataset")
    parser.add_argument("--checkpoint_path", type=str,
                        default="/home/yiwen/DDA-BERT/software/resource/model/mp_rank_00_model_states.pt")
    parser.add_argument("--config_path", type=str,
                        default="/home/yiwen/DDA-BERT/software/config/Model.yaml")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--output_dir", type=str,
                        default="/home/yiwen/AIPC/scripts/organized_attantion/data/results_old_test_data/DDA-BERT-Pubilished")
    parser.add_argument("--num_files", type=int, default=999)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--file_prefix", type=str, default="",
                        help="Only process parquet files whose basename starts with this prefix")
    return parser.parse_args()


def resolve_device(device_arg: str):
    device_arg = (device_arg or "cpu").strip().lower()
    default_cuda_idx = 0
    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA requested but not available, switching to CPU.")
            return torch.device("cpu")
        if device_arg == "cuda":
            idx = default_cuda_idx if default_cuda_idx < torch.cuda.device_count() else 0
            return torch.device(f"cuda:{idx}")
        if ":" in device_arg:
            try:
                gpu_idx = int(device_arg.split(":", 1)[1])
                if gpu_idx < 0 or gpu_idx >= torch.cuda.device_count():
                    gpu_idx = 0
                return torch.device(f"cuda:{gpu_idx}")
            except ValueError:
                return torch.device("cuda:0")
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    config["vocab"] = DDA_BERT_VOCAB
    return config


def load_model(config, device, checkpoint_path):
    model = DDA_BERT.load_pt(checkpoint_path, config)
    model.eval()
    model.to(torch.bfloat16)
    model.to(device)
    return model


def to_device(batch, device):
    """与 DDA-BERT eval_model.test_step 一致的输入处理。"""
    spectra, spectra_mask, precursors, tokens, _peptide, _label, _weight, indices = batch
    return (
        spectra.to(device, non_blocking=True).to(torch.bfloat16),
        spectra_mask.to(device, non_blocking=True).to(torch.bfloat16),
        precursors.to(device, non_blocking=True).to(torch.bfloat16),
        tokens.to(device, non_blocking=True),
        indices,
    )


def run_one_parquet(parquet_path, model, config, s2i, device, args):
    print(f"\n===== Processing: {parquet_path} =====")
    file_name = os.path.splitext(os.path.basename(parquet_path))[0]
    out_path = os.path.join(args.output_dir, f"{file_name}_pred.csv")

    if os.path.exists(out_path):
        print(f"Skip {parquet_path}: {out_path} already exists.")
        return True, file_name

    # 读取 parquet → 转换列名 → polars（DDA-BERT 发行版 dataset 只支持 polars）
    pdf = pd.read_parquet(parquet_path)
    pdf = pdf.rename(columns={"charge": "precursor_charge", "precursor_sequence": "modified_sequence"})
    if "index" not in pdf.columns:
        pdf["index"] = np.arange(len(pdf))
    if "weight" not in pdf.columns:
        pdf["weight"] = 1.0
    df = pl.from_pandas(pdf)

    n_peaks = int(config.get("n_peaks", 300))
    max_length = int(config.get("max_length", 50))

    dataset_all = SpectrumDataset(
        df=df, s2i=s2i, n_peaks=n_peaks, max_length=max_length,
        need_label=True, need_deltaRT=True, need_weight=True, need_index=True,
    )

    if args.max_samples is not None and args.max_samples > 0:
        num_samples = min(args.max_samples, len(dataset_all))
        dataset = Subset(dataset_all, list(range(num_samples)))
        print(f"Using {num_samples} samples (capped by --max_samples).")
    else:
        num_samples = len(dataset_all)
        dataset = dataset_all
        print(f"Using all {num_samples} samples.")

    if num_samples == 0:
        print(f"Skip {parquet_path}: no samples.")
        return False, file_name

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch_weight_deltaRT_index,
    )

    all_indices = []
    all_scores = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Scoring {os.path.basename(parquet_path)}"):
            spectra, spectra_mask, precursors, tokens, indices = to_device(batch, device)

            # 与 DDA-BERT eval_model.test_step 一致：bf16 autocast
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                dda_pred, _mask_pred = model.pred(spectra, spectra_mask, precursors, tokens)

            scores = dda_pred.detach().cpu().to(torch.float32).numpy().reshape(-1)
            all_scores.extend(scores.tolist())
            all_indices.extend(indices.cpu().numpy().tolist())

    if len(all_indices) == 0:
        print(f"Skip {parquet_path}: no predictions.")
        return False, file_name

    out_df = pd.DataFrame({
        "index": all_indices,
        "score": all_scores,
    })
    out_df.to_csv(out_path, header=False, index=False)
    print(f"Saved {out_path} ({len(out_df):,} rows)")
    return True, file_name


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print("Device:", device)

    if not os.path.isdir(args.val_parquet_dir):
        print(f"Validation parquet directory not found: {args.val_parquet_dir}")
        return
    if not os.path.exists(args.checkpoint_path):
        print(f"Checkpoint not found: {args.checkpoint_path}")
        return
    if not os.path.exists(args.config_path):
        print(f"Config not found: {args.config_path}. Run 'dda-bert init-config' first or ensure config/Model.yaml exists.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(args.config_path)
    s2i = {v: i for i, v in enumerate(config["vocab"])}

    print(f"Loading checkpoint: {args.checkpoint_path}")
    try:
        model = load_model(config, device, args.checkpoint_path)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    parquet_files = sorted(glob.glob(os.path.join(args.val_parquet_dir, "*.parquet")))
    if args.file_prefix:
        parquet_files = [f for f in parquet_files
                         if os.path.basename(f).startswith(args.file_prefix)]
        print(f"Filtered by prefix '{args.file_prefix}': {len(parquet_files)} files")
    if not parquet_files:
        print(f"No parquet files found in: {args.val_parquet_dir}")
        return

    total_files = len(parquet_files)
    if 0 < args.num_files < total_files:
        import random
        rng = random.Random(args.seed)
        parquet_files = rng.sample(parquet_files, args.num_files)
        parquet_files.sort()
        print(f"Randomly selected {args.num_files} / {total_files} files (seed={args.seed})")
    else:
        print(f"Found {len(parquet_files)} parquet files.")

    for parquet_path in parquet_files:
        try:
            run_one_parquet(parquet_path, model, config, s2i, device, args)
        except Exception as e:
            print(f"Failed on {parquet_path}: {e}")

    # ── 合并所有 _pred.csv → all_pred.tsv ──
    pred_files = sorted(glob.glob(os.path.join(args.output_dir, "*_pred.csv")))
    if pred_files:
        df_list = []
        for f in pred_files:
            try:
                tmp_df = pd.read_csv(f, header=None, names=["index", "score"])
                df_list.append(tmp_df)
            except Exception as e:
                print(f"Error reading {f}: {e}")

        if df_list:
            merged_df = pd.concat(df_list, ignore_index=True)
            merged_path = os.path.join(args.output_dir, "all_pred.tsv")
            merged_df.to_csv(merged_path, sep="\t", index=False)
            print(f"\nMerged TSV saved to: {merged_path} ({len(merged_df):,} rows)")

    print("All done.")


if __name__ == "__main__":
    main()
