import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import sys
sys.path.insert(0, "/home/yiwen/AIPC/scripts/organized_attantion")
from model.transformer.dataset import PROTON_MASS_AMU, SpectrumDataset, padding
from model.transformer_compare.model import MSGPT


def collate_batch_parquet(batch):
    spectra, precursor_mzs, precursor_charges, delta_rt, predicted_rt, tokens, _peptides, labels, _weights, indices = zip(*batch)

    spectra, spectra_mask = padding(list(spectra))
    tokens = torch.stack(tokens, dim=0)

    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - PROTON_MASS_AMU) * precursor_charges

    delta_rt = torch.zeros(len(delta_rt), dtype=torch.float32)
    predicted_rt = torch.zeros(len(predicted_rt), dtype=torch.float32)
    precursors = torch.vstack([precursor_masses, precursor_charges, delta_rt, predicted_rt]).T.float()

    labels = torch.tensor(labels, dtype=torch.float32)
    indices = torch.tensor(indices, dtype=torch.int64)
    return spectra, spectra_mask, precursors, tokens, labels, indices


def parse_args():
    parser = argparse.ArgumentParser(
        description="MSGPT Validation Script: parquet → _pred.csv → all_pred.tsv (matches predict_weight_zxf.py output format)"
    )

    parser.add_argument("--config", type=str,
                        default="/home/yiwen/AIPC/scripts/attantion/model.yaml")
    parser.add_argument("--val_parquet_dir", type=str,
                        default="/home/yiwen/AIPC/test_data/bas_test_dataset/")
    parser.add_argument("--checkpoint_path", type=str,
                        default="/home/yiwen/AIPC/scripts/attantion/checkpoints_compare/msgpt_epoch_8.pt")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--output_dir", type=str,
                        default="/home/yiwen/AIPC/test_results/dda_bert_compare")
    parser.add_argument("--num_files", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--file_prefix", type=str, default="",
                        help="Only process parquet files whose basename starts with this prefix, e.g. 'mzml_'")
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
    if not os.path.exists(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, "model.yaml")
        if os.path.exists(candidate):
            config_path = candidate
        else:
            print(f"Warning: Config file {config_path} not found.")
            return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_state_dict_keys(state_dict):
    normalized = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod."):]
        if nk.startswith("module."):
            nk = nk[len("module."):]
        normalized[nk] = v
    return normalized


def build_vocab_from_config(config):
    residues = config.get("residues", {})
    if residues:
        vocab = ["<pad>", "<mask>"] + list(residues.keys()) + ["<unk>"]
    else:
        vocab = ["<pad>", "<mask>", "<unk>"]
    s2i = {v: i for i, v in enumerate(vocab)}
    return vocab, s2i


def load_model(config, vocab_size, max_length, device, checkpoint_path):
    model = MSGPT(
        dim_model=int(config.get("dim_model", 768)),
        n_head=int(config.get("n_head", 16)),
        dim_feedforward=int(config.get("dim_feedforward", 1024)),
        n_layers=int(config.get("n_layers", 9)),
        dropout=float(config.get("dropout", 0.0)),
        max_length=max_length,
        vocab_size=vocab_size,
        max_charge=int(config.get("max_charge", 10)),
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    normalized_state_dict = normalize_state_dict_keys(state_dict)
    missing, unexpected = model.load_state_dict(normalized_state_dict, strict=False)
    if missing:
        print("Missing keys:", len(missing))
    if unexpected:
        print("Unexpected keys:", len(unexpected))
    loaded_count = len(normalized_state_dict) - len(unexpected)
    print(f"Loaded parameters: {loaded_count} / {len(model.state_dict())}")
    model.to(device)
    model.eval()
    return model


def run_one_parquet(parquet_path, model, config, s2i, device, args):
    print(f"\n===== Processing: {parquet_path} =====")
    file_name = os.path.splitext(os.path.basename(parquet_path))[0]
    out_path = os.path.join(args.output_dir, f"{file_name}_pred.csv")

    if os.path.exists(out_path):
        print(f"Skip {parquet_path}: {out_path} already exists.")
        return True

    df = pd.read_parquet(parquet_path)

    if "index" not in df.columns:
        df["index"] = np.arange(len(df))
    if "weight" not in df.columns:
        df["weight"] = 1.0
    if "label" not in df.columns:
        df["label"] = 1

    n_peaks = int(config.get("n_peaks", 150))
    max_length = int(config.get("max_length", 50))

    dataset_all = SpectrumDataset(
        df=df, s2i=s2i, n_peaks=n_peaks, max_length=max_length,
        need_label=True, need_deltaRT=True, need_weight=True, need_index=True,
    )

    if args.max_samples is not None and args.max_samples > 0:
        num_samples = min(args.max_samples, len(dataset_all))
        indices_subset = list(range(num_samples))
        dataset = Subset(dataset_all, indices_subset)
        print(f"Using {num_samples} samples (capped by --max_samples).")
    else:
        num_samples = len(dataset_all)
        indices_subset = None
        dataset = dataset_all
        print(f"Using all {num_samples} samples.")

    if num_samples == 0:
        print(f"Skip {parquet_path}: no samples.")
        return False

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch_parquet,
    )

    all_indices = []
    all_scores = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Validating {os.path.basename(parquet_path)}"):
            spectra, spectra_mask, precursors, tokens, _labels, indices = batch

            spectra = spectra.to(device, non_blocking=True)
            spectra_mask = spectra_mask.to(device, non_blocking=True)
            precursors = precursors.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)

            model_out = model(spectra, spectra_mask, precursors, tokens)

            if isinstance(model_out, tuple):
                if len(model_out) >= 3:
                    dda_logits = model_out[2]
                elif len(model_out) >= 1:
                    dda_logits = model_out[0]
                else:
                    raise RuntimeError("模型输出为空 tuple")
            else:
                dda_logits = model_out

            scores = torch.sigmoid(dda_logits).detach().cpu().numpy().reshape(-1)
            all_scores.extend(scores.tolist())
            all_indices.extend(indices.cpu().numpy().tolist())

    if len(all_indices) == 0:
        print(f"Skip {parquet_path}: no predictions.")
        return False

    out_df = pd.DataFrame({"index": all_indices, "score": all_scores})
    out_df.to_csv(out_path, header=False, index=False)
    print(f"Saved {out_path} ({len(out_df):,} rows)")
    return True


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

    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(args.config)
    vocab, s2i = build_vocab_from_config(config)
    max_length = int(config.get("max_length", 50))

    print(f"Loading checkpoint: {args.checkpoint_path}")
    try:
        model = load_model(config, len(vocab), max_length, device, args.checkpoint_path)
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
    if args.num_files > 0 and args.num_files < total_files:
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

    # ── 合并所有 _pred.csv → all_pred.tsv（与 predict_weight_zxf.py 一致）──
    pred_files = sorted(glob.glob(os.path.join(args.output_dir, "*_pred.csv")))
    if not pred_files:
        print("No pred files found. Skip merging.")
        return

    df_list = []
    for f in pred_files:
        try:
            tmp_df = pd.read_csv(f, header=None, names=["index", "score"])
            df_list.append(tmp_df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not df_list:
        print("No valid pred files to merge.")
        return

    merged_df = pd.concat(df_list, ignore_index=True)
    merged_path = os.path.join(args.output_dir, "all_pred.tsv")
    merged_df.to_csv(merged_path, sep="\t", index=False)
    print(f"\nMerged TSV saved to: {merged_path} ({len(merged_df):,} rows)")
    print("All done.")


if __name__ == "__main__":
    main()
