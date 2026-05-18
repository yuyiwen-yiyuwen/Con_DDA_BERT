#!/usr/bin/env python3
"""
Quick test: score a 100k subsample of tims data with msgpt_epoch_9.pt,
ensuring RT features are zeroed out. Reports 1% FDR identifications.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, "/home/yiwen/AIPC/scripts/organized_attantion")
from model.transformer_compare.model import MSGPT


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_state_dict_keys(state_dict):
    norm = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod."):]
        if nk.startswith("module."):
            nk = nk[len("module."):]
        norm[nk] = v
    return norm


def build_model(config, vocab_size):
    return MSGPT(
        dim_model=int(config.get("dim_model", 768)),
        n_head=int(config.get("n_head", 16)),
        dim_feedforward=int(config.get("dim_feedforward", 1024)),
        n_layers=int(config.get("n_layers", 9)),
        dropout=float(config.get("dropout", 0.0)),
        max_length=int(config.get("max_length", 50)),
        vocab_size=vocab_size,
        max_charge=int(config.get("max_charge", 10)),
    )


def calculate_q_values_and_fdr(scores, labels, fdr_threshold=0.01):
    """Compute q-values and count identifications at given FDR threshold."""
    idx = np.argsort(scores)[::-1]
    sorted_labels = labels[idx].astype(float)
    cum_targets = np.cumsum(sorted_labels)
    cum_decoys = np.cumsum(1.0 - sorted_labels)
    fdr = np.where(cum_targets > 0, cum_decoys / cum_targets, 0.0)
    q_values = np.minimum.accumulate(fdr[::-1])[::-1]

    eligible = np.where(q_values <= fdr_threshold)[0]
    if len(eligible) > 0:
        n_total = eligible[-1] + 1
        n_target = int(cum_targets[eligible[-1]])
    else:
        n_total = 0
        n_target = 0
    return n_target, n_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/home/yiwen/AIPC/scripts/attantion/checkpoints/msgpt_epoch_9.pt")
    parser.add_argument("--config_path", default="/home/yiwen/AIPC/scripts/attantion/model.yaml")
    parser.add_argument("--pkl_path",
                        default="/home/yiwen/AIPC/scripts/attantion/pkl_dataset_tims/tims_pkl_all/train/train.00000_train.pkl.gz")
    parser.add_argument("--n_samples", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load config & build model ---
    config = load_config(args.config_path)
    residues = config.get("residues", {})
    vocab = ["<pad>", "<mask>"] + list(residues.keys()) + ["<unk>"]
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")

    model = build_model(config, vocab_size).to(device)
    model.eval()

    # --- Load checkpoint ---
    print(f"Loading checkpoint: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    state_dict = normalize_state_dict_keys(state_dict)

    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded {len(state_dict)} parameters")

    # --- Load data & subsample ---
    print(f"Loading data: {args.pkl_path}")
    data = pd.read_pickle(args.pkl_path)
    n_available = len(data["label"])
    n_use = min(args.n_samples, n_available)
    print(f"Using {n_use:,} / {n_available:,} samples")

    # Slice first n_use
    spectra = torch.from_numpy(data["spectra"][:n_use])
    mask = torch.from_numpy(data["spectra_mask"][:n_use])
    precursors = torch.from_numpy(data["precursors"][:n_use])
    labels_raw = data["label"][:n_use]

    # --- ZERO RT features ---
    if precursors.shape[-1] > 2:
        precursors = precursors.clone()
        precursors[..., 2:] = 0.0
        print(f"RT features (cols 2-{precursors.shape[-1]-1}) explicitly zeroed.")
    else:
        print("Precursors only have {precursors.shape[-1]} dims, no RT to zero.")

    tokens = torch.from_numpy(data["tokens"][:n_use]).long()
    labels = torch.from_numpy(labels_raw.astype(np.float32))

    print(f"Target: {(labels_raw == 1).sum():,}, Decoy: {(labels_raw == 0).sum():,}")

    # --- Inference ---
    ds = TensorDataset(spectra, mask, precursors, tokens, labels)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for spec, msk, prec, tok, _ in tqdm(loader, desc="Scoring"):
            spec = spec.to(device, non_blocking=True)
            msk = msk.to(device, non_blocking=True)
            prec = prec.to(device, non_blocking=True)
            tok = tok.to(device, non_blocking=True)

            _, _, dda_logits, _ = model(spec, msk, prec, tok)
            all_probs.append(torch.sigmoid(dda_logits).cpu().numpy().reshape(-1))

    probs = np.concatenate(all_probs)

    # --- 1% FDR ---
    n_target, n_total = calculate_q_values_and_fdr(probs, labels_raw)
    print(f"\n{'='*50}")
    print(f"1% FDR results (n={n_use:,}):")
    print(f"  Target PSMs: {n_target:,}")
    print(f"  Total PSMs:  {n_total:,}")
    if n_total > 0:
        actual_fdr = round((1 - n_target / n_total) * 100, 2)
        print(f"  Actual FDR:  {actual_fdr}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
