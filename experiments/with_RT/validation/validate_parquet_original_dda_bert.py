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
from model.transformer.model import MSGPT


def collate_batch_parquet(batch):
    """将 SpectrumDataset(need_label+need_deltaRT) 的输出拼成原始 MSGPT 输入。"""
    spectra, precursor_mzs, precursor_charges, delta_rt, predicted_rt, tokens, _peptides, labels, _weights = zip(*batch)

    spectra, spectra_mask = padding(list(spectra))
    tokens = torch.stack(tokens, dim=0)

    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - PROTON_MASS_AMU) * precursor_charges

    delta_rt = torch.tensor(delta_rt)
    predicted_rt = torch.tensor(predicted_rt)
    precursors = torch.vstack([precursor_masses, precursor_charges, delta_rt, predicted_rt]).T.float()

    labels = torch.tensor(labels, dtype=torch.float32)
    return spectra, spectra_mask, precursors, tokens, labels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Original DDA-BERT(MSGPT) validation: one parquet -> one TSV without dedup"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/model.yaml",
        help="Path to config file",
    )

    parser.add_argument(
        "--val_parquet_dir",
        type=str,
        default="/home/yiwen/AIPC/test_data/bas_test_dataset/",
        help="Directory containing validation parquet files",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/mzml_pt/checkpoints_mzml/msgpt_epoch_8.pt",
        help="Path to checkpoint file",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device (cuda / cuda:N / cpu)",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum number of samples to use per parquet. <=0 means use all samples.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/yiwen/AIPC/test_results/original_dda_bert",
        help="Directory to save output TSV files",
    )

    return parser.parse_args()


def resolve_device(device_arg: str):
    device_arg = (device_arg or "cpu").strip().lower()
    default_cuda_idx = 0

    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA requested but torch.cuda.is_available() is False, switching to CPU.")
            return torch.device("cpu")

        if device_arg == "cuda":
            if default_cuda_idx < torch.cuda.device_count():
                return torch.device(f"cuda:{default_cuda_idx}")
            return torch.device("cuda:0")

        if ":" in device_arg:
            try:
                gpu_idx = int(device_arg.split(":", 1)[1])
                if gpu_idx < 0 or gpu_idx >= torch.cuda.device_count():
                    fallback_idx = default_cuda_idx if default_cuda_idx < torch.cuda.device_count() else 0
                    print(f"Invalid CUDA index {gpu_idx}, switching to cuda:{fallback_idx}.")
                    return torch.device(f"cuda:{fallback_idx}")
                return torch.device(f"cuda:{gpu_idx}")
            except ValueError:
                fallback_idx = default_cuda_idx if default_cuda_idx < torch.cuda.device_count() else 0
                print(f"Invalid device argument '{device_arg}', switching to cuda:{fallback_idx}.")
                return torch.device(f"cuda:{fallback_idx}")

        fallback_idx = default_cuda_idx if default_cuda_idx < torch.cuda.device_count() else 0
        print(f"Unsupported cuda device format '{device_arg}', switching to cuda:{fallback_idx}.")
        return torch.device(f"cuda:{fallback_idx}")

    return torch.device("cpu")


def load_config(config_path: str):
    if not os.path.exists(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, "model.yaml")
        if os.path.exists(candidate):
            config_path = candidate
        else:
            print(f"Warning: Config file {config_path} not found. Using defaults.")
            return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_state_dict_keys(state_dict):
    normalized_state_dict = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod."):]
        if nk.startswith("module."):
            nk = nk[len("module."):]
        normalized_state_dict[nk] = v
    return normalized_state_dict


def clean_sequence(seq):
    if pd.isna(seq):
        return ""
    s = str(seq)
    s = re.sub(r"\[[^\]]*\]", "", s)
    s = s.replace("n", "")
    return s


def build_vocab_from_config(config):
    residues = config.get("residues", {})
    if residues:
        vocab = ["<pad>", "<mask>"] + list(residues.keys()) + ["<unk>"]
    else:
        print("Warning: residues not in config, vocab size might be incorrect.")
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
        print("Missing keys:")
        for k in missing:
            print("  ", k)

    if unexpected:
        print("Unexpected keys:")
        for k in unexpected:
            print("  ", k)

    loaded_count = len(normalized_state_dict) - len(unexpected)
    print(f"Loaded parameters: {loaded_count} / {len(model.state_dict())}")

    model.to(device)
    model.eval()
    return model


def run_one_parquet(parquet_path, model, config, s2i, device, args):
    print(f"\n===== Processing: {parquet_path} =====")
    df = pd.read_parquet(parquet_path)

    if "label" not in df.columns:
        print(f"Skip {parquet_path}: missing 'label' column.")
        return None

    if "weight" not in df.columns:
        df["weight"] = 1.0

    n_peaks = int(config.get("n_peaks", 150))
    max_length = int(config.get("max_length", 50))

    dataset_all = SpectrumDataset(
        df=df,
        s2i=s2i,
        n_peaks=n_peaks,
        max_length=max_length,
        need_label=True,
        need_deltaRT=True,
        need_weight=True,
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
        return None

    meta_df = df.iloc[:num_samples].copy().reset_index(drop=True)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch_parquet,
    )

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Validating {os.path.basename(parquet_path)}"):
            spectra, spectra_mask, precursors, tokens, labels = batch

            spectra = spectra.to(device, non_blocking=True)
            spectra_mask = spectra_mask.to(device, non_blocking=True)
            precursors = precursors.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            dda_logits, _ = model(spectra, spectra_mask, precursors, tokens)

            probs = torch.sigmoid(dda_logits).detach().cpu().numpy().reshape(-1)
            targets = labels.detach().cpu().numpy().reshape(-1)

            all_preds.extend(probs.tolist())
            all_targets.extend(targets.tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    if len(all_targets) == 0:
        print(f"Skip {parquet_path}: no predictions.")
        return None

    out_df = pd.DataFrame(
        {
            "scan_number": (
                meta_df["scan_number"].to_numpy()
                if "scan_number" in meta_df.columns
                else np.arange(len(all_preds))
            ),
            "peptide": (
                meta_df["precursor_sequence"].astype(str).to_numpy()
                if "precursor_sequence" in meta_df.columns
                else (
                    meta_df["modified_sequence"].astype(str).to_numpy()
                    if "modified_sequence" in meta_df.columns
                    else np.array([""] * len(all_preds))
                )
            ),
            "label": all_targets.astype(int),
            "score": all_preds,
        }
    )

    base_name = os.path.splitext(os.path.basename(parquet_path))[0]
    out_path = os.path.join(args.output_dir, f"{base_name}.tsv")
    out_df.to_csv(out_path, sep="\t", index=False)

    print(f"Saved TSV: {out_path}")
    print(f"Output rows (no dedup): {len(out_df)}")
    return out_path


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
        model = load_model(
            config=config,
            vocab_size=len(vocab),
            max_length=max_length,
            device=device,
            checkpoint_path=args.checkpoint_path,
        )
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    parquet_files = sorted(glob.glob(os.path.join(args.val_parquet_dir, "*.parquet")))
    if not parquet_files:
        print(f"No parquet files found in: {args.val_parquet_dir}")
        return

    print(f"Found {len(parquet_files)} parquet files.")

    for parquet_path in parquet_files:
        try:
            run_one_parquet(
                parquet_path=parquet_path,
                model=model,
                config=config,
                s2i=s2i,
                device=device,
                args=args,
            )
        except Exception as e:
            print(f"Failed on {parquet_path}: {e}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
