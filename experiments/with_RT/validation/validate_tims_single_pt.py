import argparse
import os
import pickle

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import sys
sys.path.insert(0, "/home/yiwen/AIPC/scripts/organized_attantion")
from model.transformer.model import MSGPT


class SinglePKLDataset(Dataset):
    """Map-style dataset for a single encoded PKL file."""

    def __init__(self, pkl_path: str):
        if not os.path.isfile(pkl_path):
            raise FileNotFoundError(f"Validation PKL not found: {pkl_path}")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.spectra = torch.as_tensor(data["spectra"], dtype=torch.float32)
        self.spectra_mask = torch.as_tensor(data["spectra_mask"], dtype=torch.bool)
        self.precursors = torch.as_tensor(data["precursors"], dtype=torch.float32)
        self.tokens = torch.as_tensor(data["tokens"], dtype=torch.long)
        self.labels = torch.as_tensor(data["label"], dtype=torch.float32)

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        return (
            self.spectra[idx],
            self.spectra_mask[idx],
            self.precursors[idx],
            self.tokens[idx],
            self.labels[idx],
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Validate one MSGPT checkpoint on one tims PKL file.")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/model.yaml",
        help="Path to model config yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/checkpoints_tims/msgpt_epoch_7.pt",
        help="Path to a single checkpoint .pt file.",
    )
    parser.add_argument(
        "--val_pkl",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/pkl_dataset_tims/val/val.00000_val.pkl",
        help="Path to a single validation PKL file.",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for validation.")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device: cuda/cuda:N/cpu.")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Use only first N samples for quick validation. <=0 means all.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str):
    device_arg = (device_arg or "cpu").strip().lower()

    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA requested but not available, falling back to CPU.")
            return torch.device("cpu")

        if device_arg == "cuda":
            return torch.device("cuda")

        if ":" in device_arg:
            try:
                gpu_idx = int(device_arg.split(":", 1)[1])
                if 0 <= gpu_idx < torch.cuda.device_count():
                    return torch.device(f"cuda:{gpu_idx}")
                print(f"Invalid CUDA index: {gpu_idx}, use cuda:0.")
                return torch.device("cuda:0")
            except ValueError:
                print(f"Invalid device argument: {device_arg}, use cuda:0.")
                return torch.device("cuda:0")

        print(f"Unsupported cuda argument: {device_arg}, use cuda:0.")
        return torch.device("cuda:0")

    return torch.device("cpu")


def load_config(config_path: str):
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_model(config, vocab_size: int):
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


def normalize_state_dict_keys(state_dict):
    normalized = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod.") :]
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        normalized[nk] = v
    return normalized


def calculate_fdr_identities(probs, targets, fdr_threshold=0.01):
    idx = np.argsort(probs)[::-1]
    sorted_targets = targets[idx]

    cum_targets = np.cumsum(sorted_targets)
    cum_decoys = np.cumsum(1 - sorted_targets)
    fdr = np.where(cum_targets > 0, cum_decoys / cum_targets, 0)

    eligible = np.where(fdr <= fdr_threshold)[0]
    if len(eligible) > 0:
        return int(cum_targets[eligible[-1]])
    return 0


def main():
    args = parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Validation PKL: {args.val_pkl}")

    config = load_config(args.config)
    residues = config.get("residues", {})
    vocab = ["<pad>", "<mask>"] + list(residues.keys()) + ["<unk>"]

    dataset = SinglePKLDataset(args.val_pkl)
    if args.max_samples > 0:
        use_n = min(int(args.max_samples), len(dataset))
        dataset = Subset(dataset, list(range(use_n)))
        print(f"Using subset samples: {use_n}")
    else:
        print(f"Using all samples: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(config, vocab_size=len(vocab))

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state_dict = normalize_state_dict_keys(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("Missing keys:")
        for k in missing:
            print("  ", k)
    if unexpected:
        print("Unexpected keys:")
        for k in unexpected:
            print("  ", k)

    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            spectra, spectra_mask, precursors, tokens, labels = batch

            spectra = spectra.to(device, non_blocking=True)
            spectra_mask = spectra_mask.to(device, non_blocking=True)
            precursors = precursors.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs, _ = model(spectra, spectra_mask, precursors, tokens)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            probs = torch.sigmoid(outputs).detach().cpu().numpy().reshape(-1)
            targets = labels.detach().cpu().numpy().reshape(-1)

            all_preds.extend(probs.tolist())
            all_targets.extend(targets.tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    if len(all_targets) == 0:
        raise RuntimeError("No samples were evaluated.")

    unique_targets = np.unique(all_targets)
    auc = roc_auc_score(all_targets, all_preds) if len(unique_targets) > 1 else np.nan
    acc = accuracy_score(all_targets, (all_preds > 0.5).astype(int))
    identities_1fdr = calculate_fdr_identities(all_preds, all_targets, fdr_threshold=0.01)

    print("\n===== Validation Summary =====")
    print(f"num_samples: {len(all_targets)}")
    print(f"target_ratio: {float((all_targets > 0.5).mean()):.6f}")
    print(f"AUC: {auc:.6f}")
    print(f"ACC@0.5: {acc:.6f}")
    print(f"identities@1%FDR: {identities_1fdr}")


if __name__ == "__main__":
    main()
