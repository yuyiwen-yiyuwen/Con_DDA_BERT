import argparse
import bisect
import glob
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import sys
sys.path.insert(0, "/home/yiwen/AIPC/scripts/organized_attantion")
from model.transformer.model import MSGPT

PROTON_MASS_AMU = 1.007276


class EncodedPKLDataset(Dataset):
    """
    Validation Dataset (Map-style) for PKL files.
    Reads files on demand to save memory.
    """

    def __init__(self, pkl_files):
        self.pkl_files = sorted(pkl_files)
        if not self.pkl_files:
            raise ValueError("No pkl files found.")

        self.current_file_idx = -1
        self.file_data = None

        # file_offsets[i] 表示第 i 个文件在整个大数据集中的起始全局下标
        self.file_offsets = [0]
        self.total_samples = 0

        print("Initializing dataset and indexing files...")
        for fp in tqdm(self.pkl_files, desc="Indexing"):
            with open(fp, "rb") as f:
                data = pickle.load(f)

            num_samples = len(data["label"])
            self.total_samples += num_samples
            self.file_offsets.append(self.total_samples)

        # 最后一个是总长度，不是某个文件的起点，去掉
        self.file_offsets.pop()
        print(f"Total samples: {self.total_samples}")

    def _load_file(self, file_idx: int):
        """
        按需加载指定 pkl 文件到内存。
        """
        if self.current_file_idx != file_idx:
            fp = self.pkl_files[file_idx]
            with open(fp, "rb") as f:
                data = pickle.load(f)

            precursors = torch.as_tensor(data["precursors"], dtype=torch.float32)
            if precursors.shape[-1] > 2:
                precursors = precursors.clone()
                precursors[..., 2:] = 0
            self.file_data = {
                "spectra": torch.as_tensor(data["spectra"], dtype=torch.float32),
                "spectra_mask": torch.as_tensor(data["spectra_mask"], dtype=torch.bool),
                "precursors": precursors,
                "tokens": torch.as_tensor(data["tokens"], dtype=torch.long),
                "labels": torch.as_tensor(data["label"], dtype=torch.float32),
            }
            self.current_file_idx = file_idx

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx: int):
        """
        根据全局 idx 找到它属于哪个文件，再取出对应样本。
        """
        file_idx = bisect.bisect_right(self.file_offsets, idx) - 1
        self._load_file(file_idx)
        relative_idx = idx - self.file_offsets[file_idx]

        return (
            self.file_data["spectra"][relative_idx],
            self.file_data["spectra_mask"][relative_idx],
            self.file_data["precursors"][relative_idx],
            self.file_data["tokens"][relative_idx],
            self.file_data["labels"][relative_idx],
        )


def calculate_fdr_identities(probs, targets, fdr_threshold=0.01):
    """
    计算在指定 FDR 阈值下的鉴定数目。
    probs: 模型输出概率
    targets: 真实标签 (1=target, 0=decoy)
    """
    idx = np.argsort(probs)[::-1]
    sorted_targets = targets[idx]

    cum_targets = np.cumsum(sorted_targets)
    cum_decoys = np.cumsum(1 - sorted_targets)

    fdr = np.where(cum_targets > 0, cum_decoys / cum_targets, 0)

    eligible = np.where(fdr <= fdr_threshold)[0]
    if len(eligible) > 0:
        last_idx = eligible[-1]
        identities = cum_targets[last_idx]
    else:
        identities = 0

    return int(identities)


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


def extract_epoch_from_filename(path):
    """
    从类似 msgpt_epoch_5.pt 中提取 epoch=5
    """
    name = os.path.basename(path)
    m = re.search(r"epoch[_\-]?(\d+)", name)
    if m:
        return int(m.group(1))
    return None


def extract_loss_info_from_checkpoint(ckpt):
    """
    尽量从 checkpoint 中提取 loss 信息。
    支持多种可能命名：
    - loss
    - train_loss
    - val_loss
    - losses
    - history / metrics 中的 loss 曲线
    """
    loss_info = {}

    if not isinstance(ckpt, dict):
        return loss_info

    # 直接平铺字段
    candidate_keys = [
        "loss",
        "train_loss",
        "val_loss",
        "train_losses",
        "val_losses",
        "losses",
    ]

    for k in candidate_keys:
        if k in ckpt:
            loss_info[k] = ckpt[k]

    # history
    if "history" in ckpt and isinstance(ckpt["history"], dict):
        for k, v in ckpt["history"].items():
            if "loss" in str(k).lower():
                loss_info[f"history.{k}"] = v

    # metrics
    if "metrics" in ckpt and isinstance(ckpt["metrics"], dict):
        for k, v in ckpt["metrics"].items():
            if "loss" in str(k).lower():
                loss_info[f"metrics.{k}"] = v

    return loss_info


def to_scalar_if_possible(x):
    if isinstance(x, (float, int, np.floating, np.integer)):
        return float(x)
    if torch.is_tensor(x):
        if x.numel() == 1:
            return float(x.item())
        return None
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return float(x.reshape(-1)[0])
        return None
    return None


def to_list_if_possible(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return [float(x.item())]
        return [float(v) for v in x.reshape(-1)]
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            s = to_scalar_if_possible(v)
            if s is None:
                return None
            out.append(s)
        return out
    scalar = to_scalar_if_possible(x)
    if scalar is not None:
        return [scalar]
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all MSGPT checkpoints and plot ACC/loss curves."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="model.yaml",
        help="Path to config file",
    )

    parser.add_argument(
        "--val_data_dir",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/pkl_dataset/val",
        help="Validation data directory (pkl files)",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/mzml_pt/checkpoints_mzml",
        help="Directory containing checkpoint .pt files",
    )

    parser.add_argument(
        "--checkpoint_pattern",
        type=str,
        default="msgpt_epoch_*.pt",
        help="Glob pattern for checkpoint files",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
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
        default="cuda:1",
        help="Device (cuda / cuda:N / cpu)",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum number of validation samples to use. <=0 means use all samples.",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/val_train_results",
        help="Directory to save csv and plots",
    )

    return parser.parse_args()


def resolve_device(device_arg: str):
    device_arg = (device_arg or "cpu").strip().lower()

    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA requested but not available, switching to CPU.")
            return torch.device("cpu")

        if device_arg == "cuda":
            return torch.device("cuda")

        if ":" in device_arg:
            try:
                gpu_idx = int(device_arg.split(":", 1)[1])
                if gpu_idx < 0 or gpu_idx >= torch.cuda.device_count():
                    print(f"Invalid CUDA index {gpu_idx}, switching to cuda:0.")
                    return torch.device("cuda:0")
                return torch.device(f"cuda:{gpu_idx}")
            except ValueError:
                print(f"Invalid device argument '{device_arg}', switching to cuda:0.")
                return torch.device("cuda:0")

        print(f"Unsupported cuda device format '{device_arg}', switching to cuda:0.")
        return torch.device("cuda:0")

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


def evaluate_one_checkpoint(ckpt_path, model, loader, device):
    model_name = os.path.basename(ckpt_path)
    epoch = extract_epoch_from_filename(ckpt_path)

    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        normalized_state_dict = normalize_state_dict_keys(state_dict)

        missing, unexpected = model.load_state_dict(normalized_state_dict, strict=False)

        if missing:
            print(f"[{model_name}] Missing keys:")
            for k in missing:
                print("  ", k)

        if unexpected:
            print(f"[{model_name}] Unexpected keys:")
            for k in unexpected:
                print("  ", k)

    except Exception as e:
        print(f"Error loading checkpoint {ckpt_path}: {e}")
        return None, None

    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Validating {model_name}", leave=False):
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
        print(f"[{model_name}] No targets found.")
        return None, None

    if len(np.unique(all_targets)) > 1:
        auc = roc_auc_score(all_targets, all_preds)
    else:
        auc = np.nan

    pred_labels = (all_preds > 0.5).astype(int)
    acc = accuracy_score(all_targets, pred_labels)
    identities_1fdr = calculate_fdr_identities(all_preds, all_targets, fdr_threshold=0.01)

    result = {
        "checkpoint": model_name,
        "checkpoint_path": ckpt_path,
        "epoch": epoch if epoch is not None else -1,
        "num_samples": len(all_preds),
        "auc": auc,
        "accuracy": acc,
        "identities_1fdr": identities_1fdr,
    }

    loss_info = extract_loss_info_from_checkpoint(ckpt if isinstance(ckpt, dict) else {})
    return result, loss_info


def plot_acc_curve(metrics_df, save_path):
    df = metrics_df.copy()
    df = df.sort_values(["epoch", "checkpoint"]).reset_index(drop=True)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["accuracy"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy across Checkpoints")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_auc_curve(metrics_df, save_path):
    df = metrics_df.copy()
    df = df.sort_values(["epoch", "checkpoint"]).reset_index(drop=True)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["auc"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("Validation AUC across Checkpoints")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_identities_curve(metrics_df, save_path):
    df = metrics_df.copy()
    df = df.sort_values(["epoch", "checkpoint"]).reset_index(drop=True)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["identities_1fdr"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Identities @ 1% FDR")
    plt.title("Validation Identities@1%FDR across Checkpoints")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_checkpoint_scalar_losses(loss_scalar_records, save_path):
    if not loss_scalar_records:
        return False

    df = pd.DataFrame(loss_scalar_records)
    if df.empty:
        return False

    plt.figure(figsize=(9, 6))
    for loss_name in sorted(df["loss_name"].unique()):
        sub = df[df["loss_name"] == loss_name].sort_values(["epoch", "checkpoint"])
        plt.plot(sub["epoch"], sub["loss_value"], marker="o", label=loss_name)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Checkpoint-level Loss Values")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return True


def plot_checkpoint_loss_series(loss_series_records, output_dir):
    """
    每个 checkpoint 如果保存的是完整 loss 列表，就单独画一张图。
    """
    plotted = 0
    for item in loss_series_records:
        checkpoint = item["checkpoint"]
        epoch = item["epoch"]
        loss_name = item["loss_name"]
        values = item["values"]

        if values is None or len(values) == 0:
            continue

        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, len(values) + 1), values, marker="o")
        plt.xlabel("Step / Epoch Index")
        plt.ylabel("Loss")
        plt.title(f"{checkpoint} - {loss_name}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        safe_name = re.sub(r"[^\w\-\.]+", "_", f"epoch_{epoch}_{checkpoint}_{loss_name}.png")
        save_path = os.path.join(output_dir, safe_name)
        plt.savefig(save_path, dpi=200)
        plt.close()
        plotted += 1

    return plotted


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    metrics_csv_path = os.path.join(args.results_dir, "all_checkpoints_metrics.csv")

    device = resolve_device(args.device)
    print("Device:", device)

    config = load_config(args.config)

    residues = config.get("residues", {})
    if residues:
        vocab = ["<pad>", "<mask>"] + list(residues.keys()) + ["<unk>"]
    else:
        print("Warning: residues not in config, vocab size might be incorrect.")
        vocab = ["<pad>", "<mask>", "<unk>"]

    pkl_files = sorted(glob.glob(os.path.join(args.val_data_dir, "*.pkl")))
    if not pkl_files:
        print(f"No pkl files found in {args.val_data_dir}")
        return

    full_dataset = EncodedPKLDataset(pkl_files)

    if args.max_samples is not None and args.max_samples > 0:
        num_samples = min(args.max_samples, len(full_dataset))
    else:
        num_samples = len(full_dataset)

    subset_indices = list(range(num_samples))
    dataset = Subset(full_dataset, subset_indices)

    print(f"Using {num_samples} samples for validation.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    ckpt_pattern = os.path.join(args.checkpoint_dir, args.checkpoint_pattern)
    ckpt_files = sorted(glob.glob(ckpt_pattern), key=lambda x: (extract_epoch_from_filename(x) is None, extract_epoch_from_filename(x), x))

    if not ckpt_files:
        print(f"No checkpoint files found with pattern: {ckpt_pattern}")
        return

    print("Found checkpoints:")
    for fp in ckpt_files:
        print("  ", fp)

    # 断点续评估：读取历史评估结果并跳过已评估 checkpoint。
    existing_metrics_df = pd.DataFrame()
    evaluated_paths = set()
    evaluated_names = set()
    if os.path.exists(metrics_csv_path):
        try:
            existing_metrics_df = pd.read_csv(metrics_csv_path)
            if "checkpoint_path" in existing_metrics_df.columns:
                evaluated_paths = set(existing_metrics_df["checkpoint_path"].dropna().astype(str).tolist())
            if "checkpoint" in existing_metrics_df.columns:
                evaluated_names = set(existing_metrics_df["checkpoint"].dropna().astype(str).tolist())
        except Exception as e:
            print(f"Warning: failed to read existing metrics csv {metrics_csv_path}: {e}")
            existing_metrics_df = pd.DataFrame()

    ckpt_files_to_eval = []
    for ckpt in ckpt_files:
        ckpt_name = os.path.basename(ckpt)
        if ckpt in evaluated_paths or ckpt_name in evaluated_names:
            continue
        ckpt_files_to_eval.append(ckpt)

    already_done = len(ckpt_files) - len(ckpt_files_to_eval)
    print(
        f"Checkpoint summary: total={len(ckpt_files)}, already_evaluated={already_done}, "
        f"to_evaluate={len(ckpt_files_to_eval)}"
    )

    if len(ckpt_files_to_eval) == 0 and not existing_metrics_df.empty:
        print("All checkpoints are already evaluated. Will only regenerate plots from existing metrics.")

    model = build_model(config, vocab_size=len(vocab))

    all_results = []
    loss_scalar_records = []
    loss_series_records = []

    for ckpt_path in ckpt_files_to_eval:
        print(f"\n===== Evaluating {os.path.basename(ckpt_path)} =====")
        result, loss_info = evaluate_one_checkpoint(ckpt_path, model, loader, device)

        if result is None:
            continue

        all_results.append(result)
        print(
            f"epoch={result['epoch']} | "
            f"AUC={result['auc']:.6f} | "
            f"ACC={result['accuracy']:.6f} | "
            f"identities_1fdr={result['identities_1fdr']}"
        )

        for loss_name, loss_value in loss_info.items():
            scalar = to_scalar_if_possible(loss_value)
            if scalar is not None:
                loss_scalar_records.append(
                    {
                        "checkpoint": result["checkpoint"],
                        "epoch": result["epoch"],
                        "loss_name": loss_name,
                        "loss_value": scalar,
                    }
                )
                continue

            values = to_list_if_possible(loss_value)
            if values is not None:
                loss_series_records.append(
                    {
                        "checkpoint": result["checkpoint"],
                        "epoch": result["epoch"],
                        "loss_name": loss_name,
                        "values": values,
                    }
                )

    new_metrics_df = pd.DataFrame(all_results)

    if existing_metrics_df.empty and new_metrics_df.empty:
        print("No valid checkpoint evaluation results.")
        return

    if existing_metrics_df.empty:
        metrics_df = new_metrics_df.copy()
    elif new_metrics_df.empty:
        metrics_df = existing_metrics_df.copy()
    else:
        metrics_df = pd.concat([existing_metrics_df, new_metrics_df], ignore_index=True)

    # 去重：同一路径或同名 checkpoint 只保留最新一条。
    if "checkpoint_path" in metrics_df.columns:
        metrics_df = metrics_df.drop_duplicates(subset=["checkpoint_path"], keep="last")
    elif "checkpoint" in metrics_df.columns:
        metrics_df = metrics_df.drop_duplicates(subset=["checkpoint"], keep="last")

    sort_cols = [c for c in ["epoch", "checkpoint"] if c in metrics_df.columns]
    if sort_cols:
        metrics_df = metrics_df.sort_values(sort_cols).reset_index(drop=True)

    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nMetrics saved to: {metrics_csv_path}")

    # 画 ACC 曲线
    acc_plot_path = os.path.join(args.results_dir, "all_checkpoints_acc_curve.png")
    plot_acc_curve(metrics_df, acc_plot_path)
    print(f"ACC curve saved to: {acc_plot_path}")

    # 画 AUC 曲线
    auc_plot_path = os.path.join(args.results_dir, "all_checkpoints_auc_curve.png")
    plot_auc_curve(metrics_df, auc_plot_path)
    print(f"AUC curve saved to: {auc_plot_path}")

    # 画 identities_1fdr 曲线
    id_plot_path = os.path.join(args.results_dir, "all_checkpoints_identities_1fdr_curve.png")
    plot_identities_curve(metrics_df, id_plot_path)
    print(f"Identities curve saved to: {id_plot_path}")

    # checkpoint 中若有标量 loss，画总图
    scalar_loss_plot_path = os.path.join(args.results_dir, "all_checkpoints_scalar_loss_curve.png")
    scalar_loss_ok = plot_checkpoint_scalar_losses(loss_scalar_records, scalar_loss_plot_path)
    if scalar_loss_ok:
        print(f"Scalar loss curve saved to: {scalar_loss_plot_path}")
    else:
        print("No scalar loss found in checkpoints, skipped scalar loss curve.")

    # checkpoint 中若有 loss 列表，为每个 checkpoint 分别画图
    loss_series_dir = os.path.join(args.results_dir, "loss_series_plots")
    os.makedirs(loss_series_dir, exist_ok=True)
    num_loss_series_plots = plot_checkpoint_loss_series(loss_series_records, loss_series_dir)
    if num_loss_series_plots > 0:
        print(f"Saved {num_loss_series_plots} checkpoint loss-series plots to: {loss_series_dir}")
    else:
        print("No loss series found in checkpoints, skipped per-checkpoint loss plots.")

    print("\nDone.")
    print(metrics_df)


if __name__ == "__main__":
    main()
