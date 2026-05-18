import argparse
import glob
import gc
import os
from pathlib import Path
from types import MethodType

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import sys
sys.path.insert(0, "/home/yiwen/AIPC/scripts/organized_attantion")
from model.transformer_compare.model import MSGPT


def parse_args():
    parser = argparse.ArgumentParser(description="Score all PSM rows in wiff_all PKL files with MSGPT.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/yiwen/AIPC/scripts/organized_attantion/data/checkpoints/checkpoints_wiff_sage_select/msgpt_epoch_10.pt",
    )
    parser.add_argument(
        "--train_src_dir",
        type=str,
        default="/home/yiwen/AIPC/scripts/organized_attantion/data/dataset/wiff_all/train",
    )
    parser.add_argument(
        "--val_src_dir",
        type=str,
        default="/home/yiwen/AIPC/scripts/organized_attantion/data/dataset/wiff_all/val",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/yiwen/AIPC/scripts/organized_attantion/data/score/train_wiff_all_psm_score_epoch10",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/model.yaml",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument("--max_files", type=int, default=0, help="0 means all files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output TSV")
    return parser.parse_args()


def load_default_config():
    return {
        "dim_model": 768,
        "n_head": 12,
        "dim_feedforward": 1024,
        "n_layers": 9,
        "dropout": 0.0,
        "max_length": 50,
        "max_charge": 10,
    }


def load_config_file(config_path):
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def infer_vocab_size(config):
    residues = config.get("residues") if isinstance(config, dict) else None
    if isinstance(residues, dict) and residues:
        return len(["<pad>", "<mask>"] + list(residues.keys()) + ["<unk>"])
    return int(config.get("vocab_size", 31))


def build_model(model_config, vocab_size):
    return MSGPT(
        dim_model=int(model_config.get("dim_model", 768)),
        n_head=int(model_config.get("n_head", 12)),
        dim_feedforward=int(model_config.get("dim_feedforward", 1024)),
        n_layers=int(model_config.get("n_layers", 9)),
        dropout=float(model_config.get("dropout", 0.0)),
        max_length=int(model_config.get("max_length", 50)),
        vocab_size=int(vocab_size),
        max_charge=int(model_config.get("max_charge", 10)),
    )


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


def count_tsv_data_rows(tsv_path):
    if not os.path.exists(tsv_path):
        return 0
    line_count = 0
    with open(tsv_path, "r", encoding="utf-8") as f:
        for _ in f:
            line_count += 1
    if line_count <= 1:
        return 0
    return line_count - 1


def strip_pkl_gz_suffix(file_path: str) -> str:
    name = Path(file_path).name
    if name.endswith(".pkl.gz"):
        return name[:-len(".pkl.gz")]
    return Path(file_path).stem


def build_output_base_name(pkl_path: str, split_name: str) -> str:
    base = strip_pkl_gz_suffix(pkl_path)
    suffix = f"_{split_name}"
    if base.endswith(suffix):
        return base[:-len(suffix)]
    return base


def find_pkl_gz_files(src_dir: str):
    direct = sorted(glob.glob(os.path.join(src_dir, "*.pkl.gz")))
    if direct:
        return direct
    return sorted(glob.glob(os.path.join(src_dir, "**", "*.pkl.gz"), recursive=True))


def build_s2i_from_config(config_path):
    cfg = load_config_file(config_path)
    residues = cfg.get("residues") if isinstance(cfg, dict) else None
    if not isinstance(residues, dict) or not residues:
        raise ValueError(f"config 中缺少 residues: {config_path}")
    vocab = ["<pad>", "<mask>"] + list(residues.keys()) + ["<unk>"]
    s2i = {v: i for i, v in enumerate(vocab)}
    n_peaks = int(cfg.get("n_peaks", 100))
    return s2i, n_peaks


def attach_no_mlm_head(model):
    def _multi_task_encoder_from_decoder_no_mlm(self, decoder_output):
        pooled = self.dropout(self.relu(self.seq_pool(decoder_output.transpose(1, 2)).squeeze(-1)))
        dda_hidden = self.dropout(self.relu(self.psm_1(pooled)))
        dda_pred = self.psm_2(dda_hidden).squeeze(-1)
        return dda_pred, None

    model.mask_lm = torch.nn.Identity()
    model._multi_task_encoder_from_decoder = MethodType(_multi_task_encoder_from_decoder_no_mlm, model)


def build_infer_model(model_path, config_path, device):
    print(f"[*] Loading model from {model_path} on {device}...")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    normalized_state_dict = normalize_state_dict_keys(state_dict)

    default_cfg = load_default_config()
    file_cfg = load_config_file(config_path)
    ckpt_cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    if isinstance(ckpt_cfg, dict) and ckpt_cfg:
        model_config = ckpt_cfg
    elif isinstance(file_cfg, dict) and file_cfg:
        model_config = file_cfg
    else:
        model_config = default_cfg

    vocab_size = infer_vocab_size(model_config)
    print(f"[*] Using vocab_size={vocab_size} on {device}")
    model = build_model(model_config, vocab_size=vocab_size)

    has_mask_lm = any(k.startswith("mask_lm.") for k in normalized_state_dict.keys())
    if not has_mask_lm:
        print(f"[*] No MLM head in checkpoint on {device}, attaching simplified DDA head.")
        attach_no_mlm_head(model)

    model.load_state_dict(normalized_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def score_one_pkl(pkl_path, out_dir, model, device, batch_size, num_workers, overwrite, s2i, n_peaks):
    split_name = Path(out_dir).name
    out_name = build_output_base_name(pkl_path, split_name) + "_all_psm_score.tsv"
    out_path = os.path.join(out_dir, out_name)
    if overwrite and os.path.exists(out_path):
        os.remove(out_path)

    data_dict = pd.read_pickle(pkl_path)
    total_rows = len(data_dict["label"])
    if total_rows == 0:
        return "skipped_empty"

    existing_rows = 0 if overwrite else count_tsv_data_rows(out_path)
    if existing_rows >= total_rows:
        return "skipped_complete"

    if existing_rows > total_rows:
        os.remove(out_path)
        existing_rows = 0

    with torch.no_grad():
        spectra = torch.from_numpy(data_dict["spectra"][existing_rows:])
        mask = torch.from_numpy(data_dict["spectra_mask"][existing_rows:])
        precursors = torch.from_numpy(data_dict["precursors"][existing_rows:])
        if precursors.shape[-1] > 2:
            precursors = precursors.clone()
            precursors[..., 2:] = 0
        tokens = torch.from_numpy(data_dict["tokens"][existing_rows:])
        labels = torch.from_numpy(data_dict["label"][existing_rows:].astype(np.float32))

        index_tensor = torch.arange(existing_rows, total_rows, dtype=torch.long)
        infer_ds = TensorDataset(spectra, mask, precursors, tokens, labels, index_tensor)
        loader = DataLoader(infer_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    wrote_header = existing_rows > 0
    file_desc = strip_pkl_gz_suffix(pkl_path)

    with torch.no_grad():
        for spectra, mask, precursors, tokens, batch_labels, batch_indices in tqdm(
            loader,
            total=len(loader),
            desc=file_desc,
            leave=False,
            dynamic_ncols=True,
        ):
            spectra = spectra.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            precursors = precursors.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)

            _, _, dda_logits, _ = model(spectra, mask, precursors, tokens)
            probs = torch.sigmoid(dda_logits).cpu().numpy().reshape(-1)

            batch_df = pd.DataFrame(
                {
                    "original_index": batch_indices.cpu().numpy().reshape(-1),
                    "label": batch_labels.cpu().numpy().reshape(-1),
                    "model_score": probs,
                }
            )
            batch_df.to_csv(
                out_path,
                sep="\t",
                index=False,
                mode="a",
                header=not wrote_header,
            )
            wrote_header = True

    del loader
    del infer_ds
    del data_dict
    gc.collect()
    return "ok"


def process_split(split_name, files, out_dir, model, device, args, s2i, n_peaks):
    if len(files) == 0:
        print(f"[*] {split_name}: 未找到 pkl.gz 文件，跳过。")
        return 0, 0, 0

    if args.max_files > 0:
        files = files[: args.max_files]

    split_out_dir = os.path.join(out_dir, split_name)
    os.makedirs(split_out_dir, exist_ok=True)

    print(f"[*] {split_name}: checking completed files...")
    completed = 0
    for pkl_path in tqdm(files, desc=f"Checking {split_name}", leave=False, dynamic_ncols=True):
        out_name = build_output_base_name(pkl_path, split_name) + "_all_psm_score.tsv"
        out_path = os.path.join(split_out_dir, out_name)
        if os.path.exists(out_path):
            data_dict = pd.read_pickle(pkl_path)
            total_rows = len(data_dict["label"])
            done_rows = count_tsv_data_rows(out_path)
            if done_rows >= total_rows:
                completed += 1

    print(f"[*] {split_name}: completed files {completed}/{len(files)}")

    if completed == len(files) and not args.overwrite:
        print(f"[*] {split_name}: {completed}/{len(files)} 文件已完整预测，整套数据跳过。")
        return 0, 0, len(files)

    ok_count = 0
    skip_count = 0
    resume_count = 0
    for pkl_path in tqdm(files, desc=f"Scoring {split_name} all PSMs"):
        status = score_one_pkl(
            pkl_path=pkl_path,
            out_dir=split_out_dir,
            model=model,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            overwrite=args.overwrite,
            s2i=s2i,
            n_peaks=n_peaks,
        )
        if status == "ok":
            ok_count += 1
        elif status == "skipped_complete":
            resume_count += 1
        else:
            skip_count += 1

    return ok_count, skip_count, resume_count


def main():
    args = parse_args()

    if args.device.startswith("cuda") and (not torch.cuda.is_available()):
        print("[*] CUDA 不可用，自动回退到 CPU")
        device_str = "cpu"
    else:
        device_str = args.device
    device = torch.device(device_str)

    os.makedirs(args.out_dir, exist_ok=True)
    s2i, n_peaks = build_s2i_from_config(args.config_path)

    train_files = find_pkl_gz_files(args.train_src_dir)
    val_files = find_pkl_gz_files(args.val_src_dir)

    if len(train_files) == 0 and len(val_files) == 0:
        raise RuntimeError(
            "未找到 pkl.gz 文件，请检查 --train_src_dir / --val_src_dir。"
            f"当前 train 路径: {args.train_src_dir}；val 路径: {args.val_src_dir}。"
        )

    print(f"[*] Found train files: {len(train_files)}")
    print(f"[*] Found val files: {len(val_files)}")
    print(f"[*] Using device: {device_str}")

    model = build_infer_model(args.model_path, args.config_path, device)

    train_ok, train_skip, train_resume = process_split(
        split_name="train",
        files=train_files,
        out_dir=args.out_dir,
        model=model,
        device=device,
        args=args,
        s2i=s2i,
        n_peaks=n_peaks,
    )
    val_ok, val_skip, val_resume = process_split(
        split_name="val",
        files=val_files,
        out_dir=args.out_dir,
        model=model,
        device=device,
        args=args,
        s2i=s2i,
        n_peaks=n_peaks,
    )

    ok_count = train_ok + val_ok
    skip_count = train_skip + val_skip
    resume_count = train_resume + val_resume

    print(
        "[+] Split stats: "
        f"train(ok={train_ok}, skipped={train_skip}, already_complete={train_resume}), "
        f"val(ok={val_ok}, skipped={val_skip}, already_complete={val_resume})"
    )
    print(f"[+] Done total: ok={ok_count}, skipped={skip_count}, already_complete={resume_count}")
    print(f"[+] All TSV results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
