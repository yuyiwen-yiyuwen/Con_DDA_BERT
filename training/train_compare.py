import argparse
import glob
import math
import os
import pickle
import random
import threading
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

import sys
sys.path.insert(0, "/home/yiwen/AIPC/scripts/organized_attantion")
from model.transformer_compare.model import (
    MSGPT,
    coca_inbatch_contrastive_loss_with_decoys,
)

PROGRESS_REFRESH_SEC = 5.0


def _start_heartbeat(state: dict, stop_event: threading.Event, tag: str = "heartbeat") -> threading.Thread:
    """Print progress during long online rerank stages."""

    def _run():
        while not stop_event.wait(PROGRESS_REFRESH_SEC):
            idx = int(state.get("idx", 0))
            total = int(state.get("total", 0))
            stage = str(state.get("stage", "unknown"))
            current_file = str(state.get("file", ""))
            tqdm.write(f"[{tag}] {idx}/{total} | stage={stage} | file={current_file}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread

# 一个用于流式读取 pkl 文件批量数据的 PyTorch IterableDataset，按文件流式读取，每个文件只做一次 tensor 化
class PKLBatchIterableDataset(IterableDataset):
    # pkl_dir：存放 .pkl 文件的目录
    # batch_size：每个 batch 里要放多少条样本
    def __init__(
        self,
        pkl_dir: str,
        batch_size: int,
        pkl_files: Optional[list[str]] = None,
        shuffle_within_file: bool = False,
        shuffle_batches: bool = True,
    ):
        # 调用父类 IterableDataset 的初始化函数
        super().__init__()
        # 优先使用外部传入文件列表；否则从目录扫描。
        all_files = sorted(pkl_files) if pkl_files is not None else sorted(glob.glob(os.path.join(pkl_dir, "*.pkl")))
        if not all_files:
            raise ValueError(f"在 {pkl_dir} 下未发现 pkl 文件")

        # 把找到的所有 pkl 文件路径保存到对象属性里
        self.all_files = all_files
        # 把 batch size 保存起来，并显式转成整数
        self.batch_size = int(batch_size)
        self.epoch = 0
        self.shuffle_within_file = bool(shuffle_within_file)
        self.shuffle_batches = bool(shuffle_batches)

    # 设置 epoch
    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    # 按 worker 分文件
    def _get_files_for_this_worker(self):
        # 取当前 worker 的信息
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # 从 self.all_files 中，当前 worker 从自己的起点开始，每隔 num_workers 取一个文件
        worker_files = self.all_files[worker_id::num_workers]
        return worker_files

    # 核心方法：不断生成 batch
    def __iter__(self):
        # 调用刚才那个函数，取出“当前 worker 该读的文件”
        files = list(self._get_files_for_this_worker())

        # 每个 epoch 可复现洗牌
        rng = random.Random(self.epoch + 1234)
        # 对当前 worker 的文件列表进行原地打乱
        rng.shuffle(files)

        # 逐个处理文件
        for fp in files:
            # 以二进制只读模式打开文件
            with open(fp, "rb") as f:
                # 把 pickle 文件反序列化，读回 Python 对象
                data = pickle.load(f)

            # 一个文件读进来以后，一次性把整个文件所有字段都转成 tensor
            spectra = torch.as_tensor(data["spectra"], dtype=torch.float32)
            spectra_mask = torch.as_tensor(data["spectra_mask"], dtype=torch.bool)
            precursors = torch.as_tensor(data["precursors"], dtype=torch.float32)
            if precursors.shape[-1] > 2:
                precursors = precursors.clone()
                precursors[..., 2:] = 0
            tokens = torch.as_tensor(data["tokens"], dtype=torch.long)
            labels = torch.as_tensor(data["label"], dtype=torch.float32)
            weights = torch.as_tensor(
                data.get("weight", np.ones_like(data["label"])),
                dtype=torch.float32,
            )
            # 加载 unmask，默认全为0（即都进行mask）
            unmask = torch.as_tensor(
                data.get("unmask", np.zeros_like(data["label"])),
                dtype=torch.float32,
            )

            num_all = labels.shape[0]
            if num_all <= 0:
                continue

            # 文件内打乱样本顺序（质量锚定重排数据可关闭，保持离线排布）
            num = labels.shape[0] # 取当前文件里的样本数
            if self.shuffle_within_file:
                order = torch.randperm(num)
            else:
                order = torch.arange(num)

            # 构造 batch 切分点；默认按 batch 粒度打乱一次，保持批内局部结构
            batch_starts = list(range(0, num, self.batch_size))
            if self.shuffle_batches and len(batch_starts) > 1:
                rng.shuffle(batch_starts)

            # 按 batch 切分并 yield
            for start in batch_starts:
                idx = order[start : start + self.batch_size]
                yield (
                    spectra[idx],
                    spectra_mask[idx],
                    precursors[idx],
                    tokens[idx],
                    labels[idx],
                    weights[idx],
                    unmask[idx],
                )

# 随机种子设置
def _extract_peptide_ids(payload: dict, n: int) -> np.ndarray:
    if "peptides" in payload:
        peptides = np.asarray(payload["peptides"], dtype=object)
        if peptides.shape[0] != n:
            raise ValueError("peptides length mismatch")
        return peptides
    if "tokens" in payload:
        tokens = np.asarray(payload["tokens"])
        if tokens.shape[0] != n:
            raise ValueError("tokens length mismatch")
        return np.array([tokens[i].tobytes() for i in range(n)], dtype=object)
    raise KeyError("payload lacks peptides/tokens for batch peptide diversity")


def _build_mass_anchored_order(
    masses: np.ndarray,
    batch_size: int,
    start_window_da: float,
    expand_factor: float,
    max_window_da: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if masses.ndim != 1:
        raise ValueError(f"masses must be 1D, got shape={masses.shape}")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    n = masses.shape[0]
    if n == 0:
        return np.empty((0,), dtype=np.int64)
    sorted_idx = np.argsort(masses, kind="mergesort")
    sorted_masses = masses[sorted_idx]
    ordered = []
    cursor = 0
    total_batches = max(math.ceil(n / max(batch_size, 1)), 1)
    with tqdm(total=n, desc="online-rerank-mass-anchor", unit="sample", dynamic_ncols=True, mininterval=PROGRESS_REFRESH_SEC) as pbar:
        built_batches = 0
        while cursor < n:
            anchor_mass = float(sorted_masses[cursor])
            window = max(float(start_window_da), 1e-6)
            right = cursor
            while True:
                upper = anchor_mass + window
                new_right = int(np.searchsorted(sorted_masses, upper, side="right"))
                if (new_right - cursor) >= batch_size:
                    right = cursor + batch_size
                    break
                right = new_right
                if right >= n:
                    right = n
                    break
                if window >= max_window_da:
                    break
                window = min(window * expand_factor, max_window_da)
            if right <= cursor:
                right = min(cursor + batch_size, n)
            chunk = sorted_idx[cursor:right].copy()
            if chunk.size > 1:
                rng.shuffle(chunk)
            ordered.append(chunk)
            pbar.update(max(right - cursor, 0))
            built_batches += 1
            pbar.set_postfix(batches=f"{built_batches}/{total_batches}", anchor_mass=f"{anchor_mass:.2f}", window_da=f"{window:.2f}", refresh=False)
            cursor = right
    return np.concatenate(ordered, axis=0)


def _enhance_batch_peptide_diversity(
    global_order: np.ndarray,
    peptide_ids: np.ndarray,
    batch_size: int,
    max_search_batches: int = 8,
) -> np.ndarray:
    n_total = global_order.shape[0]
    if n_total == 0:
        return global_order
    order = global_order.copy()
    search_limit = max(batch_size * max_search_batches, batch_size)
    batch_starts = list(range(0, n_total, batch_size))
    for s in tqdm(batch_starts, desc="online-rerank-peptide-diversity", unit="batch", dynamic_ncols=True, mininterval=PROGRESS_REFRESH_SEC):
        e = min(s + batch_size, n_total)
        if (e - s) <= 1:
            continue
        seen = set()
        dup_pos = []
        unique_in_batch = set()
        for p in range(s, e):
            peptide = peptide_ids[order[p]]
            if peptide in seen:
                dup_pos.append(p)
            else:
                seen.add(peptide)
                unique_in_batch.add(peptide)
        if not dup_pos:
            continue
        k = e
        k_end = min(n_total, e + search_limit)
        for p in dup_pos:
            while k < k_end:
                candidate = peptide_ids[order[k]]
                if candidate not in unique_in_batch:
                    order[p], order[k] = order[k], order[p]
                    unique_in_batch.add(candidate)
                    k += 1
                    break
                k += 1
    return order


class OnlineMassAnchoredBatchIterableDataset(IterableDataset):
    """Compute a mass-anchored global order in memory, then yield training batches."""

    def __init__(
        self,
        input_dir: str,
        batch_size: int,
        seed: int,
        start_window_da: float,
        expand_factor: float,
        max_window_da: float,
        drop_last: bool,
        shuffle_batches: bool,
        train_batch_count: int = 4000,
    ):
        super().__init__()
        self.input_dir = input_dir
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.start_window_da = float(start_window_da)
        self.expand_factor = float(expand_factor)
        self.max_window_da = float(max_window_da)
        self.drop_last = bool(drop_last)
        self.shuffle_batches = bool(shuffle_batches)
        self.train_batch_count = int(train_batch_count)
        self.epoch = 0
        self.payloads = []
        self.lengths = []
        self.batch_ranges = []
        self._prepare_global_order()

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.batch_ranges)

    def _prepare_global_order(self):
        files = sorted(glob.glob(os.path.join(self.input_dir, "*.pkl")))
        if not files:
            raise FileNotFoundError(f"no pkl files found in {self.input_dir}")
        all_masses = []
        all_peptide_ids = []
        print(f"online mass rerank: files={len(files)}")
        print(f"online mass rerank: reading files; heartbeat every {PROGRESS_REFRESH_SEC:.0f}s")
        heartbeat_state = {"idx": 0, "total": len(files), "stage": "init", "file": "-"}
        heartbeat_stop = threading.Event()
        heartbeat_thread = _start_heartbeat(heartbeat_state, heartbeat_stop, tag="online-rerank-read")
        try:
            for i, fp in enumerate(tqdm(files, desc="online-rerank-read-files", unit="file", dynamic_ncols=True, mininterval=PROGRESS_REFRESH_SEC), start=1):
                heartbeat_state["idx"] = i
                heartbeat_state["file"] = os.path.basename(fp)
                heartbeat_state["stage"] = "pickle.load"
                with open(fp, "rb") as f:
                    payload = pickle.load(f)
                heartbeat_state["stage"] = "extract_arrays"
                precursors = np.asarray(payload["precursors"])
                if precursors.shape[-1] > 2:
                    precursors = precursors.copy()
                    precursors[..., 2:] = 0
                    payload["precursors"] = precursors
                n = int(precursors.shape[0])
                peptide_ids = _extract_peptide_ids(payload, n)
                self.payloads.append(payload)
                self.lengths.append(n)
                all_masses.append(np.asarray(precursors[:, 0], dtype=np.float64))
                all_peptide_ids.append(peptide_ids)
                if (i % 20) == 0 or i == len(files):
                    tqdm.write(f"online rerank read: {i}/{len(files)} files, samples={int(np.sum(self.lengths))}")
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1.0)
        masses = np.concatenate(all_masses, axis=0)
        peptide_ids = np.concatenate(all_peptide_ids, axis=0)
        print(f"online mass rerank: samples={int(masses.shape[0])}")
        rng = np.random.default_rng(self.seed)
        global_order = _build_mass_anchored_order(masses=masses, batch_size=self.batch_size, start_window_da=self.start_window_da, expand_factor=self.expand_factor, max_window_da=self.max_window_da, rng=rng)
        global_order = _enhance_batch_peptide_diversity(global_order=global_order, peptide_ids=peptide_ids, batch_size=self.batch_size)
        starts = np.cumsum([0] + self.lengths)
        self.src_file_ids = np.searchsorted(starts[1:], global_order, side="right").astype(np.int64)
        self.src_local_ids = (global_order - starts[self.src_file_ids]).astype(np.int64)
        n_total = int(np.sum(self.lengths))
        n_batch = n_total // self.batch_size
        if (not self.drop_last) and (n_total % self.batch_size) > 0:
            n_batch += 1
        self.batch_ranges = []
        for i in range(n_batch):
            s = i * self.batch_size
            e = min((i + 1) * self.batch_size, n_total)
            if (e - s) < self.batch_size and self.drop_last:
                continue
            self.batch_ranges.append((s, e))
        total_batch_count = len(self.batch_ranges)
        if self.train_batch_count > 0 and self.train_batch_count < total_batch_count:
            subset_rng = np.random.default_rng(self.seed + 4096)
            keep = np.sort(subset_rng.choice(total_batch_count, size=self.train_batch_count, replace=False))
            self.batch_ranges = [self.batch_ranges[int(i)] for i in keep]
            print(f"online mass rerank: selected {len(self.batch_ranges)}/{total_batch_count} batches with seed={self.seed}")
        else:
            print(f"online mass rerank: using all {total_batch_count} batches")
        print(f"online mass rerank: batches={len(self.batch_ranges)} (drop_last={self.drop_last})")

    def _gather_ndarray(self, key: str, file_ids: np.ndarray, local_ids: np.ndarray) -> np.ndarray:
        template = np.asarray(self.payloads[int(file_ids[0])][key])
        out_shape = (file_ids.shape[0],) + tuple(template.shape[1:])
        out = np.empty(out_shape, dtype=template.dtype)
        for fid in np.unique(file_ids):
            pos = np.where(file_ids == fid)[0]
            src = np.asarray(self.payloads[int(fid)][key])
            out[pos] = src[local_ids[pos]]
        return out

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise RuntimeError("online mass rerank mode requires num_workers=0")
        batch_indices = list(range(len(self.batch_ranges)))
        if self.shuffle_batches and len(batch_indices) > 1:
            rng = random.Random(self.seed + self.epoch + 2048)
            rng.shuffle(batch_indices)
        for bidx in batch_indices:
            s, e = self.batch_ranges[bidx]
            seg_file_ids = self.src_file_ids[s:e]
            seg_local_ids = self.src_local_ids[s:e]
            spectra = self._gather_ndarray("spectra", seg_file_ids, seg_local_ids)
            spectra_mask = self._gather_ndarray("spectra_mask", seg_file_ids, seg_local_ids)
            precursors = self._gather_ndarray("precursors", seg_file_ids, seg_local_ids)
            if precursors.shape[-1] > 2:
                precursors = precursors.copy()
                precursors[..., 2:] = 0
            tokens = self._gather_ndarray("tokens", seg_file_ids, seg_local_ids)
            labels = self._gather_ndarray("label", seg_file_ids, seg_local_ids)
            weights = self._gather_ndarray("weight", seg_file_ids, seg_local_ids) if "weight" in self.payloads[0] else np.ones_like(labels)
            unmask = self._gather_ndarray("unmask", seg_file_ids, seg_local_ids) if "unmask" in self.payloads[0] else np.zeros_like(labels)
            yield (
                torch.as_tensor(spectra, dtype=torch.float32),
                torch.as_tensor(spectra_mask, dtype=torch.bool),
                torch.as_tensor(precursors, dtype=torch.float32),
                torch.as_tensor(tokens, dtype=torch.long),
                torch.as_tensor(labels, dtype=torch.float32),
                torch.as_tensor(weights, dtype=torch.float32),
                torch.as_tensor(unmask, dtype=torch.float32),
            )


def set_seeds(seed: int, deterministic: bool = False) -> None:
    torch.manual_seed(seed) # 设置 PyTorch 的 CPU 随机数种子，保证所有 CPU 上的随机操作（如权重初始化、数据打乱）可复现
    torch.cuda.manual_seed_all(seed) # 设置 PyTorch 的所有 GPU 随机数种子，保证所有 GPU 上的随机操作一致可复现
    np.random.seed(seed) # 设置 NumPy 的随机数种子，保证 NumPy 相关的随机操作（如数据处理、采样）可复现
    random.seed(seed) # 设置 Python 标准库的随机数种子，保证 Python 自带的 random 模块操作可复现

    # 保证卷积操作完全可复现，速度会慢一些
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def get_torch_optimizer(optimizer):
    if isinstance(optimizer, Optimizer):
        return optimizer
    if hasattr(optimizer, "optimizer") and isinstance(optimizer.optimizer, Optimizer):
        return optimizer.optimizer
    raise TypeError(f"{type(optimizer).__name__} 不是有效的 torch.optim.Optimizer")

# 学习率调度器
class WarmupScheduler:
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_iter: int,
        max_iter: int,
        max_lr: float,
        min_lr: float,
        warmup_type: str,
        last_batch_iteration: int = -1,
    ):
        self.optimizer = get_torch_optimizer(optimizer)
        self.warmup_iter = warmup_iter
        self.max_iter = max_iter
        self.warmup_type = warmup_type
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.last_batch_iteration = last_batch_iteration
        self.org_lrs = [group["lr"] for group in self.optimizer.param_groups]
        self._last_lr = self.org_lrs

    # 返回一个学习率缩放比例因子 ratio
    def get_exponential_lr_factor(self) -> float:
        if self.last_batch_iteration <= self.warmup_iter:
            return self.last_batch_iteration / max(self.warmup_iter, 1)

        # 计算当前衰减进度
        progress = (self.last_batch_iteration - self.warmup_iter) / max(
            self.max_iter - self.warmup_iter, 1
        )

        # 把 progress 限制在 [0, 1] 范围内
        progress = min(max(progress, 0.0), 1.0)

        # 幂函数衰减
        return (1 - progress) ** 0.9

    # 余弦退火策略
    def get_cosine_lr_factor(self) -> float:
        if self.last_batch_iteration <= self.warmup_iter:
            return self.last_batch_iteration / max(self.warmup_iter, 1)

        progress = (self.last_batch_iteration - self.warmup_iter) / max(
            self.max_iter - self.warmup_iter, 1
        )
        progress = min(max(progress, 0.0), 1.0)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(progress * np.pi))
        return lr / self.max_lr

    # 负责决定当前 step 应该用哪个 ratio
    def get_lr_ratio(self) -> float:
        if self.last_batch_iteration < 0:
            return 0.0

        if self.warmup_type == "exp":
            ratio = self.get_exponential_lr_factor()
        elif self.warmup_type == "cos":
            ratio = self.get_cosine_lr_factor()
        else:
            ratio = 1.0

        ratio = min(1.0, float(ratio))
        return max(0.0, ratio)

    def get_lr(self):
        # 拿到当前 step 对应的学习率比例
        ratio = self.get_lr_ratio()
        # 保证学习率不会低于 min_lr
        return [max(float(org_lr) * ratio, self.min_lr) for org_lr in self.org_lrs]

    # 每训练一步，就调用一次它，用来更新 optimizer 的学习率
    def step(self, last_batch_iteration=None):
        if last_batch_iteration is None:
            last_batch_iteration = self.last_batch_iteration + 1

        self.last_batch_iteration = last_batch_iteration
        lrs = self.get_lr()

        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    # 返回最近一次 step 后的学习率列表
    def get_last_lr(self):
        return self._last_lr

# 接收一个配置字典，返回一个词表（vocab）和字符到索引的映射字典（s2i）
def build_vocab_and_s2i(config: dict) -> tuple[list[str], dict[str, int]]:
    # 从配置字典中获取 residues
    residues = config.get("residues", {})

    # 如果 residues 为空，说明配置文件缺少必要信息，直接报错
    if not residues:
        raise ValueError("model.yaml 中缺少 residues，无法构建词表")
    
    # 构建词表
    # 包含特殊符号 <pad>（填充）、<mask>（掩码）、所有氨基酸符号（来自 residues）、以及 <unk>（未知符号）
    vocab = ["<pad>", "<mask>"] + list(residues.keys()) + ["<unk>"]

    # 用字典推导式，把词表中的每个符号映射为唯一的整数索引（从0开始）
    s2i = {v: i for i, v in enumerate(vocab)}

    # 返回词表和索引映射
    return vocab, s2i


def build_masked_tokens_and_labels(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    mask_token_id: int,
    unk_token_id: int,
    token_mask_ratio: float,
    ignore_index: int,
    unmask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """构造 MLM 输入与标签。

    - masked_tokens: 仅二分类/MLM 分支使用。
    - tokens_label: 仅被 mask 的位置保留原 token，其余位置置为 ignore_index。
    - decoy 样本的 MLM 目标改写为 <unk>，避免把 decoy 当作正常序列重建。
    """
    masked_tokens = tokens.clone()
    tokens_label = torch.full_like(tokens, fill_value=ignore_index)

    valid_pos = tokens.ne(0)
    if unmask is not None:
        sample_allow_mask = (unmask <= 0.5).view(-1, 1)
        valid_pos = valid_pos & sample_allow_mask

    rand_mask = torch.rand(tokens.shape, device=tokens.device) < float(token_mask_ratio)
    mask_pos = valid_pos & rand_mask

    tokens_label[mask_pos] = tokens[mask_pos]
    masked_tokens[mask_pos] = int(mask_token_id)

    decoy_mask = (labels <= 0.5).view(-1, 1)
    tokens_label = torch.where(
        decoy_mask & tokens_label.ne(ignore_index),
        torch.full_like(tokens_label, int(unk_token_id)),
        tokens_label,
    )

    return masked_tokens, tokens_label

def parse_args():
    parser = argparse.ArgumentParser(description="MSGPT 本地训练脚本（加速版）")
    parser.add_argument("--config", type=str, default="/home/yiwen/AIPC/scripts/organized_attantion/config/model.yaml", help="配置文件路径")
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/pkl_dataset/train",
        help="训练数据目录，支持 pkl 或 parquet",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker 数")
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/checkpoints",
        help="模型保存目录",
    )
    parser.add_argument("--epochs", type=int, default=0, help="覆盖配置里的 epochs，0 表示不覆盖")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="训练设备，支持 cpu/cuda/cuda:0，也支持直接传 GPU 序号如 0/1",
    )
    parser.add_argument("--amp", action="store_true", help="启用 AMP 混合精度")
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="AMP 精度类型",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="启用 torch.compile（首次会有编译开销）",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="启用确定性模式（更慢，但更可复现）",
    )
    parser.add_argument(
        "--shuffle_within_file",
        action="store_true",
        help="是否打乱文件内样本顺序（默认不打乱）",
    )
    parser.add_argument(
        "--disable_shuffle_batches",
        action="store_true",
        help="关闭 batch 级顺序打乱（默认开启）",
    )
    parser.add_argument(
        "--online_mass_anchor_train",
        action="store_true",
        help="启用在线质量锚定重排训练（不落盘重排结果）",
    )
    parser.add_argument(
        "--online_input_dir",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/pkl_dataset/train",
        help="在线重排模式的输入目录（原始 pkl）",
    )
    parser.add_argument(
        "--online_start_window_da",
        type=float,
        default=1.0,
        help="在线重排初始质量窗口（Da）",
    )
    parser.add_argument(
        "--online_expand_factor",
        type=float,
        default=2.0,
        help="在线重排窗口扩展倍数",
    )
    parser.add_argument(
        "--online_max_window_da",
        type=float,
        default=64.0,
        help="在线重排窗口最大上限（Da）",
    )
    parser.add_argument(
        "--online_drop_last",
        action="store_true",
        help="在线重排是否丢弃尾部不足 batch_size 的批次",
    )
    parser.add_argument(
        "--train_batch_count",
        type=int,
        default=4000,
        help="从完整重排结果中可复现抽取多少个 batch 进入训练，<=0 表示使用全部 batch",
    )
    parser.add_argument(
        "--debug_log_interval",
        type=int,
        default=10,
        help="每隔多少个 step 记录一次详细调试日志（0 表示关闭）",
    )
    parser.add_argument(
        "--dda_loss_weight",
        type=float,
        default=0.18,
        help="DDA 损失权重，>0 时 DDA 头会被联合训练",
    )
    parser.add_argument(
        "--contrastive_weight",
        type=float,
        default=1,
        help="Contrastive 损失权重，>=0",
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=1.0,
        help="梯度裁剪阈值，<=0 表示关闭",
    )
    return parser.parse_args()


# AdamW 优化器的创建
def build_optimizer(model, learning_rate: float, weight_decay: float, device: torch.device):
    # 仅在 CUDA 上启用 fused，CPU 上退化到普通 AdamW。
    use_fused = (device.type == "cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=use_fused,
    )
    print(f"优化器: AdamW(fused={use_fused})")
    return optimizer

# 在流式读取 pkl 数据时，提前估算“一个 epoch 大概会跑多少个 batch/step”
def estimate_stream_steps_per_epoch(
    pkl_files: list[str],
    batch_size: int,
) -> int:
    # 空文件列表处理
    if not pkl_files:
        return 1

    # 初始化总步数
    total_steps = 0
    # 逐个文件处理
    for fp in pkl_files:
        with open(fp, "rb") as f:
            data = pickle.load(f)
        num_samples = int(len(data["label"]))
        total_steps += max(math.ceil(num_samples / max(batch_size, 1)), 1) # ceil 是向上取整

    return total_steps


def main():
    args = parse_args()
    train_batch_count = int(args.train_batch_count)
    debug_log_interval = int(args.debug_log_interval)
    if debug_log_interval < 0:
        raise ValueError("--debug_log_interval 不能小于 0")
    if train_batch_count < 0:
        raise ValueError("--train_batch_count 不能小于 0")
    if args.dda_loss_weight < 0:
        raise ValueError("--dda_loss_weight 不能小于 0")
    if args.contrastive_weight < 0:
        raise ValueError("--contrastive_weight 不能小于 0")
    if args.grad_clip_norm < 0:
        raise ValueError("--grad_clip_norm 不能小于 0")

    # 规范化 device 参数：支持 --device 0/1/gpu:1/cuda:1。
    if args.online_start_window_da <= 0:
        raise ValueError("--online_start_window_da must be > 0")
    if args.online_expand_factor <= 1.0:
        raise ValueError("--online_expand_factor must be > 1")
    if args.online_max_window_da <= 0:
        raise ValueError("--online_max_window_da must be > 0")

    device_arg = str(args.device).strip()
    device_arg_lower = device_arg.lower()
    if device_arg.isdigit():
        device_arg = f"cuda:{device_arg}"
        device_arg_lower = device_arg.lower()
    elif device_arg_lower.startswith("gpu"):
        gpu_suffix = device_arg_lower[3:].lstrip(":")
        if gpu_suffix.isdigit():
            device_arg = f"cuda:{gpu_suffix}"
            device_arg_lower = device_arg.lower()
        else:
            raise ValueError("--device 若使用 gpu 前缀，格式应为 gpu:N（如 gpu:0）")

    if device_arg_lower.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("当前未检测到可用 GPU，请检查 CUDA 环境，或显式传 --device cpu")
    device = torch.device(device_arg)

    # CUDA 加速项，开启混合精度，提升训练速度
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 在训练开始前，检查关键输入文件和目录是否存在，防止后续报错
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"找不到配置文件: {args.config}")
    selected_train_dir = args.train_data_dir
    if args.online_mass_anchor_train:
        selected_train_dir = args.online_input_dir
    if not os.path.isdir(selected_train_dir):
        raise FileNotFoundError(f"找不到训练数据目录: {selected_train_dir}")


    # 用 yaml 库读取配置文件（如 model.yaml），并解析为 Python 字典对象 config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)


    # 根据配置文件中的 residues 字段，生成一个“词表”（vocab）和“字符到索引的映射字典”（s2i）
    # 词表（vocab）包含 <pad>、<mask>、所有氨基酸符号、<unk>，用于后续序列编码
    # s2i 是一个字典，把每个词表中的符号映射为唯一的整数索引，便于模型输入
    vocab, s2i = build_vocab_and_s2i(config)
    # 把生成的词表（vocab）存入配置字典 config，方便后续模型初始化和数据处理时使用
    config["vocab"] = vocab

    # 获取随机种子
    seed = int(config.get("seed", 123))
    set_seeds(seed, deterministic=args.deterministic)

    # 打印系统基本参数
    print(f"使用设备: {device}")
    print(f"deterministic={args.deterministic}")
    print(f"num_workers={args.num_workers}")

    # 查找训练数据地址中所有pkl文件
    # glob库用于批量查找文件，os库用于各种文件、路径和系统操作
    # pkl_files：文件地址列表
    pkl_files = sorted(glob.glob(os.path.join(selected_train_dir, "*.pkl")))
    if not pkl_files:
        raise ValueError(
            f"在 {selected_train_dir} 下未发现 pkl 文件。"
            "当前训练入口只实现了 pkl 流式读取，请先转换数据或切换到 parquet 训练脚本。"
        )

    # 使用全量训练文件。
    # 统计具体文件数量
    use_pkl_stream = len(pkl_files) > 0

    # 从 config 字典中提取 batch_size
    batch_size = int(config.get("train_batch_size", 256))

    # 创建一个训练数据集对象 train_dataset，用于后续 DataLoader 迭代训练
    train_dataset = PKLBatchIterableDataset(
        pkl_dir=selected_train_dir,
        batch_size=batch_size,
        pkl_files=pkl_files,
        shuffle_within_file=bool(args.shuffle_within_file),
        shuffle_batches=(not args.disable_shuffle_batches),
    )

    print(
        f"读取已编码训练文件数(pkl): 总计={len(pkl_files)}"
    )
    print(
        f"shuffle_within_file={bool(args.shuffle_within_file)}, "
        f"shuffle_batches={not args.disable_shuffle_batches}"
    )
    print("使用流式批量读取模式（整文件一次 tensor 化 + worker 预取）")

    # 创建数据加载器，从 train_dataset 中读取数据
    train_loader = DataLoader(
        train_dataset,
        batch_size=None, # 不再自己组织 batch，train_dataset已经实现
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(), # 开启 pinned memory（页锁定内存）
        persistent_workers=(args.num_workers > 0), # 是否保留 worker 子进程
        prefetch_factor=(4 if args.num_workers > 0 else None), # 每个 worker 预先准备多少个 batch
    )

    # 创建 MSGPT 模型
    if args.online_mass_anchor_train:
        print("启用在线质量锚定重排训练：重排结果不落盘，重排后立即训练")
        if args.num_workers > 0:
            print("在线重排模式下将强制 num_workers=0，避免多进程重复持有大内存")
        train_dataset = OnlineMassAnchoredBatchIterableDataset(
            input_dir=args.online_input_dir,
            batch_size=batch_size,
            seed=seed,
            start_window_da=float(args.online_start_window_da),
            expand_factor=float(args.online_expand_factor),
            max_window_da=float(args.online_max_window_da),
            drop_last=bool(args.online_drop_last),
            shuffle_batches=(not args.disable_shuffle_batches),
            train_batch_count=train_batch_count,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=None,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
        )

    model = MSGPT(
        dim_model=int(config.get("dim_model", 768)),
        n_head=int(config.get("n_head", 16)),
        dim_feedforward=int(config.get("dim_feedforward", 1024)),
        n_layers=int(config.get("n_layers", 9)),
        dropout=float(config.get("dropout", 0.0)),
        max_length=int(config.get("max_length", 50)),
        vocab_size=len(vocab),
        max_charge=int(config.get("max_charge", 10)),
    ).to(device)

    print("从随机初始化开始训练，不加载启动权重")

    # 条件判断是否开启编译优化
    if args.compile and hasattr(torch, "compile"): # 检查 torch 模块里有没有 compile 这个属性/函数
        print("启用 torch.compile")
        model = torch.compile(model)

    # 定义学习率与权重衰减率
    learning_rate = float(config.get("learning_rate", 5e-5))
    weight_decay = float(config.get("weight_decay", 5e-5))

    # 创建一个用于更新模型参数的优化器（optimizer）
    optimizer = build_optimizer(model, learning_rate, weight_decay, device)

    # 定义 epoch：命令行 --epochs > 0 时覆盖配置文件。
    epochs = int(args.epochs) if int(args.epochs) > 0 else int(config.get("epochs", 25))

    # 估计 its
    if args.online_mass_anchor_train:
        steps_per_epoch = len(train_dataset)
        print(f"在线质量重排模式 steps_per_epoch={steps_per_epoch}")
    elif use_pkl_stream:
        steps_per_epoch = estimate_stream_steps_per_epoch(
            pkl_files=pkl_files,
            batch_size=batch_size,
        )
        print(f"流式模式 steps_per_epoch(自动估计)={steps_per_epoch}")
    else:
        steps_per_epoch = len(train_loader)

    # 确定学习率调度器总步数。
    scheduled_total_steps = epochs * steps_per_epoch

    # 确定学习率调度器要跑多长，以及 warmup 要持续多少步
    max_iters = max(int(scheduled_total_steps), 1)
    warmup_iters = max(int(float(config.get("warmup_ratio", 0.2)) * max_iters), 1)

    # 创建一个学习率调度器 scheduler，用来在训练过程中动态调整优化器的学习率
    scheduler = WarmupScheduler(
        optimizer=optimizer, # 把前面创建好的优化器传给调度器
        warmup_iter=warmup_iters, # warmup 阶段持续多少步
        max_iter=max_iters, # 整个训练总共要跑多少步
        max_lr=learning_rate, # 最高学习率
        min_lr=float(config.get("min_lr", 5e-6)), # 最低学习率
        warmup_type=str(config.get("warmup_strategy", "exp")), # 指数式衰减风格
    )

    dda_criterion = nn.BCEWithLogitsLoss(reduction="none") # DDA 加权损失
    # MLM 相关超参硬编码，避免通过命令行注入。
    mlm_ignore_index = 0
    mlm_mask_ratio = 0.4
    mask_loss_weight = 1.0

    mask_criterion = nn.CrossEntropyLoss(ignore_index=mlm_ignore_index)
    mask_token_id = int(s2i["<mask>"])
    unk_token_id = int(s2i["<unk>"])
    temperature_max = float(config.get("contrastive_temperature_max", 100.0))
    dda_loss_weight = float(args.dda_loss_weight)
    contrastive_weight = float(args.contrastive_weight)
    grad_clip_norm = float(args.grad_clip_norm)

    # 配置 AMP（Automatic Mixed Precision，自动混合精度训练）
    # 加速 GPU 训练并减少显存占用，同时保持数值稳定
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    print(f"AMP: {use_amp}, dtype: {args.amp_dtype}")
    print(f"dda_loss_weight={dda_loss_weight}")
    print(f"contrastive_weight={contrastive_weight}")
    print(f"mask_loss_weight={mask_loss_weight}")
    print(f"mlm_mask_ratio={mlm_mask_ratio}")
    print(f"mlm_ignore_index={mlm_ignore_index}")
    print(f"grad_clip_norm={grad_clip_norm}")

    # 确定模型权重保存目录，并确保这个目录存在
    model_save_path = args.model_save_path if args.model_save_path else str(
        config.get("model_save_path", "checkpoints")
    )
    os.makedirs(model_save_path, exist_ok=True)
    batch_pred_log_path = os.path.join(model_save_path, "batch_pred_top10.log")
    debug_metrics_log_path = os.path.join(model_save_path, "debug_metrics.csv")
    with open(batch_pred_log_path, "w", encoding="utf-8") as f:
        f.write("epoch,step,global_step,spectra_emb_norm,pep_emb_norm,target_count,decoy_count,temperature\n")
    with open(debug_metrics_log_path, "w", encoding="utf-8") as f:
        f.write(
            "epoch,step,global_step,lr,temperature,loss,contrastive_loss,dda_loss,mask_loss,"
            "contrastive_acc_s2p,contrastive_acc_p2s,pos_sim_mean,neg_sim_mean,sim_gap,"
            "contrastive_acc_s2p_target,contrastive_acc_p2s_target,pos_sim_mean_target,neg_sim_mean_target,sim_gap_target,"
            "dda_prob_mean,dda_target_prob_mean,dda_decoy_prob_mean,"
            "target_count,decoy_count,masked_token_count,masked_token_ratio,grad_norm,param_norm,spectra_emb_norm,pep_emb_norm\n"
        )

    # 初始化全局步数和累积损失
    global_step = 0
    running_loss, dda_running_loss, contrastive_running_loss, mask_running_loss = None, None, None, None

    # 按 epoch 循环训练
    for epoch in range(epochs):
        # 每个 epoch 开始时，设置流式数据集的 epoch
        if use_pkl_stream and hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)

        # 把模型切换到训练模式
        model.train()

        # 设置进度条总长度
        bar_total = None if use_pkl_stream else len(train_loader)
        # 创建 tqdm 进度条，设置 mininterval 防止刷新过快导致刷屏
        train_bar = tqdm(
            train_loader,
            total=bar_total,
            desc=f"Epoch {epoch + 1}/{epochs}",
            mininterval=1.0,
            dynamic_ncols=True,
        )
        loader_iter = train_bar

        # 遍历一个 epoch 中的每个 batch
        for step, batch in enumerate(loader_iter, start=1): # 同时得到 step 和 batch
            # 解包 batch
            if len(batch) == 7:
                spectra, spectra_mask, precursors, tokens, labels, weights, unmask = batch
            else:
                spectra, spectra_mask, precursors, tokens, labels, weights = batch
                unmask = None

            # 不再进行 MLM token mask，直接用原始肽段序列训练 contrastive + DDA。
            spectra = spectra.to(device, non_blocking=True)
            spectra_mask = spectra_mask.to(device, non_blocking=True)
            precursors = precursors.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            if unmask is not None:
                unmask = unmask.to(device, non_blocking=True)

            masked_tokens, tokens_label = build_masked_tokens_and_labels(
                tokens=tokens,
                labels=labels,
                mask_token_id=mask_token_id,
                unk_token_id=unk_token_id,
                token_mask_ratio=mlm_mask_ratio,
                ignore_index=mlm_ignore_index,
                unmask=unmask,
            )

            # 清空旧梯度
            optimizer.zero_grad(set_to_none=True)

            # 选择 autocast 的设备类型
            autocast_device = "cuda" if device.type == "cuda" else "cpu"

            # 在自动混合精度上下文中做前向和损失计算
            with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
                spectra_latent, pep_latent, dda_pred, mask_pred, temperature = model(
                    spectra,
                    spectra_mask,
                    precursors,
                    tokens,
                    binary_tokens=masked_tokens,
                    return_mask_pred=True,
                )

                # DeepSearch 风格：谱图 latent vs 肽段 latent 的双向 in-batch CE
                target_mask = labels > 0.5
                if target_mask.any():
                    target_spectra = spectra_latent[target_mask]
                    target_peps = pep_latent[target_mask]

                    # 默认将 decoy 肽向量并入谱图侧负样本池
                    decoy_mask = labels <= 0.5
                    decoy_peps = pep_latent[decoy_mask]
                    contrastive_loss = coca_inbatch_contrastive_loss_with_decoys(
                        target_spectra,
                        target_peps,
                        temperature=temperature,
                        decoy_latents=decoy_peps,
                    )
                else:
                    contrastive_loss = torch.zeros((), device=device, dtype=dda_pred.dtype)

                dda_loss_all = dda_criterion(dda_pred, labels)
                dda_loss = (dda_loss_all * weights).sum() / weights.sum().clamp_min(1e-8)

                if tokens_label.ne(mlm_ignore_index).any():
                    mask_loss = mask_criterion(mask_pred, tokens_label)
                else:
                    mask_loss = torch.zeros((), device=device, dtype=dda_pred.dtype)

                # 合成总损失：contrastive + dda + mlm
                loss = (
                    contrastive_weight * contrastive_loss
                    + dda_loss_weight * dda_loss
                    + mask_loss_weight * mask_loss
                )

            # 如果启用了 GradScaler，就走混合精度反向传播
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, norm_type=2.0)
                grad_l2_sq = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_l2_sq += float(p.grad.detach().float().pow(2).sum().item())
                grad_norm = math.sqrt(max(grad_l2_sq, 0.0))
                scaler.step(optimizer)
                scaler.update()
            # 否则走普通反向传播
            else:
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, norm_type=2.0)
                grad_l2_sq = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_l2_sq += float(p.grad.detach().float().pow(2).sum().item())
                grad_norm = math.sqrt(max(grad_l2_sq, 0.0))
                # 根据当前梯度更新模型参数
                optimizer.step()

            # 与 DeepSearch 一致，约束 temperature 的上界，防止数值过大
            with torch.no_grad():
                model.temperature.clamp_(0, math.log(max(temperature_max, 1.0)))

            # 更新学习率调度器
            scheduler.step()

            # 更新累积 Loss (Exponential Moving Average)
            global_step += 1
            if running_loss is None:
                running_loss = loss.item()
                dda_running_loss = dda_loss.item()
                contrastive_running_loss = contrastive_loss.item()
                mask_running_loss = mask_loss.item()
            else:
                running_loss = 0.99 * running_loss + (1 - 0.99) * loss.item()
                dda_running_loss = 0.99 * dda_running_loss + (1 - 0.99) * dda_loss.item()
                contrastive_running_loss = 0.99 * contrastive_running_loss + (1 - 0.99) * contrastive_loss.item()
                mask_running_loss = 0.99 * mask_running_loss + (1 - 0.99) * mask_loss.item()

            if step % 50 == 0:
                lr = scheduler.get_last_lr()[0]

                emb_norm_view1 = spectra_latent.detach().float().norm(dim=1).mean().item()
                emb_norm_view2 = pep_latent.detach().float().norm(dim=1).mean().item()
                target_count = int((labels > 0.5).sum().item())
                decoy_count = int((labels <= 0.5).sum().item())
                temperature_value = float(temperature.detach().float().item())
                with open(batch_pred_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{epoch + 1},{step},{global_step},{emb_norm_view1:.6f},{emb_norm_view2:.6f},{target_count},{decoy_count},{temperature_value:.6f}\n"
                    )
                
                train_bar.set_postfix(
                    avg=f"{running_loss:.4f}",
                    dda=f"{dda_running_loss:.4f}",
                    contrastive=f"{contrastive_running_loss:.4f}",
                    mlm=f"{mask_running_loss:.4f}",
                    lr=f"{lr:.2e}",
                )

            if debug_log_interval > 0 and (step % debug_log_interval == 0):
                with torch.no_grad():
                    lr = scheduler.get_last_lr()[0]

                    spectra_latent_f = spectra_latent.detach().float()
                    pep_latent_f = pep_latent.detach().float()
                    sim_matrix = spectra_latent_f @ pep_latent_f.t()
                    batch_n = int(sim_matrix.shape[0])
                    if batch_n > 0:
                        diag = sim_matrix.diag()
                        pos_sim_mean = float(diag.mean().item())
                        if batch_n > 1:
                            neg_mask = ~torch.eye(batch_n, dtype=torch.bool, device=sim_matrix.device)
                            neg_sim_mean = float(sim_matrix[neg_mask].mean().item())
                        else:
                            neg_sim_mean = float("nan")
                        row_argmax = sim_matrix.argmax(dim=1)
                        col_argmax = sim_matrix.argmax(dim=0)
                        gt_idx = torch.arange(batch_n, device=sim_matrix.device)
                        contrastive_acc_s2p = float((row_argmax == gt_idx).float().mean().item())
                        contrastive_acc_p2s = float((col_argmax == gt_idx).float().mean().item())
                    else:
                        pos_sim_mean = float("nan")
                        neg_sim_mean = float("nan")
                        contrastive_acc_s2p = float("nan")
                        contrastive_acc_p2s = float("nan")

                    if math.isnan(neg_sim_mean):
                        sim_gap = float("nan")
                    else:
                        sim_gap = pos_sim_mean - neg_sim_mean

                    # 仅在 target 子集上评估检索指标，和 contrastive_loss 的优化目标保持一致。
                    labels_f = labels.detach().float()
                    target_mask = labels_f > 0.5
                    target_idx = torch.where(target_mask)[0]
                    target_n = int(target_idx.numel())
                    if target_n > 0:
                        sim_target = sim_matrix.index_select(0, target_idx).index_select(1, target_idx)
                        diag_target = sim_target.diag()
                        pos_sim_mean_target = float(diag_target.mean().item())
                        if target_n > 1:
                            neg_mask_target = ~torch.eye(target_n, dtype=torch.bool, device=sim_target.device)
                            neg_sim_mean_target = float(sim_target[neg_mask_target].mean().item())
                            row_argmax_target = sim_target.argmax(dim=1)
                            col_argmax_target = sim_target.argmax(dim=0)
                            gt_idx_target = torch.arange(target_n, device=sim_target.device)
                            contrastive_acc_s2p_target = float((row_argmax_target == gt_idx_target).float().mean().item())
                            contrastive_acc_p2s_target = float((col_argmax_target == gt_idx_target).float().mean().item())
                        else:
                            neg_sim_mean_target = float("nan")
                            contrastive_acc_s2p_target = float("nan")
                            contrastive_acc_p2s_target = float("nan")
                    else:
                        pos_sim_mean_target = float("nan")
                        neg_sim_mean_target = float("nan")
                        contrastive_acc_s2p_target = float("nan")
                        contrastive_acc_p2s_target = float("nan")

                    if math.isnan(neg_sim_mean_target):
                        sim_gap_target = float("nan")
                    else:
                        sim_gap_target = pos_sim_mean_target - neg_sim_mean_target

                    probs = torch.sigmoid(dda_pred.detach().float())
                    target_mask = labels_f > 0.5
                    decoy_mask = labels_f <= 0.5
                    target_count = int(target_mask.sum().item())
                    decoy_count = int(decoy_mask.sum().item())
                    dda_prob_mean = float(probs.mean().item())
                    dda_target_prob_mean = float(probs[target_mask].mean().item()) if target_count > 0 else float("nan")
                    dda_decoy_prob_mean = float(probs[decoy_mask].mean().item()) if decoy_count > 0 else float("nan")
                    masked_token_count = int(tokens_label.ne(mlm_ignore_index).sum().item())
                    masked_token_ratio = float(masked_token_count / max(tokens_label.numel(), 1))

                    emb_norm_view1 = float(spectra_latent_f.norm(dim=1).mean().item())
                    emb_norm_view2 = float(pep_latent_f.norm(dim=1).mean().item())

                    param_l2_sq = 0.0
                    for p in model.parameters():
                        param_l2_sq += float(p.detach().float().pow(2).sum().item())
                    param_norm = math.sqrt(max(param_l2_sq, 0.0))

                    temperature_value = float(temperature.detach().float().item())

                    with open(debug_metrics_log_path, "a", encoding="utf-8") as f:
                        f.write(
                            f"{epoch + 1},{step},{global_step},{lr:.8e},{temperature_value:.6f},"
                            f"{loss.item():.6f},{contrastive_loss.item():.6f},{dda_loss.item():.6f},{mask_loss.item():.6f},"
                            f"{contrastive_acc_s2p:.6f},{contrastive_acc_p2s:.6f},{pos_sim_mean:.6f},{neg_sim_mean:.6f},{sim_gap:.6f},"
                            f"{contrastive_acc_s2p_target:.6f},{contrastive_acc_p2s_target:.6f},{pos_sim_mean_target:.6f},{neg_sim_mean_target:.6f},{sim_gap_target:.6f},"
                            f"{dda_prob_mean:.6f},{dda_target_prob_mean:.6f},{dda_decoy_prob_mean:.6f},"
                            f"{target_count},{decoy_count},{masked_token_count},{masked_token_ratio:.6f},{grad_norm:.6f},{param_norm:.6f},{emb_norm_view1:.6f},{emb_norm_view2:.6f}\n"
                        )

        ckpt_file = os.path.join(model_save_path, f"msgpt_epoch_{epoch + 1}.pt")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch + 1,
                "global_step": global_step,
            },
            ckpt_file,
        )
        print(f"已保存权重: {ckpt_file}")


if __name__ == "__main__":
    main()
