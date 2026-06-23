"""
共享训练工具模块，供 training/ 下各个训练脚本导入使用。

包含:
- 文件 I/O 辅助: _open_maybe_gz, discover_pkl_files
- 随机种子: set_seeds
- 优化器辅助: get_torch_optimizer, build_optimizer
- 学习率调度: WarmupScheduler (支持单/多 lr)
- 词表构建: build_vocab_and_s2i
- MLM 掩码: build_masked_tokens_and_labels
- 心跳日志: _start_heartbeat
- 在线质量锚定重排: _extract_peptide_ids, _build_mass_anchored_order, _enhance_batch_peptide_diversity
- 数据集类: PKLBatchIterableDataset, OnlineMassAnchoredBatchIterableDataset
- 步数估算: estimate_stream_steps_per_epoch
- 断点工具: normalize_state_dict_keys, load_pretrained_checkpoint
- 验证集收集: _gather_val_pkl_files
"""

import gzip
import glob
import math
import os
import pickle
import random
import threading
from typing import Optional, Union

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import IterableDataset
from tqdm import tqdm

PROGRESS_REFRESH_SEC = 5.0


# ==================== 文件 I/O 辅助 ====================

def _open_maybe_gz(path: str, mode: str):
    """根据后缀自动选择 open 或 gzip.open。"""
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def discover_pkl_files(pkl_dir: Union[str, list[str]]) -> list[str]:
    """扫描目录下的 .pkl 和 .pkl.gz 文件，支持单目录或多目录列表。"""
    if isinstance(pkl_dir, list):
        all_files = []
        for d in pkl_dir:
            all_files.extend(discover_pkl_files(d))
        return sorted(all_files)
    files = sorted(glob.glob(os.path.join(pkl_dir, "*.pkl")))
    files += sorted(glob.glob(os.path.join(pkl_dir, "*.pkl.gz")))
    if not files:
        raise ValueError(f"在 {pkl_dir} 下未发现 pkl/pkl.gz 文件")
    return files


# ==================== 随机种子 ====================

def set_seeds(seed: int, deterministic: bool = False) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


# ==================== 优化器辅助 ====================

def get_torch_optimizer(optimizer):
    """从 Optimizer 或 lr_scheduler wrapper 中取出底层的 Optimizer。"""
    if isinstance(optimizer, Optimizer):
        return optimizer
    if hasattr(optimizer, "optimizer") and isinstance(optimizer.optimizer, Optimizer):
        return optimizer.optimizer
    raise TypeError(f"{type(optimizer).__name__} 不是有效的 torch.optim.Optimizer")


def build_optimizer(model, learning_rate: float, weight_decay: float, device: torch.device):
    """创建 AdamW 优化器（CUDA 上启用 fused）。"""
    use_fused = (device.type == "cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=use_fused,
    )
    print(f"优化器: AdamW(fused={use_fused})")
    return optimizer


# ==================== 学习率调度器 ====================

class WarmupScheduler:
    """支持 warmup + cosine/exp 衰减的学习率调度器。

    参数 max_lr 可以是 float（所有参数组共用同一最高学习率）或
    list[float]（每个参数组独立指定最高学习率，用于分层学习率场景）。
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_iter: int,
        max_iter: int,
        max_lr: Union[float, list[float]],
        min_lr: float,
        warmup_type: str,
        last_batch_iteration: int = -1,
    ):
        self.optimizer = get_torch_optimizer(optimizer)
        self.warmup_iter = warmup_iter
        self.max_iter = max_iter
        self.warmup_type = warmup_type
        self.min_lr = min_lr
        self.last_batch_iteration = last_batch_iteration
        self.org_lrs = [group["lr"] for group in self.optimizer.param_groups]
        self._last_lr = self.org_lrs
        if isinstance(max_lr, (int, float)):
            self._max_lrs = [float(max_lr)]
        else:
            self._max_lrs = [float(v) for v in max_lr]

    def get_exponential_lr_factor(self) -> float:
        if self.last_batch_iteration <= self.warmup_iter:
            return self.last_batch_iteration / max(self.warmup_iter, 1)
        progress = (self.last_batch_iteration - self.warmup_iter) / max(
            self.max_iter - self.warmup_iter, 1
        )
        progress = min(max(progress, 0.0), 1.0)
        return (1 - progress) ** 0.9

    def get_cosine_lr_factor(self) -> float:
        if self.last_batch_iteration <= self.warmup_iter:
            return self.last_batch_iteration / max(self.warmup_iter, 1)
        progress = (self.last_batch_iteration - self.warmup_iter) / max(
            self.max_iter - self.warmup_iter, 1
        )
        progress = min(max(progress, 0.0), 1.0)
        lr = self.min_lr + 0.5 * (self._max_lrs[0] - self.min_lr) * (1 + np.cos(progress * np.pi))
        return lr / self._max_lrs[0]

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
        ratio = self.get_lr_ratio()
        return [max(float(org_lr) * ratio, self.min_lr) for org_lr in self.org_lrs]

    def step(self, last_batch_iteration=None):
        if last_batch_iteration is None:
            last_batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = last_batch_iteration
        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        return self._last_lr


# ==================== 词表构建 ====================

def build_vocab_and_s2i(config: dict) -> tuple[list[str], dict[str, int]]:
    """从 config['residues'] 构建词表和字符→索引映射。"""
    residues = config.get("residues", {})
    if not residues:
        raise ValueError("model.yaml 中缺少 residues，无法构建词表")
    vocab = ["<pad>", "<mask>"] + list(residues.keys()) + ["<unk>"]
    s2i = {v: i for i, v in enumerate(vocab)}
    return vocab, s2i


# ==================== MLM 掩码 ====================

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

    - masked_tokens: 被 mask 位置替换为 mask_token_id。
    - tokens_label: 仅被 mask 位置保留原 token，其余为 ignore_index。
    - decoy 样本的 MLM 目标改写为 <unk>。
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


# ==================== 心跳日志 ====================

def _start_heartbeat(state: dict, stop_event: threading.Event, tag: str = "heartbeat") -> threading.Thread:
    """在长时间运行的阶段定期打印进度。"""

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


# ==================== 在线质量锚定重排辅助 ====================

def _extract_peptide_ids(payload: dict, n: int) -> np.ndarray:
    """从 payload 中提取 peptide ID（优先 'peptides' 字段，fallback 到 'tokens'）。"""
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
    """按 precursor mass 排序并构造扩展窗口内的 batch 级排列。"""
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
    """通过交换重复 peptide 提升 batch 内 peptide 多样性。"""
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


# ==================== 步数估算 ====================

def estimate_stream_steps_per_epoch(
    pkl_files: list[str],
    batch_size: int,
    use_gzip: bool = False,
) -> int:
    """遍历 pkl 文件列表，估算一个 epoch 的总 step 数。"""
    if not pkl_files:
        return 1
    total_steps = 0
    open_fn = _open_maybe_gz if use_gzip else open
    for fp in pkl_files:
        with open_fn(fp, "rb") as f:
            data = pickle.load(f)
        num_samples = int(len(data["label"]))
        total_steps += max(math.ceil(num_samples / max(batch_size, 1)), 1)
    return total_steps


# ==================== 断点工具 ====================

def normalize_state_dict_keys(state_dict: dict) -> dict:
    """去掉 torch.compile (_orig_mod.) 和 DDP (module.) 包裹产生的前缀。"""
    normalized = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod."):]
        if nk.startswith("module."):
            nk = nk[len("module."):]
        normalized[nk] = v
    return normalized


def load_pretrained_checkpoint(model, checkpoint_path: str, device: torch.device):
    """加载预训练权重，兼容多种保存格式，自动跳过 shape 不匹配的参数。"""
    print(f"加载基座权重: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        elif "module" in ckpt:
            state_dict = ckpt["module"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    state_dict = normalize_state_dict_keys(state_dict)

    model_dict = model.state_dict()
    compatible_state = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            compatible_state[k] = v
        else:
            skipped.append(k)

    if skipped:
        print(f"  跳过 {len(skipped)} 个不兼容的 key: {skipped[:5]}...")

    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    loaded_count = len(compatible_state)
    total_count = len(model_dict)

    if missing:
        print(f"  缺失 key ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  多余 key ({len(unexpected)}): {unexpected[:5]}...")

    print(f"  加载参数: {loaded_count}/{total_count}")
    return model


# ==================== 验证集文件收集 ====================

def _gather_val_pkl_files(val_data_dir: str, use_gzip: bool = False) -> list[str]:
    """收集 val 目录下的 pkl 文件，排除 val.00004_val.pkl。"""
    val_files = sorted(glob.glob(os.path.join(val_data_dir, "*.pkl")))
    if use_gzip:
        val_files += sorted(glob.glob(os.path.join(val_data_dir, "*.pkl.gz")))
    exclude_name = "val.00004_val.pkl"
    val_files = [f for f in val_files if os.path.basename(f) != exclude_name]
    print(f"val 数据: 从 {val_data_dir} 收集到 {len(val_files)} 个文件（已排除 {exclude_name}）")
    return val_files


# ==================== 流式数据集 ====================

class PKLBatchIterableDataset(IterableDataset):
    """流式读取 pkl/pkl.gz 的 IterableDataset。

    按文件逐个读取，每个文件一次性 tensor 化后按 batch 切分 yield。
    支持文件内打乱和 batch 级打乱。

    Parameters
    ----------
    pkl_dir : str
        数据目录（当 pkl_files 为 None 时从此目录扫描）。
    batch_size : int
    pkl_files : list[str] or None
        显式指定文件列表，为 None 时自动扫描 pkl_dir。
    shuffle_within_file : bool
        是否打乱文件内样本顺序。
    shuffle_batches : bool
        是否打乱 batch 产出顺序。
    use_gzip : bool
        是否支持 .pkl.gz 文件（影响文件扫描和打开方式）。
    """

    def __init__(
        self,
        pkl_dir: str,
        batch_size: int,
        pkl_files: Optional[list[str]] = None,
        shuffle_within_file: bool = False,
        shuffle_batches: bool = True,
        use_gzip: bool = False,
    ):
        super().__init__()
        if pkl_files is not None:
            all_files = sorted(pkl_files)
        elif use_gzip:
            all_files = discover_pkl_files(pkl_dir)
        else:
            all_files = sorted(glob.glob(os.path.join(pkl_dir, "*.pkl")))
        if not all_files:
            raise ValueError(f"在 {pkl_dir} 下未发现 pkl 文件")
        self.all_files = all_files
        self.batch_size = int(batch_size)
        self.epoch = 0
        self.shuffle_within_file = bool(shuffle_within_file)
        self.shuffle_batches = bool(shuffle_batches)
        self.use_gzip = bool(use_gzip)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _get_files_for_this_worker(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        return self.all_files[worker_id::num_workers]

    def __iter__(self):
        files = list(self._get_files_for_this_worker())
        rng = random.Random(self.epoch + 1234)
        rng.shuffle(files)
        open_fn = _open_maybe_gz if self.use_gzip else open
        for fp in files:
            with open_fn(fp, "rb") as f:
                data = pickle.load(f)

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
            unmask = torch.as_tensor(
                data.get("unmask", np.zeros_like(data["label"])),
                dtype=torch.float32,
            )

            num_all = labels.shape[0]
            if num_all <= 0:
                continue

            num = labels.shape[0]
            if self.shuffle_within_file:
                order = torch.randperm(num)
            else:
                order = torch.arange(num)

            batch_starts = list(range(0, num, self.batch_size))
            if self.shuffle_batches and len(batch_starts) > 1:
                rng.shuffle(batch_starts)

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


# ==================== 在线质量锚定重排数据集 ====================

class OnlineMassAnchoredBatchIterableDataset(IterableDataset):
    """在线质量锚定重排 + peptide 多样性增强的 IterableDataset。

    将全部数据读入内存后，按 precursor mass 排序并用扩展窗口构造 batch，
    再通过 peptide 去重提升 batch 内多样性，最后按 batch 粒度 yield。

    Parameters
    ----------
    input_dir : str or list[str]
        输入目录（支持多数据源列表）。
    batch_size : int
    seed : int
    start_window_da : float
    expand_factor : float
    max_window_da : float
    drop_last : bool
    shuffle_batches : bool
    train_batch_count : int
        每 epoch 使用的 batch 数上限，<=0 表示全部。
    use_gzip : bool
        是否支持 .pkl.gz 文件。
    extra_pkl_files : list[str] or None
        额外加入重排的 pkl 文件路径（如 val 数据）。
    """

    def __init__(
        self,
        input_dir: Union[str, list[str]],
        batch_size: int,
        seed: int,
        start_window_da: float,
        expand_factor: float,
        max_window_da: float,
        drop_last: bool,
        shuffle_batches: bool,
        train_batch_count: int = 0,
        use_gzip: bool = False,
        extra_pkl_files: Optional[list[str]] = None,
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
        self.use_gzip = bool(use_gzip)
        self.extra_pkl_files = extra_pkl_files
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
        files = discover_pkl_files(self.input_dir)

        if self.extra_pkl_files:
            files = sorted(files + list(self.extra_pkl_files))
            seen = set()
            files = [f for f in files if not (f in seen or seen.add(f))]

        if not files:
            raise FileNotFoundError(f"no pkl files found in {self.input_dir}")

        all_masses = []
        all_peptide_ids = []
        print(f"online mass rerank: files={len(files)}")
        heartbeat_state = {"idx": 0, "total": len(files), "stage": "init", "file": "-"}
        heartbeat_stop = threading.Event()
        heartbeat_thread = _start_heartbeat(heartbeat_state, heartbeat_stop, tag="online-rerank-read")
        open_fn = _open_maybe_gz if self.use_gzip else open
        try:
            for i, fp in enumerate(tqdm(files, desc="online-rerank-read-files", unit="file", dynamic_ncols=True, mininterval=PROGRESS_REFRESH_SEC), start=1):
                heartbeat_state["idx"] = i
                heartbeat_state["file"] = os.path.basename(fp)
                heartbeat_state["stage"] = "pickle.load"
                with open_fn(fp, "rb") as f:
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
        global_order = _build_mass_anchored_order(
            masses=masses, batch_size=self.batch_size,
            start_window_da=self.start_window_da, expand_factor=self.expand_factor,
            max_window_da=self.max_window_da, rng=rng)
        global_order = _enhance_batch_peptide_diversity(
            global_order=global_order, peptide_ids=peptide_ids, batch_size=self.batch_size)
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
