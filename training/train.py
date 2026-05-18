import argparse
import glob
import math
import os
import pickle
import random
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
from model.transformer.dataset import PROTON_MASS_AMU, SpectrumDataset, padding
from model.transformer.iterable_dataset_online_parquet import mask_batch_data_by_unmask, mask_spectra_data
from model.transformer.model import MSGPT

# 一个用于流式读取 pkl 文件批量数据的 PyTorch IterableDataset，按文件流式读取，每个文件只做一次 tensor 化
class PKLBatchIterableDataset(IterableDataset):
    # pkl_dir：存放 .pkl 文件的目录
    # batch_size：每个 batch 里要放多少条样本
    def __init__(self, pkl_dir: str, batch_size: int, pkl_files: Optional[list[str]] = None):
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

            # 文件内打乱样本顺序
            num = labels.shape[0] # 取当前文件里的样本数
            order = torch.randperm(num) # 生成一个 0 ~ num-1 的随机排列

            # 按 batch 切分并 yield
            for start in range(0, num, self.batch_size):
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

def parse_args():
    parser = argparse.ArgumentParser(description="MSGPT 本地训练脚本（加速版）")
    parser.add_argument("--config", type=str, default="model.yaml", help="配置文件路径")
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
    parser.add_argument("--device", type=str, default="cuda", help="训练设备，默认 cuda")
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
        "--train_file_limit",
        type=int,
        default=0,
        help="仅使用前N个pkl文件训练（0表示使用全部文件）",
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
        num_samples = int(len(data["label"])) # 计算当前文件样本数
        total_steps += max(math.ceil(num_samples / max(batch_size, 1)), 1) # ceil 是向上取整

    return total_steps


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("当前未检测到可用 GPU，请检查 CUDA 环境，或显式传 --device cpu")
    device = torch.device(args.device)

    # CUDA 加速项，开启混合精度，提升训练速度
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 在训练开始前，检查关键输入文件和目录是否存在，防止后续报错
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"找不到配置文件: {args.config}")
    if not os.path.isdir(args.train_data_dir):
        raise FileNotFoundError(f"找不到训练数据目录: {args.train_data_dir}")


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
    pkl_files = sorted(glob.glob(os.path.join(args.train_data_dir, "*.pkl")))
    if not pkl_files:
        raise ValueError(
            f"在 {args.train_data_dir} 下未发现 pkl 文件。"
            "当前训练入口只实现了 pkl 流式读取，请先转换数据或切换到 parquet 训练脚本。"
        )

    if args.train_file_limit < 0:
        raise ValueError("--train_file_limit 不能小于 0")
    if args.train_file_limit > 0:
        pkl_files = pkl_files[: args.train_file_limit]

    # 使用全量训练文件（不再限制只读取一个文件）。
    # 统计具体文件数量
    use_pkl_stream = len(pkl_files) > 0

    # 从 config 字典中提取 batch_size
    batch_size = int(config.get("train_batch_size", 256))

    # 创建一个训练数据集对象 train_dataset，用于后续 DataLoader 迭代训练
    train_dataset = PKLBatchIterableDataset(
        pkl_dir=args.train_data_dir,
        batch_size=batch_size,
        pkl_files=pkl_files,
    )

    print(
        f"读取已编码训练文件数(pkl): 总计={len(pkl_files)}"
        f" (train_file_limit={args.train_file_limit})"
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
    if use_pkl_stream:
        steps_per_epoch = estimate_stream_steps_per_epoch(
            pkl_files=pkl_files,
            batch_size=batch_size,
        )
        print(f"流式模式 steps_per_epoch(自动估计)={steps_per_epoch}")
    else:
        steps_per_epoch = len(train_loader)

    # 确定学习率调度器要跑多长，以及 warmup 要持续多少步
    max_iters = max(epochs * steps_per_epoch, 1)
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

    dda_criterion = nn.BCEWithLogitsLoss(reduction="none") # 给 DDA 用的损失函数
    mask_criterion = nn.CrossEntropyLoss(ignore_index=0) # 给 mask token 用的损失函数
    dda_weight = float(config.get("dda_loss_weight", 0.18)) # DDA 损失在总损失里的权重
    mask_token_id = s2i["<mask>"] # 词表里 <mask> 这个 token 的编号

    # 配置 AMP（Automatic Mixed Precision，自动混合精度训练）
    # 加速 GPU 训练并减少显存占用，同时保持数值稳定
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    print(f"AMP: {use_amp}, dtype: {args.amp_dtype}")

    # 确定模型权重保存目录，并确保这个目录存在
    model_save_path = args.model_save_path if args.model_save_path else str(
        config.get("model_save_path", "checkpoints")
    )
    os.makedirs(model_save_path, exist_ok=True)
    batch_pred_log_path = os.path.join(model_save_path, "batch_pred_top10.log")
    with open(batch_pred_log_path, "w", encoding="utf-8") as f:
        f.write("epoch,step,global_step,pred_top10,label_top10\n")

    # 初始化全局步数和累积损失
    global_step = 0
    total_loss_sum = 0.0
    total_dda_loss_sum = 0.0
    total_mask_loss_sum = 0.0
    
    # 损失累积 (Target, False Target, Decoy)
    total_target_loss_sum = 0.0
    total_ft_loss_sum = 0.0
    total_decoy_loss_sum = 0.0

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
            unmask = None
            if len(batch) == 7:
                spectra, spectra_mask, precursors, tokens, labels, weights, unmask = batch
            else:
                spectra, spectra_mask, precursors, tokens, labels, weights = batch

            # 与原始 deepspeed 逻辑对齐：
            # 1) 按 unmask 控制样本级 token mask；
            # 2) decoy 的 MLM 目标使用 <unk>，避免把 decoy 当作正常序列去重建。
            if unmask is None:
                unmask = torch.zeros_like(labels)

            spectra, spectra_mask, precursors, masked_tokens, tokens_label, labels, weights = mask_batch_data_by_unmask(
                (spectra, spectra_mask, precursors, tokens, labels, weights, unmask),
                token_mask_ratio=0.4,
                device=device,
            )
            tokens_label = tokens_label.to(torch.long)

            # 与 deepspeed 训练脚本保持一致：对谱图做随机谱峰置零增强。
            spectra, spectra_mask = mask_spectra_data(
                spectra,
                spectra_mask,
                device=device,
            )

            # 清空旧梯度
            optimizer.zero_grad(set_to_none=True)

            # 选择 autocast 的设备类型
            autocast_device = "cuda" if device.type == "cuda" else "cpu"

            # 在自动混合精度上下文中做前向和损失计算
            with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
                # 把当前 batch 输入模型，得到两个输出
                dda_pred, mask_pred = model(spectra, spectra_mask, precursors, masked_tokens)

                # 当前 batch 概率输出，用于日志观察模型区分度。
                batch_probs = torch.sigmoid(dda_pred)

                # 计算每个样本的 DDA 损失
                dda_loss_all = dda_criterion(dda_pred, labels)

                # 对 DDA 损失做样本加权平均
                dda_loss = (dda_loss_all * weights).sum() / weights.sum().clamp_min(1e-8)

                # --- 细粒度损失统计 ---
                # Target: label > 0.5
                target_mask = (labels > 0.5)
                # 注意：如果某个batch不存在某类样本，loss为0
                if target_mask.any():
                    target_loss = (dda_loss_all[target_mask] * weights[target_mask]).sum() / weights[target_mask].sum().clamp_min(1e-8)
                else:
                    target_loss = torch.tensor(0.0, device=device)
                
                # False Target: label < 0.5 & weight < 0.9 (Label=0, Weight=0.3)
                ft_mask = (labels < 0.5) & (weights < 0.9)
                if ft_mask.any():
                    ft_loss = (dda_loss_all[ft_mask] * weights[ft_mask]).sum() / weights[ft_mask].sum().clamp_min(1e-8)
                else:
                    ft_loss = torch.tensor(0.0, device=device)
                
                # Decoy: label < 0.5 & weight > 0.9 (Label=0, Weight=1.0)
                decoy_mask = (labels < 0.5) & (weights > 0.9)
                if decoy_mask.any():
                    decoy_loss = (dda_loss_all[decoy_mask] * weights[decoy_mask]).sum() / weights[decoy_mask].sum().clamp_min(1e-8)
                else:
                    decoy_loss = torch.tensor(0.0, device=device)
                # ---------------------

                # 计算 mask 预测损失
                mask_loss = mask_criterion(mask_pred, tokens_label)

                # 合成总损失
                loss = mask_loss + dda_weight * dda_loss

            # 如果启用了 GradScaler，就走混合精度反向传播
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            # 否则走普通反向传播
            else:
                loss.backward()
                # 根据当前梯度更新模型参数
                optimizer.step()

            # 更新学习率调度器
            scheduler.step()

            # 更新累积 Loss (算术平均)
            global_step += 1
            total_loss_sum += loss.item()
            total_dda_loss_sum += dda_loss.item()
            total_mask_loss_sum += mask_loss.item()
            total_target_loss_sum += target_loss.item()
            total_ft_loss_sum += ft_loss.item()
            total_decoy_loss_sum += decoy_loss.item()

            if step % 50 == 0:
                lr = scheduler.get_last_lr()[0]

                pred_top10 = batch_probs.detach().float().cpu().view(-1)[:10].tolist()
                label_top10 = labels.detach().float().cpu().view(-1)[:10].tolist()
                pred_top10_str = "|".join([f"{x:.6f}" for x in pred_top10])
                label_top10_str = "|".join([f"{x:.0f}" for x in label_top10])
                with open(batch_pred_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{epoch + 1},{step},{global_step},{pred_top10_str},{label_top10_str}\n"
                    )
                
                # 计算平均值
                avg_loss = total_loss_sum / global_step
                avg_dda = total_dda_loss_sum / global_step
                avg_mask = total_mask_loss_sum / global_step
                avg_target = total_target_loss_sum / global_step
                avg_ft = total_ft_loss_sum / global_step
                avg_decoy = total_decoy_loss_sum / global_step

                train_bar.set_postfix(
                    avg=f"{avg_loss:.4f}",
                    dda=f"{avg_dda:.4f}",
                    mask=f"{avg_mask:.4f}",
                    target=f"{avg_target:.4f}",
                    false_target=f"{avg_ft:.4f}",
                    decoy=f"{avg_decoy:.4f}",
                    lr=f"{lr:.2e}",
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
