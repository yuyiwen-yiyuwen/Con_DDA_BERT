"""
带验证集 + 谱图随机峰置零增强的训练脚本（transformer 主体架构版）。

基于 train_compare_with_val_spectra_mask.py，模型替换为 model.transformer 的 MSGPT 主体架构。
transformer 版 MSGPT 不含 contrastive head / peptide_encoder / temperature，
因此训练目标为：DDA (PSM 二分类) + MLM (token 重建) 联合训练。

支持在线质量锚定重排 (Online Mass-Anchored Rerank) 来提升 batch 内肽段多样性。
在线质量锚定重排默认开启，如需关闭传 --no_online_mass_anchor。

输入示例 (单来源):
    python train_compare_with_val_spectra_mask_without_compare.py \
        --train_data_dirs /path/to/pkl_dataset/train \
        --val_data_dirs /path/to/pkl_dataset/val \
        --online_input_dirs /path/to/pkl_dataset/train \
        --model_save_path /path/to/checkpoints \
        --spectra_zero_ratio 0.1 --spectra_remain_ratio 0.1 \
        --amp --compile --device 0

输出:
    {model_save_path}/msgpt_epoch_N.pt           - 模型权重
    {model_save_path}/batch_pred_top10.log        - 每 50 step 的简要日志
    {model_save_path}/debug_metrics.csv           - 详细指标日志
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, "/home/yiwen/AIPC/scripts/organized_attantion")
from model.transformer.model import MSGPT
from model.transformer.iterable_dataset_online_parquet import mask_spectra_data
from model.transformer.training_utils import (
    set_seeds,
    WarmupScheduler,
    build_vocab_and_s2i,
    build_optimizer,
    build_masked_tokens_and_labels,
    discover_pkl_files,
    PKLBatchIterableDataset,
    OnlineMassAnchoredBatchIterableDataset,
    estimate_stream_steps_per_epoch,
)


# ===========================================================================
# 命令行参数
# ===========================================================================

def parse_args():
    """解析所有命令行参数并返回 argparse.Namespace。

    参数分为以下几组：
      - 数据路径: config, train_data_dir, val_data_dir
      - 训练超参: epochs, batch_size (在 yaml 中), num_workers, device
      - AMP/编译: amp, amp_dtype, compile, deterministic
      - 数据加载: shuffle_within_file, disable_shuffle_batches
      - 在线重排 (默认开启，--no_online_mass_anchor 关闭): online_input_dir,
        online_start_window_da, online_expand_factor, online_max_window_da,
        online_drop_last, train_batch_count
      - 损失权重: dda_loss_weight, grad_clip_norm
      - 谱图增强: spectra_zero_ratio, spectra_remain_ratio
      - 日志: debug_log_interval
    """
    p = argparse.ArgumentParser(description="MSGPT 训练（transformer 主体架构 + val + 谱图峰置零增强）")

    # ---- 数据路径 ----
    p.add_argument("--config", type=str,
                   default="/home/yiwen/AIPC/scripts/organized_attantion/config/model.yaml",
                   help="YAML 配置文件路径（含 residues/vocab/模型维度等）")
    p.add_argument("--train_data_dirs", type=str, nargs="+",
                   default=["/home/yiwen/AIPC/scripts/attantion/pkl_dataset/train"],
                   help="训练 pkl 数据目录（支持多个）")
    p.add_argument("--val_data_dirs", type=str, nargs="+",
                   default=["/home/yiwen/AIPC/scripts/attantion/pkl_dataset/val"],
                   help="验证集 pkl 目录（支持多个，排除 val.00004_val.pkl 后并入训练）")

    # ---- 训练控制 ----
    p.add_argument("--epochs", type=int, default=0,
                   help="覆盖 yaml 中的 epochs，0=使用 yaml 配置")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader 的 worker 进程数")
    p.add_argument("--device", type=str, default="cuda",
                   help="训练设备: cpu / cuda / cuda:0 / gpu:0 / 0")

    # ---- AMP & 编译 ----
    p.add_argument("--amp", action="store_true",
                   help="启用自动混合精度 (AMP)")
    p.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"],
                   help="AMP 精度类型: bf16 (推荐) 或 fp16")
    p.add_argument("--compile", action="store_true",
                   help="启用 torch.compile (首次运行有编译开销)")
    p.add_argument("--deterministic", action="store_true",
                   help="启用 cudnn deterministic 模式 (更慢但更可复现)")

    # ---- 数据加载 ----
    p.add_argument("--shuffle_within_file", action="store_true",
                   help="打乱单个 pkl 文件内的样本顺序")
    p.add_argument("--disable_shuffle_batches", action="store_true",
                   help="关闭 batch 级顺序打乱 (默认开启)")

    # ---- 在线质量锚定重排 ----
    p.add_argument("--no_online_mass_anchor", action="store_false",
                   dest="online_mass_anchor_train", default=True,
                   help="关闭在线质量锚定重排，改用流式读取模式")
    p.add_argument("--online_input_dirs", type=str, nargs="+",
                   default=["/home/yiwen/AIPC/scripts/attantion/pkl_dataset/train"],
                   help="在线重排模式的输入目录（支持多个）")
    p.add_argument("--online_start_window_da", type=float, default=1.0,
                   help="在线重排初始质量窗口 (Da) — 在该窗口内收集 batch_size 个样本")
    p.add_argument("--online_expand_factor", type=float, default=2.0,
                   help="窗口扩展倍数 — 每次窗口不够时乘以该因子扩展")
    p.add_argument("--online_max_window_da", type=float, default=64.0,
                   help="在线重排窗口最大上限 (Da)")
    p.add_argument("--online_drop_last", action="store_true",
                   help="丢弃尾部不足 batch_size 的 batch")
    p.add_argument("--train_batch_count", type=int, default=0,
                   help="每 epoch 使用的 batch 数上限 (0=全部，>0=限制数量以加速 epoch 迭代)")

    # ---- 损失权重 ----
    p.add_argument("--dda_loss_weight", type=float, default=0.18,
                   help="DDA (PSM 二分类) 损失在总损失中的权重")
    p.add_argument("--grad_clip_norm", type=float, default=1.0,
                   help="梯度裁剪的 max_norm，<=0=关闭")

    # ---- 谱图峰置零增强 ----
    p.add_argument("--spectra_zero_ratio", type=float, default=0.1,
                   help="每个样本内随机置零的谱峰比例 (0.1 = 10%% 峰被随机置零)")
    p.add_argument("--spectra_remain_ratio", type=float, default=0.1,
                   help="完全不做谱图 mask 的样本比例 (0.1 = 10%% 样本保留全部谱峰)")

    # ---- 日志 ----
    p.add_argument("--debug_log_interval", type=int, default=10,
                   help="每 N step 写一次详细 debug 指标，0=关闭")
    p.add_argument("--model_save_path", type=str,
                   default="/home/yiwen/AIPC/scripts/organized_attantion/data/checkpoints_with_val_spectra_mask_without_compare",
                   help="模型权重和日志的保存目录")
    p.add_argument("--resume_from", type=str, default="",
                   help="从指定 checkpoint 断点续训")

    return p.parse_args()


# ===========================================================================
# 参数校验 & 设备设置
# ===========================================================================

def _validate_args(args):
    """对命令行参数做取值范围校验，非法值直接抛出 ValueError。

    校验规则：
      - 非负参数: debug_log_interval, train_batch_count, dda_loss_weight, grad_clip_norm
      - 在线重排参数: start_window_da > 0, expand_factor > 1, max_window_da > 0
    """
    if args.debug_log_interval < 0:
        raise ValueError("--debug_log_interval 不能小于 0")
    if args.train_batch_count < 0:
        raise ValueError("--train_batch_count 不能小于 0")
    if args.dda_loss_weight < 0:
        raise ValueError("--dda_loss_weight 不能小于 0")
    if args.grad_clip_norm < 0:
        raise ValueError("--grad_clip_norm 不能小于 0")
    if args.online_start_window_da <= 0:
        raise ValueError("--online_start_window_da must be > 0")
    if args.online_expand_factor <= 1.0:
        raise ValueError("--online_expand_factor must be > 1")
    if args.online_max_window_da <= 0:
        raise ValueError("--online_max_window_da must be > 0")


def _resolve_device(device_arg: str) -> torch.device:
    """将用户输入的设备参数规范化为 torch.device 并配置 CUDA 加速选项。

    支持的输入格式:
      - 纯数字: "0" / "1"                    → cuda:0 / cuda:1
      - gpu 前缀: "gpu:0" / "gpu:1"         → cuda:0 / cuda:1
      - cuda 前缀: "cuda" / "cuda:0"        → 原样保留
      - cpu: "cpu"                           → cpu

    CUDA 设备会额外开启 tf32 加速 (matmul + cudnn allow_tf32=True)。
    """
    s = str(device_arg).strip().lower()
    if device_arg.strip().isdigit():
        s = f"cuda:{device_arg.strip()}"
    elif s.startswith("gpu"):
        suffix = s[3:].lstrip(":")
        if suffix.isdigit():
            s = f"cuda:{suffix}"
        else:
            raise ValueError("--device 若使用 gpu 前缀，格式应为 gpu:N")
    if s.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("未检测到可用 GPU，请检查 CUDA 环境或传 --device cpu")
    device = torch.device(s)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


# ===========================================================================
# 训练过程中的辅助函数
# ===========================================================================

def _compute_grad_norm(model) -> float:
    """计算模型所有参数梯度的 L2 范数 (标量)。"""
    sq = sum(float(p.grad.detach().float().pow(2).sum().item())
             for p in model.parameters() if p.grad is not None)
    return math.sqrt(max(sq, 0.0))


def _write_quick_log(path: str, labels, dda_pred,
                     epoch: int, step: int, global_step: int,
                     dda_loss_val: float, mask_loss_val: float, lr: float):
    """每 50 step 写一次简要日志到 CSV 文件。

    记录的列:
      epoch, step, global_step - 训练进度定位
      target_count               - 当前 batch 中 target (label>0.5) 的样本数
      decoy_count                - 当前 batch 中 decoy (label<=0.5) 的样本数
      dda_prob_mean              - DDA 预测概率均值
      dda_loss, mask_loss        - 当前 step 损失值
      lr                         - 当前学习率
    """
    n_target = int((labels > 0.5).sum().item())
    n_decoy = int((labels <= 0.5).sum().item())
    prob_mean = float(torch.sigmoid(dda_pred.detach().float()).mean().item())
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{epoch},{step},{global_step},{n_target},{n_decoy},"
                f"{prob_mean:.6f},{dda_loss_val:.6f},{mask_loss_val:.6f},{lr:.2e}\n")


def _write_debug_log(path: str, dda_pred, labels,
                     tokens_label, model, mlm_ignore_index,
                     loss_val: float, dda_val: float, mlm_val: float,
                     grad_norm: float,
                     epoch: int, step: int, global_step: int, lr: float):
    """每 debug_log_interval step 写一次详细指标到 CSV 文件。

    包含以下指标组:

    【DDA 指标】
      dda_prob_mean         - sigmoid(dda_pred) 全局均值
      dda_target_prob_mean  - target 样本的预测概率均值
      dda_decoy_prob_mean   - decoy 样本的预测概率均值

    【MLM 指标】
      masked_token_count    - 当前 batch 中被 mask 的 token 数量
      masked_token_ratio    - mask token 数 / 总 token 数

    【系统诊断】
      grad_norm             - 梯度全局 L2 范数
      param_norm            - 模型参数 L2 范数
      lr                    - 当前学习率
    """
    labels_f = labels.detach().float()

    # ---- DDA 预测概率统计 ----
    probs = torch.sigmoid(dda_pred.detach().float())
    t_mask = labels_f > 0.5
    d_mask = labels_f <= 0.5
    n_t = int(t_mask.sum().item())
    n_d = int(d_mask.sum().item())
    prob_all = float(probs.mean().item())
    prob_t = float(probs[t_mask].mean().item()) if n_t > 0 else float("nan")
    prob_d = float(probs[d_mask].mean().item()) if n_d > 0 else float("nan")

    # ---- MLM 统计 ----
    n_masked = int(tokens_label.ne(mlm_ignore_index).sum().item())
    r_masked = float(n_masked / max(tokens_label.numel(), 1))

    # ---- 系统诊断 ----
    pm_sq = sum(float(p.detach().float().pow(2).sum().item()) for p in model.parameters())
    pm_norm = math.sqrt(max(pm_sq, 0.0))

    with open(path, "a", encoding="utf-8") as f:
        f.write(
            f"{epoch},{step},{global_step},{lr:.8e},"
            f"{loss_val:.6f},{dda_val:.6f},{mlm_val:.6f},"
            f"{prob_all:.6f},{prob_t:.6f},{prob_d:.6f},"
            f"{n_t},{n_d},{n_masked},{r_masked:.6f},{grad_norm:.6f},{pm_norm:.6f}\n"
        )


# ===========================================================================
# 主训练入口
# ===========================================================================

def main():
    """MSGPT 训练主函数（transformer 主体架构）。

    流程概览:
      1. 解析参数 → 校验 → 解析 device
      2. 加载 config yaml → 构建词表 → 设置种子
      3. 数据准备: 收集 train+val 的 pkl 列表
      4. 创建 Dataset/DataLoader (流式 或 在线重排)
      5. 构建 MSGPT 模型 → torch.compile (可选)
      6. 创建 AdamW 优化器 + WarmupScheduler
      7. 配置损失函数 (DDA + MLM)
      8. 训练循环:
         a. 解包 batch → 移至 GPU
         b. mask_spectra_data() 随机置零谱峰
         c. build_masked_tokens_and_labels() 构造 MLM 输入
         d. 前向传播 → DDA + MLM 损失计算 → 总损失加权求和
         e. 反向传播 (AMP 分支) → 梯度裁剪 → optimizer.step()
         f. scheduler.step()
         g. EMA 更新 running loss
         h. 定期写日志
      9. 每个 epoch 结束保存 checkpoint
    """
    args = parse_args()
    _validate_args(args)
    device = _resolve_device(args.device)

    # ================================================================
    # 1. 加载配置文件
    # ================================================================
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"找不到配置文件: {args.config}")
    selected_train_dirs = list(args.train_data_dirs)
    if args.online_mass_anchor_train:
        selected_train_dirs = list(args.online_input_dirs)
    for d in selected_train_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"找不到训练数据目录: {d}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    vocab, s2i = build_vocab_and_s2i(config)
    config["vocab"] = vocab
    seed = int(config.get("seed", 123))
    set_seeds(seed, deterministic=args.deterministic)

    # ================================================================
    # 2. 数据准备
    # ================================================================
    val_pkl_files = discover_pkl_files(list(args.val_data_dirs))
    val_pkl_files = [f for f in val_pkl_files if os.path.basename(f) != "val.00004_val.pkl"]
    print(f"val 数据: 从 {list(args.val_data_dirs)} 收集到 {len(val_pkl_files)} 个文件（已排除 val.00004_val.pkl）")
    train_pkl_files = discover_pkl_files(selected_train_dirs)
    all_pkl_files = sorted(train_pkl_files + val_pkl_files)
    if not all_pkl_files:
        raise ValueError(f"在 {selected_train_dirs} 和 {list(args.val_data_dirs)} 下均未发现 pkl 文件。")
    print(f"使用设备: {device}")
    print(f"训练 pkl 文件: train={len(train_pkl_files)}, val={len(val_pkl_files)}, 合计={len(all_pkl_files)}")
    for d in selected_train_dirs:
        n = len(discover_pkl_files(d))
        print(f"  {d}: {n} 个文件")

    batch_size = int(config.get("train_batch_size", 256))
    use_online = args.online_mass_anchor_train
    use_pkl_stream = len(all_pkl_files) > 0

    # ================================================================
    # 3. 创建 Dataset & DataLoader
    # ================================================================
    if use_online:
        print("启用在线质量锚定重排训练")
        if args.num_workers > 0:
            print("在线重排模式下强制 num_workers=0")
        train_dataset = OnlineMassAnchoredBatchIterableDataset(
            input_dir=selected_train_dirs, batch_size=batch_size, seed=seed,
            start_window_da=float(args.online_start_window_da),
            expand_factor=float(args.online_expand_factor),
            max_window_da=float(args.online_max_window_da),
            drop_last=bool(args.online_drop_last),
            shuffle_batches=(not args.disable_shuffle_batches),
            train_batch_count=int(args.train_batch_count),
            use_gzip=True,
            extra_pkl_files=val_pkl_files,
        )
        train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0,
                                  pin_memory=torch.cuda.is_available())
    else:
        train_dataset = PKLBatchIterableDataset(
            pkl_dir=selected_train_dirs[0], batch_size=batch_size, pkl_files=all_pkl_files,
            shuffle_within_file=bool(args.shuffle_within_file),
            shuffle_batches=(not args.disable_shuffle_batches),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=None, num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=(4 if args.num_workers > 0 else None),
        )
        print("使用流式批量读取模式")

    # ================================================================
    # 4. 构建模型 & 优化器
    # ================================================================
    # transformer 版 MSGPT 主体架构:
    #   peak_encoder (MultiScalePeakEmbedding) → encoder (TransformerEncoder)
    #   → spectrum_sequence_encoder (PeptideDecoder, cross-attention)
    #   → psm_0/psm_1/psm_2 (DDA 头) + mask_lm (token 重建头)
    # 不含 contrastive head / peptide_encoder / temperature 参数
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

    resume_epoch = 0
    resume_global_step = 0
    if args.resume_from:
        print(f"从 checkpoint 恢复训练: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device)
        state_dict = ckpt["state_dict"]
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
        resume_epoch = int(ckpt.get("epoch", 0))
        resume_global_step = int(ckpt.get("global_step", 0))
        print(f"已恢复: epoch={resume_epoch}, global_step={resume_global_step}")
    else:
        print("从随机初始化开始训练，不加载启动权重")

    if args.compile and hasattr(torch, "compile"):
        print("启用 torch.compile")
        model = torch.compile(model)

    learning_rate = float(config.get("learning_rate", 5e-5))
    weight_decay = float(config.get("weight_decay", 5e-5))
    optimizer = build_optimizer(model, learning_rate, weight_decay, device)

    # ================================================================
    # 5. 学习率调度器
    # ================================================================
    epochs = int(args.epochs) if int(args.epochs) > 0 else int(config.get("epochs", 25))
    if use_online:
        steps_per_epoch = len(train_dataset)
    elif use_pkl_stream:
        steps_per_epoch = estimate_stream_steps_per_epoch(all_pkl_files, batch_size)
    else:
        steps_per_epoch = len(train_loader)
    print(f"steps_per_epoch={steps_per_epoch}")

    total_epochs = resume_epoch + epochs
    scheduled_total_steps = total_epochs * steps_per_epoch
    max_iters = max(int(scheduled_total_steps), 1)
    warmup_iters = max(int(float(config.get("warmup_ratio", 0.2)) * max_iters), 1)
    resume_offset = resume_epoch * steps_per_epoch
    scheduler = WarmupScheduler(
        optimizer=optimizer, warmup_iter=warmup_iters, max_iter=max_iters,
        max_lr=learning_rate, min_lr=float(config.get("min_lr", 5e-6)),
        warmup_type=str(config.get("warmup_strategy", "exp")),
        last_batch_iteration=resume_offset,
    )

    # ================================================================
    # 6. 损失函数 & 超参配置
    # ================================================================
    dda_criterion = nn.BCEWithLogitsLoss(reduction="none")
    mlm_ignore_index = 0
    mlm_mask_ratio = 0.4
    mask_loss_weight = 1.0
    mask_criterion = nn.CrossEntropyLoss(ignore_index=mlm_ignore_index)
    mask_token_id = int(s2i["<mask>"])
    unk_token_id = int(s2i["<unk>"])
    dda_loss_w = float(args.dda_loss_weight)
    grad_clip = float(args.grad_clip_norm)
    spectra_zero_ratio = float(args.spectra_zero_ratio)
    spectra_remain_ratio = float(args.spectra_remain_ratio)

    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    # ================================================================
    # 7. 输出目录 & 日志初始化
    # ================================================================
    save_dir = args.model_save_path or str(config.get("model_save_path", "checkpoints"))
    os.makedirs(save_dir, exist_ok=True)
    quick_log = os.path.join(save_dir, "batch_pred_top10.log")
    debug_log = os.path.join(save_dir, "debug_metrics.csv")
    log_mode = "a" if args.resume_from else "w"
    with open(quick_log, log_mode) as f:
        if log_mode == "w":
            f.write("epoch,step,global_step,target_count,decoy_count,"
                    "dda_prob_mean,dda_loss,mask_loss,lr\n")
    with open(debug_log, log_mode) as f:
        if log_mode == "w":
            f.write(
                "epoch,step,global_step,lr,loss,dda_loss,mask_loss,"
                "dda_prob_mean,dda_target_prob_mean,dda_decoy_prob_mean,"
                "target_count,decoy_count,masked_token_count,masked_token_ratio,grad_norm,param_norm\n"
            )

    print(f"AMP: {use_amp}, dtype: {args.amp_dtype}")
    print(f"dda_loss_weight={dda_loss_w}, mask_loss_weight={mask_loss_weight}")
    print(f"mlm_mask_ratio={mlm_mask_ratio}, grad_clip_norm={grad_clip}")
    print(f"spectra_zero_ratio={spectra_zero_ratio} "
          f"(随机置零 {spectra_zero_ratio*100:.0f}% 的峰), "
          f"spectra_remain_ratio={spectra_remain_ratio} "
          f"(保留 {spectra_remain_ratio*100:.0f}% 样本不 mask)")

    # ================================================================
    # 8. 训练循环
    # ================================================================
    global_step = resume_global_step
    running_loss = running_dda = running_mlm = None

    for epoch in range(resume_epoch, epochs):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        model.train()

        bar_total = None if use_pkl_stream else len(train_loader)
        train_bar = tqdm(train_loader, total=bar_total,
                         desc=f"Epoch {epoch + 1}/{epochs}",
                         mininterval=1.0, dynamic_ncols=True)

        for step, batch in enumerate(train_bar, start=1):
            # ---- 8a. 解包 batch & 异步传输到 GPU ----
            if len(batch) == 7:
                spectra, spectra_mask, precursors, tokens, labels, weights, unmask = batch
            else:
                spectra, spectra_mask, precursors, tokens, labels, weights = batch
                unmask = None

            spectra = spectra.to(device, non_blocking=True)
            spectra_mask = spectra_mask.to(device, non_blocking=True)
            precursors = precursors.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            if unmask is not None:
                unmask = unmask.to(device, non_blocking=True)

            # ---- 8b. 谱图峰置零增强 ----
            spectra, spectra_mask = mask_spectra_data(
                spectra, spectra_mask,
                remain_ratio=spectra_remain_ratio,
                spectra_zero_ratio=spectra_zero_ratio,
                device=device,
            )

            # ---- 8c. MLM token 掩码 ----
            masked_tokens, tokens_label = build_masked_tokens_and_labels(
                tokens=tokens, labels=labels, mask_token_id=mask_token_id,
                unk_token_id=unk_token_id, token_mask_ratio=mlm_mask_ratio,
                ignore_index=mlm_ignore_index, unmask=unmask,
            )

            optimizer.zero_grad(set_to_none=True)

            # ---- 8d. 前向传播 ----
            # transformer 版 MSGPT forward 返回 (dda_pred, mask_pred) 两个值
            #   dda_pred: PSM 二分类 logits (B,)
            #   mask_pred: token 分类 logits (B, vocab_size, seq_len)
            device_type = "cuda" if device.type == "cuda" else "cpu"
            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                dda_pred, mask_pred = model(spectra, spectra_mask, precursors, masked_tokens)

                # --- DDA Loss (PSM 二分类) ---
                dda_loss_all = dda_criterion(dda_pred, labels)
                dda_loss = (dda_loss_all * weights).sum() / weights.sum().clamp_min(1e-8)

                # --- MLM Loss (token 重建) ---
                if tokens_label.ne(mlm_ignore_index).any():
                    mask_loss = mask_criterion(mask_pred, tokens_label)
                else:
                    mask_loss = torch.zeros((), device=device, dtype=dda_pred.dtype)

                # 总损失 = 两项损失的加权和
                loss = dda_loss_w * dda_loss + mask_loss_weight * mask_loss

            # ---- 8e. 反向传播 ----
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, norm_type=2.0)
                grad_norm = _compute_grad_norm(model)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, norm_type=2.0)
                grad_norm = _compute_grad_norm(model)
                optimizer.step()

            scheduler.step()

            # ---- 8f. EMA 更新 ----
            global_step += 1
            if running_loss is None:
                running_loss = loss.item()
                running_dda = dda_loss.item()
                running_mlm = mask_loss.item()
            else:
                a = 0.99
                running_loss = a * running_loss + (1 - a) * loss.item()
                running_dda = a * running_dda + (1 - a) * dda_loss.item()
                running_mlm = a * running_mlm + (1 - a) * mask_loss.item()

            # ---- 8g. 定期写入日志 ----
            if step % 50 == 0:
                lr = scheduler.get_last_lr()[0]
                _write_quick_log(quick_log, labels, dda_pred,
                                 epoch + 1, step, global_step,
                                 dda_loss.item(), mask_loss.item(), lr)
                train_bar.set_postfix(
                    avg=f"{running_loss:.4f}", dda=f"{running_dda:.4f}",
                    mlm=f"{running_mlm:.4f}", lr=f"{lr:.2e}",
                )

            if args.debug_log_interval > 0 and step % args.debug_log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                _write_debug_log(
                    debug_log, dda_pred, labels,
                    tokens_label, model, mlm_ignore_index,
                    loss.item(), dda_loss.item(), mask_loss.item(),
                    grad_norm,
                    epoch + 1, step, global_step, lr,
                )

        # ---- epoch 结束: 保存 checkpoint ----
        ckpt_epoch = epoch + 1
        ckpt = os.path.join(save_dir, f"msgpt_epoch_{ckpt_epoch}.pt")
        torch.save({
            "state_dict": model.state_dict(),
            "config": config,
            "epoch": ckpt_epoch,
            "global_step": global_step,
        }, ckpt)
        print(f"已保存权重: {ckpt}")


if __name__ == "__main__":
    main()
