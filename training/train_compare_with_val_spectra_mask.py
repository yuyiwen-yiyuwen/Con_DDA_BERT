"""
带验证集 + 谱图随机峰置零增强的对比学习训练脚本。

基于 train_compare_with_val.py，并在数据加载后调用 mask_spectra_data()
随机将 spectra_zero_ratio 比例的谱峰置零，提升模型对缺失峰的鲁棒性。

训练目标：contrastive (CoCa) + DDA (PSM 二分类) + MLM (token 重建) 联合训练。
支持在线质量锚定重排 (Online Mass-Anchored Rerank) 来提升 batch 内肽段多样性。

在线质量锚定重排默认开启，如需关闭传 --no_online_mass_anchor。

输入示例 (单来源):
    python train_compare_with_val_spectra_mask.py \
        --train_data_dirs /path/to/pkl_dataset/train \
        --val_data_dirs /path/to/pkl_dataset/val \
        --online_input_dirs /path/to/pkl_dataset/train \
        --model_save_path /path/to/checkpoints \
        --spectra_zero_ratio 0.1 --spectra_remain_ratio 0.1 \
        --amp --compile --device 0

输入示例 (多来源):
    python train_compare_with_val_spectra_mask.py \
        --train_data_dirs /path/to/mzml/train /path/to/tims/train /path/to/wiff/train \
        --val_data_dirs /path/to/mzml/val /path/to/tims/val \
        --online_input_dirs /path/to/mzml/train /path/to/tims/train /path/to/wiff/train \
        --model_save_path /path/to/checkpoints \
        --spectra_zero_ratio 0.1 --spectra_remain_ratio 0.1 \
        --amp --compile --device 0

输出:
    {model_save_path}/msgpt_epoch_N.pt           - 模型权重
    {model_save_path}/batch_pred_top10.log        - 每 50 step 的简要日志
    {model_save_path}/debug_metrics.csv           - 详细指标日志
"""

import argparse
import glob
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
from model.transformer_compare.model import (
    MSGPT,
    coca_inbatch_contrastive_loss_with_decoys,
)
from model.transformer_compare.iterable_dataset_online_parquet import mask_spectra_data
from model.transformer_compare.training_utils import (
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
      - 损失权重: dda_loss_weight, contrastive_weight, grad_clip_norm
      - 谱图增强: spectra_zero_ratio, spectra_remain_ratio
      - 日志: debug_log_interval
    """
    p = argparse.ArgumentParser(description="MSGPT 训练（val + 谱图峰置零增强）")

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
    p.add_argument("--contrastive_weight", type=float, default=1.0,
                   help="Contrastive 损失权重")
    p.add_argument("--grad_clip_norm", type=float, default=1.0,
                   help="梯度裁剪的 max_norm，<=0=关闭")

    # ---- 谱图峰置零增强 (本脚本的核心新增功能) ----
    p.add_argument("--spectra_zero_ratio", type=float, default=0.1,
                   help="每个样本内随机置零的谱峰比例 (0.1 = 10%% 峰被随机置零)")
    p.add_argument("--spectra_remain_ratio", type=float, default=0.1,
                   help="完全不做谱图 mask 的样本比例 (0.1 = 10%% 样本保留全部谱峰)")

    # ---- 日志 ----
    p.add_argument("--debug_log_interval", type=int, default=10,
                   help="每 N step 写一次详细 debug 指标，0=关闭")
    p.add_argument("--model_save_path", type=str,
                   default="/home/yiwen/AIPC/scripts/organized_attantion/data/checkpoints_with_val_spectra_mask",
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
      - 非负参数: debug_log_interval, train_batch_count, dda_loss_weight,
        contrastive_weight, grad_clip_norm
      - 在线重排参数: start_window_da > 0, expand_factor > 1, max_window_da > 0
    """
    if args.debug_log_interval < 0:
        raise ValueError("--debug_log_interval 不能小于 0")
    if args.train_batch_count < 0:
        raise ValueError("--train_batch_count 不能小于 0")
    if args.dda_loss_weight < 0:
        raise ValueError("--dda_loss_weight 不能小于 0")
    if args.contrastive_weight < 0:
        raise ValueError("--contrastive_weight 不能小于 0")
    if args.grad_clip_norm < 0:
        raise ValueError("--grad_clip_norm 不能小于 0")
    # 在线重排窗口参数必须在合理范围
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
    # 纯数字 → cuda:N
    if device_arg.strip().isdigit():
        s = f"cuda:{device_arg.strip()}"
    # gpu:N → cuda:N (兼容习惯写法)
    elif s.startswith("gpu"):
        suffix = s[3:].lstrip(":")
        if suffix.isdigit():
            s = f"cuda:{suffix}"
        else:
            raise ValueError("--device 若使用 gpu 前缀，格式应为 gpu:N")
    # 要求 cuda 时检查 CUDA 是否可用
    if s.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("未检测到可用 GPU，请检查 CUDA 环境或传 --device cpu")
    device = torch.device(s)
    # CUDA 加速: 允许 tf32 矩阵乘法，在 Ampere 及以上 GPU 上有显著速度提升
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


# ===========================================================================
# 训练过程中的辅助函数
# ===========================================================================

def _compute_grad_norm(model) -> float:
    """计算模型所有参数梯度的 L2 范数 (标量)。

    遍历 model.parameters()，对每个有梯度的参数求其 grad 的 L2 平方和，
    最后取 sqrt。用于日志记录和梯度裁剪后的诊断。
    相当于 torch.nn.utils.clip_grad_norm_ 的内部计算但没有副作用。
    """
    sq = sum(float(p.grad.detach().float().pow(2).sum().item())
             for p in model.parameters() if p.grad is not None)
    return math.sqrt(max(sq, 0.0))


def _write_quick_log(path: str, spectra_latent, pep_latent, labels, temperature,
                     epoch: int, step: int, global_step: int):
    """每 50 step 写一次简要日志到 CSV 文件。

    记录的列:
      epoch, step, global_step - 训练进度定位
      spectra_emb_norm           - 谱图 embedding 的平均 L2 范数 (监控数值稳定性)
      pep_emb_norm               - 肽段 embedding 的平均 L2 范数
      target_count               - 当前 batch 中 target (label>0.5) 的样本数
      decoy_count                - 当前 batch 中 decoy (label<=0.5) 的样本数
      temperature                - 当前可学习温度参数值

    注意: label > 0.5 为 target (正样本), label <= 0.5 为 decoy (负样本)。
    """
    # embedding 范数: 监控是否出现数值爆炸/消失
    e1 = float(spectra_latent.detach().float().norm(dim=1).mean().item())
    e2 = float(pep_latent.detach().float().norm(dim=1).mean().item())
    # target/decoy 分布: 确保 batch 中正负比例正常
    n_target = int((labels > 0.5).sum().item())
    n_decoy = int((labels <= 0.5).sum().item())
    t_val = float(temperature.detach().float().item())
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{epoch},{step},{global_step},{e1:.6f},{e2:.6f},{n_target},{n_decoy},{t_val:.6f}\n")


def _write_debug_log(path: str, spectra_latent, pep_latent, dda_pred, labels,
                     tokens_label, model, temperature, mlm_ignore_index,
                     loss_val: float, contrastive_val: float, dda_val: float, mlm_val: float,
                     grad_norm: float,
                     epoch: int, step: int, global_step: int, lr: float):
    """每 debug_log_interval step 写一次详细指标到 CSV 文件。

    包含以下指标组:

    【检索指标 (全量)】
      contrastive_acc_s2p   - 谱→肽 top-1 准确率 (按行 argmax)
      contrastive_acc_p2s   - 肽→谱 top-1 准确率 (按列 argmax)
      pos_sim_mean          - 对角线 (正样本对) 余弦相似度均值
      neg_sim_mean          - 非对角线 (负样本对) 余弦相似度均值
      sim_gap               - pos - neg 相似度差距 (越大区分度越好)

    【检索指标 (仅 target 子集)】
      同上但只在 target (label>0.5) 样本上计算，与 contrastive_loss 优化目标对齐。
      后缀 _target。

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
      spectra_emb_norm      - 谱图 embedding 平均 L2 范数
      pep_emb_norm          - 肽段 embedding 平均 L2 范数
      lr, temperature       - 当前学习率 / 温度参数值
    """
    lat_s = spectra_latent.detach().float()  # (B, D)
    lat_p = pep_latent.detach().float()      # (B, D)

    # ---- 全量 cosine 相似度矩阵 & 检索准确率 ----
    # sim[i,j] = cosine(spectra_i, peptide_j)  （假设 embedding 已 L2 归一化）
    sim = lat_s @ lat_p.t()
    bn = int(sim.shape[0])
    if bn > 0:
        # 对角线 = 正样本对
        pos_mean = float(sim.diag().mean().item())
        if bn > 1:
            # 非对角线 = 负样本对 (排除对角线)
            neg_mask = ~torch.eye(bn, dtype=torch.bool, device=sim.device)
            neg_mean = float(sim[neg_mask].mean().item())
        else:
            neg_mean = float("nan")
        # s→p: 每行 (每个谱图) 的最大相似度所对应的肽段索引
        acc_s2p = float((sim.argmax(dim=1) == torch.arange(bn, device=sim.device)).float().mean().item())
        # p→s: 每列 (每个肽段) 的最大相似度所对应的谱图索引
        acc_p2s = float((sim.argmax(dim=0) == torch.arange(bn, device=sim.device)).float().mean().item())
    else:
        pos_mean = neg_mean = acc_s2p = acc_p2s = float("nan")
    gap = pos_mean - neg_mean if not math.isnan(neg_mean) else float("nan")

    # ---- target 子集指标 (仅 label>0.5 的样本) ----
    # 与 contrastive_loss 的优化目标一致，反映对真正 target 肽段的检索能力
    labels_f = labels.detach().float()
    t_idx = torch.where(labels_f > 0.5)[0]  # target 样本在 batch 中的位置
    tn = int(t_idx.numel())
    if tn > 0:
        st = sim.index_select(0, t_idx).index_select(1, t_idx)  # target×target 子矩阵
        pos_t = float(st.diag().mean().item())
        if tn > 1:
            nm_t = ~torch.eye(tn, dtype=torch.bool, device=st.device)
            neg_t = float(st[nm_t].mean().item())
            acc_s2p_t = float((st.argmax(dim=1) == torch.arange(tn, device=st.device)).float().mean().item())
            acc_p2s_t = float((st.argmax(dim=0) == torch.arange(tn, device=st.device)).float().mean().item())
        else:
            neg_t = acc_s2p_t = acc_p2s_t = float("nan")
    else:
        pos_t = neg_t = acc_s2p_t = acc_p2s_t = float("nan")
    gap_t = pos_t - neg_t if not math.isnan(neg_t) else float("nan")

    # ---- DDA 预测概率统计 ----
    # 期望: target 概率→1, decoy 概率→0
    probs = torch.sigmoid(dda_pred.detach().float())
    t_mask = labels_f > 0.5
    d_mask = labels_f <= 0.5
    n_t = int(t_mask.sum().item())
    n_d = int(d_mask.sum().item())
    prob_all = float(probs.mean().item())
    prob_t = float(probs[t_mask].mean().item()) if n_t > 0 else float("nan")
    prob_d = float(probs[d_mask].mean().item()) if n_d > 0 else float("nan")

    # ---- MLM 统计 ----
    # tokens_label 中 != ignore_index 的位置才是被 mask 的 token
    n_masked = int(tokens_label.ne(mlm_ignore_index).sum().item())
    r_masked = float(n_masked / max(tokens_label.numel(), 1))

    # ---- 系统诊断 ----
    e1 = float(lat_s.norm(dim=1).mean().item())  # spectra embedding 范数
    e2 = float(lat_p.norm(dim=1).mean().item())  # peptide embedding 范数
    # 模型所有参数的 L2 范数，用于监控权重变化幅度
    pm_sq = sum(float(p.detach().float().pow(2).sum().item()) for p in model.parameters())
    pm_norm = math.sqrt(max(pm_sq, 0.0))
    t_val = float(temperature.detach().float().item())

    with open(path, "a", encoding="utf-8") as f:
        f.write(
            f"{epoch},{step},{global_step},{lr:.8e},{t_val:.6f},"
            f"{loss_val:.6f},{contrastive_val:.6f},{dda_val:.6f},{mlm_val:.6f},"
            f"{acc_s2p:.6f},{acc_p2s:.6f},{pos_mean:.6f},{neg_mean:.6f},{gap:.6f},"
            f"{acc_s2p_t:.6f},{acc_p2s_t:.6f},{pos_t:.6f},{neg_t:.6f},{gap_t:.6f},"
            f"{prob_all:.6f},{prob_t:.6f},{prob_d:.6f},"
            f"{n_t},{n_d},{n_masked},{r_masked:.6f},{grad_norm:.6f},{pm_norm:.6f},{e1:.6f},{e2:.6f}\n"
        )


# ===========================================================================
# 主训练入口
# ===========================================================================

def main():
    """MSGPT 训练主函数。

    流程概览:
      1. 解析参数 → 校验 → 解析 device
      2. 加载 config yaml → 构建词表 → 设置种子
      3. 数据准备: 收集 train+val 的 pkl 列表
      4. 创建 Dataset/DataLoader (流式 或 在线重排)
      5. 构建 MSGPT 模型 → torch.compile (可选)
      6. 创建 AdamW 优化器 + WarmupScheduler
      7. 配置损失函数 (DDA + Contrastive + MLM)
      8. 训练循环:
         a. 解包 batch → 移至 GPU
         b. mask_spectra_data() 随机置零谱峰
         c. build_masked_tokens_and_labels() 构造 MLM 输入
         d. 前向传播 → 三项损失计算 → 总损失加权求和
         e. 反向传播 (AMP 分支) → 梯度裁剪 → optimizer.step()
         f. temperature clamp → scheduler.step()
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
    # config yaml 包含: residues, model dims, learning_rate, warmup_ratio 等
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"找不到配置文件: {args.config}")
    # 在线重排模式下，数据目录以 online_input_dirs 为准
    selected_train_dirs = list(args.train_data_dirs)
    if args.online_mass_anchor_train:
        selected_train_dirs = list(args.online_input_dirs)
    for d in selected_train_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"找不到训练数据目录: {d}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # build_vocab_and_s2i 从 config["residues"] 构建 vocab 列表和 char→id 映射
    vocab, s2i = build_vocab_and_s2i(config)
    config["vocab"] = vocab  # 回存 vocab 到 config，后续保存 checkpoint 时一并序列化
    seed = int(config.get("seed", 123))
    set_seeds(seed, deterministic=args.deterministic)

    # ================================================================
    # 2. 数据准备
    # ================================================================
    # 收集 val pkl 文件（排除 val.00004_val.pkl）
    val_pkl_files = discover_pkl_files(list(args.val_data_dirs))
    val_pkl_files = [f for f in val_pkl_files if os.path.basename(f) != "val.00004_val.pkl"]
    print(f"val 数据: 从 {list(args.val_data_dirs)} 收集到 {len(val_pkl_files)} 个文件（已排除 val.00004_val.pkl）")
    train_pkl_files = discover_pkl_files(selected_train_dirs)
    # 合并 train + val，按文件名排序保证确定性
    all_pkl_files = sorted(train_pkl_files + val_pkl_files)
    if not all_pkl_files:
        raise ValueError(f"在 {selected_train_dirs} 和 {list(args.val_data_dirs)} 下均未发现 pkl 文件。")
    print(f"使用设备: {device}")
    print(f"训练 pkl 文件: train={len(train_pkl_files)}, val={len(val_pkl_files)}, 合计={len(all_pkl_files)}")
    for d in selected_train_dirs:
        n = len(discover_pkl_files(d))
        print(f"  {d}: {n} 个文件")

    batch_size = int(config.get("train_batch_size", 256))
    use_online = args.online_mass_anchor_train  # 在线质量锚定重排模式
    use_pkl_stream = len(all_pkl_files) > 0     # 流式 pkl 读取模式

    # ================================================================
    # 3. 创建 Dataset & DataLoader
    # ================================================================
    # ---- 在线重排模式 ----
    # 将所有 pkl 文件全量读入内存，按 precursor mass 排序并构造多样化的 batch。
    # 强制 num_workers=0 (多进程会各自复制一份全量数据导致 OOM)。
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
            extra_pkl_files=val_pkl_files,  # val 数据也参与重排
        )
        train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0,
                                  pin_memory=torch.cuda.is_available())

    # ---- 流式读取模式 ----
    # 每个 pkl 文件逐文件读入、一次性 tensor 化、按 batch 切分 yield。
    # 支持 multi-worker 并行预取。
    else:
        train_dataset = PKLBatchIterableDataset(
            pkl_dir=selected_train_dirs[0], batch_size=batch_size, pkl_files=all_pkl_files,
            shuffle_within_file=bool(args.shuffle_within_file),
            shuffle_batches=(not args.disable_shuffle_batches),
        )
        # batch_size=None: PKLBatchIterableDataset 已产出固定大小的 batch
        # pin_memory=True: 加速 CPU→GPU 传输
        # persistent_workers: 避免每个 epoch 重启 worker 进程
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
    # MSGPT 核心架构:
    #   peak_encoder (MultiScalePeakEmbedding) → encoder (TransformerEncoder)
    #   → spectrum_sequence_encoder (PeptideDecoder, cross-attention)
    #   → peptide_encoder (PurePeptideEncoder)
    #   → contrastive_head (投影到对比空间) + mask_lm (token 重建头) + DDA 头 (PSM 二分类)
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

    # 断点续训：加载 checkpoint，恢复 model、epoch、global_step
    resume_epoch = 0
    resume_global_step = 0
    if args.resume_from:
        print(f"从 checkpoint 恢复训练: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device)
        state_dict = ckpt["state_dict"]
        # torch.compile 保存的权重带 _orig_mod. 前缀，加载到未编译模型需去除
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

    # torch.compile 会追踪计算图并编译为优化后的 kernel (首次编译慢，后续快)
    if args.compile and hasattr(torch, "compile"):
        print("启用 torch.compile")
        model = torch.compile(model)

    learning_rate = float(config.get("learning_rate", 5e-5))
    weight_decay = float(config.get("weight_decay", 5e-5))
    # build_optimizer 创建 AdamW，CUDA 上会启用 fused 版本以减少显存和加速
    optimizer = build_optimizer(model, learning_rate, weight_decay, device)

    # ================================================================
    # 5. 学习率调度器
    # ================================================================
    # epochs: 命令行 > 0 时覆盖 yaml 配置
    epochs = int(args.epochs) if int(args.epochs) > 0 else int(config.get("epochs", 25))
    # 在线重排模式下 len(train_dataset) 返回 batch 数量，流式模式需要遍历文件估算
    if use_online:
        steps_per_epoch = len(train_dataset)
    elif use_pkl_stream:
        steps_per_epoch = estimate_stream_steps_per_epoch(all_pkl_files, batch_size)
    else:
        steps_per_epoch = len(train_loader)
    print(f"steps_per_epoch={steps_per_epoch}")

    # 续训时 scheduler 覆盖完整训练区间（已训 + 剩余），通过 last_batch_iteration 定位
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
    # DDA: BCEWithLogitsLoss(reduction="none") 方便做样本级加权
    dda_criterion = nn.BCEWithLogitsLoss(reduction="none")
    # MLM: 忽略 pad (index=0) 位置的损失
    mlm_ignore_index = 0
    mlm_mask_ratio = 0.4          # 每个序列中 40% 的有效 token 被 mask
    mask_loss_weight = 1.0        # MLM 损失权重 (通常固定为 1.0)
    mask_criterion = nn.CrossEntropyLoss(ignore_index=mlm_ignore_index)
    mask_token_id = int(s2i["<mask>"])   # <mask> token 的 id
    unk_token_id = int(s2i["<unk>"])     # <unk> token 的 id (decoy 的 MLM 目标)
    temperature_max = float(config.get("contrastive_temperature_max", 100.0))
    dda_loss_w = float(args.dda_loss_weight)
    contrastive_w = float(args.contrastive_weight)
    grad_clip = float(args.grad_clip_norm)
    spectra_zero_ratio = float(args.spectra_zero_ratio)
    spectra_remain_ratio = float(args.spectra_remain_ratio)

    # AMP: bf16 不需要 GradScaler (bf16 动态范围大不会溢出)，fp16 需要
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
    # 写入 CSV 表头（续训时追加模式，不覆盖已有日志）
    log_mode = "a" if args.resume_from else "w"
    with open(quick_log, log_mode) as f:
        if log_mode == "w":
            f.write("epoch,step,global_step,spectra_emb_norm,pep_emb_norm,target_count,decoy_count,temperature\n")
    with open(debug_log, log_mode) as f:
        if log_mode == "w":
            f.write(
                "epoch,step,global_step,lr,temperature,loss,contrastive_loss,dda_loss,mask_loss,"
                "contrastive_acc_s2p,contrastive_acc_p2s,pos_sim_mean,neg_sim_mean,sim_gap,"
                "contrastive_acc_s2p_target,contrastive_acc_p2s_target,pos_sim_mean_target,neg_sim_mean_target,sim_gap_target,"
                "dda_prob_mean,dda_target_prob_mean,dda_decoy_prob_mean,"
                "target_count,decoy_count,masked_token_count,masked_token_ratio,grad_norm,param_norm,spectra_emb_norm,pep_emb_norm\n"
            )

    print(f"AMP: {use_amp}, dtype: {args.amp_dtype}")
    print(f"dda_loss_weight={dda_loss_w}, contrastive_weight={contrastive_w}, mask_loss_weight={mask_loss_weight}")
    print(f"mlm_mask_ratio={mlm_mask_ratio}, grad_clip_norm={grad_clip}")
    print(f"spectra_zero_ratio={spectra_zero_ratio} "
          f"(随机置零 {spectra_zero_ratio*100:.0f}% 的峰), "
          f"spectra_remain_ratio={spectra_remain_ratio} "
          f"(保留 {spectra_remain_ratio*100:.0f}% 样本不 mask)")

    # ================================================================
    # 8. 训练循环
    # ================================================================
    global_step = resume_global_step
    # running loss 用 EMA (α=0.99) 平滑，减少逐 batch 的抖动
    running_loss = running_dda = running_contrastive = running_mlm = None

    for epoch in range(resume_epoch, epochs):
        # 流式数据集需要每 epoch 传入 epoch 号以保证可复现的 shuffle
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        model.train()

        # 流式模式下 DataLoader 长度未知 (因为 __len__ 未定义)，total=None 让 tqdm 自适应
        bar_total = None if use_pkl_stream else len(train_loader)
        train_bar = tqdm(train_loader, total=bar_total,
                         desc=f"Epoch {epoch + 1}/{epochs}",
                         mininterval=1.0, dynamic_ncols=True)

        for step, batch in enumerate(train_bar, start=1):
            # ---- 8a. 解包 batch & 异步传输到 GPU ----
            # batch 可能是 6 或 7 元组 (取决于是否有 unmask 字段)
            if len(batch) == 7:
                spectra, spectra_mask, precursors, tokens, labels, weights, unmask = batch
            else:
                spectra, spectra_mask, precursors, tokens, labels, weights = batch
                unmask = None

            # non_blocking=True: 异步传输，CPU 不等待 GPU 完成拷贝即可继续执行
            spectra = spectra.to(device, non_blocking=True)
            spectra_mask = spectra_mask.to(device, non_blocking=True)
            precursors = precursors.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            if unmask is not None:
                unmask = unmask.to(device, non_blocking=True)

            # ---- 8b. 谱图峰置零增强 [本脚本核心差异] ----
            # mask_spectra_data 来自 iterable_dataset_online_parquet:
            #   - spectra_remain_ratio 比例的样本完全不做处理
            #   - 其余样本中，随机将 spectra_zero_ratio 比例的峰 (m/z + intensity) 置零
            #   - 同时更新 spectra_mask，被置零的峰在 mask 中也标为 0
            # 目的: 模拟真实实验中缺失的谱峰，提升模型对碎片离子覆盖不全的鲁棒性
            spectra, spectra_mask = mask_spectra_data(
                spectra, spectra_mask,
                remain_ratio=spectra_remain_ratio,
                spectra_zero_ratio=spectra_zero_ratio,
                device=device,
            )

            # ---- 8c. MLM token 掩码 ----
            # 在 token 序列中随机 mask 一部分位置 (mlm_mask_ratio=0.4):
            #   - 该位置 token 替换为 <mask>
            #   - tokens_label 中仅对被 mask 的位置记录原始 token id
            #   - decoy 样本的被 mask 位置标签改写为 <unk> (避免 decoy 序列被当作正常序列重建)
            #   - unmask=1 的样本跳过 mask (用于保护特定样本不被 MLM 干扰)
            masked_tokens, tokens_label = build_masked_tokens_and_labels(
                tokens=tokens, labels=labels, mask_token_id=mask_token_id,
                unk_token_id=unk_token_id, token_mask_ratio=mlm_mask_ratio,
                ignore_index=mlm_ignore_index, unmask=unmask,
            )

            # set_to_none=True: 将梯度显存释放而不是写零，减少显存碎片
            optimizer.zero_grad(set_to_none=True)

            # ---- 8d. 前向传播 ----
            # autocast 自动将前向传播中的矩阵乘法和卷积转为低精度 (bf16/fp16)
            device_type = "cuda" if device.type == "cuda" else "cpu"
            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                # forward 返回 5 个值:
                #   lat_s: 谱图 latent (B, D) — 用于 contrastive loss
                #   lat_p: 肽段 latent (B, D) — 用于 contrastive loss
                #   dda_pred: PSM 二分类 logits (B, 1)
                #   mask_pred: token 分类 logits (B, L, vocab_size)
                #   temperature: 可学习的温度参数 (标量)
                lat_s, lat_p, dda_pred, mask_pred, temperature = model(
                    spectra, spectra_mask, precursors, tokens,
                    binary_tokens=masked_tokens, return_mask_pred=True,
                )

                # --- Contrastive Loss (CoCa/DeepSearch 风格) ---
                # 在 target (label>0.5) 上做双向 in-batch cross-entropy:
                #   谱→肽: 对于每个 target 谱图，在所有 target 肽段中选最匹配的
                #   肽→谱: 对于每个 target 肽段，在所有 target 谱图中选最匹配的
                # decoy 肽段向量并入谱图侧的负样本池，扩大负样本数量
                t_mask = labels > 0.5  # target 样本
                if t_mask.any():
                    d_mask = labels <= 0.5  # decoy 样本
                    contrastive_loss = coca_inbatch_contrastive_loss_with_decoys(
                        lat_s[t_mask], lat_p[t_mask],
                        temperature=temperature, decoy_latents=lat_p[d_mask],
                    )
                else:
                    # 极端情况: batch 中没有 target 样本，contrastive loss 为 0
                    contrastive_loss = torch.zeros((), device=device, dtype=dda_pred.dtype)

                # --- DDA Loss (PSM 二分类) ---
                # BCEWithLogitsLoss(reduction="none") → 样本级加权平均
                # weights: 来自数据预处理，控制不同样本在 DDA loss 中的贡献
                dda_loss_all = dda_criterion(dda_pred, labels)
                dda_loss = (dda_loss_all * weights).sum() / weights.sum().clamp_min(1e-8)

                # --- MLM Loss (token 重建) ---
                # 仅在被 mask 的位置 (tokens_label != ignore_index) 计算交叉熵
                if tokens_label.ne(mlm_ignore_index).any():
                    mask_loss = mask_criterion(mask_pred, tokens_label)
                else:
                    mask_loss = torch.zeros((), device=device, dtype=dda_pred.dtype)

                # 总损失 = 三项损失的加权和
                loss = (contrastive_w * contrastive_loss +
                        dda_loss_w * dda_loss +
                        mask_loss_weight * mask_loss)

            # ---- 8e. 反向传播 ----
            if scaler.is_enabled():
                # fp16 + GradScaler: scale loss → backward → unscale → clip → step → update scale
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # 先 unscale 再做梯度裁剪，否则 grad norm 偏小
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, norm_type=2.0)
                grad_norm = _compute_grad_norm(model)
                scaler.step(optimizer)
                scaler.update()
            else:
                # bf16 或 fp32: 直接 backward，不需要 scale/unscale
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, norm_type=2.0)
                grad_norm = _compute_grad_norm(model)
                optimizer.step()

            # temperature 约束: clamp 到 [0, log(temperature_max)] 防止过大导致 contrastive loss 退化
            with torch.no_grad():
                model.temperature.clamp_(0, math.log(max(temperature_max, 1.0)))
            scheduler.step()

            # ---- 8f. EMA 更新 ----
            # α=0.99 的指数移动平均 → 当前值权重 1%，历史值权重 99%
            global_step += 1
            if running_loss is None:
                running_loss = loss.item()
                running_dda = dda_loss.item()
                running_contrastive = contrastive_loss.item()
                running_mlm = mask_loss.item()
            else:
                a = 0.99
                running_loss = a * running_loss + (1 - a) * loss.item()
                running_dda = a * running_dda + (1 - a) * dda_loss.item()
                running_contrastive = a * running_contrastive + (1 - a) * contrastive_loss.item()
                running_mlm = a * running_mlm + (1 - a) * mask_loss.item()

            # ---- 8g. 定期写入日志 ----
            # 每 50 step: 简要日志 + tqdm 进度条更新
            if step % 50 == 0:
                lr = scheduler.get_last_lr()[0]
                _write_quick_log(quick_log, lat_s, lat_p, labels, temperature,
                                 epoch + 1, step, global_step)
                train_bar.set_postfix(
                    avg=f"{running_loss:.4f}", contrastive=f"{running_contrastive:.4f}",
                    dda=f"{running_dda:.4f}", mlm=f"{running_mlm:.4f}", lr=f"{lr:.2e}",
                )

            # 每 debug_log_interval step: 详细 debug 指标
            if args.debug_log_interval > 0 and step % args.debug_log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                _write_debug_log(
                    debug_log, lat_s, lat_p, dda_pred, labels,
                    tokens_label, model, temperature, mlm_ignore_index,
                    loss.item(), contrastive_loss.item(), dda_loss.item(), mask_loss.item(),
                    grad_norm,
                    epoch + 1, step, global_step, lr,
                )

        # ---- epoch 结束: 保存 checkpoint ----
        # 每个 epoch 存一个独立文件，便于回溯和选择最优 epoch
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
