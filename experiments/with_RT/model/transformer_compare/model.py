from __future__ import annotations

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

import yaml
import numpy as np

from .layers import MultiScalePeakEmbedding
from .decoder import PeptideDecoder, PurePeptideEncoder


def coca_inbatch_contrastive_loss(
    spectra_latents: torch.Tensor,
    pep_latents: torch.Tensor,
    temperature: torch.Tensor | float,
) -> torch.Tensor:
    """DeepSearch 风格双向 in-batch 对比损失。"""
    if spectra_latents.ndim != 2 or pep_latents.ndim != 2:
        raise ValueError(
            f"spectra_latents/pep_latents 必须是二维张量，当前为 {spectra_latents.shape} 和 {pep_latents.shape}"
        )
    if spectra_latents.shape != pep_latents.shape:
        raise ValueError(
            f"spectra_latents/pep_latents 形状必须一致，当前为 {spectra_latents.shape} 和 {pep_latents.shape}"
        )

    n_targets = spectra_latents.shape[0]
    if n_targets == 0:
        return torch.tensor(0.0, device=spectra_latents.device, dtype=spectra_latents.dtype)

    scale = torch.clamp(torch.as_tensor(temperature, device=spectra_latents.device, dtype=spectra_latents.dtype), min=1e-6)
    logits_per_spectra = scale * (spectra_latents @ pep_latents.t())
    logits_per_pep = scale * (pep_latents @ spectra_latents.t())

    labels = torch.arange(n_targets, device=spectra_latents.device, dtype=torch.long)
    loss = (
        F.cross_entropy(logits_per_spectra, labels)
        + F.cross_entropy(logits_per_pep, labels)
    ) / 2.0
    return loss

# 带有 Decoy（诱饵/负样本） 增强的 In-batch 对比损失函数
def coca_inbatch_contrastive_loss_with_decoys(
    spectra_latents: torch.Tensor,     # target 谱图特征向量，形状为 (N_target, Dim)
    pep_latents: torch.Tensor,         # target 肽段特征向量，形状为 (N_target, Dim)
    temperature: torch.Tensor | float, # 温度参数，用于缩放相似度分数
    decoy_latents: torch.Tensor | None = None, # 可选的诱饵肽段向量，形状为 (N_decoy, Dim)
) -> torch.Tensor:
    """
    DeepSearch 风格对比损失。
    与普通版本的区别在于：支持将 decoy_latents（诱饵肽段）合并到肽段负样本池中。
    注意：这里的 spectra_latents/pep_latents 应该是 target 子集，而不是原始 batch 全量。
    """
    # 1. 维度校验
    if spectra_latents.ndim != 2 or pep_latents.ndim != 2:
        raise ValueError(f"输入必须是二维张量 (Batch, Dim)")
    if spectra_latents.shape != pep_latents.shape:
        raise ValueError(f"谱图和肽段在 target 子集上的大小及维度必须一致")

    n_targets = spectra_latents.shape[0]
    if n_targets == 0:
        return torch.tensor(0.0, device=spectra_latents.device, dtype=spectra_latents.dtype)

    # 2. 计算缩放因子 (Scale)
    # 将温度转化为 scale，并设置最小值防止除零或过大波动
    scale = torch.clamp(
        torch.as_tensor(temperature, device=spectra_latents.device, dtype=spectra_latents.dtype),
        min=1e-6,
    )

    # 3. 构造任务 A：从谱图角度寻找正确的肽段 (Spectra -> Peptides)
    if decoy_latents is not None and decoy_latents.numel() > 0:
        # 校验诱饵向量维度是否与特征维度对齐
        if decoy_latents.ndim != 2 or decoy_latents.shape[1] != pep_latents.shape[1]:
            raise ValueError(f"decoy_latents 维度不匹配")
        
        # [核心增强] 将 target 肽段 + 诱饵肽段拼接成候选池。
        pep_pool = torch.cat([pep_latents, decoy_latents], dim=0)
        
        # logits 形状: (N_target, N_target + N_decoy)
        logits_per_spectra = scale * (spectra_latents @ pep_pool.t())
    else:
        # 无 decoy 时退化为标准 in-batch 对比
        logits_per_spectra = scale * (spectra_latents @ pep_latents.t())

    # 4. 构造任务 B：从 target 肽段角度寻找正确的 target 谱图
    # 此处通常不加 decoy，logits 形状为 (N_target, N_target)
    logits_per_pep = scale * (pep_latents @ spectra_latents.t())

    # 5. 生成标签并计算交叉熵
    # labels 是 [0, 1, 2, ..., N_target-1]。
    # 对第 i 个 target 谱图，正确答案是候选池前 N_target 列中的第 i 个 target 肽段。
    labels = torch.arange(n_targets, device=spectra_latents.device, dtype=torch.long)
    
    # 计算两个方向损失的平均值：
    # loss_spectra: 谱图在混合池（包含 Decoys）中识别正确肽段的难度
    # loss_pep: 肽段在 Batch 谱图中识别正确谱图的难度
    loss = (
        F.cross_entropy(logits_per_spectra, labels)  # 包含 Decoy 干扰的损失
        + F.cross_entropy(logits_per_pep, labels)    # 标准的 In-batch 损失
    ) / 2.0
    
    return loss

# 将一个 PyTorch 的 nn.Module 模型转换为 torch.fx.GraphModule，并重新编译其 forward 方法。
# 把一个普通的 PyTorch 神经网络模型，变成一个“可以看见内部结构”的模型（GraphModule）。
# 让模型“透明化”，方便后续做处理；
def transform(m : nn.Module) -> nn.Module:
    gm : torch.fx.GraphModule = torch.fx.symbolic_trace(m)

    # Recompile the forward() method of `gm` from its Graph
    gm.recompile()

    return gm


class MaskedLanguageModel(nn.Module):
    """Predict origin token from masked input sequence."""

    def __init__(self, hidden: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

class ContrastiveProjectionHead(nn.Module):
    # 两个线性层
    def __init__(self, hidden: int, projection_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, projection_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # 对输出向量进行 L2 归一化
        return F.normalize(self.proj(x), dim=-1)

# 主要任务是处理质谱 (Spectrum) 和 肽段序列 (Peptide Tokens) 之间的关系
# 主要解决两个问题：
    # 1) 对比学习：学习谱图与肽段的共享表示空间。
    # 2) PSM (Peptide-Spectrum Matching) 二分类评分。
class MSGPT(nn.Module):
    """The MSGPT model."""

    def __init__(
            # 基础参数配置
            self,
            dim_model: int = 768, # 隐藏层维度
            n_head: int = 16, # 注意力头数
            dim_feedforward: int = 2048, # 逐步前馈神经网络的神经元数
            n_layers: int = 9, # 9层深度
            dropout: float = 0,
            max_length: int = 50,  # 肽段的最大长度
            vocab_size: int = 29,  # 词表大小
            max_charge: int = 10, # 最大电荷数
                contrastive_dim: int = 128,
    ) -> None:
        super().__init__() 
        self.dim_model = dim_model
        self.max_length = max_length

        # 光谱及其每个峰的潜在表示
        # 第一个 1：表示 batch size，后续会扩展到实际 batch
        # 第二个 1：表示序列的长度，这边只有 1
        # dim_model：每个 token 的特征维度
        self.latent_spectrum = nn.Parameter(torch.randn(1, 1, dim_model))

        # 初始化一个用于“质谱峰特征编码”的模块 peak_encoder
        self.peak_encoder = MultiScalePeakEmbedding(dim_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # spectrum sequence encoder, where no spectrum mask
        # 肽段-质谱融合层
        self.spectrum_sequence_encoder = PeptideDecoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            residues_length=vocab_size,
            max_charge=max_charge,
            hidden_size=max_length # peptide max_length
        )

        # 纯肽段编码器：CLS + 前体 + 肽段，仅用于对比学习分支。
        self.peptide_encoder = PurePeptideEncoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            residues_length=vocab_size,
            max_charge=max_charge,
            hidden_size=max_length,
        )

        # 对比学习任务头：先沿序列维聚合，再映射到对比空间
        self.contrastive_head = ContrastiveProjectionHead(
            hidden=self.dim_model,
            projection_dim=contrastive_dim,
            dropout=dropout,
        )
        # 可学习的温度参数
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # DDA任务头
        self.seq_pool = nn.Linear(self.max_length + 1, 1)
        self.psm_1 = nn.Linear(self.dim_model, 64)
        self.psm_2 = nn.Linear(64, 1)

        # peptide mask 任务头：和 transformer 目录实现保持一致
        self.mask_lm = MaskedLanguageModel(self.dim_model, vocab_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    ## 加载pkl文件
    @classmethod
    # @classmethod 是 Python 的类方法装饰器。它的作用是让方法的第一个参数变成类本身（通常命名为 cls），而不是实例对象（self）
    # 这样就可以写 MSGPT.load_ckpt，而不用model = MSGPT(...) + model.load_ckpt(path)
    def load_ckpt(cls, path: str) -> nn.Module:
        """Load model from checkpoint."""
        # 从保存的 checkpoint 文件恢复并构造一个 MSGPT 模型对象，并加载权重。
        ckpt = torch.load(path, map_location="cpu")
        config = ckpt["config"]

        # 如果有loss_fn.weight，就把它删除，避免后续加载模型时出错x
        if 'loss_fn.weight' in ckpt["state_dict"]:
            ckpt["state_dict"].pop('loss_fn.weight')
            
        # 检查 PyTorch Lightning (PTL) 的 checkpoint 格式
        # 判断 state_dict 的所有 key 是否都以 "model" 开头
        if all([x.startswith("model") for x in ckpt["state_dict"].keys()]):
            # 把所有 key 的 "model." 前缀去掉，只保留模型参数，重新构造 state_dict
            ckpt["state_dict"] = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith('model.')}

        # cls(...) 等价于 MSGPT(...)
        # 根据 config 配置参数，创建一个新的 MSGPT 模型对象
        model = cls(
            dim_model=config["dim_model"],
            n_head=config["n_head"],
            dim_feedforward=config["dim_feedforward"],
            n_layers=config["n_layers"],
            dropout=config["dropout"],
            max_length=config["max_length"],
            vocab_size=len(config["vocab"]),
            max_charge=config["max_charge"],
        )
        model.load_state_dict(ckpt["state_dict"], strict = False)
        return model, config
    
    ## 加载pkl文件，使用resume方式，用于 resume（断点续训）或某些特殊保存格式
    @classmethod
    def load_ckpt_resume(cls, path: str) -> nn.Module:
        """Load model from checkpoint."""
        ckpt = torch.load(path, map_location="cpu")
        config = ckpt["config"]

        if 'loss_fn.weight' in ckpt["model"].keys():
            ckpt["model"].pop('loss_fn.weight')

        model = cls(
            dim_model=config["dim_model"],
            n_head=config["n_head"],
            dim_feedforward=config["dim_feedforward"],
            n_layers=config["n_layers"],
            dropout=config["dropout"],
            max_length=config["max_length"],
            vocab_size=len(config["vocab"]),
            max_charge=config["max_charge"],
        )
        model.load_state_dict(ckpt["model"], strict = False)
        # 返回 model, ckpt（不是 config）
        return model, ckpt
    
    ## 加载.pt文件
    # .pt 文件是 PyTorch 框架保存模型参数或整个模型的文件
    # 从 .pt 文件加载模型参数，并用指定的 config 构造一个新的 MSGPT 模型对象。
    @classmethod
    def load_pt(cls, path: str, config: str) -> nn.Module:
        """Load model from checkpoint."""
        ckpt = torch.load(path, map_location="cpu")
        model = cls(
            dim_model=config["dim_model"],
            n_head=config["n_head"],
            dim_feedforward=config["dim_feedforward"],
            n_layers=config["n_layers"],
            dropout=config["dropout"],
            max_length=config["max_length"],
            vocab_size=len(config["vocab"]),
            max_charge=config["max_charge"],
        )

        # 假如有缺失的关键词，报错
        miss_keys = set(model.state_dict().keys()) - set(ckpt['module'].keys())
        if len(miss_keys) > 0:
            print('miss keys： ', miss_keys)
        model.load_state_dict(ckpt['module'], strict=False)
        # print(f"Model loaded with {np.sum([p.numel() for p in model.parameters()]):,d} parameters")
        return model

    # 模型的前向推理过程，作用是：给定输入数据，输出模型的预测结果。
    # 输入参数：质谱数据 (batch, n_peaks, 2), spectra_mask：质谱的 padding mask, 形状为 (batch, n_peaks),precursors：前体信息，形状为 (batch, 4), tokens：肽段序列，形状为 (batch, 50)
    # 输出：预测及过，形状为(batch, 1)
    def forward(
            self,
            spectra: Tensor,
            spectra_mask: Tensor,
            precursors: Tensor,
            tokens: Tensor,
            binary_tokens: Tensor | None = None,
            return_mask_pred: bool = False,
    ) -> Tensor:
        """模型前向传播。

        参数:
            spectra: 浮点张量 (batch, n_peaks, 2) . 2: [mz_array, int_array]
            spectra_mask: 谱图填充掩码, True 表示填充部分, 布尔张量 (batch, n_peaks)
            precursors: 浮点张量 (batch, 4) . 4: [precursor_masses, precursor_charges, deltaRT, predictedRT]
            tokens: 浮点张量 (batch, 50)
        返回:
            spectra_latent: 浮点张量 (batch, contrastive_dim)
            pep_latent: 浮点张量 (batch, contrastive_dim)
            dda_pred: 浮点张量 (batch)
            temperature: 浮点张量 (1)
        """
        spectra, spectra_mask = self._encoder(spectra, spectra_mask)

        # 用深度交互后的 decoder 输出同时构造对比学习与 DDA 特征。
        binary_tokens = tokens if binary_tokens is None else binary_tokens

        decoder_output = self.spectrum_sequence_encoder(spectra, spectra_mask, precursors, binary_tokens)

        # 纯肽段分支（不接入谱图 memory），用于构造更纯净的肽段对比向量。
        pure_pep_cls, _, _ = self.peptide_encoder(precursors, tokens)

        # 谱图侧对比向量：使用编码后谱图序列的第 0 位全局 token。
        spectra_latent = self.contrastive_head(spectra[:, 0, :])

        # 肽段侧对比向量：使用纯肽段编码器的 CLS 全局 token。
        pep_latent = self.contrastive_head(pure_pep_cls)

        dda_pred, mask_pred = self._multi_task_encoder_from_decoder(decoder_output)
        if return_mask_pred:
            return spectra_latent, pep_latent, dda_pred, mask_pred, self.temperature.exp()
        return spectra_latent, pep_latent, dda_pred, self.temperature.exp()
    
    # 用于模型微调（finetune）时的前向推理流程，编码后的特征、前体和肽段送入微调专用的解码器finetune_psm_encoder
    def finetune_forward(
            self,
            spectra: Tensor,
            spectra_mask: Tensor,
            precursors: Tensor,
            tokens: Tensor,
    ) -> Tensor:
        """模型前向传播。

        参数:
            spectra: 浮点张量 (batch, n_peaks, 2) . 2: [mz_array, int_array]
            spectra_mask: 谱图填充掩码, True 表示填充部分, 布尔张量 (batch, n_peaks)
            precursors: 浮点张量 (batch, 4) . 4: [precursor_masses, precursor_charges, deltaRT, predictedRT]
            tokens: 浮点张量 (batch, 50)
        返回:
            # PSM
            pred: 浮点张量 (batch, 1)
        """
        spectra, spectra_mask = self._encoder(spectra, spectra_mask)
        return self.finetune_encoder(spectra, spectra_mask, precursors, tokens)

    # 该方法专门用于两个任务头预测的给出
    def pred(
            self,
            spectra: Tensor,
            spectra_mask: Tensor,
            precursors: Tensor,
            tokens: Tensor,
            binary_tokens: Tensor | None = None,
            return_mask_pred: bool = False,
    ) -> Tensor:
        """模型前向传播。

        参数:
            spectra: 浮点张量 (batch, n_peaks, 2) . 2: [mz_array, int_array]
            spectra_mask: 谱图填充掩码, True 表示填充部分, 布尔张量 (batch, n_peaks)
            precursors: 浮点张量 (batch, 4) . 4: [precursor_masses, precursor_charges, deltaRT, predictedRT]
            tokens: 浮点张量 (batch, 50)
        返回:
            spectra_latent: 浮点张量 (batch, contrastive_dim)
            pep_latent: 浮点张量 (batch, contrastive_dim)
            dda_pred: 浮点张量 (batch)
            temperature: 浮点张量 (1)
        """
        with torch.no_grad(): # 表示下面的代码不会计算梯度，适合推理和评估阶段。
            if return_mask_pred:
                spectra_latent, pep_latent, dda_pred, mask_pred, temperature = self.forward(
                    spectra,
                    spectra_mask,
                    precursors,
                    tokens,
                    binary_tokens=binary_tokens,
                    return_mask_pred=True,
                )
            else:
                spectra_latent, pep_latent, dda_pred, temperature = self.forward(
                    spectra,
                    spectra_mask,
                    precursors,
                    tokens,
                    binary_tokens=binary_tokens,
                    return_mask_pred=False,
                )
            dda_pred = torch.sigmoid(dda_pred)
        if return_mask_pred:
            return spectra_latent, pep_latent, dda_pred, mask_pred, temperature
        return spectra_latent, pep_latent, dda_pred, temperature

    # 调用上文给定的self.encoder方法，对输入的质谱数据进行特征编码和自注意力处理，得到适合后续模型使用的高维特征
    def _encoder(self, spectra: Tensor, spectra_mask: Tensor) -> tuple[Tensor, Tensor]:
        # 对每个峰的 mz 和强度做特征编码
        spectra = self.peak_encoder(spectra[:, :, [0]], spectra[:, :, [1]])

        # Self-attention on latent spectra AND peaks
        # 构造一个“全局潜在光谱”特征，扩展到 batch 大小
        latent_spectra = self.latent_spectrum.expand(spectra.shape[0], -1, -1)
        # 全局特征拼接到谱图向量前面
        spectra = torch.cat([latent_spectra, spectra], dim=1)
        # 构造一个 mask，表示 latent_spectra 不是 padding
        # 创建一个形状为 (batch_size, 1) 的全零张量，并把它和spectra_mask合并
        latent_mask = torch.zeros((spectra_mask.shape[0], 1), dtype=bool, device=spectra_mask.device)
        spectra_mask = torch.cat([latent_mask, spectra_mask], dim=1).bool()

        # 调用self.encoder，返回经过特征编码和自注意力处理后的“质谱序列特征”张量
        spectra = self.encoder(spectra, src_key_padding_mask=spectra_mask)
        return spectra, spectra_mask

        # 给出 DDA 任务输出
    def _multi_task_encoder(
            self,
            spectra: Tensor,
            spectra_mask: Tensor,
            precursors: Tensor,
            tokens: Tensor,
            return_mask_pred: bool = False,
    ) -> Tensor:
        # pred： (batch , 51, 768)
        # 将之前编码好的质谱特征 (spectra)、前体信息 (precursors) 和肽段序列 (tokens) 融合。
        decoder_output = self.spectrum_sequence_encoder(spectra, spectra_mask, precursors, tokens)

        dda_pred, mask_pred = self._multi_task_encoder_from_decoder(decoder_output)
        if return_mask_pred:
            return dda_pred, mask_pred
        return dda_pred

    def _multi_task_encoder_from_decoder(self, decoder_output: Tensor) -> tuple[Tensor, Tensor]:
        """从融合后的 decoder 输出计算 DDA 与 MLM 预测。"""

        # 沿序列长度做可学习聚合得到句向量
        pooled = self.dropout(self.relu(self.seq_pool(decoder_output.transpose(1, 2)).squeeze(-1)))

        # MLM 预测 (batch, 51, vocab) -> (batch, vocab, 50)
        mask_pred = self.mask_lm(decoder_output).transpose(1, 2)[:, :, 1:]
        
        # DDA 预测
        dda_hidden = self.dropout(self.relu(self.psm_1(pooled)))
        dda_pred = self.psm_2(dda_hidden).squeeze(-1)

        return dda_pred, mask_pred
    
    # 微调时返回 pooled feature 与对比 embedding
    def finetune_encoder(
            self,
            spectra: Tensor,
            spectra_mask: Tensor,
            precursors: Tensor,
            tokens: Tensor,
    ) -> Tensor:
        # pred： (batch , 51, 768)
        decoder_output = self.spectrum_sequence_encoder(spectra, spectra_mask, precursors, tokens)

        pooled = self.dropout(self.relu(self.seq_pool(decoder_output.transpose(1, 2)).squeeze(-1)))
        contrastive_embedding = self.contrastive_head(pooled)

        return pooled, contrastive_embedding

    

if __name__ == '__main__':
    spectra = torch.randn(128, 300, 3)
    spectra_mask = torch.zeros(128, 300)
    precursors = torch.ones(128, 4) + 3
    tokens = torch.ones(128, 50)
    
    label = torch.ones(128)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    spectra = spectra.to(device).to(torch.bfloat16)
    spectra_mask = spectra_mask.to(device).to(torch.bfloat16)
    precursors = precursors.to(device).to(torch.bfloat16)
    tokens = tokens.to(device).to(torch.bfloat16)
    
    label = label.to(device).to(torch.bfloat16)
    # 初始化测试
    config_path = '/ajun/dda_bert/yaml/model.yaml'
    with open(config_path) as f_in:
        config = yaml.safe_load(f_in)

    vocab = ['<pad>', '<mask>'] + list(config["residues"].keys()) + ['<unk>']
    config["vocab"] = vocab
    
    model = MSGPT(
        dim_model=config["dim_model"],
        n_head=config["n_head"],
        dim_feedforward=config["dim_feedforward"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        max_length=config["max_length"],
        vocab_size=len(vocab),
        max_charge=config["max_charge"],
    )
    model.to(device)
    print('模型规模： {}'.format(np.sum([p.numel() for p in model.parameters()])))

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            spectra_latent, pep_latent, dda_pred, temperature = model.forward(
                spectra,
                spectra_mask,
                precursors,
                tokens,
            )

        print('spectra_latent', spectra_latent.shape)
        print('pep_latent', pep_latent.shape)
        print('dda_pred', dda_pred.shape)
        print('temperature', temperature)
