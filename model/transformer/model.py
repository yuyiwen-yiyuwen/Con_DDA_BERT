from __future__ import annotations

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

import yaml
import numpy as np

from .layers import MultiScalePeakEmbedding
from .decoder import PeptideDecoder

# 将一个 PyTorch 的 nn.Module 模型转换为 torch.fx.GraphModule，并重新编译其 forward 方法。
# 把一个普通的 PyTorch 神经网络模型，变成一个“可以看见内部结构”的模型（GraphModule）。
# 让模型“透明化”，方便后续做处理；
def transform(m : nn.Module) -> nn.Module:
    gm : torch.fx.GraphModule = torch.fx.symbolic_trace(m)

    # Recompile the forward() method of `gm` from its Graph
    gm.recompile()

    return gm

# 作用是：根据输入的隐藏特征，预测被掩盖（mask）的原始 token。
class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        # 参数 hidden：输入特征的维度（比如 Bert 输出的向量长度）。
        # 参数 vocab_size：词表大小（即要分类的类别数）。
        # 
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        # 输入 x（形状是 [batch, seq_len, hidden]）
        # 把 x 送进 self.linear，输出每个位置上属于每个 token 的分数（未经过 softmax）
        return self.linear(x)

# 主要任务是处理质谱 (Spectrum) 和 肽段序列 (Peptide Tokens) 之间的关系
# 主要解决两个问题：
    # PSM (Peptide-Spectrum Matching) 评分：判断给定的质谱和肽段是否匹配（是一个分类/打分任务）。
    # 掩码预测 (MLM)：像 BERT 一样，预测肽段序列中被掩盖的部分。
# 肽段是由氨基酸组成的序列。MLM 任务就像是在做“成语填空”，它能让模型学习到：
    # 哪些氨基酸经常出现在一起？
    # 某种电荷下的肽段在断裂时，通常会产生什么样的碎片离子？
    # 这种对序列本身规律的掌握，能显著提高模型在处理陌生序列时的打分准确性。
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

        # DDA任务的网络结构
        self.psm_0 = nn.Linear(self.max_length + 1, 1)
        self.psm_1 = nn.Linear(self.dim_model, 64)
        self.psm_2 = nn.Linear(64, 1)
        
        # peptide mask任务的网络结构
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
    ) -> Tensor:
        """Model forward pass.

        Args:
            spectra: float Tensor (batch, n_peaks, 2) . 2: [mz_array, int_array]
            spectra_mask: Spectra padding mask, True for padded indices, bool Tensor (batch, n_peaks)
            precursors: float Tensor (batch, 4) . 4: [precursor_masses, precursor_charges, deltaRT, predictedRT]
            tokens: float Tensor (batch, 50)
        Returns:
            # PSM
            pred: float Tensor (batch, 1)
        """
        spectra, spectra_mask = self._encoder(spectra, spectra_mask)
        return self._psm_encoder(spectra, spectra_mask, precursors, tokens)
    
    # 用于模型微调（finetune）时的前向推理流程，编码后的特征、前体和肽段送入微调专用的解码器finetune_psm_encoder
    def finetune_forward(
            self,
            spectra: Tensor,
            spectra_mask: Tensor,
            precursors: Tensor,
            tokens: Tensor,
    ) -> Tensor:
        """Model forward pass.

        Args:
            spectra: float Tensor (batch, n_peaks, 2) . 2: [mz_array, int_array]
            spectra_mask: Spectra padding mask, True for padded indices, bool Tensor (batch, n_peaks)
            precursors: float Tensor (batch, 4) . 4: [precursor_masses, precursor_charges, deltaRT, predictedRT]
            tokens: float Tensor (batch, 50)
        Returns:
            # PSM
            pred: float Tensor (batch, 1)
        """
        spectra, spectra_mask = self._encoder(spectra, spectra_mask)
        return self.finetune_psm_encoder(spectra, spectra_mask, precursors, tokens)

    # 该方法专门用于两个任务头预测的给出
    def pred(
            self,
            spectra: Tensor,
            spectra_mask: Tensor,
            precursors: Tensor,
            tokens: Tensor,
    ) -> Tensor:
        """Model forward pass.

        Args:
            spectra: float Tensor (batch, n_peaks, 2) . 2: [mz_array, int_array]
            spectra_mask: Spectra padding mask, True for padded indices, bool Tensor (batch, n_peaks)
            precursors: float Tensor (batch, 4) . 4: [precursor_masses, precursor_charges, deltaRT, predictedRT]
            tokens: float Tensor (batch, 50)
        Returns:
            # PSM
            dda_pred: float Tensor (batch)
            # MaskedLanguageModel 
            mask_pred: float Tensor (batch)
        """
        with torch.no_grad(): # 表示下面的代码不会计算梯度，适合推理和评估阶段。
            # 调用模型的 forward 方法，输入质谱、mask、前体、肽段，得到两个输出：
            # dda_pred：PSM 打分（肽段-质谱匹配分数）
            # mask_pred：掩码预测（MLM任务输出）
            dda_pred, mask_pred = self.forward(spectra, spectra_mask, precursors, tokens)
            sigmod = nn.Sigmoid()
            dda_pred = sigmod(dda_pred) # 把输出变成 0~1 之间的概率值
        return dda_pred, mask_pred

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

    # 给出PSM scoreing和 MLM 两个任务的
    def _psm_encoder(
            self,
            spectra: Tensor,
            spectra_mask: Tensor,
            precursors: Tensor,
            tokens: Tensor,
    ) -> Tensor:
        # pred： (batch , 51, 768)
        # 将之前编码好的质谱特征 (spectra)、前体信息 (precursors) 和肽段序列 (tokens) 融合。
        decoder_output = self.spectrum_sequence_encoder(spectra, spectra_mask, precursors, tokens)

        # pred： (batch, 51, 768) ==>  (batch, 768, 51) ==> (batch, 768, 1) ==> (batch, 768)
        # transpose(1, 2)：将形状从 (batch, 51, 768) 变为 (batch, 768, 51)，为了让全连接层 psm_0 处理长度维度。
        pred = self.dropout(self.relu(self.psm_0(decoder_output.transpose(1, 2)).squeeze()))
        
        # mask pred：(batch, 51, 768) ==> (batch, 51, 29) ==> (batch, 29, 51) ==> (batch, 29, 50)
        # self.mask_lm 是一个线性层（MaskedLanguageModel），把最后一维 768 映射到 vocab_size（比如 29）
        mask_pred = self.mask_lm(decoder_output).transpose(1, 2)[:, :, 1:]
        
        # pred：(batch, 768) ==> (batch, 64)
        dda_pred = self.dropout(self.relu(self.psm_1(pred)))

        # preds：(batch, 64) ==> (batch, 1)  ==> batch
        # squeeze() 是 PyTorch 张量（Tensor）的方法，用来去掉维度为1的轴
        dda_pred = self.psm_2(dda_pred).squeeze()

        return dda_pred, mask_pred
    
    # 用于微调的方法，只返回dda_pred
    def finetune_psm_encoder(
            self,
            spectra: Tensor,
            spectra_mask: Tensor,
            precursors: Tensor,
            tokens: Tensor,
    ) -> Tensor:
        # pred： (batch , 51, 768)
        decoder_output = self.spectrum_sequence_encoder(spectra, spectra_mask, precursors, tokens)

        # pred： (batch, 51, 768) ==>  (batch, 768, 51) ==> (batch, 768, 1) ==> (batch, 768)
        pred = self.dropout(self.relu(self.psm_0(decoder_output.transpose(1, 2)).squeeze()))

        # pred：(batch, 768) ==> (batch, 64)
        mid_dda_pred = self.dropout(self.relu(self.psm_1(pred)))

        # preds：(batch, 64) ==> (batch, 1)
        dda_pred = self.psm_2(mid_dda_pred).squeeze()

        return mid_dda_pred, dda_pred

    

if __name__ == '__main__':
    spectra = torch.randn(128, 300, 3)
    spectra_mask = torch.zeros(128, 300)
    precursors = torch.ones(128, 4) + 3
    tokens = torch.ones(128, 50)
    
    label = torch.ones(128)
    tokens_label = torch.ones(128, 50).to(torch.long)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    spectra = spectra.to(device).to(torch.bfloat16)
    spectra_mask = spectra_mask.to(device).to(torch.bfloat16)
    precursors = precursors.to(device).to(torch.bfloat16)
    tokens = tokens.to(device).to(torch.bfloat16)
    
    label = label.to(device).to(torch.bfloat16)
    tokens_label = tokens_label.to(device)

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
            dda_pred, mask_pred = model.forward(spectra, spectra_mask, precursors, tokens)

    print('dda_pred', dda_pred.shape) # 128
    print('mask_pred', mask_pred.shape) # (128, 29)
    
     # Define dda Loss function
    dda_criterion = nn.BCEWithLogitsLoss()
    dda_loss = dda_criterion(dda_pred, label.flatten())

    # Using Negative Log Likelihood Loss function for predicting the masked_token
    mask_criterion = nn.CrossEntropyLoss(ignore_index=0)
    mask_loss = mask_criterion(mask_pred, tokens_label)
    
    print('dda_loss', dda_loss)
    print('mask_loss', mask_loss)
