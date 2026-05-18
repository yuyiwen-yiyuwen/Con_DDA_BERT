"""Base Transformer models for working with mass spectra and peptides"""
import re
import einops
import pandas as pd
import numpy as np

import torch
from torch import nn

# 一个继承自
class NumEmbeddings(nn.Module):
    def __init__(
            self,
            n_features: int, # 默认768，告诉归一化层（BatchNorm）需要的维度
            d_embedding: int, # 嵌入维度
            embedding_arch: list, # 一个列表，定义特征嵌入层的结构和顺序
            d_feature: int,
    ) -> None:
        super().__init__()
        # 确保 embedding_arch 不为空，并且只包含允许的层类型。
        assert embedding_arch
        assert set(embedding_arch) <= {
            'linear', # 2维投影到768维，自定义
            'shared_linear', # 2维投影到768维，标准
            'relu', # max(0,x)
            'layernorm', # 层归一化
            'batchnorm', # 批量归一化
        }

        # NLinear_ =  NLinear
        # 定义一个空列表 layers，用于存放后续要堆叠的神经网络层
        layers: list[nn.Module] = []

        # 条件判断使用 nn.Linear 还是 NLinearMemoryEfficient
        if embedding_arch[0] == 'linear':
            assert d_embedding is not None
            layers.append(
                NLinearMemoryEfficient(n_features, d_feature, d_embedding)
            )
        elif embedding_arch[0] == 'shared_linear':
            layers.append(
                nn.Linear(d_feature, d_embedding)
            )
        # 用来动态更新输入维度
        d_current = d_embedding

        for x in embedding_arch[1:]:
            layers.append(
                nn.ReLU()
                if x == 'relu'
                else NLinearMemoryEfficient(n_features, d_current, d_embedding)  # type: ignore[code]
                if x == 'linear'
                else nn.Linear(d_current, d_embedding)  # type: ignore[code]
                if x == 'shared_linear'
                else nn.LayerNorm([n_features, d_current])
                if x == 'layernorm'
                else nn.BatchNorm1d(n_features)
                if x == 'batchnorm'
                else nn.Identity()
            )
            if x in ['linear']:
                d_current = d_embedding
            assert not isinstance(layers[-1], nn.Identity)
        self.d_embedding = d_current
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# 专门为质量（Mass）设计的编码器。
# 采用了与 Transformer 位置编码类似的正弦（Sine）和余弦（Cosine）波机制
class MassEncoder(torch.nn.Module):
    """Encode mass values using sine and cosine waves.

    Parameters
    ----------
    dim_model : int
        The number of features to output.
    min_wavelength : float
        The minimum wavelength to use.
    max_wavelength : float
        The maximum wavelength to use.
    """

    # min_wavelength 是模型能够“看清”的最小刻度
    # max_wavelength 是模型能够“看清”的最大刻度
    def __init__(self, dim_model, min_wavelength=0.001, max_wavelength=10000):
        """Initialize the MassEncoder"""
        super().__init__()

        # 确定正余弦数目
        n_sin = int(dim_model / 2)
        n_cos = dim_model - n_sin

        # 以min_wavelength存在与否
        if min_wavelength:
            base = min_wavelength / (2 * np.pi)
            scale = max_wavelength / min_wavelength
        else:
            base = 1
            scale = max_wavelength / (2 * np.pi)

        # 和下面的位置编码的相同部分一样，但多了一个base
        sin_term = base * scale ** (
            torch.arange(0, n_sin).float() / (n_sin - 1)
        )
        cos_term = base * scale ** (
            torch.arange(0, n_cos).float() / (n_cos - 1)
        )

        self.register_buffer("sin_term", sin_term)
        self.register_buffer("cos_term", cos_term)

    def forward(self, X):
        """Encode m/z values.

        Parameters
        ----------
        X : torch.Tensor of shape (n_masses)
            The masses to embed.

        Returns
        -------
        torch.Tensor of shape (n_masses, dim_model)
            The encoded features for the mass spectra.
        """
        sin_mz = torch.sin(X / self.sin_term)
        cos_mz = torch.cos(X / self.cos_term)
        return torch.cat([sin_mz, cos_mz], axis=-1)

class PositionalEncoder(torch.nn.Module):
    #  采用正弦/余弦位置编码，用于给序列（如肽段序列）中的每个位置编码，帮助 Transformer 感知“顺序信息”。
    """The positional encoder for sequences.

    Parameters
    ----------
    dim_model : int
        The number of features to output.
    """

    def __init__(self, dim_model, max_wavelength=10000):
        """Initialize the MzEncoder"""
        super().__init__()

        # 规定sin和cos的波的数量
        n_sin = int(dim_model / 2)
        n_cos = dim_model - n_sin

        # scale 决定了所有波长的“基准”，后面会用它做指数扩展
        scale = max_wavelength / (2 * np.pi) # max_wavelength：最大波长

        # 构造原件，后面会用position除以它
        # torch.arange(0, n_sin).float()：生成一个从 0 到 n_sin-1 的序列
        # / (n_sin - 1)：变成均匀分布在0-1之间的小数
        sin_term = scale ** (torch.arange(0, n_sin).float() / (n_sin - 1))
        cos_term = scale ** (torch.arange(0, n_cos).float() / (n_cos - 1))
        self.register_buffer("sin_term", sin_term)
        self.register_buffer("cos_term", cos_term)

    def forward(self, X):
        """Encode positions in a sequence.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, n_sequence, n_features)
            The first dimension should be the batch size (i.e. each is one
            peptide) and the second dimension should be the sequence (i.e.
            each should be an amino acid representation).

        Returns
        -------
        torch.Tensor of shape (batch_size, n_sequence, n_features)
            The encoded features for the mass spectra.
        """
        # 生成一个长度为序列长度的整数序列
        # X.shape[1]决定向量长度，X 的 shape 是 [batch_size, n_sequence, n_features]
        pos = torch.arange(X.shape[1]).type_as(self.sin_term)
        # 扩展到batch维度
        pos = einops.repeat(pos, "n -> b n", b=X.shape[0])
        # 把位置索引再扩展到特征维度，n_features/2
        sin_in = einops.repeat(pos, "b n -> b n f", f=len(self.sin_term))
        cos_in = einops.repeat(pos, "b n -> b n f", f=len(self.cos_term))

        # 计算encoded，即位置编码
        sin_pos = torch.sin(sin_in / self.sin_term)
        cos_pos = torch.cos(cos_in / self.cos_term)
        encoded = torch.cat([sin_pos, cos_pos], axis=2)
        return encoded + X

class PeptideDecoder(torch.nn.Module):
    """用于肽段序列的Transformer解码器。

    参数说明
    ----------
    dim_model : int, 可选
        质谱峰潜在特征的维度。
    n_head : int, 可选
        每层的注意力头数。dim_model 必须能被 n_head 整除。
    dim_feedforward : int, 可选
        Transformer层中全连接层的维度。
    n_layers : int, 可选
        Transformer层的数量。
    dropout : float, 可选
        所有层的dropout概率。
    pos_encoder : bool, 可选
        是否对氨基酸序列使用位置编码。
    residues: Dict 或 str {"massivekb", "canonical"}, 可选
        氨基酸字典及其质量。默认只包含20种标准氨基酸，半胱氨酸为烷基化形式。如果为"massivekb"，则包含MassIVE-KB中的修饰。也可自定义字典。
    """

    def __init__(
        self,
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=1,
        dropout=0.1,
        residues_length=20,
        max_charge=5,
        hidden_size=50, # tokens的最大长度
    ):
        """初始化PeptideDecoder"""
        super().__init__()

        self.dim_model = dim_model
        self.hidden_size = hidden_size
        # 位置编码，用于给肽段序列加上顺序信息
        # 使用PositionalEncoder创建一个实例
        self.pos_encoder = PositionalEncoder(self.dim_model)
        
        # 电荷 == 向量
        self.charge_encoder = torch.nn.Embedding(max_charge, self.dim_model)
        
        # 残基库，添加$的占位符
        # 将氨基酸映射为向量
        self.aa_encoder = torch.nn.Embedding(
            residues_length,
            dim_model,
            padding_idx=0, # 指定0为padding，权重不更新
        )
        
        # Additional model components
        self.mass_encoder = MassEncoder(self.dim_model)
        layer = torch.nn.TransformerDecoderLayer(
            d_model=self.dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )

        self.transformer_decoder = torch.nn.TransformerDecoder(
            layer,
            num_layers=n_layers,
        )
        
        # 数值型变量embedding
        embedding_arch = ['shared_linear', 'batchnorm', 'relu']
        # n_features: embedding维度；d_feature：输入维度；d_embedding：输出维度
        self.num_embeddings = NumEmbeddings(n_features=768, d_embedding=768,
                                            embedding_arch=embedding_arch,
                                            d_feature=2)
        

    def forward(self, memory, memory_key_padding_mask, precursors, tokens):
        """对一组肽段序列预测下一个氨基酸。

        参数说明
        ----------
        tokens : torch.Tensor，形状为 (batch_size, n_peaks, dim_model)
            需要预测下一个氨基酸的部分肽段序列。也可以是token索引。
        precursors : torch.Tensor，形状为 (batch_size, 4)
            每个串联质谱的前体质量（第0列）、电荷（第1列）、deltaRT（第2列）、预测RT（第3列）。
        memory : torch.Tensor，形状为 (batch_size, n_peaks, dim_model)
            来自TransformerEncoder（如SpectrumEncoder）的表示。
        memory_key_padding_mask : torch.Tensor，形状为 (batch_size, n_peaks)
            指示memory哪些元素是padding的掩码。

        返回
        -------
        scores : torch.Tensor，形状为 (batch_size, len_sequence, n_amino_acids)
            最后线性层的原始输出。可用Softmax变换为每种氨基酸的预测概率。
        """
        # Prepare mass, charge, deltaRT, predictedRT
        masses = self.mass_encoder(precursors[:, None, [0]]) # precursors[:, 0].unsqueeze(-1).unsqueeze(-1)
        charges = self.charge_encoder(precursors[:, 1].int() - 1) # charge范围为[0, max_charge-1] (batch, 1)
        # deltaRT 和 predictedRT，是一个两维特征
        rt = self.num_embeddings(precursors[:, 2:]) # (batch, 2) ==> (batch, 1)
        precursors = masses + charges[:, None, :] + rt[:, None, :] # (batch, 1, 768)
        
        # token encoder
        tokens = self.aa_encoder(tokens.int()) # (batch, 50, 768)

        # Feed through model:
        tgt = torch.cat([precursors, tokens], dim=1) # (batch, 51, 768)
        tgt_key_padding_mask = tgt.sum(axis=2) == 0  # (batch, 51)
        tgt = self.pos_encoder(tgt)
        
        # tgt.shape[1] ==> 51
        tgt_mask = generate_no_mask(self.hidden_size + 1).type_as(precursors)

        # (batch, 51, 768)
        decoder_output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask.bool(),
            tgt_key_padding_mask=tgt_key_padding_mask.bool(),
            memory_key_padding_mask=memory_key_padding_mask.bool(),
        )
        return decoder_output


def generate_no_mask(sz):
    """生成无掩码的序列。
    参数说明
    ----------
    sz : int
        目标序列的长度。
    """
    mask = torch.zeros(sz, sz).float()
    return mask
