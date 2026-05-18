# 本代码的作用
# 1.PositionalEncoding (位置编码)：利用正弦和余弦函数（Standard sinusoidal）生成位置向量
# 2.MultiScalePeakEmbedding (多尺度质谱峰编码)：质谱的 m/z（质量电荷比）通过正弦/余弦变换（类似位置编码，但用于连续数值），把数值映射到高维空间
# 3.tie_encoder_decoder_weights (权重共享函数)：递归地将 encoder 和 decoder 中相同名称或结构的模块权重“绑定”
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch import Tensor
from typing import List

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) 

        position = torch.arange(max_len).unsqueeze(1) # 生成位置索引
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Positional encoding forward pass.

        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiScalePeakEmbedding(nn.Module):
    """Multi-scale sinusoidal embedding based on Voronov et. al."""
    # 基于 Voronov 等人提出的多尺度正弦嵌入方法，用于高精度编码质谱峰。

    def __init__(self, h_size: int, dropout: float = 0) -> None:
        super().__init__()
        self.h_size = h_size  # 保存隐藏层维度（特征长度）

        # 1. 定义第一个 MLP 模块
        # 作用：在 m/z 转化为正弦波形后，进行深度非线性特征提取
        self.mlp = nn.Sequential(
            nn.Linear(h_size, h_size), # 输入波形特征，输出相同维度的处理后特征
            nn.ReLU(),                 # 激活函数，引入非线性
            nn.Dropout(dropout),       # 防止过拟合
            nn.Linear(h_size, h_size), # 第二层线性变换
            nn.Dropout(dropout),
        )

        # 2. 定义最终输出的 Head 模块
        # 注意：这里的输入维度是 h_size + 1，那个 +1 是为了给“强度(Intensity)”留位置
        self.head = nn.Sequential(
            nn.Linear(h_size + 1, h_size), # 将“加工后的m/z特征”与“强度”融合，映射回 h_size
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_size, h_size),
            nn.Dropout(dropout),
        )

        # 3. 生成多尺度频率序列
        # torch.logspace(-2, -3, ...) 在 10^-2 到 10^-3 之间生成对数间隔的频率
        # 作用：不同频率对应不同精度。高频捕捉微小质量差，低频捕捉大范围分布。
        freqs = 2 * np.pi / torch.logspace(-2, -3, int(h_size / 2), dtype=torch.float64)
        self.register_buffer("freqs", freqs) # 表示为 buffer，这样它会随模型移动到 GPU 但不会被更新梯度

    def forward(self, mz_values: Tensor, intensities: Tensor) -> Tensor:
        """Encode peaks."""
        # 第一步：将原始 m/z 数值编码为高维正余弦波形
        x = self.encode_mass(mz_values) 
        
        # 数据类型转换，确保精度一致
        if x.dtype != mz_values.dtype:
            x = x.to(mz_values.dtype)
        
        # 第二步：使用第一个 MLP 对 m/z 的波形特征进行编码
        x = self.mlp(x)
        
        # 第三步：特征拼接
        # 将加工好的 m/z 特征与原始的强度(intensities) 在最后一个维度拼接到一起
        x = torch.cat([x, intensities], axis=2) # axis=2
        
        # 第四步：输出处理
        # 通过 head 模块，输出最终融合了质量和强度的峰特征向量
        return self.head(x)

    def encode_mass(self, x: Tensor) -> Tensor:
        """Encode mz."""
        # 1. 频率缩放：将每个 m/z 数字与所有预设频率相乘
        # x 的 shape 为 (batch, n_peaks, 1)，freqs 的扩展后与之匹配
        x = self.freqs[None, None, :] * x 
        
        # 2. 正余弦变换：
        # 同时计算所有缩放后数值的 sin 和 cos，然后在最后一个维度拼接
        # 这样 1 个数字就被变成了 h_size 个数字（h_size/2 个 sin 和 h_size/2 个 cos）
        x = torch.cat([torch.sin(x), torch.cos(x)], axis=2)
        
        # 3. 返回浮点数张量
        return x.float()

def tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, skip_key:str):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        print(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
        decoder_pointer: nn.Module,
        encoder_pointer: nn.Module,
        module_name: str,
        uninitialized_encoder_weights: List[str],
        skip_key: str,
        depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias
            # print(module_name+' is tied')
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                        encoder_modules
                    ) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key)
