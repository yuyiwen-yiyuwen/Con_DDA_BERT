from __future__ import annotations

import re
import os

import numpy as np
import pandas as pd
import polars as pl
import spectrum_utils.spectrum as sus
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F

# from constants import PROTON_MASS_AMU
PROTON_MASS_AMU = 1.007276


# 把 DataFrame 里每一行原始 PSM ，转换成 numpy 形式
class SpectrumDataset(Dataset): # 继承自 Dataset
    def __init__(
        self,
        df: pd.DataFrame | pl.DataFrame,
        s2i: dict,
        n_peaks: int = 200,
        max_length=50,
        min_mz: float = 50.0,
        max_mz: float = 2500.0,
        min_intensity: float = 0.01,
        remove_precursor_tol: float = 2.0,
        reverse_peptide: bool = True,
        annotated: bool = True,

        need_label: bool = False,
        need_file_name: bool = False,
        need_index: bool = False,
        need_weight: bool = False,
        need_deltaRT: bool = False,
        need_unmask: bool = False,
        need_no_fdr01_target: bool = False,
        need_psm_id: bool = False,
    ) -> None:
        super().__init__()
        self.df = df
        self.s2i = s2i
        self.n_peaks = n_peaks
        self.max_length = max_length
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.remove_precursor_tol = remove_precursor_tol
        self.min_intensity = min_intensity
        self.annotated = annotated
        self.need_label = need_label
        self.need_index = need_index
        self.need_weight = need_weight
        self.need_deltaRT = need_deltaRT
        self.need_unmask = need_unmask
        self.need_no_fdr01_target = need_no_fdr01_target
        self.need_psm_id = need_psm_id

        # 判断 DataFrame 类型
        if isinstance(df, pd.DataFrame):
            self.data_type = "pd"
        elif isinstance(df, pl.DataFrame):
            self.data_type = "pl"
        else:
            raise Exception(f"Unsupported data type {type(df)}")

    # 返回样本数
    def __len__(self) -> int:
        return int(self.df.shape[0])

    # 输入：idx ，输出：这一行对应的模型样本
    def __getitem__(self, idx: int) -> tuple[Tensor, float, int, Tensor | list[str]]:
        # 初始化 peptide
        peptide = ""
        
        if self.data_type == "pl":
            mz_array = torch.Tensor(self.df[idx, "mz_array"].to_list())
            int_array = torch.Tensor(self.df[idx, "intensity_array"].to_list())
            precursor_mz = self.df[idx, "precursor_mz"]
            precursor_charge = self.df[idx, "precursor_charge"]
            peptide = self.df[idx, "modified_sequence"]
            if self.need_label:
                label = self.df[idx, "label"]
            if self.need_index:
                index = self.df[idx, "index"]
            if self.need_psm_id:
                psm_id = self.df[idx, "psm_id"]
                # lowHz = self.df[idx, "lowHz"]    # edit_by_zxf_20241104
            if self.need_weight:
                weight = self.df[idx, "weight"]
            if self.need_deltaRT:
                # 兼容两种列名：delta_rt_model（新）和 delta_rt（旧）
                if "delta_rt_model" in self.df.columns:
                    deltaRT = self.df[idx, "delta_rt_model"] or 0.0
                elif "delta_rt" in self.df.columns:
                    deltaRT = self.df[idx, "delta_rt"] or 0.0
                else:
                    deltaRT = 0.0
                predictedRT = self.df[idx, "predicted_rt"] if "predicted_rt" in self.df.columns else 0.0
                predictedRT = predictedRT or 0.0
            if self.need_unmask:
                unmask = self.df[idx, "unmask"] or 0
            if self.need_no_fdr01_target:
                no_fdr01_target = self.df[idx, "no_fdr01_target"] or 0
        else:
            # 先取第 idx 行
            row = self.df.iloc[idx]

            # 读取 PSM 数组
            mz_array = torch.Tensor(row["mz_array"])
            int_array = torch.Tensor(row["intensity_array"])

            # 读取前体信息
            precursor_mz = row["precursor_mz"]
            precursor_charge = row["charge"]

            # 读 peptide
            peptide = row["precursor_sequence"]

            # 按需读取剩余字段
            if self.need_label:
                label = row["label"]
            if self.need_index:
                index = row["index"]
            if self.need_psm_id:
                psm_id = row["psm_id"]
            if self.need_weight:
                weight = row["weight"]
            if self.need_deltaRT:
                # 兼容两种列名：delta_rt_model 和 delta_rt
                if "delta_rt_model" in row.index:
                    deltaRT = row["delta_rt_model"] or 0.0
                elif "delta_rt" in row.index:
                    deltaRT = row["delta_rt"] or 0.0
                else:
                    deltaRT = 0.0

                if "predicted_rt" in row.index:
                    predictedRT = row["predicted_rt"] or 0.0
                else:
                    predictedRT = 0.0
            if self.need_unmask:
                unmask = row["unmask"] or 0
            if self.need_no_fdr01_target:
                no_fdr01_target = row["no_fdr01_target"] or 0

        # 把原始峰数组变成固定长度、标准化后的谱图张量
        spectrum = self._process_peaks(mz_array, int_array, precursor_mz, precursor_charge)

        # 序列 token 化
        tokens = self._tokenize(peptide)
        
        # 按不同开关组合返回不同内容
        if self.need_no_fdr01_target:
            return spectrum, precursor_mz, precursor_charge, deltaRT, predictedRT, tokens, peptide, label, weight, no_fdr01_target
        elif (self.need_unmask) and (self.need_psm_id):
            return spectrum, precursor_mz, precursor_charge, tokens, label, weight, unmask, psm_id
        elif self.need_unmask:
            return spectrum, precursor_mz, precursor_charge, deltaRT, predictedRT, tokens, peptide, label, weight, unmask
        elif (self.need_label) and (self.need_deltaRT) and (self.need_weight) and (self.need_index):
            return spectrum, precursor_mz, precursor_charge, deltaRT, predictedRT, tokens, peptide, label, weight, index
        elif (self.need_label) and (self.need_deltaRT) and (self.need_weight):
            return spectrum, precursor_mz, precursor_charge, deltaRT, predictedRT, tokens, peptide, label, weight
        elif (self.need_label) and (self.need_weight):
            return spectrum, precursor_mz, precursor_charge, tokens, peptide, label, weight
        elif (self.need_label) and (self.need_index):
            return spectrum, precursor_mz, precursor_charge, tokens, peptide, label, index
        elif self.need_label:
            return spectrum, precursor_mz, precursor_charge, tokens, peptide, label
        else:
            return spectrum, precursor_mz, precursor_charge, tokens, peptide

    # 把原始的 mz_array 和 intensity_array 处理成固定形状谱图张量
    def _process_peaks(
        self,
        mz_array: Tensor,
        int_array: Tensor,
        precursor_mz: Tensor,
        precursor_charge: Tensor,
    ) -> Tensor:
        """Preprocess the spectrum by removing noise peaks and scaling the peak intensities.

        Parameters
        ----------
        mz_array : numpy.ndarray of shape (n_peaks,)
            The spectrum peak m/z values.
        int_array : numpy.ndarray of shape (n_peaks,)
            The spectrum peak intensity values.

        Returns
        -------
        torch.Tensor of shape (n_peaks, 2)
            A tensor of the spectrum with the m/z and intensity peak values.
        """
        
        # 构造 MsmsSpectrum
        spectrum = sus.MsmsSpectrum(
            "",
            precursor_mz,
            precursor_charge,
            np.array(mz_array).astype(np.float32),
            np.array(int_array).astype(np.float32),
        )
        try:
            # 第一步：限制 m/z 范围
            spectrum.set_mz_range(self.min_mz, self.max_mz)

            # 第二步：检查是否有峰
            if len(spectrum.mz) == 0:
                raise ValueError
            
            # 第三步：去除 precursor peak
            spectrum.remove_precursor_peak(self.remove_precursor_tol, "Da")

            # 第四步：再检查一次是否为空
            if len(spectrum.mz) == 0:
                raise ValueError

            # 第五步：按强度过滤，并保留最多 n_peaks
            spectrum.filter_intensity(self.min_intensity, self.n_peaks)

            # 第六步：再检查一次是否为空
            if len(spectrum.mz) == 0:
                raise ValueError

            # 第七步：强度缩放
            spectrum.scale_intensity("root", 1)

            # 第八步：L2 归一化
            intensities = spectrum.intensity / np.linalg.norm(spectrum.intensity)

            # 第九步：组装成 (n, 2) 的张量
            spec_tensor = torch.tensor(np.array([spectrum.mz, intensities])).T.float()

            # 第十步：统一长度到 n_peaks
            if spec_tensor.shape[0] >= self.n_peaks:
                return spec_tensor[: self.n_peaks]

            pad_rows = self.n_peaks - spec_tensor.shape[0]
            return F.pad(spec_tensor, (0, 0, 0, pad_rows), "constant", 0.0)
        except ValueError:
            # Replace invalid spectra by a dummy spectrum.
            dummy = torch.zeros((self.n_peaks, 2), dtype=torch.float32)
            dummy[0] = torch.tensor([0.0, 1.0], dtype=torch.float32)
            return dummy

    # 肽段编码详解
    def _tokenize(self, sequence):
        """Transform a peptide sequence into tokens

        Parameters
        ----------
        sequence : str
            A peptide sequence.

        Returns
        -------
        torch.Tensor
            The token for each amino acid in the peptide sequence.
        """
        # 第一步：把 I 替换成 L，把 N 端修饰替换成 X
        sequence = sequence.replace("I", "L").replace('n[42]', 'X')
        
        # 第二步：兼容不同修饰写法
        sequence = sequence.replace('cC', 'C[57.02]')\
                           .replace('oxM', 'M[15.99]')\
                           .replace('M(ox)', 'M[15.99]')\
                           .replace('deamN', 'N[.98]')\
                           .replace('deamQ', 'Q[.98]')\
                           .replace('a', 'X')
        
        # 第三步：按氨基酸 token 切分
        # 在“前面有任意一个字符，后面是大写字母”的位置切分
        sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)

        # 第四步：映射为 token id
        tokens = torch.tensor([self.s2i[aa] for aa in sequence])

        # 第五步：padding 到固定长度
        tokens = F.pad(tokens, (0, self.max_length - tokens.shape[0]), 'constant', 0)# padding
        return tokens


def padding(data):
    ll = torch.tensor([x.shape[0] for x in data], dtype=torch.long)
    data = nn.utils.rnn.pad_sequence(data, batch_first=True)
    data_mask = torch.arange(data.shape[1], dtype=torch.long)[None, :] >= ll[:, None]
    return data, data_mask


def collate_batch(
    batch: list[tuple[Tensor, float, int, Tensor, Tensor]]
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Collate batch of samples."""
    spectra, precursor_mzs, precursor_charges, peptides, tokens, label = zip(*batch)

    # Pad spectra
    spectra, spectra_mask = padding(spectra)

    # stack tokens
    tokens = torch.stack(tokens, dim=0)

    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - PROTON_MASS_AMU) * precursor_charges
    precursors = torch.vstack([precursor_masses, precursor_charges]).T.float()
    label = torch.tensor(label).to(torch.float)   

    return spectra, spectra_mask, precursors, tokens, label


def collate_batch_index_weight(
    batch: list[tuple[Tensor, float, int, Tensor, Tensor]]
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Collate batch of samples."""
    spectra, precursor_mzs, precursor_charges, tokens, label, index, weight = zip(*batch)

    # Pad spectra
    spectra, spectra_mask = padding(spectra)
    
    # stack tokens
    tokens = torch.stack(tokens, dim=0)

    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - PROTON_MASS_AMU) * precursor_charges
    precursors = torch.vstack([precursor_masses, precursor_charges]).T.float()
    
    label = torch.tensor(label).to(torch.float)
    index = torch.tensor(index).to(torch.float)
    weight = torch.tensor(weight).to(torch.float)

    return spectra, spectra_mask, precursors, tokens, label, index, weight


def collate_batch_weight(
    batch: list[tuple[Tensor, float, int, Tensor, Tensor]]
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Collate batch of samples."""
    spectra, precursor_mzs, precursor_charges, tokens, peptides, label, weight = zip(*batch)

    # Pad spectra
    spectra, spectra_mask = padding(spectra)
    
    # stack tokens
    tokens = torch.stack(tokens, dim=0)

    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - PROTON_MASS_AMU) * precursor_charges
    precursors = torch.vstack([precursor_masses, precursor_charges]).T.float()
    
    label = torch.tensor(label).to(torch.float)
    weight = torch.tensor(weight).to(torch.float)

    return spectra, spectra_mask, precursors, tokens, peptides, label, weight

def collate_batch_weight_unmask(
    batch: list[tuple[Tensor, float, int, Tensor, Tensor]]
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Collate batch of samples."""
    spectrum, precursor_mzs, precursor_charges, tokens, label, weight, unmask = zip(*batch)

    # Pad spectra
    spectra, spectra_mask = padding(spectrum)
    
    # stack tokens
    tokens = torch.stack(tokens, dim=0)

    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - PROTON_MASS_AMU) * precursor_charges
    precursors = torch.vstack([precursor_masses, precursor_charges]).T.float()
    
    label = torch.tensor(label).to(torch.float)
    weight = torch.tensor(weight).to(torch.float)
    unmask = torch.tensor(unmask).to(torch.float)

    return spectra, spectra_mask, precursors, tokens, label, weight, unmask


def collate_batch_weight_unmask_psmID(
    batch: list[tuple[Tensor, float, int, Tensor, Tensor]]
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Collate batch of samples."""
    spectrum, precursor_mzs, precursor_charges, tokens, label, weight, unmask, psm_id = zip(*batch)

    # Pad spectra
    spectra, spectra_mask = padding(spectrum)
    
    # stack tokens
    tokens = torch.stack(tokens, dim=0)

    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - PROTON_MASS_AMU) * precursor_charges
    precursors = torch.vstack([precursor_masses, precursor_charges]).T.float()
    
    label = torch.tensor(label).to(torch.float)
    weight = torch.tensor(weight).to(torch.float)
    unmask = torch.tensor(unmask).to(torch.float)
    unmask = torch.nan_to_num(unmask, nan=0)

    return spectra, spectra_mask, precursors, tokens, label, weight, unmask, psm_id


def collate_batch_weight_deltaRT(
    batch: list[tuple[Tensor, float, int, Tensor, Tensor]]
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    输入batch是一个包含多个样本的列表，每个样本是一个元组：
    (谱图张量, 前体m/z, 前体电荷, deltaRT, predictedRT, 序列token, 肽段, 标签, 权重)
    或者 (包含 unmask):
    (谱图张量, 前体m/z, 前体电荷, deltaRT, predictedRT, 序列token, 肽段, 标签, 权重, unmask)

    返回：
    - spectra: padded后的谱图张量(batch, n_peaks, 2)
    - spectra_mask: mask张量(batch, n_peaks)，True表示padding部分
    - precursors: 前体信息(batch, 4)，依次为质量、电荷、deltaRT、predictedRT
    - tokens: 序列token(batch, max_length)
    - peptide: 肽段序列(batch, str)
    - label: 标签(batch, float)
    - weight: 权重(batch, float)
    - (可选) unmask: unmask标记(batch, float)
    """
    # 解包batch，根据长度判断是否包含 unmask
    if len(batch[0]) == 10:
        spectrum, precursor_mzs, precursor_charges, deltaRT, predictedRT, tokens, peptide, label, weight, unmask = zip(*batch)
        has_unmask = True
    else:
        spectrum, precursor_mzs, precursor_charges, deltaRT, predictedRT, tokens, peptide, label, weight = zip(*batch)
        has_unmask = False
        unmask = None
    
    print('input spectrum: ', spectrum[0].shape)  # 打印第一个样本的谱图形状

    # 谱图padding到统一长度，返回mask
    spectra, spectra_mask = padding(spectrum)

    # tokens堆叠成(batch, max_length)张量
    tokens = torch.stack(tokens, dim=0)

    # 前体m/z和电荷转为张量
    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    # 计算前体质量：去除质子质量后乘电荷
    precursor_masses = (precursor_mzs - PROTON_MASS_AMU) * precursor_charges

    # deltaRT和predictedRT转为张量
    deltaRT = torch.tensor(deltaRT)
    predictedRT = torch.tensor(predictedRT)

    # 前体信息合并为(batch, 4)张量
    precursors = torch.vstack([precursor_masses, precursor_charges, deltaRT, predictedRT]).T.float()

    # 标签和 weight 转为float张量
    label = torch.tensor(label).to(torch.float)
    weight = torch.tensor(weight).to(torch.float)

    if has_unmask:
        unmask = torch.tensor(unmask).to(torch.float)
        return spectra, spectra_mask, precursors, tokens, peptide, label, weight, unmask
    
    return spectra, spectra_mask, precursors, tokens, peptide, label, weight
    weight = torch.tensor(weight).to(torch.float)

    print('output spectra: ', spectra.shape, 'label: ', label.shape)  # 打印输出张量形状
    return spectra, spectra_mask, precursors, tokens, peptide, label, weight


def collate_batch_weight_deltaRT_index(
    batch: list[tuple[Tensor, float, int, Tensor, Tensor]]
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Collate batch of samples."""
    spectrum, precursor_mzs, precursor_charges, deltaRT, predictedRT, tokens, peptide, label, weight, index = zip(*batch)

    # Pad spectra
    spectra, spectra_mask = padding(spectrum)
    
    # stack tokens
    tokens = torch.stack(tokens, dim=0)

    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - PROTON_MASS_AMU) * precursor_charges

    deltaRT = torch.tensor(deltaRT)
    predictedRT = torch.tensor(predictedRT)
    precursors = torch.vstack([precursor_masses, precursor_charges, deltaRT, predictedRT]).T.float()
    
    label = torch.tensor(label).to(torch.float)
    weight = torch.tensor(weight).to(torch.float)
    index = torch.tensor(index).to(torch.float)

    return spectra, spectra_mask, precursors, tokens, peptide, label, weight, index


def collate_batch_weight_deltaRT_unmask(
    batch: list[tuple[Tensor, float, int, Tensor, Tensor]]
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Collate batch of samples."""
    spectrum, precursor_mzs, precursor_charges, deltaRT, predictedRT, tokens, peptide, label, weight, unmask = zip(*batch)

    # Pad spectra
    spectra, spectra_mask = padding(spectrum)
    
    # stack tokens
    tokens = torch.stack(tokens, dim=0)

    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - PROTON_MASS_AMU) * precursor_charges

    deltaRT = torch.tensor(deltaRT)
    predictedRT = torch.tensor(predictedRT)
    precursors = torch.vstack([precursor_masses, precursor_charges, deltaRT, predictedRT]).T.float()
    
    label = torch.tensor(label).to(torch.float)
    weight = torch.tensor(weight).to(torch.float)
    unmask = torch.tensor(unmask).to(torch.float)

    return spectra, spectra_mask, precursors, tokens, peptide, label, weight, unmask


def collate_batch_weight_deltaRT_no_fdr01_target(
    batch: list[tuple[Tensor, float, int, Tensor, Tensor]]
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Collate batch of samples."""
    spectrum, precursor_mzs, precursor_charges, deltaRT, predictedRT, tokens, peptide, label, weight, no_fdr01_target = zip(*batch)

    # Pad spectra
    spectra, spectra_mask = padding(spectrum)
    
    # stack tokens
    tokens = torch.stack(tokens, dim=0)

    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - PROTON_MASS_AMU) * precursor_charges

    deltaRT = torch.tensor(deltaRT)
    predictedRT = torch.tensor(predictedRT)
    precursors = torch.vstack([precursor_masses, precursor_charges, deltaRT, predictedRT]).T.float()
    
    label = torch.tensor(label).to(torch.float)
    weight = torch.tensor(weight).to(torch.float)
    no_fdr01_target = torch.tensor(no_fdr01_target).to(torch.float)

    return spectra, spectra_mask, precursors, tokens, peptide, label, weight, no_fdr01_target


def collate_batch_index_deltaRT(
    batch: list[tuple[Tensor, float, int, Tensor, Tensor]]
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Collate batch of samples."""
    spectrum, precursor_mz, precursor_charge, deltaRT, peptide, tokens, label, index = zip(*batch)

    # Pad spectra
    spectra, spectra_mask = padding(spectra)
    
    # stack tokens
    tokens = torch.stack(tokens, dim=0)

    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - PROTON_MASS_AMU) * precursor_charges

    deltaRT = torch.tensor(deltaRT)
    precursors = torch.vstack([precursor_masses, precursor_charges, deltaRT]).T.float()
    
    label = torch.tensor(label).to(torch.float)
    index = torch.tensor(index).to(torch.float)

    return spectra, spectra_mask, precursors, tokens, label, index

def mkdir_p(dirs, delete=True):
    """
    make a directory (dir) if it doesn't exist
    """    
    # 如果文件夹不存在，则递归新建
    if not os.path.exists(dirs):
        try:
            # 递归创建文件夹
            os.makedirs(dirs)
        except:
            pass

    return True, 'OK'
