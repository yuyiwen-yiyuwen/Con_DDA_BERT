# 整体流程
    # 读取 3 份 parquet 数据, raw, sage与fp
    # 构造统一的 PSM 标识 psm_id, 用 scan + sequence 拼接
    # 把 target (label == 1) 和 decoy (label == 0)分开
    # 把 sage-target 和 fp 结果做交集：只保留 Sage target 中，同时也出现在 fp 文件里的 PSM
    # 挑选数量接近的 decoy
    # 与原始谱图合并、做过滤、保留必要列，然后保存

# 导入模块部分
import pandas as pd
import os
import shutil
import numpy as np
import argparse

# 命令行参数部分, 运行脚本时，需要从命令行传入 4 个路径参数
    # -raw：原始谱图 parquet 文件路径
    # -sage_sr：Sage 搜索结果 parquet 文件路径
    # -fp_sr：FP 结果 parquet 文件路径
    # -parquet_path：最终输出 parquet 文件路径

parser = argparse.ArgumentParser()

parser.add_argument('-raw', type=str, default="", help="")
parser.add_argument('-sage_sr', type=str, default="", help="")
parser.add_argument('-fp_sr', type=str, default="", help="")
parser.add_argument('-parquet_path', type=str, default="", help="")

args = parser.parse_args()

# 肽段修饰字符串清洗/替换函数（后面没有用到这个）
def clean_psm_func(peptide, residues_dict):
    for key, value in residues_dict.items():
        if value not in peptide:
            peptide = peptide.replace(key, value)
    return peptide

# 读取原始谱图数据
raw_df = pd.read_parquet(args.raw)
raw_df['scan'] = raw_df['scan'].astype(int)

# 读取sage数据
sage_df = pd.read_parquet(args.sage_sr)
sage_df['scan'] = sage_df['scan'].astype(int)
sage_df['psm_id'] = sage_df['scan'].astype(str) + '_' + sage_df['precursor_sequence']
sage_df_target = sage_df[sage_df['label']==1]
sage_df_decoy = sage_df[sage_df['label']==0]

# 读取fp数据
fp_df = pd.read_parquet(args.fp_sr)
fp_df['psm_id'] = fp_df['scan'].astype(str) + '_' + fp_df['detect_sequence']
# 删除 scan 列
fp_df = fp_df.drop(columns=['scan'], axis=1)

# 生成parquet文件
# 以 psm_id 为键, 只保留同时出现在 sage_df_target 和 fp_df 中的 PSM
target_sr_df = sage_df_target.merge(fp_df, on='psm_id', how='inner')
# 把 decoy_num 设成 target 的数量
decoy_num = len(target_sr_df)

# 选 decoy 样本
# 按 sage_discriminant_score 降序排序
decoy_df_sorted = sage_df_decoy.sort_values(by='sage_discriminant_score',ascending=False).reset_index(drop=True)
# 如果 decoy 不够多，就全拿
if len(decoy_df_sorted) <= decoy_num:
    decoy_df_need = decoy_df_sorted
# 如果 decoy 足够多，就做“半高分 + 半随机”
else:
    decoy_df_high_score = decoy_df_sorted.iloc[:int(decoy_num/2)] # 选取前一半（分数最高的）decoy
    decoy_df_low_score = decoy_df_sorted.iloc[int(decoy_num/2):].sample(n=int(decoy_num/2),random_state=42) # 剩下的 decoy中，随机抽取另一半
    decoy_df_need = pd.concat([decoy_df_high_score,decoy_df_low_score], axis=0, ignore_index=True)
sr_df_need = pd.concat([target_sr_df, decoy_df_need], axis=0, ignore_index=True)
# 与原始谱图按 scan 合并
# 只保留两个表（DataFrame）中都存在的 psm_id 对应的行，其他不匹配的行会被丢弃
parquet_df = sr_df_need.merge(raw_df, on='scan', how='inner')

# 一致性检查
assert len(parquet_df) == len(sr_df_need), f"parquet df is {len(parquet_df)}, sr df need is {len(sr_df_need)}"
# 过滤前先打印数量
print(f'before the cut, psm num is {len(parquet_df)}')

# 序列清洗：把修饰标记去掉，只保留氨基酸字母本身
parquet_df['cleaned_sequence'] = parquet_df['precursor_sequence'].str.replace('n[42]', '').str.replace('N[.98]', 'N').str.replace('Q[.98]', 'Q').str.replace('M[15.99]', 'M').str.replace('C[57.02]', 'C')
parquet_df['sequence_len'] = parquet_df['cleaned_sequence'].apply(len)

# 条件筛选
parquet_df = parquet_df[(parquet_df['sequence_len']<=50)&(parquet_df['sequence_len']>=7)]
parquet_df = parquet_df[(parquet_df['charge']<=5)&(parquet_df['charge']>=2)]

# 过滤后打印数量
print(f'after the cut, psm num is {len(parquet_df)}')
# 保留最终需要的列
parquet_df = parquet_df[['scan','precursor_mz','charge','rt','mz_array','intensity_array','precursor_sequence','label','predicted_rt', 'delta_rt','sage_discriminant_score','spectrum_q']]

# 加权重列
parquet_df['weight'] = 1


# 保存
parquet_df.to_parquet(args.parquet_path)