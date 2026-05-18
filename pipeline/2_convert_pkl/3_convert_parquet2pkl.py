# 把 .parquet 格式的质谱训练数据，转换成模型更容易直接加载的 .pkl 缓存文件

# 导入库
import os
import os.path
import random
random.seed(123)

import logging
from optparse import OptionParser
import numpy as np
import pandas as pd
import yaml
import math
import re
import glob

import torch
import torch.utils.data as pt_data
import pickle
from collections import defaultdict
from sklearn.model_selection import train_test_split
from random import sample
from multiprocessing import Process
import polars as pl

import sys
sys.path.insert(0, "/home/yiwen/AIPC/scripts/organized_attantion")

try:
    from model.transformer.dataset import SpectrumDataset, collate_batch_weight_deltaRT
except ImportError:
    import sys
    sys.path.append("/home/yiwen")
    from AIPC.scripts.attantion.transformer.dataset import SpectrumDataset, collate_batch_weight_deltaRT

# 日志输出
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


# 目录创建函数，假如目录不存在就创建
def mkdir_p(dirs):
    if not os.path.exists(dirs):
        os.mkdir(dirs)
    return True, 'OK'


# 查找 ipc 文件，本函数没有用到
def gen_train_data(options, s2i, n_peaks):
    final_res = os.popen('find %s/* -type f -name "*.ipc"' % (options.file_dir)).read()
    file_path_list = list(set([v for v in final_res.split('\n') if (v not in [''])]))
    logger.info('total: %s, %s' % (len(file_path_list), file_path_list))

    file_name_list = [item.split("/")[-1].split(".")[0] for item in file_path_list]
    logger.info("file_name len: %s, %s" % (len(file_name_list), file_name_list[:1]))  

    # === 修改点 ===
    file_dir = options.save_dir if options.save_dir else os.path.join(options.file_dir, options.task_name)
    mkdir_p(file_dir)
    # =============

    if options.ncores > 1:
        processes = []
        sublength = int(len(file_path_list) / options.ncores) 
        for i in range(0,len(file_path_list), sublength):
            process = Process(target=construct_data_set, args=(file_path_list[i:(i+sublength)],
                              file_name_list[i:(i+sublength)],
                              file_dir,
                              s2i,
                              n_peaks,
                              options.task_name,
                              options.batch_size))
            processes.append(process)
            process.start()
        for process in processes:
            process.join()
    else:
        construct_data_set(file_path_list,
                           file_name_list,
                           file_dir,
                           s2i,
                           n_peaks,
                           options.task_name,
                           options.batch_size)

    logger.info("train data generated")

# 把一组 parquet 文件转换为 pkl 文件
def construct_data(data_path_list, # 输入 parquet 文件路径列表
                   filename_list, # 对应的文件名列表
                   feat_dir, # 输出
                   s2i, # 词表映射，字符串 token 到整数 id
                   n_peaks, # 每条谱图保留的峰数上限
                   task_name): # 任务名，用于拼接输出文件名

    # 每次处理列表中的一个 parquet
    for file_index in range(len(data_path_list)):
        # 构造输出文件名
        output_file_path = os.path.join(feat_dir, f'%s_%s.pkl' % (filename_list[file_index], task_name))
        # 如果已经存在就跳过
        if os.path.exists(output_file_path):
            continue
        
        df = pd.read_parquet(data_path_list[file_index])
        
        # 构建 SpectrumDataset
        ds = SpectrumDataset(df, s2i, n_peaks, need_label=True, need_weight=True, need_deltaRT=True, need_unmask=True)
        logger.info('load file: %s' % (data_path_list[file_index]))

        # 预处理spectra、peptide、precusor
        # 修改为接收可能的 unmask 返回值
        collate_res = collate_batch_weight_deltaRT(ds)
        if len(collate_res) == 8:
            spectra, spectra_mask, precursors, tokens, peptides, label, weight, unmask = collate_res
        else:
            spectra, spectra_mask, precursors, tokens, peptides, label, weight = collate_res
            unmask = torch.zeros_like(label)
        if precursors.shape[-1] > 2:
            precursors = precursors.clone()
            precursors[..., 2:] = 0

        # 统计 target / decoy 数量
        target_num = torch.count_nonzero(label).item()
        decoy_num = len(df) - target_num
        logger.info("decoy: {}, target: {}, total: {}".format(decoy_num, target_num, len(df)))

        name_base = feat_dir + '/' + filename_list[file_index]
        logger.info('save %s , len: %d' % (name_base, len(df)))

        # 保存为pkl文件
        out_dict = {'spectra':spectra.numpy(),
                    'spectra_mask':spectra_mask.numpy(),
                    'precursors':precursors.numpy(),
                    'tokens':tokens.numpy(),
                    'peptides':peptides,
                    'label':label.numpy(),
                    'weight': weight.numpy(),
                    'unmask': unmask.numpy()}

        output_pkl = open(os.path.join(feat_dir, f'%s_%s.pkl' % (filename_list[file_index], task_name)), "wb")
        output_pkl.write(pickle.dumps(out_dict, protocol=4))
        output_pkl.close()


def convert_data(options, s2i, n_peaks):
    file_path_list = glob.glob('%s/*parquet' % (options.file_dir))
    logger.info('total: %s, %s' % (len(file_path_list), file_path_list[:2]))

    file_name_list = [os.path.basename(item).replace(".parquet", "") for item in file_path_list]
    logger.info("file_name len: %s, %s" % (len(file_name_list), file_name_list[:2]))  

    # === 修改点 ===
    file_dir = options.save_dir if options.save_dir else os.path.join(options.file_dir, options.task_name)
    mkdir_p(file_dir)
    # =============

    if options.ncores > 1:
        processes = []
        
        ncores = min(len(file_path_list), int(options.ncores))
        sublength = int(len(file_path_list) / ncores) 
        for i in range(0,len(file_path_list), sublength):
            process = Process(target=construct_data, args=(file_path_list[i:(i+sublength)],
                              file_name_list[i:(i+sublength)],
                              file_dir,
                              s2i,
                              n_peaks,
                              options.task_name))
            processes.append(process)
            process.start()
        for process in processes:
            process.join()
    else:
        construct_data(file_path_list,
                       file_name_list,
                       file_dir,
                       s2i,
                       n_peaks,
                       options.task_name)

    logger.info("pkl data generated")

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--file_dir", type="string", default="/ajun/MS_GPT_Dataset/shuffle_decoy_sage/dataset51_target_top08/false_target_02/",
                      help=".parquet directory")
    parser.add_option("--config", type="string", default="/zhangxiaofan/DDA_BERT_deltaRT/test_data/model.yaml",
                      help=".parquet directory")
    parser.add_option("--task_name", type="string", default="target_top08_ft_02", help="task_name")
    parser.add_option("--ncores", type="int", default=30, help="number of CPU cores, range is [1, 20]")
    parser.add_option("--save_dir", type="string", default="", help="pkl save directory (if not set, use file_dir/task_name)")

    (options, args) = parser.parse_args()
    logger.info('getdata begin!!!, task_name: %s' % (options.task_name))
    
    #加载数据
    config_path = options.config
    with open(config_path) as f_in:
        config = yaml.safe_load(f_in)

    vocab = ['<pad>', '<mask>'] + list(config["residues"].keys()) + ['<unk>']
    config["vocab"] = vocab
    s2i = {v: i for i, v in enumerate(vocab)}
    logging.info(f"Vocab: {s2i}, n_peaks: {config['n_peaks']}")
    
    convert_data(options, s2i, config['n_peaks'])
    logger.info('getdata end!!!!')

### done
# cd /zhangxiaofan/DDA_BERT_deltaRT/test_data/;python 3_convert_parquet2pkl.py --file_dir /zhangxiaofan/DDA_BERT_deltaRT/test_data/test_mzml_parquet_split --task_name test_16m_mzml --ncores 20 --save_dir /zhangxiaofan/DDA_BERT_deltaRT/test_data/test_mzml_pkl_split
