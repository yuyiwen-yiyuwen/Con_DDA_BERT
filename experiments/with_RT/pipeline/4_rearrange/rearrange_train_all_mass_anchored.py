"""
该脚本旨在将原始的训练数据（pkl 格式）重新排列为“质量锚定”（Mass-Anchored）布局。

关键策略：
1. 质量锚定布局：相比于完全随机或简单的质量排序，该策略通过动态窗口搜索，确保每个训练 batch 内的样本在母离子质量（precursor mass）上尽量接近，
   同时相邻的 batch 之间在质量空间上也具有连续性。这有助于模型在具有相似物理特性的样本上进行对比学习。
2. 动态窗口扩展：在构建 batch 时，如果当前质量窗口内的样本不足一个 batch_size，算法会自动按比例扩展窗口（expand_factor），直到满足数量要求或达到上限（max_window_da）。
3. 批内肽段去重：通过向后搜索并交换样本，尽量确保同一个 batch 内不出现重复的肽段（peptide），从而增加训练的多样性。
4. 离线重排与随机化写出：将所有输入文件加载并建立全局索引，计算重排后的全局顺序，最后随机挑选指定数量的 batch 并并行写出为独立的小 pkl 文件。
"""

import argparse  # 导入参数解析库
import gzip  # 导入 gzip 支持 .pkl.gz
import glob  # 导入文件匹配库
import os  # 导入操作系统接口库
import pickle  # 导入序列化库
import concurrent.futures  # 导入并发库，用于多线程保存
from typing import Dict, List  # 导入类型提示

import numpy as np  # 导入数值计算库

# 定义命令行参数解析函数
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 train_all 的 pkl 离线重排为质量锚定布局（train_all_rerank）"
    )
    # 输入原始 pkl 文件的所在目录
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/pkl_dataset/train",
        help="原始训练 pkl 目录",
    )
    # 输出重排后 pkl 文件的目录
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/pkl_dataset/train_all_rerank",
        help="输出重排后的 pkl 目录",
    )
    # 每个训练批次的大小
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="训练 batch 大小；重排后每个连续 batch 会尽量质量相近",
    )
    # 查找质量相近样本时的初始搜索半径（道尔顿 Da）
    parser.add_argument(
        "--start_window_da",
        type=float,
        default=1.0,
        help="初始质量窗口（Da）",
    )
    # 当窗口内样本不足时，窗口宽度扩大的倍数
    parser.add_argument(
        "--expand_factor",
        type=float,
        default=2.0,
        help="候选不足时窗口扩展倍数",
    )
    # 质量窗口搜索的最大限制
    parser.add_argument(
        "--max_window_da",
        type=float,
        default=64.0,
        help="质量窗口最大上限（Da）",
    )
    # 默认打乱每个 batch 内部顺序，无需参数
    # 随机种子，确保结果可复现
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="随机种子",
    )
    # 调试用：限制读取的原始文件数量
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="仅处理前 N 个文件（0 表示全部）",
    )
    # 是否允许直接覆盖已存在的输出文件夹内容
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖输出目录内同名文件",
    )
    # 如果最后剩余的样本不足一个 batch_size，是否丢弃
    parser.add_argument(
        "--drop_last",
        action="store_true",
        help="是否丢弃最后一个不足 batch_size 的尾批次",
    )
    # 输出文件名的前缀，如 batch_0000.pkl
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="batch",
        help="输出小 pkl 文件名前缀",
    )
    # 并行保存文件时的线程数
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="保存时的并行进程数",
    )
    # 从所有生成的 batch 中随机抽取多少个进行保存
    parser.add_argument(
        "--num_batches",
        type=int,
        default=4000,
        help="随机挑选生成的 batch 数量（0 表示全部）",
    )
    # 如果已有部分文件，是否跳过它们
    parser.add_argument(
        "--resume",
        action="store_true",
        help="开启断点续传，跳过已存在的 batch 文件",
    )
    return parser.parse_args()


# 构建质量锚定顺序
def build_mass_anchored_order(
    masses: np.ndarray,  # 全局质量数组
    batch_size: int,  # 批大小
    start_window_da: float,  # 初始窗口
    expand_factor: float,  # 扩展因子
    max_window_da: float,  # 最大窗口
    rng: np.random.Generator,  # 随机数生成器
    shuffle_within_batch: bool,  # 批内打乱
) -> np.ndarray:
    """按质量排序后构造连续 batch，候选不足时自动扩窗。"""
    if masses.ndim != 1:  # 必须是一维数组
        raise ValueError(f"masses 必须是一维数组，当前 shape={masses.shape}")
    if batch_size <= 0:  # 批大小必须合法
        raise ValueError("batch_size 必须大于 0")

    n = masses.shape[0]  # 总样本数
    if n == 0:  # 无样本处理
        return np.empty((0,), dtype=np.int64)

    # 对所有质量进行升序排序，并获取索引（使用归并排序保证稳定性）
    sorted_idx = np.argsort(masses, kind="mergesort")
    sorted_masses = masses[sorted_idx]  # 排序后的质量列表

    ordered: List[np.ndarray] = []  # 存放生成的 batch 索引块
    cursor = 0  # 当前处理到的全局位置指针

    while cursor < n:  # 循环分配样本直到耗尽
        anchor_mass = float(sorted_masses[cursor])  # 以当前最轻的样本质量为锚点
        window = max(float(start_window_da), 1e-6)  # 设定初始搜索窗口

        right = cursor  # 确定当前 batch 的右边界
        while True:
            upper = anchor_mass + window  # 计算窗口上限
            # 使用二分查找快速定位窗口内样本的边界
            new_right = int(np.searchsorted(sorted_masses, upper, side="right"))
            
            if (new_right - cursor) >= batch_size:  # 如果窗口内样本数已经足够一个 batch
                right = cursor + batch_size  # 取走 batch_size 个样本
                break

            right = new_right  # 暂时将右边界设为当前窗口末尾
            if right >= n:  # 如果已经到达数据末尾
                right = n
                break

            if window >= max_window_da:  # 如果窗口已经扩到最大，强制结束
                break

            window = min(window * expand_factor, max_window_da)  # 否则，扩大窗口寻找更多样本

        if right <= cursor:  # 防御性代码，防止死循环
            right = min(cursor + batch_size, n)

        chunk = sorted_idx[cursor:right].copy()  # 提取该 batch 的索引序列
        if shuffle_within_batch and chunk.size > 1:  # 如果需要，打乱该 batch 内部的顺序
            rng.shuffle(chunk)

        ordered.append(chunk)  # 将该 batch 存入列表
        cursor = right  # 指针移动到下一个位置

    return np.concatenate(ordered, axis=0)  # 返回拼接后的全局重排序列


# 根据新顺序重新排列 payload 字典数据的辅助工具
def reorder_payload(payload: Dict, order: np.ndarray) -> Dict:
    out: Dict = {}
    n = order.shape[0]
    for k, v in payload.items():
        if isinstance(v, np.ndarray):  # 处理数组
            if v.shape[0] == n:
                out[k] = v[order]
            else:
                out[k] = v
            continue
        if isinstance(v, list):  # 处理列表
            if len(v) == n:
                out[k] = [v[i] for i in order.tolist()]
            else:
                out[k] = v
            continue
        out[k] = v  # 其它元数据直接复制
    return out


# 检查输出目录有效性
def _check_output_dir(output_dir: str, overwrite: bool, resume: bool = False) -> None:
    if not os.path.exists(output_dir):  # 目录不存在则合法
        return
    if resume:  # 断点续传不需要检查覆盖
        return
    has_pkl = any(name.endswith(".pkl") for name in os.listdir(output_dir))
    if has_pkl and (not overwrite):  # 防止非故意覆盖
        raise FileExistsError(f"输出目录已有 pkl 文件: {output_dir}，如需覆盖请加 --overwrite")


# 从 payload 中提取肽段标识符，用于去重
def _extract_peptide_ids(payload: Dict, n: int) -> np.ndarray:
    if "peptides" in payload:  # 优先使用明确的肽段字符串
        peps = np.asarray(payload["peptides"], dtype=object)
        if peps.shape[0] != n:
            raise ValueError(f"peptides 长度异常")
        return peps

    if "tokens" in payload:  # 如果没有字符串，则将 token 列表转为 bytes 作为哈希键
        tokens = np.asarray(payload["tokens"])
        if tokens.shape[0] != n:
            raise ValueError(f"tokens 长度异常")
        return np.array([tokens[i].tobytes() for i in range(n)], dtype=object)

    raise KeyError("数据中缺少肽段信息，无法去重")


# 批量读取原始文件，将数据聚合
def _load_all_payloads(files: List[str]) -> tuple[List[Dict], List[int], np.ndarray, np.ndarray]:
    payloads: List[Dict] = []  # 存放所有原始字典
    lengths: List[int] = []  # 存放每个文件的样本数
    all_masses: List[np.ndarray] = []  # 聚合所有质量
    all_peptide_ids: List[np.ndarray] = []  # 聚合所有肽段 ID

    for i, in_file in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] 读取 {os.path.basename(in_file)}")
        open_fn = gzip.open if in_file.endswith(".gz") else open
        with open_fn(in_file, "rb") as f:
            payload = pickle.load(f)  # 加载 pkl

        if "precursors" not in payload:
            raise KeyError(f"文件缺少 precursors 字段: {in_file}")

        precursors = np.asarray(payload["precursors"])
        n = int(precursors.shape[0])
        peptide_ids = _extract_peptide_ids(payload, n)  # 提取 ID
        
        payloads.append(payload)
        lengths.append(n)
        all_masses.append(np.asarray(precursors[:, 0], dtype=np.float64))  # 提取第 0 列作为主质量
        all_peptide_ids.append(peptide_ids)

    if not all_masses:
        return payloads, lengths, np.empty((0,), dtype=np.float64), np.empty((0,), dtype=object)

    # 垂直拼接所有加载的数据
    return (
        payloads,
        lengths,
        np.concatenate(all_masses, axis=0),
        np.concatenate(all_peptide_ids, axis=0),
    )


# 批内肽段多样性增强策略
def _enhance_batch_peptide_diversity(
    global_order: np.ndarray,  # 全局顺序
    peptide_ids: np.ndarray,  # 全局 ID
    batch_size: int,  # 批大小
    max_search_batches: int = 8,  # 向后搜索进行交换的最大范围
) -> np.ndarray:
    n_total = global_order.shape[0]
    if n_total == 0: return global_order

    order = global_order.copy() # 拷贝一份顺序，避免原地修改
    search_limit = max(batch_size * max_search_batches, batch_size) # 计算最大搜索范围（样本数）

    for s in range(0, n_total, batch_size):  # 按 batch 遍历
        e = min(s + batch_size, n_total)
        if (e - s) <= 1: continue # 如果 batch 只有 1 个或为空，跳过

        seen = set()  # 记录当前 batch 已出现的肽段
        dup_pos: List[int] = []  # 记录 batch 内重复肽段的样本位置
        unique_in_batch = set() # 记录 batch 内所有唯一肽段

        # 遍历 batch 内每个样本，判断肽段是否重复。
        for p in range(s, e):
            pep = peptide_ids[order[p]]
            if pep in seen:
                dup_pos.append(p)  # 标记为重复
            else:
                seen.add(pep)
                unique_in_batch.add(pep)

        if not dup_pos: continue  # 无重复则跳过

        k = e  # 从下一个 batch 开始搜索可交换的对象
        k_end = min(n_total, e + search_limit) # 限制最大查找范围
        for p in dup_pos:  # 尝试替换每一个重复样本
            found = False # 标记是否找到可交换的样本
            while k < k_end:
                cand_pep = peptide_ids[order[k]] # 获取候选样本的肽段 ID
                if cand_pep not in unique_in_batch:  # 如果后续库里的样本在当前 batch 没出现过
                    order[p], order[k] = order[k], order[p]  # 交换
                    unique_in_batch.add(cand_pep)
                    found = True
                    k += 1
                    break
                k += 1
            if not found: continue

    return order


# 确定哪些 key 是逐行对应的（即每个样本一个值）
# 后续在重新组装 batch 时，可以只对这些 key 做行级索引提取，保证数据对齐
def _row_aligned_keys(payloads: List[Dict], lengths: List[int]) -> List[str]:
    keys = list(payloads[0].keys())
    aligned: List[str] = []
    for k in keys:
        ok = True
        for payload, n in zip(payloads, lengths):
            if k not in payload:
                ok = False; break
            v = payload[k]
            # 简单判断特征：长度与样本数相等
            if isinstance(v, np.ndarray):
                if v.ndim == 0 or v.shape[0] != n: ok = False; break
            elif isinstance(v, list):
                if len(v) != n: ok = False; break
            else: ok = False; break
        if ok: aligned.append(k)
    return aligned


# 从多个源 payload 中捞取数组类型的样本
def _gather_shard_value_ndarray(
    key: str, payloads: List[Dict], file_ids: np.ndarray, local_ids: np.ndarray,
) -> np.ndarray:
    template = np.asarray(payloads[int(file_ids[0])][key])
    out_shape = (file_ids.shape[0],) + tuple(template.shape[1:])
    out = np.empty(out_shape, dtype=template.dtype)

    for fid in np.unique(file_ids):  # 按文件 ID 分组，减少访问开销
        pos = np.where(file_ids == fid)[0]
        src = np.asarray(payloads[int(fid)][key])
        out[pos] = src[local_ids[pos]]
    return out


# 从多个源 payload 中捞取列表类型的样本
def _gather_shard_value_list(
    key: str, payloads: List[Dict], file_ids: np.ndarray, local_ids: np.ndarray,
) -> List:
    out = [None] * file_ids.shape[0]
    for fid in np.unique(file_ids):
        pos = np.where(file_ids == fid)[0]
        src = payloads[int(fid)][key]
        for p in pos:
            out[int(p)] = src[int(local_ids[int(p)])]
    return out


# 构造一个 batch 的完整字典
def _gather_batch_payload(
    payloads: List[Dict], aligned_keys: List[str], file_ids: np.ndarray, local_ids: np.ndarray,
) -> Dict:
    out_payload: Dict = {}
    for k in payloads[0].keys():
        if k in aligned_keys:  # 如果是逐行数据，则按索引提取
            sample_value = payloads[0][k]
            if isinstance(sample_value, np.ndarray):
                out_payload[k] = _gather_shard_value_ndarray(k, payloads, file_ids, local_ids)
            else:
                out_payload[k] = _gather_shard_value_list(k, payloads, file_ids, local_ids)
        else:  # 如果是全局元数据（如 config），则直接取第一个
            out_payload[k] = payloads[0][k]
    return out_payload


# 并行任务单元：提取并保存一个 batch 文件
def _save_one_batch(
    out_i: int,
    batch_idx: int,
    batch_ranges: List[tuple[int, int]],
    src_file_ids: np.ndarray,
    src_local_ids: np.ndarray,
    payloads: List[Dict],
    aligned_keys: List[str],
    output_dir: str,
    output_prefix: str,
) -> None:
    s, e = batch_ranges[int(batch_idx)]
    seg_file_ids = src_file_ids[s:e]
    seg_local_ids = src_local_ids[s:e]
    # 提取数据
    out_payload = _gather_batch_payload(payloads, aligned_keys, seg_file_ids, seg_local_ids)

    out_file = os.path.join(output_dir, f"{output_prefix}_{out_i:07d}.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(out_payload, f, protocol=pickle.HIGHEST_PROTOCOL)


# 分块写出管理函数
def _write_batches_as_small_pkls(
    output_dir: str,
    output_prefix: str,
    payloads: List[Dict],
    lengths: List[int],
    global_order: np.ndarray,
    batch_size: int,
    drop_last: bool,
    rng: np.random.Generator,
    num_workers: int = 4,
    resume: bool = False,
    num_batches_to_save: int = 0,
) -> None:
    aligned_keys = _row_aligned_keys(payloads, lengths)
    n_total = int(np.sum(lengths))
    if n_total == 0 or global_order.size == 0: return

    # 计算全局偏移以便定位文件
    starts = np.cumsum([0] + lengths)
    src_file_ids = np.searchsorted(starts[1:], global_order, side="right").astype(np.int64)
    src_local_ids = (global_order - starts[src_file_ids]).astype(np.int64)

    # 划分 batch 范围
    n_batch = n_total // batch_size
    if (not drop_last) and (n_total % batch_size) > 0: n_batch += 1

    batch_ranges: List[tuple[int, int]] = []
    for i in range(n_batch):
        s, e = i * batch_size, min((i + 1) * batch_size, n_total)
        if (e - s) < batch_size and drop_last: continue
        batch_ranges.append((s, e))

    if not batch_ranges: return

    # 随机打乱输出顺序，以便后续即便只选取部分 pkl，也是随机抽样的
    order = np.arange(len(batch_ranges), dtype=np.int64)
    rng.shuffle(order)

    # 如果有抽取限制
    if num_batches_to_save > 0:
        order = order[:num_batches_to_save]

    pending_tasks = []
    for out_i, batch_idx in enumerate(order):
        out_file = os.path.join(output_dir, f"{output_prefix}_{out_i:07d}.pkl")
        if resume and os.path.exists(out_file): continue
        pending_tasks.append((out_i, int(batch_idx)))

    if not pending_tasks:
        print("所有文件已存在。"); return

    print(f"待处理批次: {len(pending_tasks)}，并行写出中...")
    
    # 使用线程池执行保存任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_save_one_batch, out_i, b_idx, batch_ranges, src_file_ids, src_local_ids, payloads, aligned_keys, output_dir, output_prefix): out_i for out_i, b_idx in pending_tasks}
        
        try:
            from tqdm import tqdm
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Saving batches"):
                fut.result()
        except ImportError:
            for i, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
                fut.result()
                if i % 100 == 0: print(f"已保存 {i}/{len(pending_tasks)}")


# 主入口
def main() -> None:
    args = parse_args()

    # 1. 扫描文件
    pkl_files = glob.glob(os.path.join(args.input_dir, "*.pkl"))
    pkl_gz_files = glob.glob(os.path.join(args.input_dir, "*.pkl.gz"))
    files = sorted(set(pkl_files + pkl_gz_files))
    if not files: raise FileNotFoundError(f"输入目录为空")
    if args.max_files > 0: files = files[:args.max_files]

    # 2. 检查目录
    os.makedirs(args.output_dir, exist_ok=True)
    _check_output_dir(args.output_dir, args.overwrite, args.resume)

    print(f"输入文件数: {len(files)}, 设置: batch={args.batch_size}")

    # 3. 读取并聚合所有数据到内存
    payloads, lengths, all_masses, all_peptide_ids = _load_all_payloads(files)
    total_samples = int(all_masses.shape[0])
    print(f"总样本数: {total_samples}")

    # 4. 执行质量锚定重排
    rng = np.random.default_rng(args.seed)
    global_order = build_mass_anchored_order(
        masses=all_masses, batch_size=args.batch_size, 
        start_window_da=args.start_window_da, expand_factor=args.expand_factor, 
        max_window_da=args.max_window_da, rng=rng, shuffle_within_batch=True
    )

    # 5. 执行肽段去重增强
    print("去重增强中...")
    global_order = _enhance_batch_peptide_diversity(
        global_order=global_order, peptide_ids=all_peptide_ids, batch_size=args.batch_size
    )

    # 6. 保存为小文件
    print("保存小文件中...")
    _write_batches_as_small_pkls(
        output_dir=args.output_dir, output_prefix=args.output_prefix,
        payloads=payloads, lengths=lengths, global_order=global_order,
        num_batches_to_save=args.num_batches, batch_size=args.batch_size,
        drop_last=args.drop_last, rng=rng, num_workers=args.num_workers, resume=args.resume
    )

    print("完成！")


if __name__ == "__main__":
    main()
