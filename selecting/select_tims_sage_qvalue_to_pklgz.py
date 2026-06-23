"""
功能：per-file Sage q-value筛选timsTOF PSM，构造train/val pkl.gz（3阶段：sage lookup→对齐pkl→materialize）
输入：
    --tims_root /home/yiwen/AIPC/database/tims
    --pkl_dir .../pkl_dataset_tims/tims_pkl_all
    --train_output_dir .../data/tims_sage_select/train
    --val_output_dir .../data/tims_sage_select/val
输出：
    data/dataset/tims_sage_select/train/train.XXXXX.pkl.gz
    data/dataset/tims_sage_select/val/val.XXXXX.pkl.gz
"""

import argparse
import gzip
import glob
import heapq
import os
import pickle
import random
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# 质子质量常量（与 model.transformer.dataset 完全一致）
PROTON_MASS_AMU = 1.007276


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按 per-file sage_discriminant_score q-value 筛选 pkl.gz 数据"
    )
    parser.add_argument(
        "--tims_root", type=str,
        default="/home/yiwen/AIPC/database/tims",
        help="原始 tims parquet 目录（含 sage_discriminant_score）",
    )
    parser.add_argument(
        "--pkl_dir", type=str,
        default="/home/yiwen/AIPC/scripts/attantion/pkl_dataset_tims/tims_pkl_all",
        help="pkl.gz 文件根目录，内含 train/ 和 val/ 子目录",
    )
    parser.add_argument(
        "--score_col", type=str,
        default="sage_discriminant_score",
        help="用于计算 q-value 的列名",
    )
    parser.add_argument(
        "--train_output_dir", type=str,
        default="/home/yiwen/AIPC/scripts/organized_attantion/data/tims_sage_select/train",
        help="train 分块输出目录",
    )
    parser.add_argument(
        "--val_output_dir", type=str,
        default="/home/yiwen/AIPC/scripts/organized_attantion/data/tims_sage_select/val",
        help="val 分块输出目录",
    )
    parser.add_argument("--rows_per_file", type=int, default=1_000_000)
    parser.add_argument("--q_threshold", type=float, default=0.01)
    parser.add_argument("--val_target_count", type=int, default=50_000)
    parser.add_argument("--val_decoy_count", type=int, default=50_000)
    parser.add_argument(
        "--selected_decoy_count", type=int, default=0,
        help="decoy 半高半随筛选总数，0=自动匹配 qualifying target 数量 (上限 20M)",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pkl_compresslevel", type=int, default=6)
    parser.add_argument("--max_orig_files", type=int, default=0,
                        help="限制处理的原始 parquet 数量，0=全部")
    return parser.parse_args()


# ───────────────────── 工具 ─────────────────────

def _heap_push_topk(heap, item, k):
    if k <= 0:
        return
    if len(heap) < k:
        heapq.heappush(heap, item)
    elif item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


def _warn_if_short(got: int, need: int, tag: str) -> None:
    if got < need:
        print(f"警告: {tag} 需求 {need}，实际仅 {got}")


def _normalize_sequence(seq: str) -> str:
    """模拟 normalize_precursor_sequence_column：统一原始 parquet 与 pkl.gz 的 PTM 格式。"""
    s = str(seq)
    # N-term acetyl: [+42]- → n[42]
    s = s.replace("[+42]-", "n[42]")
    # Carbamidomethylation / Oxidation → 统一为 pkl.gz 的简写格式
    s = s.replace("C[+57.0216]", "C[57.02]")
    s = s.replace("M[+15.9949]", "M[15.99]")
    # Deamidation
    s = s.replace("N[+0.9840]", "N[.98]")
    s = s.replace("Q[+0.9840]", "Q[.98]")
    # Legacy aliases
    s = s.replace("cC", "C[57.02]")
    s = s.replace("oxM", "M[15.99]")
    s = s.replace("M(ox)", "M[15.99]")
    s = s.replace("deamN", "N[.98]")
    s = s.replace("deamQ", "Q[.98]")
    s = s.replace("a", "X")
    return s


def _clean_sequence(seq: str) -> str:
    """先做格式归一化，再去除所有修饰标签，得到纯氨基酸序列。"""
    if seq is None:
        return ""
    s = _normalize_sequence(seq)
    s = s.replace("n[42]", "")
    s = s.replace("N[.98]", "N").replace("Q[.98]", "Q")
    s = s.replace("M[15.99]", "M").replace("C[57.02]", "C")
    s = re.sub(r"\[[^\]]+\]", "", s)
    return s


def _to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    if pd.isna(x):
        return []
    return [x]


# ──────────── per-file q-value 计算 ────────────

def compute_q_values(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """对单个文件的 scores/labels 计算 per-PSM q-value (降序→累积FDR→suffix-min)。"""
    n = len(scores)
    if n == 0:
        return np.empty(0, dtype=np.float32)

    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_labels = labels[order]
    is_target = (sorted_labels > 0.5)

    cum_t = np.cumsum(is_target, dtype=np.int64)
    cum_d = np.arange(1, n + 1, dtype=np.int64) - cum_t

    with np.errstate(divide="ignore", invalid="ignore"):
        fdr = np.where(cum_t > 0, cum_d.astype(np.float64) / cum_t.astype(np.float64), np.inf)
    q_sorted = np.minimum.accumulate(fdr[::-1])[::-1]

    q_values = np.empty(n, dtype=np.float32)
    q_values[order] = q_sorted.astype(np.float32)
    return q_values


# ──── 阶段A: 从原始 parquet 展平 PSM，建 (key → sage_score + orig_fid) 映射 ────

def build_psm_sage_lookup(
    tims_root: str, score_col: str, max_files: int,
) -> Dict[Tuple, Tuple[float, float]]:
    """遍历所有原始 parquet，展平为 PSM 级别，建立查找表。

    每个原始文件独立计算 per-file q-value（基于全量 PSM），q-value 随 sage_score 一同存入 lookup。

    Returns:
        lookup: {(cleaned_pep, precursor_mass, charge, label): (sage_score, q_value)}
    """
    parquet_files = sorted(glob.glob(os.path.join(tims_root, "*.parquet")))
    if max_files > 0:
        parquet_files = parquet_files[:max_files]

    print(f"原始 parquet 文件: {len(parquet_files)} 个")

    lookup: Dict[Tuple, Tuple[float, float]] = {}
    total_psms = 0

    for fid, parquet_path in enumerate(tqdm(parquet_files, desc="展平+per-file q-value")):
        df = pd.read_parquet(parquet_path)

        # 先收集该文件所有 PSM 的 (sage_score, label, key)
        file_psms: List[Tuple[float, float, Tuple]] = []

        for row in df.itertuples(index=False):
            base_mz = getattr(row, "precursor_mz", 0.0)
            peptides = _to_list(getattr(row, "peptide"))
            charges = _to_list(getattr(row, "charge"))
            labels = _to_list(getattr(row, "label"))
            scores = _to_list(getattr(row, score_col))

            cand_len = min(len(peptides), len(charges), len(labels), len(scores))
            if cand_len == 0:
                continue

            for i in range(cand_len):
                pep = peptides[i]
                if pep is None or (isinstance(pep, float) and np.isnan(pep)):
                    continue
                try:
                    chg = int(float(charges[i]))
                    lbl = int(float(labels[i]))
                    sc = float(scores[i])
                except (ValueError, TypeError):
                    continue
                if pd.isna(sc) or pd.isna(chg) or chg <= 0:
                    continue

                cleaned = _clean_sequence(str(pep))
                if len(cleaned) == 0:
                    continue

                # 模拟 collate 中 float64→float32 计算路径：
                # torch.tensor(np.float64)→float64 tensor → 计算 → .float()→float32
                precursor_mass = np.float32(
                    (np.float64(base_mz) - np.float64(PROTON_MASS_AMU)) * np.float64(chg)
                )
                mass_rounded = round(float(precursor_mass), 3)

                key = (cleaned, mass_rounded, chg, lbl)
                file_psms.append((sc, lbl, key))

        total_psms += len(file_psms)

        # 在该文件内计算 per-file q-value
        if file_psms:
            scores_arr = np.array([s for s, _, _ in file_psms], dtype=np.float32)
            labels_arr = np.array([l for _, l, _ in file_psms], dtype=np.float32)
            qvs = compute_q_values(scores_arr, labels_arr)

            for i, (sc, lbl, key) in enumerate(file_psms):
                qv = float(qvs[i])
                # 同一 key 多次出现时，保留 sage_score 更高的那个
                if key not in lookup or sc > lookup[key][0]:
                    lookup[key] = (sc, qv)

        del df, file_psms

    print(f"总 PSM: {total_psms}, 唯一 key: {len(lookup)}")
    return lookup


# ──── 阶段B: 遍历 pkl.gz，对齐 sage_score，计算全域 q-value 并选择 ────

def align_and_select(
    pkl_dir: str,
    sage_lookup: Dict[Tuple, Tuple[float, float]],
    q_threshold: float,
    val_target_count: int,
    val_decoy_count: int,
    selected_decoy_count: int,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], int, int, int, int]:
    """遍历 pkl.gz 文件，对齐 sage 分数/q-value，执行选择逻辑，返回 per-file keep_mask。

    q-value 来自阶段A在原始文件全量PSM上预计算的结果，不再重新计算。
    """
    pkl_files = sorted(glob.glob(os.path.join(pkl_dir, "*.pkl.gz")))
    for sub in ["train", "val"]:
        sub_dir = os.path.join(pkl_dir, sub)
        if os.path.isdir(sub_dir):
            pkl_files.extend(sorted(glob.glob(os.path.join(sub_dir, "*.pkl.gz"))))

    if not pkl_files:
        raise FileNotFoundError(f"未找到 pkl.gz: {pkl_dir}")
    print(f"pkl.gz 文件: {len(pkl_files)} 个")

    # ── 收集所有 PSMs 的 sage_score、per-file q_value、label ──
    all_scores: List[float] = []
    all_qvalues: List[float] = []
    all_labels: List[int] = []
    all_pkl_indices: List[Tuple[int, int]] = []  # (pkl_fid, row_idx)
    pkl_row_counts: List[int] = []

    for pkl_fid, pkl_path in enumerate(tqdm(pkl_files, desc="对齐 sage score + q-value")):
        with gzip.open(pkl_path, "rb") as f:
            data = pickle.load(f)

        peptides = data["peptides"]
        precursors = data["precursors"]  # (N, 4): [mass, charge, delta_rt, predicted_rt]
        labels_arr = data["label"]
        n = len(labels_arr)
        pkl_row_counts.append(n)

        for i in range(n):
            pep = peptides[i]
            cleaned = _clean_sequence(str(pep))
            mass = round(float(precursors[i, 0]), 3)
            chg = int(precursors[i, 1])
            lbl = int(labels_arr[i] > 0.5)
            key = (cleaned, mass, chg, lbl)

            entry = sage_lookup.get(key)
            if entry is not None:
                sage_sc, qv = entry
            else:
                sage_sc, qv = -1.0, 1.0

            all_scores.append(sage_sc)
            all_qvalues.append(qv)
            all_labels.append(lbl)
            all_pkl_indices.append((pkl_fid, i))

    total_psms = len(all_scores)
    scores_arr = np.array(all_scores, dtype=np.float32)
    q_values_arr = np.array(all_qvalues, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.float32)

    matched = (scores_arr >= 0).sum()
    print(f"对齐完成: {total_psms} PSMs, matched={matched}, unmatched={total_psms - matched}")

    # ── 确定 selected decoy 总量 ──
    qualifying_targets = int(((labels_arr > 0.5) & (q_values_arr <= q_threshold)).sum())
    print(f"qualifying targets (per-file q<={q_threshold}): {qualifying_targets}")

    if selected_decoy_count > 0:
        sel_decoy_total = selected_decoy_count
    else:
        sel_decoy_total = min(max(qualifying_targets, val_decoy_count), 20_000_000)
        print(f"[auto] selected_decoy_count = {sel_decoy_total}")

    high_n = sel_decoy_total // 2
    rand_n = sel_decoy_total - high_n

    print(
        f"采样计划: target per-file-q<={q_threshold} → val{val_target_count}, 其余→train; "
        f"decoy(共选{sel_decoy_total}: 高分{high_n}+随机{rand_n}) → val{val_decoy_count}, 其余→train"
    )

    # ── 选择逻辑 ──
    rng = random.Random(seed)

    # val target: reservoir 从 qualified targets 中抽
    t_seen = 0
    val_target_selected: List[int] = []  # global indices
    qual_target_indices = np.where((labels_arr > 0.5) & (q_values_arr <= q_threshold))[0]
    for gi in qual_target_indices:
        t_seen += 1
        if len(val_target_selected) < val_target_count:
            val_target_selected.append(int(gi))
        else:
            j = rng.randint(0, t_seen - 1)
            if j < val_target_count:
                val_target_selected[j] = int(gi)

    # decoy high: heap top-k by sage_score
    decoy_high_heap: List[Tuple[float, int]] = []
    decoy_indices = np.where(labels_arr <= 0.5)[0]
    for gi in decoy_indices:
        _heap_push_topk(decoy_high_heap, (float(scores_arr[gi]), int(gi)), high_n)
    decoy_high_list = sorted(decoy_high_heap, key=lambda x: x[0], reverse=True)
    decoy_high_set = {gi for _, gi in decoy_high_list}

    # decoy random: reservoir 排除 decoy_high
    high_thr = min((s for s, _ in decoy_high_list), default=float("inf"))
    decoy_random_list: List[int] = []
    d_seen = 0
    rng2 = random.Random(seed + 1)
    for gi in decoy_indices:
        if gi in decoy_high_set:
            continue
        sc = float(scores_arr[gi])
        if sc > high_thr:
            continue
        d_seen += 1
        if len(decoy_random_list) < rand_n:
            decoy_random_list.append(int(gi))
        else:
            j = rng2.randint(0, d_seen - 1)
            if j < rand_n:
                decoy_random_list[j] = int(gi)

    all_selected_decoys_gi = [gi for _, gi in decoy_high_list] + decoy_random_list
    rng3 = random.Random(seed + 2)
    val_decoy_selected = rng3.sample(
        all_selected_decoys_gi,
        min(val_decoy_count, len(all_selected_decoys_gi)),
    )
    val_decoy_set = set(val_decoy_selected)
    train_decoy_selected = [gi for gi in all_selected_decoys_gi if gi not in val_decoy_set]

    # train targets: qualified 排除 val
    val_target_set = set(val_target_selected)
    train_target_selected = [int(gi) for gi in qual_target_indices if gi not in val_target_set]

    _warn_if_short(len(val_target_selected), val_target_count, "val_target")
    _warn_if_short(len(decoy_high_list), high_n, "decoy_high")
    _warn_if_short(len(decoy_random_list), rand_n, "decoy_random")
    _warn_if_short(len(val_decoy_selected), val_decoy_count, "val_decoy")

    print(f"val_targets={len(val_target_selected)}, val_decoys={len(val_decoy_selected)}, "
          f"train_targets={len(train_target_selected)}, train_decoys={len(train_decoy_selected)}")

    # ── 构建 per-pkl keep_mask ──
    pkl_keep_masks: Dict[str, np.ndarray] = {}

    # 收集所有要保留的 global index，按 pkl_fid 分组
    keep_by_pkl: Dict[int, List[int]] = defaultdict(list)
    all_keep_gi = set(val_target_selected)
    all_keep_gi.update(val_decoy_selected)
    all_keep_gi.update(train_target_selected)
    all_keep_gi.update(train_decoy_selected)

    for gi in all_keep_gi:
        pfid, ridx = all_pkl_indices[gi]
        keep_by_pkl[pfid].append(ridx)

    for pkl_fid, n_rows in enumerate(pkl_row_counts):
        pkl_path = pkl_files[pkl_fid]
        mask = np.zeros(n_rows, dtype=bool)
        rows_to_keep = keep_by_pkl.get(pkl_fid, [])
        if rows_to_keep:
            mask[rows_to_keep] = True
        pkl_keep_masks[pkl_path] = mask

    n_train_t = len(train_target_selected)
    n_train_d = len(train_decoy_selected)
    n_val_t = len(val_target_selected)
    n_val_d = len(val_decoy_selected)

    return pkl_keep_masks, n_train_t, n_train_d, n_val_t, n_val_d


# ──── 阶段C: 从 pkl.gz 回捞并写出 ────

class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _load_pklgz(path: str):
    with gzip.open(path, "rb") as f:
        return _CompatUnpickler(f).load()


def _find_row_aligned_keys(payload):
    n = len(payload["label"])
    return [k for k, v in payload.items()
            if (isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] == n)
            or (isinstance(v, (list, tuple)) and len(v) == n)]


def _finalize_payload(first_payload, store, keys):
    out = {}
    ks = set(keys)
    for k, v in first_payload.items():
        if k not in ks:
            out[k] = v
            continue
        if isinstance(v, np.ndarray):
            out[k] = np.empty((0,) + v.shape[1:], dtype=v.dtype) if len(store[k]) == 0 else np.concatenate(store[k], axis=0)
        elif isinstance(v, tuple):
            out[k] = tuple(store[k])
        else:
            out[k] = store[k]
    return out


def _shuffle_payload(payload, seed):
    n = len(payload["label"])
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    for k, v in list(payload.items()):
        if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] == n:
            payload[k] = v[order]
        elif isinstance(v, list) and len(v) == n:
            payload[k] = [v[int(i)] for i in order]
        elif isinstance(v, tuple) and len(v) == n:
            payload[k] = tuple(v[int(i)] for i in order)
    return payload


def materialize_and_write(
    pkl_keep_masks: Dict[str, np.ndarray],
    pkl_files: List[str],
    train_output_dir: str,
    val_output_dir: str,
    rows_per_file: int,
    compresslevel: int,
    seed: int,
):
    """从 pkl.gz 回捞选中行，按 train/val 分别写分块 pkl.gz。

    根据文件路径中是否包含 'val' 来决定归属。"""
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(val_output_dir, exist_ok=True)

    # 分别收集 train 和 val 的行
    train_store = defaultdict(list)
    val_store = defaultdict(list)
    train_first = {}
    val_first = {}
    train_aligned = []
    val_aligned = []
    train_rows = 0
    val_rows = 0
    n_train_t, n_train_d = 0, 0
    n_val_t, n_val_d = 0, 0

    for pkl_path in tqdm(pkl_files, desc="回捞 pkl.gz"):
        mask = pkl_keep_masks.get(pkl_path)
        if mask is None or mask.sum() == 0:
            continue

        payload = _load_pklgz(pkl_path)
        idx_arr = np.where(mask)[0]

        is_val = "val" in os.path.basename(pkl_path).lower()

        if is_val:
            store = val_store
            first = val_first
            aligned = val_aligned
        else:
            store = train_store
            first = train_first
            aligned = train_aligned

        if not first:
            for k in payload:
                first[k] = payload[k]
            aligned[:] = _find_row_aligned_keys(payload)
            for k in aligned:
                _ = store[k]

        # 提取行
        n = len(payload["label"])
        for k in aligned:
            v = payload[k]
            if isinstance(v, np.ndarray):
                store[k].append(v[idx_arr])
            elif isinstance(v, (list, tuple)):
                store[k].extend([v[int(i)] for i in idx_arr])

        lbl = payload["label"]
        if isinstance(lbl, np.ndarray):
            batch_labels = lbl[idx_arr]
            if is_val:
                n_val_t += int((batch_labels > 0.5).sum())
                n_val_d += int((batch_labels <= 0.5).sum())
                val_rows += len(idx_arr)
            else:
                n_train_t += int((batch_labels > 0.5).sum())
                n_train_d += int((batch_labels <= 0.5).sum())
                train_rows += len(idx_arr)

    # 写出 train
    if train_first:
        train_payload = _finalize_payload(train_first, train_store, train_aligned)
        train_payload = _shuffle_payload(train_payload, seed + 100)
        _write_chunks(train_payload, train_output_dir, "train", rows_per_file, compresslevel)

    # 写出 val
    if val_first:
        val_payload = _finalize_payload(val_first, val_store, val_aligned)
        val_payload = _shuffle_payload(val_payload, seed + 101)
        _write_chunks(val_payload, val_output_dir, "val", rows_per_file, compresslevel)

    print(f"train: target={n_train_t}, decoy={n_train_d}, total={train_rows} → {train_output_dir}")
    print(f"val:   target={n_val_t}, decoy={n_val_d}, total={val_rows} → {val_output_dir}")


def _write_chunks(payload, out_dir, prefix, rows_per_file, compresslevel):
    total = len(payload["label"])
    if total == 0:
        return
    cid = 0
    for s in range(0, total, rows_per_file):
        e = min(s + rows_per_file, total)
        path = os.path.join(out_dir, f"{prefix}.{cid:05d}.pkl.gz")
        chunk = {}
        n = len(payload["label"])
        for k, v in payload.items():
            if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] == n:
                chunk[k] = v[s:e]
            elif isinstance(v, list) and len(v) == n:
                chunk[k] = v[s:e]
            elif isinstance(v, tuple) and len(v) == n:
                chunk[k] = tuple(v[s:e])
            else:
                chunk[k] = v
        with gzip.open(path, "wb", compresslevel=compresslevel) as f:
            pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
        cid += 1


# ───────────────────── 主流程 ─────────────────────

def main():
    args = parse_args()

    if not (0.0 <= args.q_threshold <= 1.0):
        raise ValueError("q_threshold 必须在 [0,1]")
    for v in ["val_target_count", "val_decoy_count"]:
        if getattr(args, v) < 0:
            raise ValueError(f"{v} 必须 >= 0")

    # 检查输出目录
    for out_dir in [args.train_output_dir, args.val_output_dir]:
        if os.path.exists(out_dir) and os.listdir(out_dir):
            if not args.overwrite:
                raise FileExistsError(f"输出目录非空: {out_dir}，请加 --overwrite")

    # ── 阶段A: 从原始 parquet 建 sage_score + q-value 查找表 ──
    print("=" * 60)
    print("阶段A: 从原始 parquet 提取 sage_discriminant_score 并计算 per-file q-value")
    print("=" * 60)
    sage_lookup = build_psm_sage_lookup(
        args.tims_root, args.score_col, args.max_orig_files,
    )

    # ── 阶段B: 对齐到 pkl.gz 并选择 ──
    print("\n" + "=" * 60)
    print("阶段B: 对齐 sage_score + q-value 到 pkl.gz 并执行选择")
    print("=" * 60)
    pkl_keep_masks, nt, nd, vt, vd = align_and_select(
        args.pkl_dir,
        sage_lookup,
        args.q_threshold,
        args.val_target_count,
        args.val_decoy_count,
        args.selected_decoy_count,
        args.seed,
    )

    # ── 阶段C: 回捞并写出 ──
    print("\n" + "=" * 60)
    print("阶段C: 从 pkl.gz 回捞选中行并写出")
    print("=" * 60)
    all_pkl_files = sorted(glob.glob(os.path.join(args.pkl_dir, "*.pkl.gz")))
    for sub in ["train", "val"]:
        sub_dir = os.path.join(args.pkl_dir, sub)
        if os.path.isdir(sub_dir):
            all_pkl_files.extend(sorted(glob.glob(os.path.join(sub_dir, "*.pkl.gz"))))

    materialize_and_write(
        pkl_keep_masks,
        all_pkl_files,
        args.train_output_dir,
        args.val_output_dir,
        args.rows_per_file,
        args.pkl_compresslevel,
        args.seed,
    )

    print("\n全部完成。")


if __name__ == "__main__":
    main()
