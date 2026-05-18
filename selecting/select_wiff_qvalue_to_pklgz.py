"""
按全局 q-value 从 train_wiff_all_psm_score_epoch10 筛选 wiff_all 数据，卡 1% FDR。
所有 PSM 均作为训练集（不分割 val）。
"""

import argparse
import gzip
import glob
import heapq
import os
import pickle
import random
import sys
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按全局 q-value 卡 1% FDR 筛选 wiff_all，全部作为训练集输出"
    )
    parser.add_argument(
        "--score_dir", type=str,
        default="/home/yiwen/AIPC/scripts/organized_attantion/data/score/train_wiff_all_psm_score_epoch10",
        help="打分结果根目录，内含 train/ 和 val/ 子目录",
    )
    parser.add_argument(
        "--train_pkl_dir", type=str,
        default="/home/yiwen/AIPC/scripts/organized_attantion/data/dataset/wiff_all/train",
        help="train pkl 文件所在目录",
    )
    parser.add_argument(
        "--val_pkl_dir", type=str,
        default="/home/yiwen/AIPC/scripts/organized_attantion/data/dataset/wiff_all/val",
        help="val pkl 文件所在目录",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/home/yiwen/AIPC/scripts/organized_attantion/data/dataset/wiff_sage_select",
        help="输出目录",
    )
    parser.add_argument("--rows_per_file", type=int, default=1_000_000)
    parser.add_argument("--q_threshold", type=float, default=0.01)
    parser.add_argument("--selected_decoy_count", type=int, default=0,
                        help="decoy 半高半随筛选总数，0=自动匹配 qualifying target 数量 (上限 20M)")
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--max_score_files", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--compresslevel", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _open_maybe_gz(path: str, mode: str):
    if path.endswith(".gz"): return gzip.open(path, mode)
    return open(path, mode)


def discover_score_files(score_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(score_dir, "*_all_psm_score.tsv")))
    for sub in ["train", "val"]:
        sub_dir = os.path.join(score_dir, sub)
        if os.path.isdir(sub_dir):
            files.extend(sorted(glob.glob(os.path.join(sub_dir, "*_all_psm_score.tsv"))))
    if not files:
        raise FileNotFoundError(f"未找到 score 文件: {score_dir}")
    return files


def score_to_pkl_name(score_file: str) -> str:
    base = os.path.basename(score_file)
    prefix = base.replace("_all_psm_score.tsv", "")
    return f"{prefix}_val" if prefix.startswith("val.") else f"{prefix}_train"


def build_pkl_path_map(pkl_dir: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(pkl_dir, "*.pkl"))):
        out[os.path.basename(path).replace(".pkl", "")] = path
    for path in sorted(glob.glob(os.path.join(pkl_dir, "*.pkl.gz"))):
        out[os.path.basename(path).replace(".pkl.gz", "")] = path
    return out


# ───────────────────── 工具 ─────────────────────

def _heap_push_topk(heap, item, k):
    if k <= 0: return
    if len(heap) < k:
        heapq.heappush(heap, item)
    elif item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


# ──────────────────── PASS1: 全局 q-value（精确计算）────────────────────

def compute_global_qvalue_mapping(
    score_files, chunksize,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    score_chunks = []
    label_chunks = []

    for fid, score_path in enumerate(score_files):
        print(f"[PASS1] {fid + 1}/{len(score_files)} {os.path.basename(score_path)}")
        for chunk in pd.read_csv(score_path, sep="\t", usecols=["label", "model_score"], chunksize=chunksize):
            score_chunks.append(chunk["model_score"].to_numpy(dtype=np.float32, copy=True))
            label_chunks.append(chunk["label"].to_numpy(dtype=np.float32, copy=True))

    scores = np.concatenate(score_chunks, dtype=np.float32)
    labels = np.concatenate(label_chunks, dtype=np.float32)
    del score_chunks, label_chunks

    total_t = int((labels > 0.5).sum())
    total_d = int((labels <= 0.5).sum())
    print(f"[PASS1] 总计 {len(scores)} 条记录，开始排序...")

    order = np.argsort(scores, kind="mergesort")[::-1]
    scores_sorted = scores[order]
    is_target = (labels[order] > 0.5)
    del scores, labels, order

    cum_t = np.cumsum(is_target, dtype=np.int64)
    cum_d = np.arange(1, len(is_target) + 1, dtype=np.int64) - cum_t

    with np.errstate(divide="ignore", invalid="ignore"):
        fdr = np.where(cum_t > 0, cum_d.astype(np.float64) / cum_t.astype(np.float64), np.inf)
    q_val_all = np.minimum.accumulate(fdr[::-1])[::-1]

    q01_positions = np.where(q_val_all <= 0.01)[0]
    below = int(cum_t[q01_positions[-1]]) if len(q01_positions) > 0 else 0

    unique_scores, unique_idx = np.unique(scores_sorted, return_index=True)
    unique_q = q_val_all[unique_idx].astype(np.float32)
    neg_unique_scores = -unique_scores[::-1]
    unique_q = unique_q[::-1].astype(np.float32)

    del scores_sorted, is_target, cum_t, cum_d, fdr, q_val_all

    print(f"[qvalue] total target={total_t}, decoy={total_d}; targets@q<=0.01: {below}")
    return unique_scores, neg_unique_scores, unique_q, total_t, total_d, below


def score_to_q_value(score, neg_unique_scores, unique_q):
    sc = np.clip(score, 0.0, 1.0 - 1e-12)
    idx = np.searchsorted(neg_unique_scores, -sc, side="right")
    result = np.ones(sc.shape, dtype=np.float32)
    valid = idx > 0
    result[valid] = unique_q[idx[valid] - 1]
    return result


# ──────────────────── PASS2: decoy 高分筛选 ────────────────────

def pass2_decoy_high(
    score_files, decoy_high_count, chunksize,
) -> List[Tuple[float, int, int]]:
    heap: List[Tuple[float, int, int]] = []
    for fid, score_path in enumerate(score_files):
        print(f"[PASS2] {fid + 1}/{len(score_files)} {os.path.basename(score_path)}")
        for chunk in pd.read_csv(score_path, sep="\t",
                                  usecols=["original_index", "label", "model_score"],
                                  chunksize=chunksize):
            idx = chunk["original_index"].to_numpy(dtype=np.int64, copy=False)
            lbl = chunk["label"].to_numpy(dtype=np.float32, copy=False)
            sc = chunk["model_score"].to_numpy(dtype=np.float32, copy=False)
            for p in np.where(lbl <= 0.5)[0]:
                _heap_push_topk(heap, (float(sc[p]), fid, int(idx[p])), decoy_high_count)
    heap.sort(key=lambda x: x[0], reverse=True)
    return heap


# ──────────────────── PASS3: decoy 随机筛选（排除 high）────────────────────

def pass3_decoy_random(
    score_files, chunksize, seed, random_count,
    high_decoy_items: Sequence[Tuple[float, int, int]],
) -> List[Tuple[float, int, int]]:
    if random_count <= 0: return []
    high_set = {(fid, idx) for _, fid, idx in high_decoy_items}
    high_thr = min((s for s, _, _ in high_decoy_items), default=float("inf"))
    rng = random.Random(seed)
    sampled: List[Tuple[float, int, int]] = []
    seen = 0

    for fid, score_path in enumerate(score_files):
        print(f"[PASS3] {fid + 1}/{len(score_files)} {os.path.basename(score_path)}")
        for chunk in pd.read_csv(score_path, sep="\t",
                                  usecols=["original_index", "label", "model_score"],
                                  chunksize=chunksize):
            idx = chunk["original_index"].to_numpy(dtype=np.int64, copy=False)
            lbl = chunk["label"].to_numpy(dtype=np.float32, copy=False)
            sc = chunk["model_score"].to_numpy(dtype=np.float32, copy=False)

            for p in np.where(lbl <= 0.5)[0]:
                key = (fid, int(idx[p]))
                score = float(sc[p])
                if score > high_thr: continue
                if score == high_thr and key in high_set: continue
                seen += 1
                item = (score, key[0], key[1])
                if len(sampled) < random_count:
                    sampled.append(item)
                else:
                    j = rng.randint(0, seen - 1)
                    if j < random_count:
                        sampled[j] = item
    return sampled


# ──────────────────── PASS4: train targets ────────────────────

def pass4_collect_train_targets(
    score_files, neg_unique_scores, unique_q, q_threshold, chunksize,
) -> Dict[int, List[int]]:
    by_file: Dict[int, List[int]] = defaultdict(list)
    for fid, score_path in enumerate(score_files):
        print(f"[PASS4] {fid + 1}/{len(score_files)} {os.path.basename(score_path)}")
        for chunk in pd.read_csv(score_path, sep="\t",
                                  usecols=["original_index", "label", "model_score"],
                                  chunksize=chunksize):
            idx = chunk["original_index"].to_numpy(dtype=np.int64, copy=False)
            lbl = chunk["label"].to_numpy(dtype=np.float32, copy=False)
            sc = chunk["model_score"].to_numpy(dtype=np.float32, copy=False)
            qv = score_to_q_value(sc, neg_unique_scores, unique_q)
            qual = (lbl > 0.5) & (qv <= float(q_threshold))
            for p in np.where(qual)[0]:
                by_file[fid].append(int(idx[p]))
    return dict(by_file)


# ─────────────────── 回捞 pkl ───────────────────

def _find_row_aligned_keys(payload):
    if "label" not in payload: raise KeyError("缺少 label")
    n = len(payload["label"])
    return [k for k, v in payload.items()
            if (isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] == n)
            or (isinstance(v, (list, tuple)) and len(v) == n)]


def _append_values(store, payload, keys, indices):
    for k in keys:
        v = payload[k]
        if isinstance(v, np.ndarray): store[k].append(v[indices])
        else: store[k].extend([v[int(i)] for i in indices])


def _finalize_payload(first_payload, store, keys):
    out, ks = {}, set(keys)
    for k, v in first_payload.items():
        if k not in ks: out[k] = v; continue
        if isinstance(v, np.ndarray):
            out[k] = np.empty((0,) + v.shape[1:], dtype=v.dtype) if len(store[k]) == 0 else np.concatenate(store[k], axis=0)
        elif isinstance(v, tuple): out[k] = tuple(store[k])
        else: out[k] = store[k]
    return out


def _load_pickle(path):
    with _open_maybe_gz(path, "rb") as f: return _CompatUnpickler(f).load()


def _materialize_and_write_incremental(
    by_file, pkl_path_map, all_score_files,
    rows_per_file, output_dir, prefix, compresslevel,
):
    os.makedirs(output_dir, exist_ok=True)

    first, aligned = {}, []
    store = defaultdict(list)
    current_rows = 0
    chunk_id = 0
    total_written = 0
    n_target, n_decoy = 0, 0

    for fid in sorted(by_file.keys()):
        pkl_name = score_to_pkl_name(all_score_files[fid])
        if pkl_name not in pkl_path_map:
            raise FileNotFoundError(f"缺少 pkl: {pkl_name}")
        payload = _load_pickle(pkl_path_map[pkl_name])
        if not first:
            first = payload
            aligned = _find_row_aligned_keys(payload)
            for k in aligned:
                _ = store[k]

        idx_arr = np.array(sorted(set(by_file[fid])), dtype=np.int64)
        if idx_arr.size == 0:
            continue

        n = len(payload["label"])
        if idx_arr[0] < 0 or idx_arr[-1] >= n:
            raise IndexError(
                f"越界: {os.path.basename(pkl_path_map[pkl_name])} "
                f"min={idx_arr[0]} max={idx_arr[-1]} n={n}"
            )

        start = 0
        while start < len(idx_arr):
            remaining = rows_per_file - current_rows
            end = min(start + remaining, len(idx_arr))
            batch = idx_arr[start:end]

            _append_values(store, payload, aligned, batch)

            lbl = payload["label"]
            if isinstance(lbl, np.ndarray):
                batch_labels = lbl[batch]
                n_target += int((batch_labels > 0.5).sum())
                n_decoy += int((batch_labels <= 0.5).sum())
            else:
                for i in batch:
                    if lbl[int(i)] > 0.5: n_target += 1
                    else: n_decoy += 1

            current_rows += len(batch)

            if current_rows >= rows_per_file:
                chunk_payload = _finalize_payload(first, store, aligned)
                path = os.path.join(output_dir, f"{prefix}.{chunk_id:05d}.pkl.gz")
                with gzip.open(path, "wb", compresslevel=compresslevel) as f:
                    pickle.dump(chunk_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                chunk_id += 1
                total_written += current_rows

                store = defaultdict(list)
                for k in aligned:
                    _ = store[k]
                current_rows = 0

            start = end

    if current_rows > 0:
        chunk_payload = _finalize_payload(first, store, aligned)
        path = os.path.join(output_dir, f"{prefix}.{chunk_id:05d}.pkl.gz")
        with gzip.open(path, "wb", compresslevel=compresslevel) as f:
            pickle.dump(chunk_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        chunk_id += 1
        total_written += current_rows

    if not first:
        raise RuntimeError("未提取到样本")

    return chunk_id, total_written, n_target, n_decoy


def _check_output_dir(output_dir, overwrite):
    if not os.path.exists(output_dir): return
    if any(name.endswith(".pkl") or name.endswith(".pkl.gz") for name in os.listdir(output_dir)):
        if not overwrite: raise FileExistsError(f"输出目录已有 pkl: {output_dir}，请加 --overwrite")


def _merge_by_file(a: dict, b: dict) -> dict:
    merged = defaultdict(list, {int(k): list(v) for k, v in a.items()})
    for fid, idxs in b.items(): merged[int(fid)].extend(idxs)
    return dict(merged)


# ──────────────────────────── 主流程 ────────────────────────────

def main():
    args = parse_args()
    if args.rows_per_file <= 0: raise ValueError("rows_per_file 必须 > 0")
    if not (0.0 <= args.q_threshold <= 1.0): raise ValueError("q_threshold 必须在 [0,1]")

    all_score_files = discover_score_files(args.score_dir)
    print(f"score 文件: {len(all_score_files)} 个")
    if args.max_score_files > 0:
        all_score_files = all_score_files[:args.max_score_files]

    pkl_map = build_pkl_path_map(args.train_pkl_dir)
    v_map = build_pkl_path_map(args.val_pkl_dir)
    pkl_map.update(v_map)
    print(f"pkl 映射: train={len(build_pkl_path_map(args.train_pkl_dir))}, val={len(v_map)}, 合并={len(pkl_map)}")

    _check_output_dir(args.output_dir, args.overwrite)

    # ── PASS1: q-value 映射 ──
    unique_scores, neg_unique_scores, unique_q, total_t, total_d, targets_below = \
        compute_global_qvalue_mapping(all_score_files, args.chunksize)

    # 确定 selected decoy 总量
    if args.selected_decoy_count > 0:
        sel_decoy_total = args.selected_decoy_count
    else:
        sel_decoy_total = min(targets_below, 20_000_000)
        print(f"[auto] selected_decoy_count = {sel_decoy_total} (qualifying_targets≈{targets_below})")

    sel_decoy_total = min(sel_decoy_total, 20_000_000)
    high_n = sel_decoy_total // 2
    rand_n = sel_decoy_total - high_n

    print(
        f"采样计划: target(q<={args.q_threshold}) → 全部 train; "
        f"decoy(共选{sel_decoy_total}: 高分{high_n}+随机{rand_n}) → 全部 train"
    )

    # ── PASS2: decoy 高分 ──
    decoy_high = pass2_decoy_high(all_score_files, high_n, args.chunksize)
    if len(decoy_high) < high_n:
        print(f"警告: decoy_high 需求 {high_n}，实际仅 {len(decoy_high)}")
    print(f"PASS2 完成: decoy_high={len(decoy_high)}")

    # ── PASS3: decoy 随机（排除 high）──
    decoy_random = pass3_decoy_random(
        all_score_files, args.chunksize, args.seed, rand_n, decoy_high,
    )
    if len(decoy_random) < rand_n:
        print(f"警告: decoy_random 需求 {rand_n}，实际仅 {len(decoy_random)}")
    print(f"PASS3 完成: decoy_random={len(decoy_random)}")

    # ── 全部 decoy 进 train ──
    all_decoys = decoy_high + decoy_random
    print(f"train decoy 总量: {len(all_decoys)}")

    # ── PASS4: train targets ──
    train_target_by_file = pass4_collect_train_targets(
        all_score_files, neg_unique_scores, unique_q, args.q_threshold, args.chunksize,
    )
    n_train_t = sum(len(v) for v in train_target_by_file.values())
    print(f"train targets: {n_train_t}")

    # ── 组装 train decoy by_file ──
    train_decoy_by_file: Dict[int, List[int]] = defaultdict(list)
    for _, fid, idx in all_decoys:
        train_decoy_by_file[fid].append(int(idx))

    # ── 写出 train ──
    print("\n--- 回捞并写出 train ---")
    nt, train_total, train_n_target, train_n_decoy = _materialize_and_write_incremental(
        _merge_by_file(train_target_by_file, train_decoy_by_file),
        pkl_map, all_score_files,
        args.rows_per_file, args.output_dir, "train", args.compresslevel,
    )

    print(f"\ntrain: {nt} 块, {train_total} 行 → {args.output_dir}")
    print(f"target={train_n_target}, decoy={train_n_decoy}")


if __name__ == "__main__":
    main()
