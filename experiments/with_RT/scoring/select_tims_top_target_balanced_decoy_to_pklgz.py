"""
按模型分数从 train_tims_all_psm_score 中筛选数据，并构造 train/val 两套样本：
1) target: 全局按分数从高到低选取；
2) decoy: 采用“一半高分 + 一半随机”策略；
3) 先按需求切分为 train/val，再在本脚本中调用 rearrange_train_all_mass_anchored.py 做质量锚定重排；
4) 最终输出为分块 pkl（每块默认 100 万行）。
"""

import argparse
import gzip
import glob
import heapq
import os
import pickle
import random
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按分数抽样并输出 train/val 质量锚定分块数据"
    )
    parser.add_argument(
        "--score_dir",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/train_tims_all_psm_score",
        help="*_all_psm_score.tsv 所在目录",
    )
    parser.add_argument(
        "--pkl_dir",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/pkl_dataset_tims/tims_pkl_all/train",
        help="train.*.pkl 或 train.*.pkl.gz 所在目录",
    )
    parser.add_argument(
        "--val_score_dir",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/train_tims_all_psm_score/val",
        help="val *_all_psm_score.tsv 所在目录",
    )
    parser.add_argument(
        "--val_pkl_dir",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/pkl_dataset_tims/tims_pkl_all/val",
        help="val train.*.pkl 或 train.*.pkl.gz 所在目录",
    )
    parser.add_argument(
        "--train_output_dir",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/pkl_dataset_tims/tims_pkl_all/train_select",
        help="train 分块输出目录",
    )
    parser.add_argument(
        "--val_output_dir",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/pkl_dataset_tims/tims_pkl_all/val_select",
        help="val 分块输出目录",
    )
    parser.add_argument(
        "--rearrange_script",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/rearrange_train_all_mass_anchored.py",
        help="质量锚定重排脚本路径",
    )
    parser.add_argument(
        "--rows_per_file",
        type=int,
        default=1_000_000,
        help="重排后每个输出文件的行数（作为 batch_size）",
    )

    # 目标数量：默认按 4:1 分配为 train=1000万, val=250万
    parser.add_argument("--train_target_count", type=int, default=10_000_000)
    parser.add_argument("--val_target_count", type=int, default=2_500_000)

    # decoy 数量：默认与 target 1:1（train=1000万, val=250万）
    parser.add_argument("--train_decoy_count", type=int, default=10_000_000)
    parser.add_argument("--val_decoy_count", type=int, default=2_500_000)

    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--max_score_files", type=int, default=0, help="仅处理前N个score文件，0表示全部")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--compresslevel", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=4, help="rearrange 脚本并行写出线程数")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖输出目录已有文件",
    )
    parser.add_argument(
        "--keep_tmp",
        action="store_true",
        help="保留中间临时目录（默认自动删除）",
    )
    return parser.parse_args()


class _CompatUnpickler(pickle.Unpickler):
    """兼容将 numpy._core.* 保存的旧 pickle。"""

    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _open_maybe_gz(path: str, mode: str):
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def discover_score_files(score_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(score_dir, "*_all_psm_score.tsv")))
    if not files:
        raise FileNotFoundError(f"未找到 score 文件: {score_dir}")
    return files


def score_to_pkl_name(score_file: str) -> str:
    base = os.path.basename(score_file)
    prefix = base.replace("_all_psm_score.tsv", "")
    if prefix.startswith("val."):
        return f"{prefix}_val"
    return f"{prefix}_train"


def build_pkl_path_map(pkl_dir: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(pkl_dir, "*.pkl"))):
        out[os.path.basename(path).replace(".pkl", "")] = path
    for path in sorted(glob.glob(os.path.join(pkl_dir, "*.pkl.gz"))):
        out[os.path.basename(path).replace(".pkl.gz", "")] = path
    if not out:
        raise FileNotFoundError(f"未找到 pkl/pkl.gz 文件: {pkl_dir}")
    return out


def _heap_push_topk(heap: List[Tuple[float, int, int]], item: Tuple[float, int, int], k: int) -> None:
    if k <= 0:
        return
    if len(heap) < k:
        heapq.heappush(heap, item)
        return
    if item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


def pass1_collect_topk(
    score_files: Sequence[str],
    target_count: int,
    decoy_high_count: int,
    chunksize: int,
) -> Tuple[List[Tuple[float, int, int]], List[Tuple[float, int, int]], int]:
    target_heap: List[Tuple[float, int, int]] = []
    decoy_high_heap: List[Tuple[float, int, int]] = []
    total_decoy = 0

    for fid, score_path in enumerate(score_files):
        print(f"[PASS1] {fid + 1}/{len(score_files)} {os.path.basename(score_path)}")
        for chunk in pd.read_csv(
            score_path,
            sep="\t",
            usecols=["original_index", "label", "model_score"],
            chunksize=chunksize,
        ):
            idx = chunk["original_index"].to_numpy(dtype=np.int64, copy=False)
            lbl = chunk["label"].to_numpy(dtype=np.float32, copy=False)
            sc = chunk["model_score"].to_numpy(dtype=np.float32, copy=False)

            target_mask = lbl > 0.5
            decoy_mask = ~target_mask
            total_decoy += int(decoy_mask.sum())

            for p in np.where(target_mask)[0]:
                _heap_push_topk(target_heap, (float(sc[p]), fid, int(idx[p])), target_count)
            for p in np.where(decoy_mask)[0]:
                _heap_push_topk(decoy_high_heap, (float(sc[p]), fid, int(idx[p])), decoy_high_count)

    return target_heap, decoy_high_heap, total_decoy


def pass2_reservoir_random_decoy(
    score_files: Sequence[str],
    chunksize: int,
    random_count: int,
    high_score_items: Sequence[Tuple[float, int, int]],
    seed: int,
) -> List[Tuple[float, int, int]]:
    if random_count <= 0:
        return []

    high_set = {(fid, idx) for _, fid, idx in high_score_items}
    high_threshold = min((s for s, _, _ in high_score_items), default=float("inf"))

    rng = random.Random(seed)
    sampled: List[Tuple[float, int, int]] = []
    seen = 0

    for fid, score_path in enumerate(score_files):
        print(f"[PASS2] {fid + 1}/{len(score_files)} {os.path.basename(score_path)}")
        for chunk in pd.read_csv(
            score_path,
            sep="\t",
            usecols=["original_index", "label", "model_score"],
            chunksize=chunksize,
        ):
            idx = chunk["original_index"].to_numpy(dtype=np.int64, copy=False)
            lbl = chunk["label"].to_numpy(dtype=np.float32, copy=False)
            sc = chunk["model_score"].to_numpy(dtype=np.float32, copy=False)

            for p in np.where(lbl <= 0.5)[0]:
                key = (fid, int(idx[p]))
                score = float(sc[p])

                if score > high_threshold:
                    continue
                if score == high_threshold and key in high_set:
                    continue

                seen += 1
                item = (score, key[0], key[1])
                if len(sampled) < random_count:
                    sampled.append(item)
                else:
                    j = rng.randint(0, seen - 1)
                    if j < random_count:
                        sampled[j] = item

    return sampled


def _find_row_aligned_keys(payload: Dict) -> List[str]:
    if "label" not in payload:
        raise KeyError("pkl 数据缺少 label 字段")
    n = len(payload["label"])
    aligned = []
    for k, v in payload.items():
        if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] == n:
            aligned.append(k)
        elif isinstance(v, (list, tuple)) and len(v) == n:
            aligned.append(k)
    return aligned


def _append_values(store: Dict[str, List], payload: Dict, keys: Iterable[str], indices: np.ndarray) -> None:
    for k in keys:
        v = payload[k]
        if isinstance(v, np.ndarray):
            store[k].append(v[indices])
        else:
            store[k].extend([v[int(i)] for i in indices])


def _finalize_payload(first_payload: Dict, store: Dict[str, List], keys: Iterable[str]) -> Dict:
    out: Dict = {}
    key_set = set(keys)
    for k, v in first_payload.items():
        if k not in key_set:
            out[k] = v
            continue
        if isinstance(v, np.ndarray):
            if len(store[k]) == 0:
                out[k] = np.empty((0,) + v.shape[1:], dtype=v.dtype)
            else:
                out[k] = np.concatenate(store[k], axis=0)
        elif isinstance(v, tuple):
            out[k] = tuple(store[k])
        else:
            out[k] = store[k]
    return out


def _load_pickle(path: str) -> Dict:
    with _open_maybe_gz(path, "rb") as f:
        return _CompatUnpickler(f).load()


def materialize_selected_payload(
    selected_items: Sequence[Tuple[float, int, int]],
    pkl_path_map: Dict[str, str],
    all_score_files: Sequence[str],
) -> Dict:
    by_file: Dict[int, List[int]] = defaultdict(list)
    for _, fid, original_index in selected_items:
        by_file[fid].append(int(original_index))

    first_payload: Dict = {}
    aligned_keys: List[str] = []
    store: Dict[str, List] = defaultdict(list)

    for fid in sorted(by_file.keys()):
        score_name = score_to_pkl_name(all_score_files[fid])
        if score_name not in pkl_path_map:
            raise FileNotFoundError(f"未找到对应 pkl 文件: {score_name}")

        pkl_path = pkl_path_map[score_name]
        print(f"[LOAD] {os.path.basename(pkl_path)}")
        payload = _load_pickle(pkl_path)

        if not first_payload:
            first_payload = payload
            aligned_keys = _find_row_aligned_keys(payload)
            for k in aligned_keys:
                _ = store[k]

        file_n = len(payload["label"])
        idx = np.array(sorted(set(by_file[fid])), dtype=np.int64)
        if idx.size == 0:
            continue
        if idx[0] < 0 or idx[-1] >= file_n:
            raise IndexError(
                f"索引越界: file={os.path.basename(pkl_path)}, min={idx[0]}, max={idx[-1]}, n={file_n}"
            )

        _append_values(store, payload, aligned_keys, idx)

    if not first_payload:
        raise RuntimeError("未提取到任何样本")

    return _finalize_payload(first_payload, store, aligned_keys)


def _shuffle_row_aligned(payload: Dict, seed: int) -> Dict:
    labels = np.asarray(payload["label"])
    n = int(labels.shape[0])
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


def _write_payload_pkl_gz(payload: Dict, out_path: str, compresslevel: int) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(out_path, "wb", compresslevel=compresslevel) as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _check_output_dir(output_dir: str, overwrite: bool) -> None:
    if not os.path.exists(output_dir):
        return
    has_pkl = any(name.endswith(".pkl") or name.endswith(".pkl.gz") for name in os.listdir(output_dir))
    if has_pkl and (not overwrite):
        raise FileExistsError(f"输出目录已有 pkl 文件: {output_dir}，请加 --overwrite")


def _run_rearrange(
    rearrange_script: str,
    input_dir: str,
    output_dir: str,
    rows_per_file: int,
    num_workers: int,
    seed: int,
    overwrite: bool,
    output_prefix: str,
) -> None:
    cmd = [
        sys.executable,
        rearrange_script,
        "--input_dir",
        input_dir,
        "--output_dir",
        output_dir,
        "--batch_size",
        str(rows_per_file),
        "--num_batches",
        "0",
        "--num_workers",
        str(max(1, int(num_workers))),
        "--seed",
        str(seed),
        "--output_prefix",
        output_prefix,
    ]
    if overwrite:
        cmd.append("--overwrite")

    print("[REARRANGE] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _slice_with_warn(items: List[Tuple[float, int, int]], start: int, need: int, tag: str) -> List[Tuple[float, int, int]]:
    got = items[start : start + need]
    if len(got) < need:
        print(f"警告: {tag} 需求 {need}，实际仅 {len(got)}")
    return got


def main() -> None:
    args = parse_args()

    if args.rows_per_file <= 0:
        raise ValueError("rows_per_file 必须 > 0")

    for vname in [
        "train_target_count",
        "val_target_count",
        "train_decoy_count",
        "val_decoy_count",
    ]:
        if getattr(args, vname) < 0:
            raise ValueError(f"{vname} 必须 >= 0")
    score_files = discover_score_files(args.score_dir)
    val_score_files = discover_score_files(args.val_score_dir)

    if args.max_score_files > 0:
        score_files = score_files[: args.max_score_files]
        val_score_files = val_score_files[: args.max_score_files]
        
    pkl_path_map = build_pkl_path_map(args.pkl_dir)
    # 合并 val 的 pkl 映射
    pkl_path_map.update(build_pkl_path_map(args.val_pkl_dir))

    _check_output_dir(args.train_output_dir, args.overwrite)
    _check_output_dir(args.val_output_dir, args.overwrite)

    print(
        "采样计划: "
        f"train(target={args.train_target_count}, decoy={args.train_decoy_count}), "
        f"val(target={args.val_target_count}, decoy={args.val_decoy_count})"
    )

    # 分别为 train 和 val 采样
    print("开始采样 Train 数据...")
    train_target_heap, train_decoy_high_heap, total_train_decoy_all = pass1_collect_topk(
        score_files=score_files,
        target_count=args.train_target_count,
        decoy_high_count=args.train_decoy_count // 2,
        chunksize=args.chunksize,
    )
    
    print("开始采样 Val 数据...")
    val_target_heap, val_decoy_high_heap, total_val_decoy_all = pass1_collect_topk(
        score_files=val_score_files,
        target_count=args.val_target_count,
        decoy_high_count=args.val_decoy_count // 2,
        chunksize=args.chunksize,
    )

    train_targets = sorted(train_target_heap, key=lambda x: x[0], reverse=True)
    train_decoy_high = sorted(train_decoy_high_heap, key=lambda x: x[0], reverse=True)
    
    val_targets = sorted(val_target_heap, key=lambda x: x[0], reverse=True)
    val_decoy_high = sorted(val_decoy_high_heap, key=lambda x: x[0], reverse=True)

    # 分别进行随机 Decoy 采样
    train_decoy_rand_need = args.train_decoy_count - len(train_decoy_high)
    train_decoy_rand = pass2_reservoir_random_decoy(
        score_files=score_files,
        chunksize=args.chunksize,
        random_count=train_decoy_rand_need,
        high_score_items=train_decoy_high,
        seed=args.seed,
    )
    
    val_decoy_rand_need = args.val_decoy_count - len(val_decoy_high)
    val_decoy_rand = pass2_reservoir_random_decoy(
        score_files=val_score_files,
        chunksize=args.chunksize,
        random_count=val_decoy_rand_need,
        high_score_items=val_decoy_high,
        seed=args.seed + 1,
    )

    train_items = train_targets + train_decoy_high + train_decoy_rand
    val_items = val_targets + val_decoy_high + val_decoy_rand

    print(
        f"完成采样: train={len(train_items)} (target={len(train_targets)}, decoy={len(train_decoy_high)+len(train_decoy_rand)}), "
        f"val={len(val_items)} (target={len(val_targets)}, decoy={len(val_decoy_high)+len(val_decoy_rand)})"
    )

    # 回捞 payload
    train_payload = materialize_selected_payload(train_items, pkl_path_map, score_files)
    val_payload = materialize_selected_payload(val_items, pkl_path_map, val_score_files)

    # 临时写入两个大 pkl.gz，随后调用 rearrange 输出 100 万行分块
    tmp_base = tempfile.mkdtemp(prefix="select_tims_mass_anchor_")
    train_in_dir = os.path.join(tmp_base, "train_in")
    val_in_dir = os.path.join(tmp_base, "val_in")
    os.makedirs(train_in_dir, exist_ok=True)
    os.makedirs(val_in_dir, exist_ok=True)

    train_big = os.path.join(train_in_dir, "selected_train.pkl.gz")
    val_big = os.path.join(val_in_dir, "selected_val.pkl.gz")
    _write_payload_pkl_gz(train_payload, train_big, args.compresslevel)
    _write_payload_pkl_gz(val_payload, val_big, args.compresslevel)

    # 调用质量锚定重排
    os.makedirs(args.train_output_dir, exist_ok=True)
    os.makedirs(args.val_output_dir, exist_ok=True)

    _run_rearrange(
        rearrange_script=args.rearrange_script,
        input_dir=train_in_dir,
        output_dir=args.train_output_dir,
        rows_per_file=args.rows_per_file,
        num_workers=args.num_workers,
        seed=args.seed,
        overwrite=args.overwrite,
        output_prefix="train_select",
    )
    _run_rearrange(
        rearrange_script=args.rearrange_script,
        input_dir=val_in_dir,
        output_dir=args.val_output_dir,
        rows_per_file=args.rows_per_file,
        num_workers=args.num_workers,
        seed=args.seed + 1,
        overwrite=args.overwrite,
        output_prefix="val_select",
    )

    train_labels = np.asarray(train_payload["label"])
    val_labels = np.asarray(val_payload["label"])
    print(
        f"train label统计: target={(train_labels > 0.5).sum()}, decoy={(train_labels <= 0.5).sum()}, total={len(train_labels)}"
    )
    print(
        f"val label统计: target={(val_labels > 0.5).sum()}, decoy={(val_labels <= 0.5).sum()}, total={len(val_labels)}"
    )
    print(f"train 分块目录: {args.train_output_dir}")
    print(f"val 分块目录: {args.val_output_dir}")

    if args.keep_tmp:
        print(f"保留临时目录: {tmp_base}")
    else:
        shutil.rmtree(tmp_base, ignore_errors=True)


if __name__ == "__main__":
    main()
