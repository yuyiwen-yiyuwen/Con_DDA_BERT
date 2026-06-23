"""
功能：收集所有PSM得分建立全局q-value查找表，按q阈值筛选PSM并输出pkl.gz
输入：
    --train_score_dir .../data/score/train_tims_all_psm_score_epoch10/train
    --val_score_dir .../data/score/train_tims_all_psm_score_epoch10/val
    --train_pkl_dir .../pkl_dataset_tims/tims_pkl_all/train
    --output_dir 输出根目录
输出：
    {output_dir}/qvalue_index/{split}_qvalue_index.tsv
    {output_dir}/dataset/{split}/{split}_q0.01.XXXXX.pkl.gz
    {output_dir}/qvalue_dataset_summary.csv
"""

import argparse
import gzip
import glob
import os
import pickle
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


class _CompatUnpickler(pickle.Unpickler):
    """Compatible with pickle files saved with legacy numpy._core.* modules."""

    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute global q-values from all *_all_psm_score.tsv files and build "
            "a new selected PKL.GZ dataset by q-value threshold."
        )
    )
    parser.add_argument(
        "--train_score_dir",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/train_tims_all_psm_score/train",
        help="Directory containing train *_all_psm_score.tsv files.",
    )
    parser.add_argument(
        "--val_score_dir",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/train_tims_all_psm_score/val",
        help="Directory containing val *_all_psm_score.tsv files.",
    )
    parser.add_argument(
        "--train_pkl_dir",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/pkl_dataset_tims/tims_pkl_all/train",
        help="Directory containing train source .pkl/.pkl.gz files.",
    )
    parser.add_argument(
        "--val_pkl_dir",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/pkl_dataset_tims/tims_pkl_all/val",
        help="Directory containing val source .pkl/.pkl.gz files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/yiwen/AIPC/scripts/attantion/pkl_dataset_tims/tims_pkl_all/qvalue_select",
        help="Output root directory.",
    )
    parser.add_argument("--q_threshold", type=float, default=0.01, help="Keep PSM rows with q_value <= this threshold.")
    parser.add_argument("--chunksize", type=int, default=500_000, help="Chunk size for reading score TSV.")
    parser.add_argument(
        "--rows_per_file",
        type=int,
        default=1_000_000,
        help="Rows per output pkl.gz chunk for each split.",
    )
    parser.add_argument("--compresslevel", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def discover_score_files(score_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(score_dir, "*_all_psm_score.tsv")))
    if not files:
        raise FileNotFoundError(f"No score files found in: {score_dir}")
    return files


def build_pkl_path_map(pkl_dir: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(pkl_dir, "*.pkl"))):
        out[os.path.basename(path).replace(".pkl", "")] = path
    for path in sorted(glob.glob(os.path.join(pkl_dir, "*.pkl.gz"))):
        out[os.path.basename(path).replace(".pkl.gz", "")] = path
    if not out:
        raise FileNotFoundError(f"No pkl/pkl.gz files found in: {pkl_dir}")
    return out


def score_to_pkl_name(score_file: str) -> str:
    base = os.path.basename(score_file)
    prefix = base.replace("_all_psm_score.tsv", "")
    if prefix.startswith("val."):
        return f"{prefix}_val"
    return f"{prefix}_train"


def compute_q_values_from_scores(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute target-decoy q-values from descending model scores."""
    if scores.size == 0:
        return np.empty((0,), dtype=np.float32)

    order = np.argsort(scores)[::-1]
    y = labels[order].astype(np.int8)

    cum_t = np.cumsum(y)
    cum_d = np.cumsum(1 - y)
    fdr = np.where(cum_t > 0, cum_d / cum_t, np.inf)
    q_sorted = np.minimum.accumulate(fdr[::-1])[::-1]

    q = np.empty_like(q_sorted, dtype=np.float32)
    q[order] = q_sorted.astype(np.float32)
    return q


def _open_maybe_gz(path: str, mode: str):
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def _load_pickle(path: str) -> Dict:
    with _open_maybe_gz(path, "rb") as f:
        return _CompatUnpickler(f).load()


def _find_row_aligned_keys(payload: Dict) -> List[str]:
    if "label" not in payload:
        raise KeyError("Payload missing 'label' field")
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


def materialize_selected_payload(
    selected_by_score_file: Dict[str, np.ndarray],
    pkl_path_map: Dict[str, str],
) -> Dict:
    first_payload: Dict = {}
    aligned_keys: List[str] = []
    store: Dict[str, List] = defaultdict(list)

    for score_file in sorted(selected_by_score_file.keys()):
        idx = np.array(sorted(set(int(i) for i in selected_by_score_file[score_file])), dtype=np.int64)
        if idx.size == 0:
            continue

        pkl_key = score_to_pkl_name(score_file)
        if pkl_key not in pkl_path_map:
            raise FileNotFoundError(f"Missing source pkl for score file: {score_file} (expected key: {pkl_key})")

        pkl_path = pkl_path_map[pkl_key]
        payload = _load_pickle(pkl_path)

        if not first_payload:
            first_payload = payload
            aligned_keys = _find_row_aligned_keys(payload)
            for k in aligned_keys:
                _ = store[k]

        n = len(payload["label"])
        if idx[0] < 0 or idx[-1] >= n:
            raise IndexError(
                f"Index out of bounds for {os.path.basename(pkl_path)}: min={idx[0]}, max={idx[-1]}, n={n}"
            )

        _append_values(store, payload, aligned_keys, idx)

    if not first_payload:
        raise RuntimeError("No rows were selected; check q_threshold and inputs.")

    return _finalize_payload(first_payload, store, aligned_keys)


def slice_payload(payload: Dict, start: int, end: int) -> Dict:
    out: Dict = {}
    n = len(payload["label"])
    for k, v in payload.items():
        if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] == n:
            out[k] = v[start:end]
        elif isinstance(v, list) and len(v) == n:
            out[k] = v[start:end]
        elif isinstance(v, tuple) and len(v) == n:
            out[k] = tuple(v[start:end])
        else:
            out[k] = v
    return out


def write_payload_chunks(payload: Dict, out_dir: str, prefix: str, rows_per_file: int, compresslevel: int) -> int:
    os.makedirs(out_dir, exist_ok=True)
    total = len(payload["label"])
    if total == 0:
        return 0

    chunk_id = 0
    for start in range(0, total, rows_per_file):
        end = min(start + rows_per_file, total)
        part = slice_payload(payload, start, end)
        out_path = os.path.join(out_dir, f"{prefix}.{chunk_id:05d}.pkl.gz")
        with gzip.open(out_path, "wb", compresslevel=compresslevel) as f:
            pickle.dump(part, f, protocol=pickle.HIGHEST_PROTOCOL)
        chunk_id += 1
    return chunk_id


def ensure_can_write(path: str, overwrite: bool) -> None:
    if not os.path.exists(path):
        return
    if overwrite:
        return
    if os.path.isdir(path) and os.listdir(path):
        raise FileExistsError(f"Output path exists and is not empty: {path}. Use --overwrite to continue.")


def collect_split_arrays(score_files: Sequence[str], chunksize: int):
    file_names: List[str] = []
    file_id_parts: List[np.ndarray] = []
    idx_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    score_parts: List[np.ndarray] = []

    for fid, score_path in enumerate(tqdm(score_files, desc="Reading scores")):
        base = os.path.basename(score_path)
        file_names.append(base)

        for chunk in pd.read_csv(
            score_path,
            sep="\t",
            usecols=["original_index", "label", "model_score"],
            chunksize=chunksize,
        ):
            idx = chunk["original_index"].to_numpy(dtype=np.int64, copy=False)
            lbl = (chunk["label"].to_numpy(dtype=np.float32, copy=False) > 0.5).astype(np.int8)
            sc = chunk["model_score"].to_numpy(dtype=np.float32, copy=False)

            file_ids = np.full(idx.shape[0], fid, dtype=np.int32)

            file_id_parts.append(file_ids)
            idx_parts.append(idx.copy())
            label_parts.append(lbl.copy())
            score_parts.append(sc.copy())

    if not idx_parts:
        return file_names, np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int8), np.empty((0,), dtype=np.float32)

    file_ids = np.concatenate(file_id_parts)
    indices = np.concatenate(idx_parts)
    labels = np.concatenate(label_parts)
    scores = np.concatenate(score_parts)
    return file_names, file_ids, indices, labels, scores


def process_split(
    split_name: str,
    score_dir: str,
    pkl_dir: str,
    output_dir: str,
    q_threshold: float,
    chunksize: int,
    rows_per_file: int,
    compresslevel: int,
) -> Dict[str, object]:
    print(f"\n===== Processing split: {split_name} =====")
    score_files = discover_score_files(score_dir)
    pkl_map = build_pkl_path_map(pkl_dir)

    qvalue_out = os.path.join(output_dir, "qvalue_index")
    os.makedirs(qvalue_out, exist_ok=True)
    qvalue_path = os.path.join(qvalue_out, f"{split_name}_qvalue_index.tsv")
    if os.path.exists(qvalue_path):
        os.remove(qvalue_path)

    selected_by_score_file: Dict[str, np.ndarray] = {}

    total_rows = 0
    selected_rows = 0
    selected_targets = 0
    selected_decoys = 0
    wrote_header = False

    for score_path in tqdm(score_files, desc=f"{split_name} per-file qvalue"):
        file_name = os.path.basename(score_path)
        _file_names, _file_ids, indices, labels, scores = collect_split_arrays([score_path], chunksize)

        if scores.size == 0:
            continue

        q_values = compute_q_values_from_scores(scores, labels)
        selected_mask = q_values <= float(q_threshold)

        per_file_df = pd.DataFrame(
            {
                "score_file": np.full(indices.shape[0], file_name, dtype=object),
                "original_index": indices,
                "label": labels,
                "model_score": scores,
                "q_value": q_values,
                "selected": selected_mask.astype(np.int8),
            }
        )
        per_file_df.to_csv(
            qvalue_path,
            sep="\t",
            index=False,
            mode="a",
            header=not wrote_header,
        )
        wrote_header = True

        if selected_mask.any():
            selected_by_score_file[file_name] = indices[selected_mask]

        total_rows += int(indices.shape[0])
        selected_rows += int(selected_mask.sum())
        selected_targets += int(((labels == 1) & selected_mask).sum())
        selected_decoys += int(((labels == 0) & selected_mask).sum())

    if total_rows == 0:
        raise RuntimeError(f"No score rows found in split {split_name}")

    selected_payload = materialize_selected_payload(selected_by_score_file, pkl_map)

    split_out_dir = os.path.join(output_dir, "dataset", split_name)
    os.makedirs(split_out_dir, exist_ok=True)
    n_chunks = write_payload_chunks(
        payload=selected_payload,
        out_dir=split_out_dir,
        prefix=f"{split_name}_q{str(q_threshold).replace('.', 'p')}",
        rows_per_file=rows_per_file,
        compresslevel=compresslevel,
    )

    print(f"[{split_name}] total_rows={total_rows}")
    print(f"[{split_name}] selected_rows={selected_rows} @ per-file q<={q_threshold}")
    print(f"[{split_name}] selected_targets={selected_targets}, selected_decoys={selected_decoys}")
    print(f"[{split_name}] qvalue index file: {qvalue_path}")
    print(f"[{split_name}] output chunks: {n_chunks} in {split_out_dir}")

    return {
        "split": split_name,
        "score_files": len(score_files),
        "total_rows": total_rows,
        "selected_rows": selected_rows,
        "selected_targets": selected_targets,
        "selected_decoys": selected_decoys,
        "q_threshold": float(q_threshold),
        "qvalue_index_file": qvalue_path,
        "output_dir": split_out_dir,
        "output_chunks": n_chunks,
    }


def main() -> None:
    args = parse_args()

    if args.rows_per_file <= 0:
        raise ValueError("rows_per_file must be > 0")
    if args.chunksize <= 0:
        raise ValueError("chunksize must be > 0")
    if not (0.0 <= args.q_threshold <= 1.0):
        raise ValueError("q_threshold must be in [0, 1]")

    ensure_can_write(args.output_dir, args.overwrite)
    os.makedirs(args.output_dir, exist_ok=True)

    summary_rows = []
    summary_rows.append(
        process_split(
            split_name="train",
            score_dir=args.train_score_dir,
            pkl_dir=args.train_pkl_dir,
            output_dir=args.output_dir,
            q_threshold=args.q_threshold,
            chunksize=args.chunksize,
            rows_per_file=args.rows_per_file,
            compresslevel=args.compresslevel,
        )
    )
    summary_rows.append(
        process_split(
            split_name="val",
            score_dir=args.val_score_dir,
            pkl_dir=args.val_pkl_dir,
            output_dir=args.output_dir,
            q_threshold=args.q_threshold,
            chunksize=args.chunksize,
            rows_per_file=args.rows_per_file,
            compresslevel=args.compresslevel,
        )
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, "qvalue_dataset_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n===== Done =====")
    print(f"Summary file: {summary_path}")


if __name__ == "__main__":
    main()
