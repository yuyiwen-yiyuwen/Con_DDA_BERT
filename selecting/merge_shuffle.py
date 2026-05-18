import os, pickle, gzip, gc
import numpy as np
from tqdm import tqdm

SOURCE_DIRS = [
    "/home/yiwen/AIPC/scripts/organized_attantion/data/dataset/mzml_sage_select",
    "/home/yiwen/AIPC/scripts/organized_attantion/data/dataset/tims_sage_select",
    "/home/yiwen/AIPC/scripts/organized_attantion/data/dataset/wiff_sage_select",
]
OUTPUT_DIR = "/home/yiwen/AIPC/scripts/organized_attantion/data/dataset/mzml_tims_wiff_sage_0.01_select"
CHUNK_SIZE = 1_000_000
SHUFFLE_BATCH = 2_000_000  # batch size for memory-safe shuffling


def load_file(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    else:
        with open(path, "rb") as f:
            return pickle.load(f)


def save_chunk(data_dict, path):
    with gzip.open(path, "wb", compresslevel=2) as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def collect_files(dirs):
    files = []
    for d in dirs:
        if not os.path.exists(d):
            print(f"Warning: directory not found: {d}")
            continue
        for root, _, names in os.walk(d):
            for name in sorted(names):
                if name.endswith(".pkl.gz") or name.endswith(".pkl"):
                    files.append(os.path.join(root, name))
    return files


def shuffle_array_inverse(arr, indices, desc="Shuffling"):
    """Shuffle arr using inverse permutation to avoid OOM.
    output[indices[i]] = arr[i]  processed in batches.
    """
    total = len(arr)
    out = np.empty_like(arr)
    for start in tqdm(range(0, total, SHUFFLE_BATCH), desc=desc):
        end = min(start + SHUFFLE_BATCH, total)
        batch_idx = indices[start:end]
        out[batch_idx] = arr[start:end]
    return out


def shuffle_list_inverse(lst, indices, desc="Shuffling"):
    """Same inverse-permutation shuffle for a Python list."""
    total = len(lst)
    out = [None] * total
    for start in tqdm(range(0, total, SHUFFLE_BATCH), desc=desc):
        end = min(start + SHUFFLE_BATCH, total)
        for j, item in zip(indices[start:end], lst[start:end]):
            out[j] = item
    return out


def merge_and_shuffle():
    files = collect_files(SOURCE_DIRS)
    print(f"Found {len(files)} source files")

    # ---- Pass 1: scan record counts ----
    file_info = []  # (path, n_records)
    sample_data = None
    for fp in tqdm(files, desc="Scanning files"):
        data = load_file(fp)
        n = len(data["label"])
        file_info.append((fp, n))
        if sample_data is None:
            sample_data = {k: v for k, v in data.items()}
        del data

    total = sum(n for _, n in file_info)
    n_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"Total records: {total:,}")
    print(f"Output chunks: {n_chunks}")

    array_keys = [k for k in sample_data if k != "peptides"]

    # ---- Pre-allocate merged arrays ----
    print("Pre-allocating merged arrays...")
    merged = {}
    for k in array_keys:
        shape = (total,) + sample_data[k].shape[1:]
        merged[k] = np.empty(shape, dtype=sample_data[k].dtype)
        print(f"  {k}: {merged[k].shape}  ({merged[k].nbytes / 1024**3:.2f} GB)")

    # Reserve list for peptides
    peptides_all = [None] * total
    del sample_data

    # ---- Pass 2: fill pre-allocated arrays ----
    offset = 0
    for fp, n in tqdm(file_info, desc="Loading and copying"):
        data = load_file(fp)
        end = offset + n
        for k in array_keys:
            merged[k][offset:end] = data[k]
        for j, pep in enumerate(data["peptides"]):
            peptides_all[offset + j] = pep
        offset = end
        del data

    total_bytes = sum(v.nbytes for v in merged.values())
    print(f"Memory after loading: ~{total_bytes / 1024**3:.1f} GB")

    # ---- Generate shuffle indices ----
    rng = np.random.default_rng(42)
    indices = rng.permutation(total)
    print(f"Shuffle index memory: {indices.nbytes / 1024**3:.2f} GB")

    # ---- Shuffle arrays using inverse permutation (memory-safe) ----
    for k in array_keys:
        merged[k] = shuffle_array_inverse(merged[k], indices, desc=f"Shuffle {k}")

    # ---- Shuffle peptides ----
    peptides_all = shuffle_list_inverse(peptides_all, indices, desc="Shuffle peptides")
    del indices
    gc.collect()

    # ---- Save chunks ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i in tqdm(range(n_chunks), desc="Saving chunks"):
        start = i * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, total)

        chunk = {k: merged[k][start:end] for k in array_keys}
        chunk["peptides"] = tuple(peptides_all[start:end])

        out_path = os.path.join(OUTPUT_DIR, f"train.{i:05d}.pkl.gz")
        save_chunk(chunk, out_path)
        del chunk

    print(f"\nDone! {n_chunks} chunks saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    merge_and_shuffle()
