"""从 tims_sage_select/val 随机取 10 万留作验证集，其余并入训练集。"""
import gzip
import os
import pickle
import numpy as np

VAL_DIR = "/home/yiwen/AIPC/scripts/organized_attantion/data/tims_sage_select/val"
TRAIN_DIR = "/home/yiwen/AIPC/scripts/organized_attantion/data/tims_sage_select/train"
VAL_KEEP = 100_000
ROWS_PER_FILE = 1_000_000
SEED = 42

rng = np.random.default_rng(SEED)


def load_pklgz(path):
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh)


def save_pklgz(data, path):
    with gzip.open(path, "wb", compresslevel=6) as fh:
        pickle.dump(data, fh)


def merge_dicts(dicts):
    """Concatenate a list of data dicts along axis 0."""
    merged = {}
    for key in dicts[0]:
        vals = [d[key] for d in dicts]
        if isinstance(vals[0], np.ndarray):
            merged[key] = np.concatenate(vals, axis=0)
        elif isinstance(vals[0], tuple):
            # merge tuples by concatenating
            merged[key] = sum(vals, ())
        else:
            merged[key] = vals[0]
    return merged


def slice_dict(data, indices):
    """Extract rows at given indices."""
    sub = {}
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            sub[key] = val[indices]
        elif isinstance(val, tuple):
            sub[key] = tuple(val[i] for i in indices)
        else:
            sub[key] = val
    return sub


def main():
    # 1. Load all val files
    val_files = sorted(
        f for f in os.listdir(VAL_DIR) if f.endswith(".pkl.gz")
    )
    print(f"Loading {len(val_files)} val files...")
    all_data = [load_pklgz(os.path.join(VAL_DIR, f)) for f in val_files]
    merged = merge_dicts(all_data)
    n_total = len(merged["label"])
    print(f"Total val rows: {n_total:,}")

    # 2. Shuffle and split
    perm = rng.permutation(n_total)
    keep_idx = perm[:VAL_KEEP]
    merge_idx = perm[VAL_KEEP:]

    keep_data = slice_dict(merged, keep_idx)
    merge_data = slice_dict(merged, merge_idx)
    print(f"Keep (new val): {VAL_KEEP:,} rows")
    print(f"Merge to train: {len(merge_idx):,} rows")

    # 3. Remove old val files, write new single val file
    for f in val_files:
        os.remove(os.path.join(VAL_DIR, f))
        print(f"  Removed old: {f}")

    save_pklgz(keep_data, os.path.join(VAL_DIR, "val.00000.pkl.gz"))
    print(f"  Wrote: val.00000.pkl.gz ({VAL_KEEP:,} rows)")

    # 4. Split merge data into 1M-row chunks and write as train files
    train_existing = sorted(
        f for f in os.listdir(TRAIN_DIR) if f.endswith(".pkl.gz")
    )
    next_idx = len(train_existing)
    n_merge = len(merge_idx)

    for start in range(0, n_merge, ROWS_PER_FILE):
        end = min(start + ROWS_PER_FILE, n_merge)
        chunk = slice_dict(merge_data, np.arange(start, end))
        out_name = f"train.{next_idx:05d}.pkl.gz"
        save_pklgz(chunk, os.path.join(TRAIN_DIR, out_name))
        print(f"  Wrote: {out_name} ({end - start:,} rows)")
        next_idx += 1

    print("Done.")


if __name__ == "__main__":
    main()
