import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
import re

def get_common_prefix(path, suffix="labels"):
    base = os.path.basename(os.path.normpath(path))
    head, _, tail = base.rpartition('_')
    return head if tail == suffix else base

def extract_slice_id(filename):
    """Extracts trailing digits before .tif/.tiff."""
    match = re.search(r'_(\d{3,6})\.tif{1,2}$', filename.lower())
    return match.group(1) if match else None

def match_files_by_name(raw_dir, label_dir):
    """Match raw and label files by shared slice ID (_###.tif)."""
    raw_files = {
        extract_slice_id(os.path.basename(f)): f
        for f in glob(os.path.join(raw_dir, "*.tif"))
    }
    label_files = {
        extract_slice_id(os.path.basename(f)): f
        for f in glob(os.path.join(label_dir, "*.tif"))
    }

    # Remove unmatched or malformed filenames
    raw_files = {k: v for k, v in raw_files.items() if k is not None}
    label_files = {k: v for k, v in label_files.items() if k is not None}

    common_keys = sorted(set(raw_files).intersection(label_files))
    if not common_keys:
        print(f"No matching slice IDs between {raw_dir} and {label_dir}")
        return [], []

    # Warn about unmatched files
    missing_in_raw = set(label_files) - set(raw_files)
    missing_in_label = set(raw_files) - set(label_files)

    if missing_in_raw:
        print(f"Missing in raw: {sorted(missing_in_raw)}")
    if missing_in_label:
        print(f"Missing in label: {sorted(missing_in_label)}")

    slice_paths = [raw_files[k] for k in common_keys]
    label_paths = [label_files[k] for k in common_keys]
    return slice_paths, label_paths

def save_memmaps(group_name, slice_paths, label_paths, output_dir):
    h, w = cv2.imread(slice_paths[0], cv2.IMREAD_GRAYSCALE).shape
    volume = np.memmap(
        os.path.join(output_dir, f"{group_name}.mmap"),
        dtype=np.uint16,
        shape=(len(slice_paths), h, w),
        mode="w+",
    )
    volume_mask = np.memmap(
        os.path.join(output_dir, f"{group_name}_mask.mmap"),
        dtype=np.uint8,
        shape=(len(label_paths), h, w),
        mode="w+",
    )

    for i, (s_path, l_path) in tqdm(
        enumerate(zip(slice_paths, label_paths)),
        total=len(slice_paths),
        desc=group_name,
    ):
        slice_img = cv2.imread(s_path, cv2.IMREAD_UNCHANGED)
        label_img = cv2.imread(l_path, cv2.IMREAD_UNCHANGED)

        if label_img is None:
            raise ValueError(f"Could not read label file: {l_path}")

        # Force to grayscale if needed
        if label_img.ndim == 3:
            label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)

        # Binarize
        label_img = (label_img > 127).astype(np.uint8) * 255

        # Count occurrences of each value
        count_0 = np.count_nonzero(label_img == 0)
        count_255 = np.count_nonzero(label_img == 255)

        # Ensure 255 (foreground) is the majority class
        if count_0 > count_255:
            label_img = 255 - label_img

        volume[i] = slice_img
        volume_mask[i] = label_img

    volume.flush()
    volume_mask.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--path",
        required=True,
        help="Path to root folder containing subject folders and matching *_labels folders",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output directory for .mmap files",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Discover subject dirs:
    # - must be a directory
    # - must NOT end with '_labels'
    # - must have a paired '<name>_labels' directory
    subject_folders = []
    for d in sorted(os.listdir(args.path)):
        full = os.path.join(args.path, d)
        if not os.path.isdir(full):
            continue
        if d.endswith("_labels"):
            continue

        label_dir = os.path.join(args.path, f"{d}_labels")
        if not os.path.isdir(label_dir):
            print(f"Skipping {d}: label dir not found: {label_dir}")
            continue

        subject_folders.append(d)

    print("Subjects:", subject_folders)

    for subject in subject_folders:
        raw_dir = os.path.join(args.path, subject)
        label_dir = os.path.join(args.path, f"{subject}_labels")

        print(f"Processing {subject}")
        print("  raw_dir  :", raw_dir)
        print("  label_dir:", label_dir)

        slice_paths, label_paths = match_files_by_name(raw_dir, label_dir)
        if not slice_paths or not label_paths:
            continue

        group_name = subject
        save_memmaps(group_name, slice_paths, label_paths, args.output)
