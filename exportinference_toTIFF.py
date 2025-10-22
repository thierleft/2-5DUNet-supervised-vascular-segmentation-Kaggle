#!/usr/bin/env python3
"""
CPU-only saver for segmentation predictions stored in a memory-mapped 3D volume.

This script is designed to run on a CPU node (e.g., an HPC login/CPU partition)
and massively parallelize writing thousands of per-slice TIFF masks from a
float16 memmap volume produced by a prior GPU inference step.

Key features
------------
- **No GPU / torch required** (pure Python + NumPy + optional tifffile/OpenCV).
- **Process-level parallelism** across slices using concurrent.futures.
- Reads the memmap read-only; each worker opens it once (per process).
- Optional CSV of RLE-encoded masks (like Kaggle format) for each slice.
- Flexible naming: use a list of names, mirror an input image folder, or
  default to zero-padded indices.
- Optional compressed TIFF via tifffile (zlib) to save space.
- Supports chunked/arrayed processing via `--start`/`--end` for HPC job arrays.

Assumptions
-----------
- The memmap file contains a SINGLE-CHANNEL prediction volume of shape (Z, Y, X)
  with dtype float16 in range ~[0,1]. This matches the aggregated mask you
  built in your inference script at `args.mask_path`.

Examples
--------
1) Basic (OpenCV writer):
   python save_mmap_to_tifs.py \
       --mmap /home/lefebvre/storage/vasc/inference_output_OG/kidney_5_mask.mmap \
       --shape 4000 39000 39000 \
       --out /home/lefebvre/storage/vasc/inference_output_OG/TIF/kidney_5 \
       --threshold 0.5 --nprocs 32

2) With original slice names from input folder (mirrors raw names, faster than reading full images):
   python save_mmap_to_tifs.py \
       --mmap /home/lefebvre/storage/vasc/inference_output_OG/kidney_5_mask.mmap \
       --input-folder /data/raw/kidney_5/images \
       --out /home/lefebvre/storage/vasc/inference_output_OG/TIF/kidney_5

3) tifffile writer with compression + CSV RLE:
   python save_mmap_to_tifs.py \
       --mmap /home/lefebvre/storage/vasc/inference_output_OG/kidney_5_mask.mmap \
       --shape 4000 39000 39000 \
       --out /home/lefebvre/storage/vasc/inference_output_OG/TIF/kidney_5 \
       --csv /home/lefebvre/storage/vasc/inference_output_OG/CSV/kidney_5.csv \
       --group kidney_5 --tifffile --compress zlib --nprocs 48

4) Split work across multiple array jobs (e.g., SLURM):
   # job 0
   python save_mmap_to_tifs.py --mmap ... --shape Z Y X --out ... --start 0   --end 1000
   # job 1
   python save_mmap_to_tifs.py --mmap ... --shape Z Y X --out ... --start 1000 --end 2000
   ...

"""
from __future__ import annotations
import os
import sys
import json
import argparse
import math
import numpy as np
from typing import Optional, Sequence, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

# Optional backends for writing TIFFs
_HAS_TIFFFILE = False
_HAS_OPENCV = False
try:
    import tifffile as tiff  # type: ignore
    _HAS_TIFFFILE = True
except Exception:
    pass
try:
    import cv2  # type: ignore
    _HAS_OPENCV = True
except Exception:
    pass

# ----------------------------- RLE utilities ----------------------------- #

def rle_encode(img: np.ndarray) -> str:
    """Run-length encode a binary mask (0/255 or 0/1).

    Parameters
    ----------
    img : np.ndarray
        2D array where non-zero == mask.

    Returns
    -------
    str
        Space-separated RLE string.
    """
    # ensure 1D vector of 0/1
    pixels = (img.flatten() > 0).astype(np.uint8)
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    if runs.size == 0:
        return "1 0"
    return " ".join(str(x) for x in runs)

# --------------------------- Global worker state -------------------------- #

_G_MEMMAP: Optional[np.memmap] = None
_G_SHAPE: Optional[Tuple[int, int, int]] = None
_G_THRESHOLD: float = 0.5
_G_USE_TIFFFILE: bool = False
_G_TIFF_COMPRESSION: Optional[str] = None
_G_OUTPUT_DIR: Optional[Path] = None


def _init_worker(memmap_path: str,
                 shape: Sequence[int],
                 threshold: float,
                 use_tifffile: bool,
                 tiff_compression: Optional[str],
                 output_dir: str) -> None:
    """Initializer for worker processes: open the memmap once per process."""
    global _G_MEMMAP, _G_SHAPE, _G_THRESHOLD, _G_USE_TIFFFILE, _G_TIFF_COMPRESSION, _G_OUTPUT_DIR
    _G_SHAPE = tuple(int(s) for s in shape)  # (Z, Y, X)
    _G_THRESHOLD = float(threshold)
    _G_USE_TIFFFILE = bool(use_tifffile)
    _G_TIFF_COMPRESSION = tiff_compression
    _G_OUTPUT_DIR = Path(output_dir)
    _G_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _G_MEMMAP = np.memmap(memmap_path, mode="r", dtype=np.float16, shape=_G_SHAPE)


# ------------------------------ I/O helpers ------------------------------- #

def _write_tif(path: Path, image_uint8: np.ndarray) -> None:
    """Write a single-channel 8-bit image to TIFF using tifffile or OpenCV."""
    if _G_USE_TIFFFILE and _HAS_TIFFFILE:
        if _G_TIFF_COMPRESSION is not None:
            tiff.imwrite(str(path), image_uint8, compression=_G_TIFF_COMPRESSION)
        else:
            tiff.imwrite(str(path), image_uint8)
    else:
        if not _HAS_OPENCV:
            raise RuntimeError("Neither tifffile nor opencv-python is available for writing TIFFs.")
        # OpenCV uses extension to choose codec; .tif works, compression not exposed.
        ok = cv2.imwrite(str(path), image_uint8)
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed for {path}")


# ------------------------------ Worker task ------------------------------- #

def _process_one(idx: int, name: str | None) -> Tuple[int, str, Optional[str]]:
    """Read slice `idx` from the global memmap, threshold, save TIFF, return CSV info.

    Returns (idx, slice_id, rle_or_None)
    """
    assert _G_MEMMAP is not None and _G_SHAPE is not None and _G_OUTPUT_DIR is not None
    # Read as float32 to avoid precision surprises in comparison
    plane = np.asarray(_G_MEMMAP[idx], dtype=np.float32)
    # Threshold to binary 0/255
    mask = (plane > _G_THRESHOLD).astype(np.uint8) * 255

    # Name handling
    slice_id = name if name is not None else f"slice_{idx:06d}"
    out_path = _G_OUTPUT_DIR / f"{slice_id}.tif"

    _write_tif(out_path, mask)

    # Return RLE string lazily; parent decides if it writes CSV
    rle = rle_encode(mask)
    return idx, slice_id, rle


# ------------------------------- Main logic -------------------------------- #

def _read_shape_from_json(meta_path: Path) -> Optional[Tuple[int, int, int]]:
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        z, y, x = meta.get("shape", [None, None, None])
        if all(isinstance(v, int) for v in (z, y, x)):
            return int(z), int(y), int(x)
    except Exception:
        pass
    return None


def _collect_slice_names(input_folder: Optional[str], names_list: Optional[str], z: int) -> list[str]:
    names: list[str] = []
    if names_list:
        with open(names_list, "r") as f:
            for line in f:
                nm = line.strip()
                if nm:
                    names.append(nm)
        if len(names) != z:
            raise ValueError(f"names_list length {len(names)} != Z {z}")
        return names

    if input_folder:
        # Mirror base names of sorted .tif files in a folder; we do NOT read images.
        tif_paths = sorted(Path(input_folder).glob("*.tif"))
        if len(tif_paths) != z:
            # Soft warning, fallback to indices
            print(f"[warn] Found {len(tif_paths)} TIFs in input_folder but Z={z}; using index-based names.", file=sys.stderr)
        else:
            return [p.stem for p in tif_paths]

    # Default: zero-padded indices
    width = max(6, int(math.ceil(math.log10(max(1, z)))))
    return [f"slice_{i:0{width}d}" for i in range(z)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CPU-parallel saver for float16 memmap predictions -> TIFF + optional CSV RLE")
    p.add_argument("--mmap", required=True, help="Path to float16 memmap file (shape Z,Y,X)")
    p.add_argument("--out", required=True, help="Output directory for TIFF slices")

    # Shape resolution strategy
    p.add_argument("--shape", nargs=3, type=int, metavar=("Z", "Y", "X"), help="Shape of the memmap volume (Z Y X)")
    p.add_argument("--meta-json", type=str, default=None, help="Optional path to JSON with {'shape':[Z,Y,X]} if --shape not given")

    # Naming
    p.add_argument("--input-folder", type=str, default=None, help="Folder of original slice images to mirror names (sorted *.tif)")
    p.add_argument("--names-list", type=str, default=None, help="Text file with one slice name per line (without extension)")

    # Processing
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold on float predictions (default: 0.5)")
    p.add_argument("--nprocs", type=int, default=os.cpu_count() or 1, help="Number of worker processes")
    p.add_argument("--chunksize", type=int, default=8, help="Task chunk size per worker submit")
    p.add_argument("--start", type=int, default=0, help="Start slice index (inclusive)")
    p.add_argument("--end", type=int, default=None, help="End slice index (exclusive); default = Z")

    # TIFF writing
    p.add_argument("--tifffile", action="store_true", help="Use tifffile writer instead of OpenCV")
    p.add_argument("--compress", type=str, default=None, choices=["zlib", "lzma", "zstd"], help="Compression codec for tifffile writer")

    # CSV RLE
    p.add_argument("--csv", type=str, default=None, help="Optional path to write CSV of RLEs")
    p.add_argument("--group", type=str, default=None, help="Dataset/group prefix for CSV IDs (id = group_name)")

    # Cleanup
    p.add_argument("--delete-mmap", action="store_true", help="Delete memmap after successful export")

    # Environment
    p.add_argument("--omp-threads", type=int, default=1, help="Set OMP_NUM_THREADS to avoid thread oversubscription")

    args = p.parse_args()
    return args


def main() -> None:
    args = parse_args()

    # Enforce CPU-friendly threading behavior for underlying libs
    os.environ.setdefault("OMP_NUM_THREADS", str(args.omp_threads))

    mmap_path = Path(args.mmap)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve shape
    shape: Optional[Tuple[int, int, int]] = None
    if args.shape is not None:
        shape = tuple(int(s) for s in args.shape)
    elif args.meta_json is not None:
        shape = _read_shape_from_json(Path(args.meta_json))
    else:
        # Try sidecar JSON next to the memmap: e.g., kidney_5_mask.mmap.json
        shape = _read_shape_from_json(mmap_path.with_suffix(mmap_path.suffix + ".json"))

    if shape is None:
        raise ValueError("You must provide --shape Z Y X or a valid --meta-json / sidecar JSON with {'shape':[Z,Y,X]}.")

    z, y, x = shape
    if z <= 0 or y <= 0 or x <= 0:
        raise ValueError(f"Invalid shape: {shape}")

    # Compute slice indices to process
    start = int(args.start)
    end = int(args.end) if args.end is not None else z
    if not (0 <= start <= z) or not (0 <= end <= z) or not (start < end):
        raise ValueError(f"Invalid start/end range: start={start}, end={end}, Z={z}")

    num_slices = end - start

    # Determine slice names
    names = _collect_slice_names(args.input_folder, args.names_list, z)

    # Validate dependencies per flags
    use_tifffile = bool(args.tifffile)
    if use_tifffile and not _HAS_TIFFFILE:
        raise RuntimeError("--tifffile requested but 'tifffile' is not installed.")
    if not use_tifffile and not _HAS_OPENCV:
        # We will fallback to tifffile if available
        if _HAS_TIFFFILE:
            use_tifffile = True
        else:
            raise RuntimeError("No TIFF writer available: install either 'tifffile' or 'opencv-python'.")

    # Prepare CSV accumulation if requested
    want_csv = args.csv is not None
    csv_rows: list[Tuple[str, str]] = []  # (id, rle)

    # Init process pool; each worker opens the memmap once
    initializer = partial(
        _init_worker,
        memmap_path=str(mmap_path),
        shape=shape,
        threshold=float(args.threshold),
        use_tifffile=use_tifffile,
        tiff_compression=args.compress,
        output_dir=str(out_dir),
    )

    # Submit tasks
    indices = list(range(start, end))
    with ProcessPoolExecutor(max_workers=args.nprocs, initializer=initializer) as ex:
        # Use chunks of indices to reduce overhead
        futures = []
        for i in range(start, end, args.chunksize):
            chunk = list(range(i, min(i + args.chunksize, end)))
            # Submit each index in the chunk as a separate task
            for idx in chunk:
                futures.append(ex.submit(_process_one, idx, names[idx] if names else None))

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Saving TIFFs"):
            idx, slice_id, rle = fut.result()
            if want_csv and rle is not None:
                sample_id = slice_id if args.group is None else f"{args.group}_{slice_id}"
                csv_rows.append((sample_id, rle))

    # Write CSV if asked
    if want_csv:
        import csv
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "rle"])
            # deterministic order by slice index
            # reconstruct order from names/indices in range [start, end)
            for idx in range(start, end):
                sid = names[idx] if names else f"slice_{idx:06d}"
                sample_id = sid if args.group is None else f"{args.group}_{sid}"
                # find row
                # we can store in a dict for faster lookup if huge; for simplicity:
                # (since we keep order, we rebuild directly by recomputing RLE here would be costly; so build a map)
            # Build a map instead to ensure performance for very large Z
        # Rebuild efficiently
        id_to_rle = {r[0]: r[1] for r in csv_rows}
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "rle"])
            for idx in range(start, end):
                sid = names[idx] if names else f"slice_{idx:06d}"
                sample_id = sid if args.group is None else f"{args.group}_{sid}"
                rle = id_to_rle.get(sample_id, "1 0")
                w.writerow([sample_id, rle])
        print(f"[ok] CSV written: {csv_path}")

    # Optional cleanup
    if args.delete_mmap:
        try:
            mmap_path.unlink()
            sidecar = mmap_path.with_suffix(mmap_path.suffix + ".json")
            if sidecar.exists():
                sidecar.unlink()
        except Exception as e:
            print(f"[warn] Failed to delete memmap or sidecar: {e}", file=sys.stderr)

    print("[done] Export completed.")


if __name__ == "__main__":
    main()
