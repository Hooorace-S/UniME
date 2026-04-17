import argparse
import json
import math

from collections.abc import Sequence
from pathlib import Path

import nibabel as nib  # pylint: disable=import-error
import numpy as np

SplitRatios = tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process BraTS 2023 raw data.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory that contains BraTS 2023 cases (BraTS-GLI-*-* folders).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/BRATS2023"),
        help="Destination directory for processed numpy arrays.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="BraTS2023",
        help="Identifier prefix used for output file names.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=128,
        help="Minimum spatial size enforced during cropping.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess cases even if output files already exist.",
    )
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        default=(0.7, 0.2, 0.1),
        metavar=("TRAIN", "TEST", "VAL"),
        help="Ratios for train/test/val splits, applied after processing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2023,
        help="Random seed used for dataset splits.",
    )
    parser.add_argument(
        "--skip-splits",
        action="store_true",
        help="Only process volumes; do not generate train/test/val files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of cases to process (for debugging).",
    )
    parser.add_argument(
        "--write-mapping",
        action="store_true",
        help="Save mapping from output ids to source case folders (mapping.json).",
    )
    return parser.parse_args()


def sup_128(start: int, stop: int, min_size: int, upper_bound: int) -> tuple[int, int]:
    """Pad the crop window (exclusive stop) so the span reaches at least `min_size` voxels."""
    span = stop - start
    if span >= min_size or upper_bound <= min_size:
        return max(0, start), min(upper_bound, stop)

    deficit = min_size - span
    pad_before = deficit // 2
    pad_after = deficit - pad_before

    start = max(0, start - pad_before)
    stop = min(upper_bound, stop + pad_after)

    span = stop - start
    if span >= min_size:
        return start, stop

    if start == 0:
        stop = min(upper_bound, min_size)
        return start, stop
    if stop == upper_bound:
        start = max(0, upper_bound - min_size)
        return start, stop

    needed = min_size - span
    extra_before = min(start, needed)
    start -= extra_before
    needed -= extra_before
    stop = min(upper_bound, stop + needed)
    return start, stop


def find_crop_bounds(mask: np.ndarray, min_size: int) -> tuple[slice, slice, slice]:
    coords = np.where(mask)
    if coords[0].size == 0:
        depth, height, width = mask.shape
        cx, cy, cz = depth // 2, height // 2, width // 2
        half = min_size // 2
        x_min = max(0, cx - half)
        y_min = max(0, cy - half)
        z_min = max(0, cz - half)
        x_max = min(depth, x_min + min_size)
        y_max = min(height, y_min + min_size)
        z_max = min(width, z_min + min_size)
        return slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max)

    x_min, x_max = int(coords[0].min()), int(coords[0].max()) + 1
    y_min, y_max = int(coords[1].min()), int(coords[1].max()) + 1
    z_min, z_max = int(coords[2].min()), int(coords[2].max()) + 1

    depth, height, width = mask.shape
    x_min, x_max = sup_128(x_min, x_max, min_size, depth)
    y_min, y_max = sup_128(y_min, y_max, min_size, height)
    z_min, z_max = sup_128(z_min, z_max, min_size, width)

    return slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max)


def normalize_channels(volume: np.ndarray) -> np.ndarray:
    """Z-score normalize each channel within the non-zero brain region."""
    out = volume.copy()
    mask = np.sum(out, axis=0) > 0
    if not np.any(mask):
        return out
    for idx in range(out.shape[0]):
        channel = out[idx]
        region = channel[mask]
        mean = region.mean()
        std = region.std()
        if std < 1e-6:
            std = 1.0
        out[idx] = (channel - mean) / std
    return out


def load_modalities(case_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    case_id = case_dir.name
    suffix_map: dict[str, str] = {
        "flair": "-t2f.nii.gz",
        "t1ce": "-t1c.nii.gz",
        "t1": "-t1n.nii.gz",
        "t2": "-t2w.nii.gz",
        "seg": "-seg.nii.gz",
    }
    arrays: dict[str, np.ndarray] = {}
    for key, suffix in suffix_map.items():
        fname = case_dir / f"{case_id}{suffix}"
        if not fname.exists():
            raise FileNotFoundError(f"Missing modality {key} for {case_id}: {fname}")
        img = nib.load(str(fname))
        data = img.get_fdata(dtype=np.float32)
        arrays[key] = data
    vol = np.stack(
        (arrays["flair"], arrays["t1ce"], arrays["t1"], arrays["t2"]),
        axis=0,
    )
    seg = arrays["seg"].astype(np.uint8)
    return vol, seg


def process_case(
    case_dir: Path,
    out_id: str,
    out_vol_dir: Path,
    out_seg_dir: Path,
    min_size: int,
    overwrite: bool,
) -> bool:
    vol_path = out_vol_dir / f"{out_id}_vol.npy"
    seg_path = out_seg_dir / f"{out_id}_seg.npy"
    if not overwrite and vol_path.exists() and seg_path.exists():
        return False

    vol, seg = load_modalities(case_dir)
    mask = np.any(vol != 0, axis=0)
    xs, ys, zs = find_crop_bounds(mask, min_size=min_size)

    vol_crop = vol[:, xs, ys, zs]
    vol_norm = normalize_channels(vol_crop)
    vol_out = np.transpose(vol_norm, (1, 2, 3, 0)).astype(np.float32)

    seg_crop = seg[xs, ys, zs].astype(np.uint8)
    seg_crop[seg_crop == 4] = 3

    np.save(vol_path, vol_out)
    np.save(seg_path, seg_crop)
    return True


def generate_splits(
    output_dir: Path,
    base_ids: Sequence[str],
    ratios: SplitRatios,
    seed: int,
) -> None:
    train_ratio, test_ratio, val_ratio = ratios
    total = train_ratio + test_ratio + val_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-6):
        raise ValueError("Split ratios must sum to 1.0")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(base_ids))
    rng.shuffle(indices)

    n_total = len(base_ids)
    n_train = int(round(n_total * train_ratio))
    n_test = int(round(n_total * test_ratio))

    test_start = n_train
    val_start = n_train + n_test

    parts = {
        "train": indices[:n_train],
        "test": indices[test_start:test_start + n_test],
        "val": indices[val_start:],
    }

    for split_name, split_indices in parts.items():
        lines = [base_ids[i] for i in split_indices]
        target = output_dir / f"{split_name}.txt"
        with target.open("w", encoding="ascii") as f:
            for line in lines:
                f.write(f"{line}\n")


def main() -> None:
    args = parse_args()
    raw_dir: Path = args.raw_dir
    out_dir: Path = args.out_dir
    out_vol_dir = out_dir / "vol"
    out_seg_dir = out_dir / "seg"
    out_vol_dir.mkdir(parents=True, exist_ok=True)
    out_seg_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    if args.limit > 0:
        case_dirs = case_dirs[: args.limit]
    if not case_dirs:
        raise RuntimeError(f"No case folders found in {raw_dir}")

    processed_ids: list[str] = []
    mapping: dict[str, str] = {}
    for idx, case_dir in enumerate(case_dirs):
        out_id = f"{args.prefix}_{idx:05d}"
        changed = process_case(
            case_dir=case_dir,
            out_id=out_id,
            out_vol_dir=out_vol_dir,
            out_seg_dir=out_seg_dir,
            min_size=args.min_size,
            overwrite=args.overwrite,
        )
        processed_ids.append(out_id)
        mapping[out_id] = case_dir.name
        status = "processed" if changed else "skipped"
        print(f"[{idx+1:04d}/{len(case_dirs):04d}] {case_dir.name} -> {out_id} ({status})")

    if args.write_mapping:
        mapping_path = out_dir / "mapping.json"
        with mapping_path.open("w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)
        print(f"Saved mapping to {mapping_path}")

    if not args.skip_splits:
        train_ratio, test_ratio, val_ratio = args.split_ratios
        ratios: SplitRatios = (train_ratio, test_ratio, val_ratio)
        generate_splits(
            output_dir=out_dir,
            base_ids=processed_ids,
            ratios=ratios,
            seed=args.seed,
        )
        print("Generated train/test/val splits.")


if __name__ == "__main__":
    main()
