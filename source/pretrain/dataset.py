import os
from collections import OrderedDict
from typing import Callable, TypedDict, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data._utils.collate import default_collate

from source.config import MASK_ARRAY
from source.dataset.augmentation import Compose, RandCrop3D, get_pretrain_transforms
from source.pretrain.parse import PretrainConfig

Location3D = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
PretrainSample = tuple[torch.Tensor, Location3D]
PretrainAuxSample = tuple[torch.Tensor, Location3D, torch.Tensor, str]
PretrainBatch = tuple[torch.Tensor, list[Location3D]]
PretrainAuxBatch = tuple[torch.Tensor, list[Location3D], torch.Tensor, list[str]]


class LoaderKwargs(TypedDict, total=False):
    pin_memory: bool
    num_workers: int
    generator: torch.Generator
    worker_init_fn: Callable[[int], None]
    persistent_workers: bool
    prefetch_factor: int
    collate_fn: Callable[[object], object]


def crop3d_with_loc(volume: np.ndarray, transform: Compose) -> tuple[np.ndarray, Location3D]:
    """
    Apply the given transform to `volume` and extract the 3D crop location when
    a RandCrop3D op is present. Location is reported in (D, H, W) order to match
    downstream Missing expectations.
    """
    cropped_volume = cast(np.ndarray, transform(volume))
    loc: Location3D = ((0, cropped_volume.shape[2]), (0, cropped_volume.shape[0]), (0, cropped_volume.shape[1]))

    if isinstance(transform, Compose):
        for op in transform.ops:
            if isinstance(op, RandCrop3D) and op.start is not None:
                sh, sw, sd = op.start
                kh, kw, kd = op.size
                loc = ((sd, sd + kd), (sh, sh + kh), (sw, sw + kw))
                break

    return cropped_volume, loc


class RepeatedPermutationSampler(Sampler[int]):
    """
    Sample concatenated shuffled dataset passes until a fixed sample budget is reached.
    """

    def __init__(
        self,
        data_source: Dataset,
        num_samples: int,
        generator: torch.Generator | None = None,
    ) -> None:
        # if len(data_source) <= 0:
        #     raise ValueError("RepeatedPermutationSampler requires a non-empty dataset.")
        # if int(num_samples) <= 0:
        #     raise ValueError(f"num_samples must be > 0, got {num_samples}.")
        self.data_source = data_source
        self.num_samples = int(num_samples)
        self.generator = generator

    def __iter__(self):
        # Static typing only: Dataset is not guaranteed to be Sized, but this sampler requires len(data_source).
        dataset_len = len(self.data_source)  # type: ignore[arg-type]
        remaining = self.num_samples
        while remaining > 0:
            permutation = torch.randperm(dataset_len, generator=self.generator).tolist()
            take = min(remaining, dataset_len)
            yield from permutation[:take]
            remaining -= take

    def __len__(self) -> int:
        return self.num_samples


def _pretrain_collate(batch: object) -> object:
    if isinstance(batch, tuple) and len(batch) in {2, 4} and torch.is_tensor(batch[0]):
        return batch
    return default_collate(batch)


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    torch.set_num_threads(1)


def _build_loader_kwargs(args: PretrainConfig, generator: torch.Generator) -> LoaderKwargs:
    kwargs: LoaderKwargs = {
        "pin_memory": True,
        "num_workers": args.num_workers,
        "generator": generator,
    }
    if args.num_workers > 0:
        kwargs["worker_init_fn"] = _seed_worker
        kwargs["persistent_workers"] = bool(args.persistent_workers)
        kwargs["prefetch_factor"] = int(args.prefetch_factor)
    return kwargs


class PretrainDataset(Dataset):
    """
    Unified BraTS pretraining dataset returning volume, crop location, mask, and sample name.
    """

    def __init__(
        self,
        root: str,
        file_path: str,
        crop_size: int = 128,
        transform: Compose | None = None,
        max_cached_handles: int = 32,
        include_auxiliary: bool = True,
    ) -> None:
        self.names = self._read_names(file_path)
        self.volpaths = [os.path.join(root, "vol", f"{name}_vol.npy") for name in self.names]
        self.transform = transform if transform is not None else get_pretrain_transforms(crop_size=crop_size)
        self.mask_array = MASK_ARRAY
        self._max_cached_handles = max(1, int(max_cached_handles))
        self._volume_cache: OrderedDict[str, np.memmap] = OrderedDict()
        self.include_auxiliary = bool(include_auxiliary)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> PretrainSample | PretrainAuxSample:
        volume, loc = self._build_sample(index)
        if not self.include_auxiliary:
            return volume, loc
        return volume, loc, torch.from_numpy(self._sample_mask()), self.names[index]

    def __getitems__(self, indices: list[int]) -> PretrainBatch | PretrainAuxBatch:
        if not indices:
            if self.include_auxiliary:
                return torch.empty(0), [], torch.empty(0), []
            return torch.empty(0), []

        first_volume, first_loc = self._build_sample(indices[0])
        batch_size = len(indices)
        images = torch.empty((batch_size,) + tuple(first_volume.shape), dtype=first_volume.dtype)
        locs = [first_loc]
        masks = torch.empty(0, dtype=torch.bool)
        names: list[str] = []

        images[0].copy_(first_volume)
        if self.include_auxiliary:
            first_mask = torch.from_numpy(self._sample_mask())
            masks = torch.empty((batch_size,) + tuple(first_mask.shape), dtype=first_mask.dtype)
            names = [self.names[indices[0]]]
            masks[0].copy_(first_mask)

        for batch_idx, sample_idx in enumerate(indices[1:], start=1):
            volume, loc = self._build_sample(sample_idx)
            images[batch_idx].copy_(volume)
            locs.append(loc)
            if self.include_auxiliary:
                masks[batch_idx].copy_(torch.from_numpy(self._sample_mask()))
                names.append(self.names[sample_idx])

        if self.include_auxiliary:
            return images, locs, masks, names
        return images, locs

    def _sample_mask(self) -> np.ndarray:
        idx = np.random.randint(0, len(self.mask_array))
        return self.mask_array[idx].astype(np.bool_, copy=False)

    def _build_sample(self, index: int) -> tuple[torch.Tensor, Location3D]:
        volume_handle = self._get_volume_handle(index)
        cropped_volume, loc = crop3d_with_loc(volume_handle, self.transform)

        cropped_volume = np.transpose(cropped_volume, (3, 2, 0, 1)).copy()
        volume = torch.from_numpy(cropped_volume)
        return volume, loc

    def _get_volume_handle(self, index: int) -> np.memmap:
        path = self.volpaths[index]
        cached = self._volume_cache.pop(path, None)
        if cached is not None:
            self._volume_cache[path] = cached
            return cached

        handle = np.load(path, mmap_mode="c")
        self._volume_cache[path] = handle
        if len(self._volume_cache) > self._max_cached_handles:
            _, evicted = self._volume_cache.popitem(last=False)
            mmap = getattr(evicted, "_mmap", None)
            close = getattr(mmap, "close", None)
            if callable(close):
                close()
        return handle

    @staticmethod
    def _read_names(file_path: str) -> list[str]:
        with open(file_path, "r", encoding="utf-8") as f:
            return sorted([ln.strip() for ln in f.readlines() if ln.strip()])

    def close(self) -> None:
        while self._volume_cache:
            _, handle = self._volume_cache.popitem(last=False)
            mmap = getattr(handle, "_mmap", None)
            close = getattr(mmap, "close", None)
            if callable(close):
                close()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass


def setup_pretrain_dataloader(
    args: PretrainConfig,
    *,
    full_dataset_pass: bool = False,
    num_processes: int = 1,
) -> DataLoader:
    """
    Build the pretraining dataloader using the unified dataset.
    """
    root = os.path.join(args.data_path, args.dataset_name)
    match args.split_type:
        case "Normal":
            train_file = os.path.join(root, "train.txt")
        case "Split1":
            train_file = os.path.join(root, "train1.txt")
        case "Split2":
            train_file = os.path.join(root, "train2.txt")
        case "Split3":
            train_file = os.path.join(root, "train3.txt")
        case _:
            raise ValueError(f"Invalid split type: {args.split_type}")

    if args.dataset_name in {"BRATS2020", "BRATS2021"} and args.train_ratio is not None:
        if args.split_type != "Normal":
            raise ValueError("train_ratio is only supported when split_type is Normal")
        ratio_file = os.path.join(root, f"train_ratio{args.train_ratio}.txt")
        if not os.path.exists(ratio_file):
            raise FileNotFoundError(f"Few-shot split not found: {ratio_file}")
        train_file = ratio_file

    dataset = PretrainDataset(root, train_file, crop_size=args.crop_size, include_auxiliary=False)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(args.seed))
    kwargs = _build_loader_kwargs(args, generator=loader_generator)
    kwargs["collate_fn"] = _pretrain_collate

    if full_dataset_pass:
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, **kwargs)

    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(int(args.seed) + 1)
    global_batch_count = max(1, int(num_processes))
    sampler = RepeatedPermutationSampler(
        dataset,
        num_samples=int(args.iter_per_epoch) * int(args.batch_size) * global_batch_count,
        generator=sampler_generator,
    )
    return DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, drop_last=True, **kwargs)
