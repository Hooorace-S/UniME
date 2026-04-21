import os
import random
from typing import TypedDict, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from source.dataset.dataset import BratsTrainDataset, BratsValidationDataset, BratsTestDataset
from source.config import TrainingConfig


class LoaderKwargs(TypedDict, total=False):
    pin_memory: bool
    num_workers: int
    generator: torch.Generator
    worker_init_fn: Callable[[int], None]
    persistent_workers: bool
    prefetch_factor: int


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.set_num_threads(1)


def _build_loader_kwargs(args: TrainingConfig, generator: torch.Generator) -> LoaderKwargs:
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


def setup_dataloader(args: TrainingConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load data for training, validation and testing.

    Args:
        args (TrainingConfig): Training configuration.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Data loaders for train, valid and test.
    """
    root = os.path.join(args.data_path, args.dataset_name)
    match args.split_type:
        case "Normal":
            train_file = os.path.join(root, "train.txt")
            val_file = os.path.join(root, "val.txt")
            test_file = os.path.join(root, "test.txt")
        case "Split1":
            train_file = os.path.join(root, "train1.txt")
            val_file = os.path.join(root, "val1.txt")
            test_file = os.path.join(root, "test1.txt")
        case "Split2":
            train_file = os.path.join(root, "train2.txt")
            val_file = os.path.join(root, "val2.txt")
            test_file = os.path.join(root, "test2.txt")
        case "Split3":
            train_file = os.path.join(root, "train3.txt")
            val_file = os.path.join(root, "val3.txt")
            test_file = os.path.join(root, "test3.txt")
        case _:
            raise ValueError(f"Invalid split type: {args.split_type}")

    if args.dataset_name in {"BRATS2020", "BRATS2021", "BRATS2023"} and args.train_ratio is not None:
        if args.split_type != "Normal":
            raise ValueError("train_ratio is only supported when split_type is Normal")
        train_ratio_file = os.path.join(root, f"train_ratio{args.train_ratio}.txt")
        if not os.path.exists(train_ratio_file):
            raise FileNotFoundError(
                f"Few-shot split file not found: {train_ratio_file}. "
                "Generate it via the provided scripts before training."
            )
        train_file = train_ratio_file

    datasets = {
        "train": BratsTrainDataset(root, train_file, args.num_classes, args.crop_size),
        "val": BratsValidationDataset(root, val_file, args.num_classes),
        "test": BratsTestDataset(root, test_file, args.num_classes),
    }

    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(args.seed))
    dataloader_kwargs = _build_loader_kwargs(args, generator=loader_generator)

    train_loader = DataLoader(
        datasets['train'], batch_size=args.batch_size, shuffle=True, **dataloader_kwargs
    )
    val_loader = DataLoader(
        datasets['val'], batch_size=1, shuffle=False, **dataloader_kwargs
    )
    test_loader = DataLoader(
        datasets['test'], batch_size=1, shuffle=False, **dataloader_kwargs
    )

    return (train_loader, val_loader, test_loader)
