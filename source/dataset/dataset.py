import os
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from source.dataset.augmentation import Compose
from source.dataset.augmentation import (
    get_train_transforms,
    get_test_transforms,
    get_val_transforms
)
from source.config import MASK_ARRAY


class BaseDataset(Dataset):
    """
    Base dataset class for BraTS data loading from txt files.

    Args:
        root (str): Root directory containing 'vol' and 'seg' folders.
        file_path (str): Path to the txt file containing data names.
        num_classes (int): Number of classes for segmentation.
        transform (Callable): Transform function defined in augmentation.py
    """
    def __init__(
        self,
        root: str,
        file_path: str,
        num_classes: int = 4,
        transform: Compose | None = None,
        max_cached_handles: int = 32,
    ) -> None:
        self.num_classes = num_classes
        self.transform = transform
        self.mask_array = MASK_ARRAY
        self._max_cached_handles = max(1, int(max_cached_handles))
        self._volume_cache: OrderedDict[str, np.memmap] = OrderedDict()
        self._label_cache: OrderedDict[str, np.memmap] = OrderedDict()

        self.names = self._load_txt_file(file_path)
        self.volpaths = [os.path.join(root, 'vol', name + '_vol.npy') for name in self.names]
        self.segpaths = [os.path.join(root, 'seg', name + '_seg.npy') for name in self.names]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        volume, label, name = self._load_data(index)
        transform = self.transform
        assert transform is not None

        # data augmentation
        volume, label = transform(
            [volume, label]
        )

        volume = volume.astype(np.float32, copy=False)
        label = label.astype(np.int64, copy=False)

        # channel first & one-hot encoding
        volume = volume.transpose(3, 0, 1, 2)
        label = self._one_hot_encoding(label, self.num_classes)
        volume = np.ascontiguousarray(volume)
        label = np.ascontiguousarray(label)

        # random mask select
        mask = self._get_random_mask(mask_array=self.mask_array)

        # np.ndarray to tensor
        volume = torch.from_numpy(volume)
        label = torch.from_numpy(label)
        mask = torch.from_numpy(mask)

        return volume, label, mask, name

    def __len__(self) -> int:
        return len(self.names)

    def _load_txt_file(self, file_path: str) -> list[str]:
        with open(file_path, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f.readlines() if line.strip()]
        return sorted(names)

    def _load_data(self, index: int) -> tuple[np.ndarray, np.ndarray, str]:
        """
        Load volume and segmentation data.
        Args:
            index (int): The index of the data

        Returns:
            Tuple[np.ndarray, np.ndarray, str]: The volume, segmentation and name
            volume (np.ndarray): The volume with shape (H, W, D, C)
            label (np.ndarray): The label with shape (H, W, D)
            name (str): The name of the data
        """
        volume = self._get_volume_handle(index)
        label = self._get_label_handle(index)
        name = self.names[index]

        return volume, label, name

    def _get_volume_handle(self, index: int) -> np.memmap:
        return self._get_cached_handle(self.volpaths[index], self._volume_cache)

    def _get_label_handle(self, index: int) -> np.memmap:
        return self._get_cached_handle(self.segpaths[index], self._label_cache)

    def _get_cached_handle(self, path: str, cache: OrderedDict[str, np.memmap]) -> np.memmap:
        cached = cache.pop(path, None)
        if cached is not None:
            cache[path] = cached
            return cached

        handle = np.load(path, mmap_mode="c")
        cache[path] = handle
        if len(cache) > self._max_cached_handles:
            _, evicted = cache.popitem(last=False)
            mmap = getattr(evicted, "_mmap", None)
            close = getattr(mmap, "close", None)
            if callable(close):
                close()
        return handle

    def close(self) -> None:
        while self._volume_cache:
            _, handle = self._volume_cache.popitem(last=False)
            mmap = getattr(handle, "_mmap", None)
            close = getattr(mmap, "close", None)
            if callable(close):
                close()
        while self._label_cache:
            _, handle = self._label_cache.popitem(last=False)
            mmap = getattr(handle, "_mmap", None)
            close = getattr(mmap, "close", None)
            if callable(close):
                close()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass

    def _one_hot_encoding(self, label: np.ndarray, num_classes: int) -> np.ndarray:
        """
        One-hot encoding for the label.
        Args:
            label (np.ndarray): The label with shape (H, W, D)

        Returns:
            np.ndarray: The one-hot encoded label with shape (num_classes, H, W, D)
        """
        invalid_mask = (label < 0) | (label >= num_classes)
        if np.any(invalid_mask):
            invalid_values = np.unique(label[invalid_mask]).tolist()
            raise ValueError(
                f"Invalid label values {invalid_values}, expected range [0, {num_classes - 1}]"
            )

        one_hot = np.eye(num_classes, dtype=np.float32)[label.reshape(-1)]
        one_hot = one_hot.reshape(label.shape + (num_classes,))
        one_hot = np.transpose(one_hot, (3, 0, 1, 2))  # [num_classes, H, W, D]
        return one_hot

    def _get_random_mask(self, mask_array: np.ndarray) -> np.ndarray:
        """
        Get a random mask from the mask array.
        Args:
            mask_array (np.ndarray): The mask array with shape (num_modals)
        """
        mask_id = np.random.choice(len(mask_array), 1)
        return mask_array[mask_id].squeeze(0)


class BratsTrainDataset(BaseDataset):
    """BraTS training dataset with augmentation."""
    def __init__(
        self,
        root: str,
        file_path: str,
        num_classes: int = 4,
        crop_size: int = 80,
        max_cached_handles: int = 32,
    ) -> None:
        transform = get_train_transforms(crop_size=crop_size)
        super().__init__(
            root,
            file_path,
            num_classes=num_classes,
            transform=transform,
            max_cached_handles=max_cached_handles,
        )


class BratsValidationDataset(BaseDataset):
    """BraTS validation dataset with augmentation."""
    def __init__(
        self,
        root: str,
        file_path: str,
        num_classes: int = 4,
        max_cached_handles: int = 32,
    ) -> None:
        transform = get_val_transforms()
        super().__init__(
            root,
            file_path,
            num_classes=num_classes,
            transform=transform,
            max_cached_handles=max_cached_handles,
        )


class BratsTestDataset(BaseDataset):
    """BraTS test dataset with augmentation."""
    def __init__(
        self,
        root: str,
        file_path: str,
        num_classes: int = 4,
        max_cached_handles: int = 32,
    ) -> None:
        transform = get_test_transforms()
        super().__init__(
            root,
            file_path,
            num_classes=num_classes,
            transform=transform,
            max_cached_handles=max_cached_handles,
        )
