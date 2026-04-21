import random
import warnings
from collections import OrderedDict
from typing import Sequence

import numpy as np
from scipy.ndimage import rotate

ImagePair = list[np.ndarray]
TransformInput = np.ndarray | ImagePair


class Base:
    """Base transformation class for 3D medical images."""
    def sample(self, *shape: int) -> list[int]:
        """Sample transformation parameters based on input shape.

        Args:
            *shape: Input spatial dimensions

        Returns:
            List of output dimensions
        """
        return list(shape)

    def tf(self, img: np.ndarray, k: int = 0) -> np.ndarray:  # noqa
        """Apply transformation to input image.

        Args:
            img: Input image
            k: Index for paired transformations (0=image, 1=label)

        Returns:
            Transformed image
        """
        _ = k  # silence unused-argument lint while preserving API
        return img

    def __call__(
        self, img: TransformInput, dim: int = 3, reuse: bool = False
    ) -> TransformInput:
        """Call the transformation on input data.

        Args:
            img: Input image or list of [image, label]
            dim: Number of spatial dimensions
            reuse: Whether to reuse previously sampled parameters

        Returns:
            Transformed image(s)
        """
        if not reuse:
            im: np.ndarray = img[0] if not isinstance(img, np.ndarray) else img
            # For unbatched data, use the first `dim` spatial dimensions (H, W, D)
            shape = im.shape[:dim]
            self.sample(*shape)

        if not isinstance(img, np.ndarray):
            return [self.tf(x, k) for k, x in enumerate(img)]

        return self.tf(img)

    def __str__(self) -> str:
        return 'Identity()'


# Alias for Identity transform
Identity = Base


class Compose(Base):
    """Compose multiple transformations together."""
    def __init__(self, ops: Base | Sequence[Base]) -> None:
        if isinstance(ops, Base):
            self.ops: tuple[Base, ...] = (ops,)
        else:
            self.ops = tuple(ops)

    def sample(self, *shape: int) -> list[int]:
        shape_list = list(shape)
        for op in self.ops:
            shape_list = op.sample(*shape_list)
        return shape_list

    def tf(self, img: np.ndarray, k: int = 0) -> np.ndarray:
        result = img
        for op in self.ops:
            result = op.tf(result, k)
        return result

    def __call__(
        self, img: TransformInput, dim: int = 3, reuse: bool = False
    ) -> TransformInput:
        result = img
        for op in self.ops:
            result = op(result, dim=dim, reuse=reuse)
        return result

    def __str__(self) -> str:
        ops = ', '.join([str(op) for op in self.ops])
        return f'Compose([{ops}])'


class RandCrop3D(Base):
    """Randomly crop a 3D volume with specified dimensions."""
    def __init__(self, size: int | tuple[int, int, int] | list[int]) -> None:
        self.size = [size, size, size] if isinstance(size, int) else list(size)
        self.start: list[int] | None = None

    def sample(self, *shape: int) -> list[int]:
        assert len(self.size) == 3, "Size must specify 3 dimensions (H,W,D)"
        # shape corresponds to (H, W, D)
        self.start = [random.randint(0, max(0, s - i)) for i, s in zip(self.size, shape)]
        return self.size.copy()

    def tf(self, img: np.ndarray, k: int = 0) -> np.ndarray:
        # Build slices depending on whether the input has channel dim (image) or not (label)
        assert self.start is not None
        sh, sw, sd = self.start
        kh, kw, kd = self.size
        if img.ndim == 4:  # (H, W, D, C)
            slices = (slice(sh, sh + kh), slice(sw, sw + kw), slice(sd, sd + kd), slice(None))
        else:  # (H, W, D)
            slices = (slice(sh, sh + kh), slice(sw, sw + kw), slice(sd, sd + kd))
        return img[slices]

    def __str__(self) -> str:
        return f'RandCrop3D({self.size})'


class RandCrop3DByPosNegLabel(RandCrop3D):
    """Randomly crop a 3D volume using WT-aware pos/neg label sampling."""

    def __init__(
        self,
        size: int | tuple[int, int, int] | list[int],
        pos: float = 1.0,
        neg: float = 1.0,
        image_threshold: float | None = None,
        allow_smaller: bool = False,
        fg_cache_size: int = 32,
        bg_sample_attempts: int = 512,
        bg_fallback_chunk_voxels: int = 1_000_000,
    ) -> None:
        super().__init__(size=size)
        if pos < 0 or neg < 0:
            raise ValueError(f"pos and neg must be nonnegative, got pos={pos} neg={neg}.")
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.requested_size = self.size.copy()
        self.pos_ratio = pos / (pos + neg)
        self.image_threshold = image_threshold
        self.allow_smaller = bool(allow_smaller)
        self.fg_cache_size = max(0, int(fg_cache_size))
        self.bg_sample_attempts = max(1, int(bg_sample_attempts))
        self.bg_fallback_chunk_voxels = max(1, int(bg_fallback_chunk_voxels))
        self._fg_cache: OrderedDict[tuple[str, tuple[int, ...], str], np.ndarray] = OrderedDict()
        self._warned: set[str] = set()

    def _resolve_crop_size(self, shape: Sequence[int]) -> list[int]:
        crop_size = self.requested_size.copy()
        for idx, (requested, actual) in enumerate(zip(crop_size, shape)):
            if actual < requested:
                if not self.allow_smaller:
                    raise ValueError(
                        "The size of the proposed random crop ROI is larger than the image size, "
                        f"got ROI size {tuple(self.requested_size)} and label image size {tuple(shape)} respectively."
                    )
                crop_size[idx] = actual
        return crop_size

    def _center_bounds(self, shape: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        crop_size = np.asarray(self.size, dtype=np.int64)
        spatial_shape = np.asarray(shape, dtype=np.int64)
        valid_start = np.floor_divide(crop_size, 2)
        valid_end = np.subtract(spatial_shape + 1, crop_size / 2.0).astype(np.int64)
        valid_end = np.maximum(valid_end, valid_start + 1)
        return valid_start, valid_end

    def _correct_crop_center(self, center: Sequence[int], shape: Sequence[int]) -> list[int]:
        valid_start, valid_end = self._center_bounds(shape)
        return [
            int(min(max(coord, start), end - 1))
            for coord, start, end in zip(center, valid_start, valid_end)
        ]

    @staticmethod
    def _unravel_index3(flat_index: int, shape: Sequence[int]) -> list[int]:
        flat_index = int(flat_index)
        width_depth = int(shape[1]) * int(shape[2])
        h = flat_index // width_depth
        rem = flat_index - h * width_depth
        w = rem // int(shape[2])
        d = rem - w * int(shape[2])
        return [int(h), int(w), int(d)]

    def _label_cache_key(
        self,
        label: np.ndarray,
    ) -> tuple[str, tuple[int, ...], str] | None:
        filename = getattr(label, "filename", None)
        if filename is None:
            return None
        return (str(filename), tuple(int(x) for x in label.shape), str(label.dtype))

    def _foreground_indices(self, label: np.ndarray) -> np.ndarray:
        key = self._label_cache_key(label)
        if key is not None:
            cached = self._fg_cache.pop(key, None)
            if cached is not None:
                self._fg_cache[key] = cached
                return cached

        fg_indices = np.flatnonzero(label != 0)
        if label.size <= np.iinfo(np.uint32).max:
            fg_indices = fg_indices.astype(np.uint32, copy=False)

        if key is not None and self.fg_cache_size > 0:
            self._fg_cache[key] = fg_indices
            while len(self._fg_cache) > self.fg_cache_size:
                self._fg_cache.popitem(last=False)

        return fg_indices

    def _passes_image_threshold(
        self,
        image: np.ndarray,
        center: tuple[int, int, int],
    ) -> bool:
        if self.image_threshold is None:
            return True
        return bool(np.any(image[center] > self.image_threshold))

    def _sample_positive_center(
        self,
        fg_indices: np.ndarray,
        shape: Sequence[int],
    ) -> list[int]:
        flat_index = fg_indices[random.randrange(fg_indices.size)]
        center = self._unravel_index3(int(flat_index), shape)
        return self._correct_crop_center(center, shape)

    def _sample_background_center_fast(
        self,
        image: np.ndarray,
        label: np.ndarray,
        shape: Sequence[int],
    ) -> list[int] | None:
        valid_start, valid_end = self._center_bounds(shape)
        for _ in range(self.bg_sample_attempts):
            center = tuple(
                random.randrange(int(valid_start[idx]), int(valid_end[idx]))
                for idx in range(3)
            )
            if label[center] != 0:
                continue
            if not self._passes_image_threshold(image, center):
                continue
            return [int(coord) for coord in center]
        return None

    def _sample_background_flat_index_chunked(
        self,
        image: np.ndarray,
        label: np.ndarray,
    ) -> int | None:
        flat_label = label.reshape(-1)
        flat_image = None
        if self.image_threshold is not None:
            if image.ndim != 4:
                raise ValueError(
                    "image_threshold requires image with shape (H, W, D, C), "
                    f"got image shape {image.shape}."
                )
            flat_image = image.reshape((-1, image.shape[-1]))

        selected: int | None = None
        seen = 0
        for start in range(0, label.size, self.bg_fallback_chunk_voxels):
            end = min(start + self.bg_fallback_chunk_voxels, label.size)
            mask = np.asarray(flat_label[start:end]) == 0
            if flat_image is not None:
                mask &= np.any(
                    np.asarray(flat_image[start:end]) > self.image_threshold,
                    axis=-1,
                )
            count = int(mask.sum())
            if count == 0:
                continue
            if random.randrange(seen + count) < count:
                local_indices = np.flatnonzero(mask)
                selected = start + int(local_indices[random.randrange(count)])
            seen += count
        return selected

    def _sample_background_center_exact(
        self,
        image: np.ndarray,
        label: np.ndarray,
        shape: Sequence[int],
    ) -> list[int] | None:
        flat_index = self._sample_background_flat_index_chunked(image, label)
        if flat_index is None:
            return None
        center = self._unravel_index3(flat_index, shape)
        return self._correct_crop_center(center, shape)

    def _warn_once(self, code: str, message: str) -> None:
        if code in self._warned:
            return
        warnings.warn(message, UserWarning, stacklevel=3)
        self._warned.add(code)

    def _sample_start_from_pair(self, image: np.ndarray, label: np.ndarray) -> None:
        if label.ndim != 3:
            raise ValueError(f"Expected label with shape (H, W, D), got {label.shape}.")
        if image.shape[:3] != label.shape[:3]:
            raise ValueError(
                "Image and label spatial shapes must match, "
                f"got image shape {image.shape} and label shape {label.shape}."
            )

        shape = tuple(int(x) for x in label.shape[:3])
        self.size = self._resolve_crop_size(shape)
        fg_indices = self._foreground_indices(label)
        has_fg = fg_indices.size > 0
        want_pos = random.random() < self.pos_ratio

        center: list[int] | None = None
        if want_pos and has_fg:
            center = self._sample_positive_center(fg_indices, shape)
        else:
            if want_pos and not has_fg:
                self._warn_once(
                    "no_foreground",
                    "No foreground voxels found; falling back to background sampling.",
                )
            center = self._sample_background_center_fast(image, label, shape)
            if center is None:
                center = self._sample_background_center_exact(image, label, shape)
            if center is None:
                if has_fg:
                    self._warn_once(
                        "no_background",
                        "No valid background voxels found; falling back to foreground sampling.",
                    )
                    center = self._sample_positive_center(fg_indices, shape)
                else:
                    raise ValueError("No sampling location available.")

        assert center is not None
        self.start = [coord - size // 2 for coord, size in zip(center, self.size)]

    def sample(self, *shape: int) -> list[int]:
        self.size = self._resolve_crop_size(shape)
        return super().sample(*shape)

    def __call__(
        self, img: TransformInput, dim: int = 3, reuse: bool = False
    ) -> TransformInput:
        if reuse or isinstance(img, np.ndarray):
            return super().__call__(img, dim=dim, reuse=reuse)

        image, label = img
        self._sample_start_from_pair(image=image, label=label)
        return [self.tf(x, k) for k, x in enumerate(img)]

    def __str__(self) -> str:
        return (
            "RandCrop3DByPosNegLabel("
            f"size={self.requested_size}, pos_ratio={self.pos_ratio:.3f}, "
            f"image_threshold={self.image_threshold}, allow_smaller={self.allow_smaller})"
        )


class RandomRotation(Base):
    """Apply random 3D rotation to the input volume."""
    def __init__(self, angle_spectrum: int = 10):
        assert isinstance(angle_spectrum, int), "Angle spectrum must be an integer"
        self.angle_spectrum = angle_spectrum
        # For unbatched data, spatial dims are (H=0, W=1, D=2)
        self.axes = ((0, 1), (1, 2), (0, 2))
        self.axes_buffer: tuple[int, int] | None = None
        self.angle_buffer: int | None = None

    def sample(self, *shape: int) -> list[int]:
        self.axes_buffer = self.axes[random.randint(0, 2)]
        self.angle_buffer = random.randint(-self.angle_spectrum, self.angle_spectrum)
        return list(shape)

    def tf(self, img: np.ndarray, k: int = 0) -> np.ndarray:
        # Labels must fill with background so rotation padding does not create invalid class ids.
        fill_value = 0 if k == 1 else -1
        assert self.axes_buffer is not None
        assert self.angle_buffer is not None

        # Unbatched rotation. For images with channels, rotate each channel.
        if img.ndim == 4 and k == 0:  # (H, W, D, C)
            result = np.empty_like(img)
            channels = img.shape[3]
            for c in range(channels):
                result[:, :, :, c] = rotate(
                    img[:, :, :, c],
                    self.angle_buffer,
                    axes=self.axes_buffer,
                    reshape=False,
                    order=0,
                    mode='constant',
                    cval=fill_value
                )
            return result
        else:
            return rotate(
                img,
                self.angle_buffer,
                axes=self.axes_buffer,
                reshape=False,
                order=0,
                mode='constant',
                cval=fill_value
            )

    def __str__(self) -> str:
        return f'RandomRotion(axes={self.axes_buffer}, Angle:{self.angle_buffer})'


class RandomFlip(Base):
    """
    Randomly flip the volume along multiple axes.

    Args:
        axis: Axis to flip along
    """
    def __init__(self):
        # For unbatched data, spatial dims are (0, 1, 2)
        self.axis = (0, 1, 2)
        self.flip_state = 0

    def sample(self, *shape: int) -> list[int]:
        self.flip_state = random.randint(0, 7)
        return list(shape)

    def tf(self, img: np.ndarray, k: int = 0) -> np.ndarray:
        if self.flip_state & 1:
            img = np.flip(img, axis=self.axis[0])
        if self.flip_state & 2:
            img = np.flip(img, axis=self.axis[1])
        if self.flip_state & 4:
            img = np.flip(img, axis=self.axis[2])
        return img


class RandomIntensityChange(Base):
    """Randomly change intensity with shift and scale factors."""
    def __init__(self, factor: tuple[float, float]) -> None:
        shift, scale = factor
        assert (shift > 0) and (scale > 0), "Shift and scale must be positive"
        self.shift = shift
        self.scale = scale
        self.shift_range = (-self.shift, self.shift)
        self.scale_range = (1.0 - self.scale, 1.0 + self.scale)

    def tf(self, img: np.ndarray, k: int = 0) -> np.ndarray:
        if k == 1:
            return img

        height = img.shape[0]
        channels = img.shape[-1]

        shift_factor = np.random.uniform(
            self.shift_range[0],
            self.shift_range[1],
            size=(height, 1, 1, channels)
        )

        scale_factor = np.random.uniform(
            self.scale_range[0],
            self.scale_range[1],
            size=(height, 1, 1, channels)
        )

        return img * scale_factor + shift_factor


class NumpyType(Base):
    """
    Convert array to specified numpy dtype.

    Args:
        types: Tuple of numpy dtypes
        num: Index of the type to use
    """
    def __init__(self, types: tuple[np.dtype, np.dtype], num: int = -1) -> None:
        self.types = tuple(np.dtype(t) if not isinstance(t, np.dtype) else t for t in types)
        self.num = num

    def tf(self, img: np.ndarray, k: int = 0) -> np.ndarray:
        if self.num > 0 and k >= self.num:
            return img

        dtype_idx = min(k, len(self.types) - 1)
        return img.astype(self.types[dtype_idx], copy=False)

    def __str__(self) -> str:
        s = ', '.join([str(s) for s in self.types])
        return f'NumpyType(({s}))'


def get_train_transforms(crop_size: int = 80) -> Compose:
    """
    Construct the training transforms.
    """
    return Compose(
        [
            RandCrop3DByPosNegLabel(size=(crop_size, crop_size, crop_size)),
            RandomRotation(angle_spectrum=10),
            RandomFlip(),
            RandomIntensityChange(factor=(0.1, 0.1)),
            NumpyType(types=(np.dtype(np.float32), np.dtype(np.int64)))
        ]
    )


def get_test_transforms() -> Compose:
    """
    Construct the test transforms.
    """
    return Compose(
        [
            NumpyType(types=(np.dtype(np.float32), np.dtype(np.int64)))
        ]
    )


def get_val_transforms() -> Compose:
    """
    Construct the validation transforms.
    """
    return Compose(
        [
            NumpyType(types=(np.dtype(np.float32), np.dtype(np.int64)))
        ]
    )


def get_pretrain_transforms(crop_size: int = 128) -> Compose:
    """
    Construct the pretraining transforms.
    """
    return Compose(
        [
            RandCrop3D(size=(crop_size, crop_size, crop_size)),
            NumpyType(types=(np.dtype(np.float32), np.dtype(np.int64)))
        ]
    )
