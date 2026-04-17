import random
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
            RandCrop3D(size=(crop_size, crop_size, crop_size)),
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
