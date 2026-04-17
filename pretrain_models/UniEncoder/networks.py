from typing import Optional, Tuple, cast

import torch
from torch import nn

from models.Backbone.networks import EVA02
from pretrain_models.UniEncoder.mask_tools import tokenizer, apply_mask

Location3DTorch = Tuple[
    Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
]


class UniEncoderClass(nn.Module):
    """
    Wrapper around UniEncoder with learnable prior.
    """
    def __init__(
        self,
        backbone: nn.Module,
        *,
        in_channels: int = 4,
        pre_train: bool = True,
        crop_size: int = 96,
        patch_size: int = 8,
        patch_mask_ratio: float = 0.75,
        original_shape: Tuple[int, int, int] = (176, 205, 154),
        num_mask_modalities: int = 3
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pre_train = pre_train
        self.num_modals = in_channels
        self.patch_size = patch_size
        self.mask_ratio = patch_mask_ratio
        self.num_mask_modalities = num_mask_modalities
        self.crop_size = crop_size

        # learnable prior to replace masked regions
        self.learnable_prior = nn.Parameter(
            torch.randn((1, in_channels, original_shape[0], original_shape[1], original_shape[2]))
        )
        # dummy input for mask sampling
        self.register_buffer(
            "dummy_input",
            tokenizer(
                torch.ones((1, in_channels, crop_size, crop_size, crop_size)),
                patch_size=patch_size
            ),
            persistent=False
        )

    def forward(
        self, x: torch.Tensor, location: Optional[Location3DTorch] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert location is not None
        batch_size = x.shape[0]
        # For static typing only, as register_buffer does not preserve type information
        dummy_input = cast(torch.Tensor, self.dummy_input)

        mask = apply_mask(
            batch_size=batch_size,
            patch_size=self.patch_size,
            patch_mask_ratio=self.mask_ratio,
            raw_input=dummy_input,
            num_mask_modalities=self.num_mask_modalities,
            use_patch_mask=True,
            crop_size=self.crop_size
        )
        (d0, _d1), (h0, _h1), (w0, _w1) = location
        depth, height, width = x.shape[2:]

        prior = self.learnable_prior.expand(batch_size, -1, -1, -1, -1)
        d_idx = d0[:, None] + torch.arange(depth, device=x.device, dtype=d0.dtype)[None, :]
        h_idx = h0[:, None] + torch.arange(height, device=x.device, dtype=h0.dtype)[None, :]
        w_idx = w0[:, None] + torch.arange(width, device=x.device, dtype=w0.dtype)[None, :]

        lim = prior.gather(
            2,
            d_idx[:, None, :, None, None].expand(-1, prior.shape[1], -1, prior.shape[3], prior.shape[4]),
        )
        lim = lim.gather(
            3,
            h_idx[:, None, None, :, None].expand(-1, lim.shape[1], lim.shape[2], -1, lim.shape[4]),
        )
        lim = lim.gather(
            4,
            w_idx[:, None, None, None, :].expand(-1, lim.shape[1], lim.shape[2], lim.shape[3], -1),
        )
        x = x * mask + lim * (1 - mask)
        return self.backbone(x), lim


class UniEncoder(UniEncoderClass):
    """
    Masked-pretraining wrapper for UniEncoder with:
        - num_register_tokens: 4
        - patch_drop_rate: 0.75
        - crop_size: 96
        - patch_size: 8
        - embed_dim: 864
        - depth: 16
        - num_heads: 12
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_register_tokens: int = 4,
        patch_drop_rate: float = 0.0,
        pre_train: bool = True,
        crop_size: int = 96,
        patch_size: int = 8,
        patch_mask_ratio: float = 0.75,
        original_shape: Tuple[int, int, int] = (176, 205, 154),
        num_mask_modalities: int = 3,
        embed_dim: int = 864,
        depth: int = 16,
        num_heads: int = 12
    ):
        super().__init__(
            backbone=EVA02(
                in_channels=in_channels,
                out_channels=out_channels,
                crop_shape=(crop_size, crop_size, crop_size),
                patch_shape=(patch_size, patch_size, patch_size),
                tokens_spatial_shape=(
                    crop_size // patch_size, crop_size // patch_size, crop_size // patch_size
                ),
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                num_register_tokens=num_register_tokens,
                patch_drop_rate=patch_drop_rate
            ),
            in_channels=in_channels,
            pre_train=pre_train,
            crop_size=crop_size,
            patch_size=patch_size,
            patch_mask_ratio=patch_mask_ratio,
            original_shape=original_shape,
            num_mask_modalities=num_mask_modalities,
        )


class UniEncoderBase(UniEncoderClass):
    """
    Masked-pretraining wrapper for UniEncoder with:
        - num_register_tokens: 4
        - patch_drop_rate: 0.75
        - crop_size: 96
        - patch_size: 8
        - embed_dim: 864
        - depth: 16
        - num_heads: 12
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_register_tokens: int = 4,
        patch_drop_rate: float = 0.0,
        pre_train: bool = True,
        crop_size: int = 96,
        patch_size: int = 8,
        patch_mask_ratio: float = 0.75,
        original_shape: Tuple[int, int, int] = (176, 205, 154),
        num_mask_modalities: int = 3,
        embed_dim: int = 864,
        depth: int = 16,
        num_heads: int = 12
    ):
        super().__init__(
            backbone=EVA02(
                in_channels=in_channels,
                out_channels=out_channels,
                crop_shape=(crop_size, crop_size, crop_size),
                patch_shape=(patch_size, patch_size, patch_size),
                tokens_spatial_shape=(
                    crop_size // patch_size, crop_size // patch_size, crop_size // patch_size
                ),
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                num_register_tokens=num_register_tokens,
                patch_drop_rate=patch_drop_rate
            ),
            in_channels=in_channels,
            pre_train=pre_train,
            crop_size=crop_size,
            patch_size=patch_size,
            patch_mask_ratio=patch_mask_ratio,
            original_shape=original_shape,
            num_mask_modalities=num_mask_modalities,
        )


class UniEncoderSmall(UniEncoderClass):
    """
    Masked-pretraining wrapper for UniEncoder with:
        - num_register_tokens: 4
        - patch_drop_rate: 0.75
        - crop_size: 96
        - patch_size: 8
        - embed_dim: 792
        - depth: 12
        - num_heads: 12
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_register_tokens: int = 4,
        patch_drop_rate: float = 0.0,
        pre_train: bool = True,
        crop_size: int = 96,
        patch_size: int = 8,
        patch_mask_ratio: float = 0.75,
        original_shape: Tuple[int, int, int] = (176, 205, 154),
        num_mask_modalities: int = 3,
        embed_dim: int = 792,
        depth: int = 12,
        num_heads: int = 12
    ):
        super().__init__(
            backbone=EVA02(
                in_channels=in_channels,
                out_channels=out_channels,
                crop_shape=(crop_size, crop_size, crop_size),
                patch_shape=(patch_size, patch_size, patch_size),
                tokens_spatial_shape=(
                    crop_size // patch_size, crop_size // patch_size, crop_size // patch_size
                ),
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                num_register_tokens=num_register_tokens,
                patch_drop_rate=patch_drop_rate
            ),
            in_channels=in_channels,
            pre_train=pre_train,
            crop_size=crop_size,
            patch_size=patch_size,
            patch_mask_ratio=patch_mask_ratio,
            original_shape=original_shape,
            num_mask_modalities=num_mask_modalities,
        )


class UniEncoderTiny(UniEncoderClass):
    """
    Masked-pretraining wrapper for UniEncoder with:
        - num_register_tokens: 4
        - patch_drop_rate: 0.75
        - crop_size: 96
        - patch_size: 8
        - embed_dim: 600
        - depth: 12
        - num_heads: 10
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_register_tokens: int = 4,
        patch_drop_rate: float = 0.0,
        pre_train: bool = True,
        crop_size: int = 96,
        patch_size: int = 8,
        patch_mask_ratio: float = 0.75,
        original_shape: Tuple[int, int, int] = (176, 205, 154),
        num_mask_modalities: int = 3,
        embed_dim: int = 600,
        depth: int = 12,
        num_heads: int = 10
    ):
        super().__init__(
            backbone=EVA02(
                in_channels=in_channels,
                out_channels=out_channels,
                crop_shape=(crop_size, crop_size, crop_size),
                patch_shape=(patch_size, patch_size, patch_size),
                tokens_spatial_shape=(
                    crop_size // patch_size, crop_size // patch_size, crop_size // patch_size
                ),
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                num_register_tokens=num_register_tokens,
                patch_drop_rate=patch_drop_rate
            ),
            in_channels=in_channels,
            pre_train=pre_train,
            crop_size=crop_size,
            patch_size=patch_size,
            patch_mask_ratio=patch_mask_ratio,
            original_shape=original_shape,
            num_mask_modalities=num_mask_modalities,
        )


class UniEncoderNano(UniEncoderClass):
    """
    Masked-pretraining wrapper for UniEncoder with:
        - num_register_tokens: 4
        - patch_drop_rate: 0.75
        - crop_size: 96
        - patch_size: 8
        - embed_dim: 384
        - depth: 12
        - num_heads: 8
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_register_tokens: int = 4,
        patch_drop_rate: float = 0.0,
        pre_train: bool = True,
        crop_size: int = 96,
        patch_size: int = 8,
        patch_mask_ratio: float = 0.75,
        original_shape: Tuple[int, int, int] = (176, 205, 154),
        num_mask_modalities: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 8
    ):
        super().__init__(
            backbone=EVA02(
                in_channels=in_channels,
                out_channels=out_channels,
                crop_shape=(crop_size, crop_size, crop_size),
                patch_shape=(patch_size, patch_size, patch_size),
                tokens_spatial_shape=(
                    crop_size // patch_size, crop_size // patch_size, crop_size // patch_size
                ),
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                num_register_tokens=num_register_tokens,
                patch_drop_rate=patch_drop_rate
            ),
            in_channels=in_channels,
            pre_train=pre_train,
            crop_size=crop_size,
            patch_size=patch_size,
            patch_mask_ratio=patch_mask_ratio,
            original_shape=original_shape,
            num_mask_modalities=num_mask_modalities,
        )
