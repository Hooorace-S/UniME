from typing import Tuple

from models.Backbone.backbone import ViT
from models.Backbone.config import BackboneConfig


class EVA02(ViT):
    """
    EVA092 Style ViT model with <medium> configuration:
        - layers: 16
        - num_heads: 12
        - embed_dim: 864
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        crop_shape: Tuple[int, int, int],
        patch_shape: Tuple[int, int, int],
        tokens_spatial_shape: Tuple[int, int, int],
        embed_dim: int = 864,
        depth: int = 16,
        num_heads: int = 12,
        num_register_tokens: int = 0,
        patch_drop_rate: float = 0.0,
    ) -> None:
        super().__init__(
            BackboneConfig(
                in_channels=in_channels,
                out_channels=out_channels,
                crop_shape=crop_shape,
                patch_shape=patch_shape,
                tokens_spatial_shape=tokens_spatial_shape,
                embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                drop_path_rate=0.2, layer_scale_init_value=0.1,
                num_register_tokens=num_register_tokens,
                patch_drop_rate=patch_drop_rate,
            )
        )
