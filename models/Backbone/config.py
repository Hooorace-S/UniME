from dataclasses import dataclass
from typing import Tuple


@dataclass
class BackboneConfig:
    """
    Configuration for the Backbone.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels, i.e. number of classes
        crop_shape (Tuple[int, int, int]): shape of the crop patch, i.e. (D, H, W)
        embed_dim (int): dimension of the embedding
        depth (int): number of layers
        num_heads (int): number of heads for the attention
        patch_shape (Tuple[int, int, int]): shape of the patch, i.e. (D, H, W)
        drop_path_rate (float): dropout rate for the drop path
        patch_drop_rate (float): dropout rate for the patch drop (applied during training; evaluation keeps all patches)
        mlp_dropout (float): dropout rate for the MLP
        attn_dropout (float): dropout rate for the attention
        proj_dropout (float): dropout rate for the projection
        layer_scale_init_value (float): initial value for the layer scale
        rope_base (float): base temperature for the RoPE3D
        use_learned_pos_embed (bool): whether to use the learned positional embedding
        use_rotary_pos_embed (bool): whether to use the rotary positional embedding
        num_register_tokens (int): number of register tokens
    """
    in_channels: int = 1
    out_channels: int = 4
    crop_shape: Tuple[int, int, int] = (80, 80, 80)
    patch_shape: Tuple[int, int, int] = (8, 8, 8)
    embed_dim: int = 396
    depth: int = 12
    num_heads: int = 6
    drop_path_rate: float = 0.2
    patch_drop_rate: float = 0.05
    return_mask: bool = False
    mlp_dropout: float = 0.0
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    layer_scale_init_value: float = 0.1
    rope_base: float = 10000.0
    use_learned_pos_embed: bool = True
    tokens_spatial_shape: Tuple[int, int, int] = (10, 10, 10)
    use_rotary_pos_embed: bool = True
    num_register_tokens: int = 0
