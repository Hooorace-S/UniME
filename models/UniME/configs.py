from typing import TypedDict, cast

from models.Backbone.config import BackboneConfig


LAYER_DECAY_DEFAULT: float = 0.75


class ScaleSpec(TypedDict):
    embed_dim: int
    depth: int
    num_heads: int
    basic_dims: int


_SCALE_SPECS: dict[str, ScaleSpec] = {
    "Original": {"embed_dim": 864, "depth": 16, "num_heads": 12, "basic_dims": 16},
    "Base": {"embed_dim": 864, "depth": 16, "num_heads": 12, "basic_dims": 32},
    "Small": {"embed_dim": 792, "depth": 12, "num_heads": 12, "basic_dims": 32},
    "Tiny": {"embed_dim": 600, "depth": 12, "num_heads": 10, "basic_dims": 24},
    "Nano": {"embed_dim": 384, "depth": 12, "num_heads": 8, "basic_dims": 16},
}


def _build_scale_config(*, embed_dim: int, depth: int, num_heads: int) -> BackboneConfig:
    return BackboneConfig(
        in_channels=4,
        out_channels=4,
        crop_shape=(96, 96, 96),
        patch_shape=(8, 8, 8),
        tokens_spatial_shape=(12, 12, 12),
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=0.2,
        layer_scale_init_value=0.1,
        num_register_tokens=4,
        patch_drop_rate=0.0,
    )


def get_scale_spec(size: str) -> ScaleSpec:
    key = size.title()
    if key not in _SCALE_SPECS:
        raise ValueError(f"Unknown UniEncoder size '{size}'. Supported sizes: {sorted(_SCALE_SPECS.keys())}.")
    return cast(ScaleSpec, dict(_SCALE_SPECS[key]))


def get_scale_config(size: str) -> BackboneConfig:
    spec = get_scale_spec(size)
    return _build_scale_config(
        embed_dim=spec["embed_dim"],
        depth=spec["depth"],
        num_heads=spec["num_heads"],
    )


OriginalScaleConfig = get_scale_config("Original")
BaseScaleConfig = get_scale_config("Base")
SmallScaleConfig = get_scale_config("Small")
TinyScaleConfig = get_scale_config("Tiny")
NanoScaleConfig = get_scale_config("Nano")
