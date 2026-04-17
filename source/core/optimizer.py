from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from source.config import TrainingConfig
    from source.pretrain.parse import PretrainConfig


def setup_optimizer(args: TrainingConfig | PretrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    """
    Setup optimizer for training with AdamW.

    Args:
        args (TrainingConfig): Training configuration
        model (nn.Module): Model

    Returns:
        torch.optim.Optimizer: Optimizer
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )
    return optimizer


def _uniencoder_layer_info(param_name: str, depth: int) -> tuple[int, str]:
    """
    Map UniEncoder parameter names to (layer_idx, label) pairs.
    """
    stem_prefixes = (
        "tokenizer",
        "pos_embed",
        "register_tokens",
        "mask_token",
        "patch_drop",
        "rope",
    )
    if param_name.startswith(stem_prefixes):
        return 0, "pre_blocks"

    if param_name.startswith("blocks"):
        parts = param_name.split(".")
        if len(parts) > 1:
            try:
                index = int(parts[1])
            except ValueError:
                index = depth - 1
            index = max(0, min(index, depth - 1))
            return index + 1, f"blocks.{index}"
        return 1, "blocks.0"

    if param_name.startswith("after_trans_norm"):
        return depth + 1, "after_trans_norm"

    # Fallback to final layer to keep overall schedule stable.
    return depth + 1, "after_trans_norm"


def _build_uniencoder_layer_scales(
    depth: int,
    layer_decay: float,
    backbone_multiplier: float = 1.0
) -> list[float]:
    """
    Create geometric layer scales following the FT-CLIP layer decay recipe.

    The sequence applies ``layer_decay ** (num_layers + 1 - i)`` (with ``i`` starting
    at 1) scaled by an optional backbone multiplier. The final element is forced to
    ``1.0`` so that the classification head keeps the scheduled base learning rate.
    """
    num_layers = depth + 2  # stem + transformer blocks + after_trans_norm
    scales: list[float] = []
    for idx in range(num_layers):
        exponent = (num_layers) - (idx + 1)
        scale = (layer_decay ** exponent) * backbone_multiplier
        scales.append(scale)

    # Ensure the head (last entry) keeps the base learning rate
    if scales:
        scales[-1] = 1.0
    return scales


def setup_lldr_optimizer(args: TrainingConfig, model: nn.Module) -> torch.optim.Optimizer:
    """
    Setup AdamW optimizer with layer-wise LR decay for UniEncoder parameters.
    """
    uni_encoder_wrapper = getattr(model, "uni_encoder", None)
    uni_encoder_module = getattr(uni_encoder_wrapper, "tok_uniencoder", None)
    if uni_encoder_module is None:
        raise ValueError("TokUniEncoder module is required to enable UniME LLDR.")

    if not getattr(model, "layerwise_lr_decay_enabled", False):
        raise ValueError("LLDR requested but UniME is not in finetune mode or layer_decay is unset.")

    layer_decay_value: float | None = getattr(model, "layer_decay", None)
    if layer_decay_value is None:
        layer_decay_value = getattr(args, "layer_decay", None)

    if layer_decay_value is None:
        raise ValueError("Layer decay value must be provided to configure LLDR.")

    layer_decay_value = float(layer_decay_value)

    if layer_decay_value <= 0:
        raise ValueError("layer_decay must be a positive float.")

    depth = getattr(getattr(uni_encoder_module, "config", None), "depth", None)
    if depth is None:
        raise ValueError("TokUniEncoder configuration depth is unavailable for LLDR.")

    layer_scales = _build_uniencoder_layer_scales(depth, layer_decay_value)
    id_to_label: dict[int, str] = {
        0: "pre_blocks",
        **{idx + 1: f"blocks.{idx}" for idx in range(depth)},
        depth + 1: "after_trans_norm",
    }
    uni_encoder_prefix = "uni_encoder.tok_uniencoder."

    uni_encoder_groups: dict[int, dict[str, list[object]]] = defaultdict(lambda: {"params": [], "names": []})
    general_params: list[torch.nn.Parameter] = []
    general_names: list[str] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith(uni_encoder_prefix):
            sub_name = name[len(uni_encoder_prefix):]
            layer_id, derived_label = _uniencoder_layer_info(sub_name, depth)
            layer_id = min(layer_id, len(layer_scales) - 1)
            uni_encoder_groups[layer_id]["params"].append(param)
            uni_encoder_groups[layer_id]["names"].append(name)
            id_to_label.setdefault(layer_id, derived_label)
        else:
            general_params.append(param)
            general_names.append(name)

    param_groups: list[dict[str, object]] = []
    for layer_id in sorted(uni_encoder_groups):
        params = uni_encoder_groups[layer_id]["params"]
        if not params:
            continue
        label = id_to_label.get(layer_id, f"layer_{layer_id}")
        param_groups.append({
            "params": params,
            "weight_decay": args.weight_decay,
            "lr_scale": layer_scales[layer_id]
        })

    if general_params:
        param_groups.append({
            "params": general_params,
            "weight_decay": args.weight_decay,
            "lr_scale": 1.0
        })

    if not param_groups:
        raise ValueError("No trainable parameters collected while configuring LLDR optimizer.")

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.base_lr,
        betas=(0.9, 0.999),
        weight_decay=0.0  # per-group weight decay already specified
    )
    layer_schedule: list[dict[str, object]] = []
    for layer_id, scale in enumerate(layer_scales):
        label = id_to_label.get(layer_id, f"layer_{layer_id}")
        names = (
            sorted(str(name) for name in uni_encoder_groups[layer_id]["names"])
            if layer_id in uni_encoder_groups else
            []
        )
        layer_schedule.append({
            "layer_id": layer_id,
            "label": label,
            "lr_scale": scale,
            "parameter_names": names
        })

    setattr(optimizer, "unime_layer_scales", layer_scales)
    setattr(optimizer, "unime_layer_schedule", layer_schedule)
    setattr(optimizer, "unime_non_uni_encoder_names", sorted(general_names))

    return optimizer
