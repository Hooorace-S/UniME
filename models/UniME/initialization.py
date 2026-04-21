from __future__ import annotations

from torch import nn

from models.UniME.blocks import ECABlock


def init_unime_conv_module(module: nn.Module) -> None:
    if isinstance(module, nn.Conv3d):
        nn.init.kaiming_normal_(
            module.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(
            module.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, (nn.InstanceNorm3d, nn.BatchNorm3d, nn.GroupNorm)):
        if getattr(module, "weight", None) is not None:
            nn.init.ones_(module.weight)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def init_seg_head(head: nn.Conv3d, std: float = 1e-3) -> None:
    nn.init.trunc_normal_(head.weight, std=std)
    if head.bias is not None:
        nn.init.zeros_(head.bias)


def reset_eca_identity(root: nn.Module) -> None:
    for module in root.modules():
        if isinstance(module, ECABlock):
            module.reset_parameters()


def zero_init_encoder_residual_stages(encoder: nn.Module) -> None:
    for stage in encoder.stages:
        final_block = stage["conv"].block
        norm = final_block[1]

        if isinstance(norm, nn.InstanceNorm3d) and norm.weight is not None:
            nn.init.zeros_(norm.weight)
            if norm.bias is not None:
                nn.init.zeros_(norm.bias)
        else:
            conv = final_block[0]
            if isinstance(conv, nn.Conv3d):
                nn.init.zeros_(conv.weight)
                if conv.bias is not None:
                    nn.init.zeros_(conv.bias)


def initialize_cnn_encoder(encoder: nn.Module) -> None:
    encoder.apply(init_unime_conv_module)
    zero_init_encoder_residual_stages(encoder)


def initialize_cnn_decoder(
    decoder: nn.Module,
    *,
    head_init_std: float = 1e-3,
) -> None:
    decoder.apply(init_unime_conv_module)
    reset_eca_identity(decoder)

    # init_seg_head(decoder.stage_head, std=head_init_std)
    # for stage in decoder.stages:
    #     init_seg_head(stage["seg_head"], std=head_init_std)


def initialize_auxiliary_regularizer(
    regularizer: nn.Module,
    *,
    head_init_std: float = 1e-3,
) -> None:
    regularizer.apply(init_unime_conv_module)
    # init_seg_head(regularizer.seg_head, std=head_init_std)
