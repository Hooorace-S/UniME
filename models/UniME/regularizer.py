from typing import cast

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.UniME.blocks import ConvBlock


class AuxiliaryRegularizer(nn.Module):
    """
    Auxiliary Regularizer for single modality.
    """
    def __init__(self, num_classes: int = 4, basic_dims: int = 16, use_checkpoint: bool = False) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.stages = nn.ModuleList([
            self._create_up_stage(basic_dims * 8, basic_dims * 4),   # Stage 3
            self._create_up_stage(basic_dims * 4, basic_dims * 2),   # Stage 2
            self._create_up_stage(basic_dims * 2, basic_dims),       # Stage 1
        ])

        self.seg_head = nn.Conv3d(
            in_channels=basic_dims,
            out_channels=num_classes,
            kernel_size=1, stride=1, padding=0, bias=True
        )

    def _create_up_stage(self, in_ch: int, out_ch: int) -> nn.ModuleDict:
        return nn.ModuleDict({
            'upsample': nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            'up_reduction': ConvBlock(in_ch, out_ch),
            'cat_reduction': ConvBlock(out_ch * 2, out_ch),
            'conv': ConvBlock(out_ch, out_ch, kernel_size=1, padding=0)
        })

    def _stage_forward(self, stage: nn.ModuleDict, x: torch.Tensor, enc_feat: torch.Tensor) -> torch.Tensor:
        x_up = stage['upsample'](x)
        x_up = stage['up_reduction'](x_up)
        x_cat = torch.cat((x_up, enc_feat), dim=1)
        return stage['conv'](stage['cat_reduction'](x_cat))

    def _checkpoint_stage_forward(
        self,
        stage: nn.ModuleDict,
        x: torch.Tensor,
        enc_feat: torch.Tensor,
    ) -> torch.Tensor:
        # Static typing only: checkpoint() preserves the wrapped function's runtime return type,
        # but Pylance/Pyright cannot infer the tuple precisely here.
        return cast(
            torch.Tensor,
            checkpoint(
                self._stage_forward,
                stage,
                x,
                enc_feat,
                use_reentrant=False,
            )
        )

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x0 (torch.Tensor): (B, basic_dims * num_modals, D, H, W).
            x1 (torch.Tensor): (B, basic_dims*2 * num_modals, D/2, H/2, W/2).
            x2 (torch.Tensor): (B, basic_dims*4 * num_modals, D/4, H/4, W/4).
            x3 (torch.Tensor): (B, embed_dims, D/8, H/8, W/8).

        Returns:
            torch.Tensor: (B, num_classes, D, H, W).
        """
        inputs = [x2, x1, x0]
        x = x3

        for stage, enc_feat in zip(self.stages, inputs):
            stage = cast(nn.ModuleDict, stage)
            if self.use_checkpoint and self.training:
                x = self._checkpoint_stage_forward(stage, x, enc_feat)
            else:
                x = self._stage_forward(stage, x, enc_feat)

        pred = self.seg_head(x)
        return pred
