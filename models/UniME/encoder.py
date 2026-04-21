from typing import Tuple, cast

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.UniME.blocks import ConvBlock

EncoderFeatures = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class CNNEncoder(nn.Module):
    """
    CNN Encoder for extracting multi-scale features.
    """
    def __init__(
        self,
        in_channels: int = 1,
        basic_dims: int = 16,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        # Construct encoder stages
        self.stages = nn.ModuleList([
            self._create_stage(in_channels, basic_dims, stride=1),
            self._create_stage(basic_dims, basic_dims * 2),
            self._create_stage(basic_dims * 2, basic_dims * 4),
            self._create_stage(basic_dims * 4, basic_dims * 8)
        ])

    def _create_stage(self, in_ch: int, out_ch: int, stride: int = 2) -> nn.ModuleDict:
        return nn.ModuleDict({
            'down': ConvBlock(in_ch, out_ch, stride=stride),
            'convnext': ConvBlock(out_ch, out_ch),
            'conv': ConvBlock(out_ch, out_ch, norm_affine=True)
        })

    def _stage_forward(self, stage: nn.ModuleDict, x: torch.Tensor) -> torch.Tensor:
        x_down = stage['down'](x)
        x_next = stage['convnext'](x_down)
        return stage['conv'](x_next) + x_down

    def _checkpoint_stage_forward(
        self,
        stage: nn.ModuleDict,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # Static typing only: checkpoint() preserves the wrapped function's runtime return type,
        # but Pylance/Pyright cannot infer the tuple precisely here.
        return cast(
            torch.Tensor,
            checkpoint(
                self._stage_forward,
                stage,
                x,
                use_reentrant=False,
            )
        )

    def forward(self, x: torch.Tensor) -> EncoderFeatures:
        """
        Args:
            x (torch.Tensor): Input tensor shape (B, C, D, H, W)

        Returns:
            Tuple of tensors from each encoder stage with decreasing spatial resolutions.
        """
        outputs: list[torch.Tensor] = []

        for stage in self.stages:
            stage = cast(nn.ModuleDict, stage)
            if self.use_checkpoint and self.training:
                x = self._checkpoint_stage_forward(stage, x)
            else:
                x = self._stage_forward(stage, x)
            outputs.append(x)

        return outputs[0], outputs[1], outputs[2], outputs[3]
