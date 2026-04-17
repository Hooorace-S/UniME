from typing import Tuple, cast

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.UniME.blocks import ConvBlock, FusionBlock, FusionBlockVariants

DecoderSupervisions = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class CNNDecoder(nn.Module):
    """
    CNN Decoder with modal fusion and deep supervision.
    """
    def __init__(
        self,
        num_classes: int = 4,
        num_modals: int = 4,
        basic_dims: int = 16,
        embed_dims: int = 1056,
        use_checkpoint: bool = False
    ) -> None:
        """
        Args:
            num_classes (int): The number of classes
            num_modals (int): The number of modalities
            basic_dims (int): The number of channels in the intermediate tensor
            embed_dims (int): The number of the embed dims in EVA-02 Style ViT model.
            use_checkpoint (bool): Whether to enable gradient checkpointing on decoder stages.
        """
        super().__init__()

        self.num_modals = num_modals
        self.num_cls = num_classes
        self.use_checkpoint = use_checkpoint

        self.stages = nn.ModuleList([
            self._create_decoder_stage(basic_dims * 8, basic_dims * 4),   # stage 2
            self._create_decoder_stage(basic_dims * 4, basic_dims * 2),   # stage 1
            self._create_decoder_stage(basic_dims * 2, basic_dims),       # stage 0
        ])

        self.stage_fusion = FusionBlockVariants(
            in_channels=embed_dims,
            out_channels=basic_dims * 8
        )
        self.stage_head = nn.Conv3d(
            in_channels=basic_dims * 8,
            out_channels=num_classes,
            kernel_size=1
        )

        self.upsamples = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True),
            nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True),
        ])

    def _create_decoder_stage(self, in_channels: int, out_channels: int) -> nn.ModuleDict:
        return nn.ModuleDict({
            "fusion": FusionBlock(out_channels, num_modals=self.num_modals),
            "up_reduction": ConvBlock(in_channels, out_channels),
            "cat_reduction": ConvBlock(out_channels * 2, out_channels),
            "conv": ConvBlock(out_channels, out_channels, kernel_size=1, padding=0),
            "seg_head": nn.Conv3d(out_channels, self.num_cls, kernel_size=1)
        })

    def _decoder_stage_forward(
        self,
        stage: nn.ModuleDict,
        prev_x: torch.Tensor,
        enc_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        up_x = stage["up_reduction"](self.upsamples[0](prev_x))
        fusion_x = stage["fusion"](enc_x)
        cat_x = torch.cat((fusion_x, up_x), dim=1)
        decoder_x = stage["conv"](stage["cat_reduction"](cat_x))
        pred = stage["seg_head"](decoder_x)
        return decoder_x, pred

    def _fusion_stage_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage_fusion(x)

    def _checkpoint_decoder_stage(
        self,
        stage: nn.ModuleDict,
        prev_x: torch.Tensor,
        enc_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Static typing only: checkpoint() preserves the wrapped function's runtime return type,
        # but Pylance/Pyright cannot infer the tuple precisely here.
        return cast(
            Tuple[torch.Tensor, torch.Tensor],
            checkpoint(
                self._decoder_stage_forward,
                stage,
                prev_x,
                enc_x,
                use_reentrant=False,
            )
        )

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor
    ) -> tuple[torch.Tensor, DecoderSupervisions]:
        """
        Args:
            x0 (torch.Tensor): (B, basic_dims * num_modals, D, H, W).
            x1 (torch.Tensor): (B, basic_dims*2 * num_modals, D/2, H/2, W/2).
            x2 (torch.Tensor): (B, basic_dims*4 * num_modals, D/4, H/4, W/4).
            x3 (torch.Tensor): (B, embed_dims, D/8, H/8, W/8).

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
            - pred0: Main prediction with shape (B, num_cls, D, H, W).
            - deep_supervisions: Deep-supervision outputs.
        """
        outputs: list[torch.Tensor] = []
        # Stage 3 (lowest resolution)
        if self.use_checkpoint and self.training:
            fusion_x = checkpoint(
                self._fusion_stage_forward,
                x3,
                use_reentrant=False
            )
        else:
            fusion_x = self.stage_fusion(x3)
        pred3 = self.stage_head(fusion_x)
        prev_x = fusion_x
        outputs.append(pred3)

        # Stages 2, 1, 0
        for _, (stage, enc_x) in enumerate(zip(self.stages, [x2, x1, x0])):
            stage = cast(nn.ModuleDict, stage)
            if self.use_checkpoint and self.training:
                decoder_x, pred = self._checkpoint_decoder_stage(stage, prev_x, enc_x)
            else:
                decoder_x, pred = self._decoder_stage_forward(stage, prev_x, enc_x)

            outputs.append(pred)
            prev_x = decoder_x

        pred0 = outputs[-1]
        low_res_outputs = outputs[:-1]
        deep_supervisions = [up(pred) for pred, up in zip(low_res_outputs[::-1], self.upsamples)]
        return pred0, (deep_supervisions[0], deep_supervisions[1], deep_supervisions[2])
