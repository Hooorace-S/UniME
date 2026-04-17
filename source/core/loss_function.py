from typing import Tuple

import torch
from torch import nn


class LossStrategy(nn.Module):
    """
    Loss strategy for incomplete multi-modal segmentation.

    Tree types of outputs are included to compute loss:
        - fusion_output: main output of the model with shape (N, num_classes, H, W, D).
        - deep_supervision_outputs: deep supervision outputs of the model with shape (N, num_classes, H, W, D), default to 4.
        - auxiliary_outputs: auxiliary outputs (of different modalities) of the model with shape (N, num_classes, H, W, D).

    Args:
        required_auxiliary_losses (bool):
         - Whether to include auxiliary losses in the loss calculation.
         - Default is False, i.e., do not include auxiliary losses.
    """
    def __init__(self, required_auxiliary: bool = False) -> None:
        super().__init__()
        self.required_auxiliary = required_auxiliary
        self.dice_loss = DiceLoss(include_background=False)
        self.weighted_ce_loss = WeightedCrossEntropyLoss()

    def forward(
        self,
        fusion_output: torch.Tensor,
        deep_supervision_outputs: Tuple[torch.Tensor, ...],
        auxiliary_outputs: Tuple[torch.Tensor, ...],
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate loss for incomplete multi-modal segmentation.

        Args:
            fusion_output (torch.Tensor): Fusion output with shape (N, num_classes, H, W, D).
            deep_supervision_outputs (Tuple[torch.Tensor, ...]): Deep supervision outputs with shape (N, num_classes, H, W, D).
            auxiliary_outputs (Tuple[torch.Tensor, ...]): Auxiliary outputs with shape (N, num_classes, H, W, D).

        <IMPORTANT>: ALL outputs are assumed <WITHOUT> SoftMax or LogSoftMax.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - loss_fusion: Loss for fusion output.
                - loss_deep_supervision: Loss for deep supervision outputs (sum of losses for all levels).
                - loss_auxiliary: Loss for auxiliary outputs (sum of losses for all modalities).
        """
        loss_fusion = self.dice_loss(fusion_output, target) + self.weighted_ce_loss(fusion_output, target)
        zero = torch.tensor(0.0, device=fusion_output.device, dtype=fusion_output.dtype)
        loss_deep_supervision = sum(
            (
                self.dice_loss(output, target) + self.weighted_ce_loss(output, target)
                for output in deep_supervision_outputs
            ),
            start=zero
        )
        if self.required_auxiliary:
            loss_auxiliary = sum(
                (
                    self.dice_loss(output, target) + self.weighted_ce_loss(output, target)
                    for output in auxiliary_outputs
                ),
                start=zero
            )
            return loss_fusion, loss_deep_supervision, loss_auxiliary
        return loss_fusion, loss_deep_supervision, zero


class DiceLoss(nn.Module):
    """
    Standard multi-class mean Dice loss for 3D medical image segmentation.

    Args:
        include_background (bool):
         - Whether to include background class (channel 0) in loss.
         - Default is False, i.e., ignore background.
        eps (float):
         - Smoothing constant to avoid division by zero.
    """
    def __init__(self, include_background: bool = False, eps: float = 1e-4):
        super().__init__()
        self.include_background = include_background
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss for 3D volumes.

        Args:
            logits (torch.Tensor): Output <logits> with shape (N, num_classes, H, W, D).
            target (torch.Tensor): Ground truth with shape (N, num_classes, H, W, D).

        Returns:
            torch.Tensor: Scalar Dice loss.
        """
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)

        # Compute per-channel Dice over batch and spatial dims
        dims = (0, 2, 3, 4)  # (batch, _, H, W, D)
        intersection = (probs * target).sum(dim=dims)
        probs_sum = probs.sum(dim=dims)
        target_sum = target.sum(dim=dims)
        dice_per_classes = (
            (2.0 * intersection + self.eps) / (probs_sum + target_sum + self.eps)
        )

        # Optionally exclude background (channel 0) when more than 1 class exists
        if not self.include_background and num_classes > 1:
            dice_per_classes = dice_per_classes[1:]

        loss_per_classes = 1.0 - dice_per_classes
        return loss_per_classes.mean()


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted cross entropy loss for 3D medical image segmentation.

    Notes:
         - The weights are computed based on the number of voxels in each class for each sample;
         - Rare classes are assigned higher weights;
         - w_{b,c} = 1 - (#voxels of class c in sample b) / (total voxels in sample b)

    Args:
        eps (float): Smoothing constant to avoid division by zero.
    """
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Output <logits> with shape (N, num_classes, H, W, D).
            target (torch.Tensor): Ground truth with shape (N, num_classes, H, W, D).

        Returns:
            torch.Tensor: Scalar weighted cross entropy loss.
        """
        spatial_dims = (2, 3, 4)
        # Stable log-probabilities for (B, _, H, W, D)
        log_probs = torch.log_softmax(logits, dim=1)

        # class-frequency weights per sample
        counts = target.sum(dim=spatial_dims, keepdim=True)
        weights = 1.0 - (
            counts / counts.sum(dim=1, keepdim=True).clamp_min(self.eps)
        )

        weighted_ce_loss = (- weights * target * log_probs).sum(dim=1).mean()
        return weighted_ce_loss
