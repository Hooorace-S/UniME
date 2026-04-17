from typing import Tuple

import torch
import numpy as np


def evaluate_single_sample(
    output_classes: torch.Tensor,
    target_classes: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the performance of the segmentation for BraTS dataset.

    <IMPORTANT>:
        - In original BraTS dataset, the classes are:
            - 0: background
            - 1: NCR/NET
            - 2: Edema
            - 4: Enhancing tumor
          But we have converted the classes in data processing stage to:
            - 0: background
            - 1: NCR/NET
            - 2: Edema
            - 3: Enhancing tumor
        - The standard evaluation areas in BraTS are:
            - WT: (label == 1) | (label == 2) | (label == 3)
            - TC: (label == 1) | (label == 3)
            - ET: (label == 3)

    Args:
        output_classes (torch.Tensor): The output classes with shape (N, H, W, D).
        target_classes (torch.Tensor): The target classes with shape (N, H, W, D).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - dice_separate: original dice score for each class (N, 3) for NCR/NET, Edema, ET
            - dice_evaluation: dice score for each evaluation area (N, 4) for WT, TC, ET, ET_postprocessed
    """
    eps = 1e-8
    dims = (1, 2, 3)  # Spatial dimensions (H, W, D)

    # Create binary masks for each class
    pred_ncr = (output_classes == 1).float()
    target_ncr = (target_classes == 1).float()

    pred_edema = (output_classes == 2).float()
    target_edema = (target_classes == 2).float()

    pred_et = (output_classes == 3).float()
    target_et = (target_classes == 3).float()

    # Compute dice scores for individual classes
    dice_ncr = _compute_dice_score(pred_ncr, target_ncr, dims, eps)
    dice_edema = _compute_dice_score(pred_edema, target_edema, dims, eps)
    dice_et = _compute_dice_score(pred_et, target_et, dims, eps)

    # Post-processing for enhancing tumor (threshold at 500 voxels)
    pred_et_post = torch.where(
        torch.sum(pred_et, dim=dims, keepdim=True) < 500,
        torch.zeros_like(pred_et),
        pred_et
    )
    dice_et_post = _compute_dice_score(pred_et_post, target_et, dims, eps)

    # Compute evaluation regions
    # Whole Tumor (WT): Union of all tumor classes (1, 2, 3)
    pred_wt = pred_ncr + pred_edema + pred_et
    target_wt = target_ncr + target_edema + target_et
    dice_wt = _compute_dice_score(pred_wt, target_wt, dims, eps)

    # Tumor Core (TC): NCR/NET + Enhancing Tumor (1, 3)
    pred_tc = pred_ncr + pred_et
    target_tc = target_ncr + target_et
    dice_tc = _compute_dice_score(pred_tc, target_tc, dims, eps)

    # Stack results
    dice_separate = torch.stack([dice_ncr, dice_edema, dice_et], dim=1)
    dice_evaluation = torch.stack([dice_wt, dice_tc, dice_et, dice_et_post], dim=1)

    return dice_separate.cpu().numpy(), dice_evaluation.cpu().numpy()


def _compute_dice_score(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    dims: Tuple[int, ...] = (1, 2, 3),
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute Dice score for binary masks.

    Args:
        pred_mask (torch.Tensor): Predicted binary mask with shape (N, H, W, D)
        target_mask (torch.Tensor): Target binary mask with shape (N, H, W, D)
        dims (Tuple[int, ...]): Dimensions to sum over, default to (1, 2, 3)
        eps (float): Small constant to avoid division by zero

    Returns:
        Dice score tensor with shape (N, )
    """
    intersection = torch.sum(2 * pred_mask * target_mask, dim=dims) + eps
    union = torch.sum(pred_mask, dim=dims) + torch.sum(target_mask, dim=dims) + eps
    return intersection / union
