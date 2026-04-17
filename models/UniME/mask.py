import torch
from torch import nn


class MaskModal(nn.Module):
    """
    Modal Mask for <Incomplete modality tumor segmentation>
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor with shape (B, K, C, H, W, Z), K is the number of modalities
            mask: boolean mask tensor with shape (B, K)
        Returns:
            output tensor with shape (B, K*C, H, W, Z)

        <NOTE>:
            - True modality will be visible to the model.
            - False modality will be masked out.
        """
        batch_size, num_modalities, channels, height, width, depth = x.size()

        # Validate mask shape
        if mask.shape != (batch_size, num_modalities):
            raise ValueError(f"Expected mask shape {(batch_size, num_modalities)}, got {mask.shape}")

        # Expand mask dimensions to match input tensor
        mask_expanded = mask.view(batch_size, num_modalities, 1, 1, 1, 1)

        # Apply mask directly without intermediate tensor
        masked_x = x * mask_expanded

        # Reshape to merge modality and channel dimensions
        output = masked_x.view(batch_size, num_modalities * channels, height, width, depth)

        return output
