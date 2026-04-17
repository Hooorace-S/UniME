from typing import Tuple

import torch
from torch import nn


class Tokenizer3D(nn.Module):
    """
    Tokenizer by using a single conv where kernel_size=stride=patch_size
    """
    def __init__(
        self, in_channels: int, embed_dim: int, patch_size: Tuple[int, int, int] = (8, 8, 8)
    ) -> None:
        """
        Args:
            in_channels (int): The number of channels in the input tensor.
            embed_dim (int): The dimension of the output tensor.
            patch_size (Tuple[int, int, int], optional): The size of the patch.
        """
        super().__init__()
        k = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=k, stride=k, padding=0, bias=True)
        self.patch = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input tensor with shape (B, C, D, H, W).

        Returns:
            torch.Tensor: The output tensor with shape (B, embed_dim, D', H', W').

        <NOTE>:
            - D' = H' = W' = D // patch_size = H // patch_size = W // patch_size
            - The sequence length for further transformer is D' * H' * W'
        """
        return self.proj(x)
