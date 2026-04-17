from typing import Optional, Tuple

import torch
from torch import nn


class DropPath(nn.Module):
    """
    DropPath module. Used to randomly drop out some of the input tokens.
    """

    def __init__(self, drop_prob: float = 0.0):
        """
        Args:
            drop_prob (float, optional): The probability of dropping out some of the input tokens.
        """
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input tensor with any shape.

        Returns:
            torch.Tensor: The output tensor with the same shape as the input tensor.
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        sample_mask = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            sample_mask = sample_mask.div_(keep_prob)
        return x * sample_mask


class SwiGLU(nn.Module):
    """
    Eva-02 style MLP block.
    """
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        hidden_features = int(round((8.0 / 3.0) * dim))
        self.fc1_g = nn.Linear(dim, hidden_features, bias=True)
        self.fc1_x = nn.Linear(dim, hidden_features, bias=True)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_features)
        self.fc2 = nn.Linear(hidden_features, dim, bias=True)
        self.drop2 = nn.Dropout(dropout)

    def init_weights(self) -> None:
        """
        Initialize the weights of the module.
        """
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        if self.fc1_g.bias is not None:
            nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        LayerNorm -> (W1 x) * SiLU(W2 x) -> LayerNorm -> Linear -> Dropout

        Args:
            x (torch.Tensor): The input tensor with shape (B, seq_len, dim).

        Returns:
            torch.Tensor: The output tensor with shape (B, seq_len, dim).
        """
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LayerNormNd(nn.Module):
    """
    Layernorm for N-dimensional tensor.
    """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape (B, C, *spatial_dims)

        Returns:
            torch.Tensor: output tensor with shape (B, C, *spatial_dims)
        """
        x = x.permute(0, *range(2, x.ndim), 1)
        x = self.norm(x)
        x = x.permute(0, x.ndim - 1, *range(1, x.ndim - 1))
        return x


def restore_full_sequence(
    tokens_seq: torch.Tensor,
    mask_token: torch.Tensor,
    keep_indices: Optional[torch.Tensor],
    num_patches: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Restore the full sequence from the tokens and keep indices.

    Args:
        tokens_seq (torch.Tensor): input tokens sequence with shape (batch_size, num_kept, dim)
        mask_token (torch.Tensor): <learnable> mask token with shape (1, 1, dim)
        keep_indices (torch.Tensor | None): keep indices with shape (batch_size, num_kept).
            When ``None`` (e.g. during evaluation or when patch dropout is disabled), the
            input sequence is returned unchanged.
        num_patches (int): number of patches in the full sequence, i.e. the full sequence length

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: restored sequence with shape (batch_size, num_patches, dim),
        restored mask with shape (batch_size, num_patches), where True indicates the patch is kept
    """
    batch_size, num_kept, _ = tokens_seq.shape
    device = tokens_seq.device

    if keep_indices is None:
        if num_kept != num_patches:
            msg = (
                "restore_full_sequence received keep_indices=None but the provided "
                f"tokens sequence length ({num_kept}) does not match num_patches ({num_patches})."
            )
            raise ValueError(msg)
        restored_mask = torch.ones(batch_size, num_patches, dtype=torch.bool, device=device)
        return tokens_seq, restored_mask

    restored = mask_token.repeat(batch_size, num_patches, 1)
    restored_mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)

    # batch generate indices for scatter
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(-1).repeat(1, num_kept)
    restored[batch_indices, keep_indices] = tokens_seq
    restored_mask[batch_indices, keep_indices] = True

    return restored, restored_mask
