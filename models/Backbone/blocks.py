from typing import Optional, Tuple

from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F

from models.Backbone.rope_3d import apply_rope
from models.Backbone.utils import DropPath, SwiGLU


class MHSA3D(nn.Module):
    """
    Multi-head self-attention for 3D images, optimized using RoPE3D.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        num_register_tokens: int = 0
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "embed_dim must be divisible by num_heads."
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        self.num_register_tokens = num_register_tokens

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.post_attn_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        rope_embed: torch.Tensor | None = None,
        spatial_shape: Tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape (B, L, C)
            rope_embed (torch.Tensor | None): optional precomputed RoPE embedding.
            spatial_shape (Tuple[int, int, int] | None): spatial grid shape (H, W, D).

        Returns:
            torch.Tensor: output tensor with shape (B, L, C)
        """
        qkv = rearrange(
            self.qkv(x),
            'b l (three h d) -> three b h l d',
            three=3, h=self.num_heads, d=self.head_dim
        )
        q, k, v = qkv.unbind(0)  # (batch_size, num_heads, seq_len, head_dim)

        if rope_embed is not None:
            assert spatial_shape is not None, "spatial_shape must be provided when using RoPE3D."
            q, k = apply_rope(q, k, rope_embed, self.num_register_tokens)

        attn_out = F.scaled_dot_product_attention(  # pylint: disable=not-callable
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,
        )
        attn_out = rearrange(
            attn_out, 'b h l d -> b l (h d)',
        ).contiguous()

        attn_out = self.post_attn_norm(attn_out)
        attn_out = self.proj_drop(self.proj(attn_out))

        return attn_out


class EVABlock(nn.Module):
    """
    EVA-02 block with:
        - EVA-02 style Multi-head Self-attention with RoPE3D support;
        - EVA-02 style MLP, i.e. SwiGLU with double LayerNorm;
        - DropPath for each residual branch;
        - LayerScale trick on both attn and MLP branches.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dropout: float,
        attn_dropout: float,
        proj_dropout: float,
        drop_path: float,
        layer_scale_init_value: float | None,
        num_register_tokens: int
    ) -> None:
        """
        Args:
            dim (int): dimension of the input and output
            num_heads (int): number of heads for the attention
            mlp_dropout (float): dropout rate for the MLP
            attn_dropout (float): dropout rate for the attention
            proj_dropout (float): dropout rate for the projection
            drop_path (float): dropout rate for the drop path (both attn and MLP)
            layer_scale_init_value (float | None): initial value for the layer scale
            num_register_tokens (int): number of register tokens
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MHSA3D(
            dim=dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            num_register_tokens=num_register_tokens,
        )
        self.drop_path1 = DropPath(drop_path)

        self.ln2 = nn.LayerNorm(dim)
        self.mlp = SwiGLU(dim=dim, dropout=mlp_dropout)
        self.mlp.init_weights()
        self.gamma1: nn.Parameter | None = None
        self.gamma2: nn.Parameter | None = None
        if layer_scale_init_value is not None:
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.drop_path2 = DropPath(drop_path)

    def forward(
        self,
        x: torch.Tensor,
        rope_embed: torch.Tensor | None = None,
        spatial_shape: Tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape (batch_size, seq_len, dim)
            rope_embed (torch.Tensor | None): optional precomputed RoPE embedding.
            spatial_shape (Tuple[int, int, int] | None): spatial grid shape (H, W, D).

        Returns:
            torch.Tensor: output tensor with shape (batch_size, seq_len, dim)
        """
        # attention branch
        attn_out = self.attn(self.ln1(x), rope_embed=rope_embed, spatial_shape=spatial_shape)
        if self.gamma1 is not None:
            attn_out = self.gamma1 * attn_out
        x = x + self.drop_path1(attn_out)

        # MLP branch
        mlp_out = self.mlp(self.ln2(x))
        if self.gamma2 is not None:
            mlp_out = self.gamma2 * mlp_out
        x = x + self.drop_path2(mlp_out)
        return x


class ConvBlock(nn.Module):
    """
    Basic Blocks with <Conv> style for modern architecture.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='reflect', bias=False
            ),
            nn.InstanceNorm3d(num_features=out_channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape (B, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor with shape (B, C, D, H, W)
        """
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding
            ),
            ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size, stride=1, padding=padding
            )
        )
        self.skip: Optional[nn.Module] = None
        if stride != 1 or in_channels != out_channels:
            modules = []
            if stride != 1:
                modules.append(nn.AvgPool3d(kernel_size=stride, stride=stride, ceil_mode=False))
            if in_channels != out_channels:
                modules.append(
                    nn.Conv3d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1, stride=1, padding=0, bias=False
                    )
                )
            self.skip = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape (B, C, D, H, W)
        """
        identity = x
        if self.skip is not None:
            identity = self.skip(x)
        return self.main(x) + identity
