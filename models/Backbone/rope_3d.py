from dataclasses import dataclass

from typing import Optional, Tuple

import torch
from timm.layers.pos_embed_sincos import apply_rot_embed_cat, RotaryEmbeddingCat


@dataclass
class RoPE3DCache:
    """Cache container for 3D RoPE embeddings."""
    spatial_shape: Tuple[int, int, int]
    dtype: torch.dtype
    device: torch.device
    embed: torch.Tensor


class RoPE3D:
    """
    3D Rotary Positional Embedding.

    For every pixel (x, y, z) in the image, we do three separate rotations along X, Y, Z axes:
        - X-axis rotation -> Y-axis rotation -> Z-axis rotation
    """
    def __init__(self, rope_dim: int, rope_base: float = 10000.0):
        """
        Args:
            rope_dim (int): rotary embedding dimension.
            rope_base (float, optional): base temperature for rotary embedding.

        <NOTE>: In our setting, <rope_dim = head_dim / 1.5>.
        """
        self.rope = RotaryEmbeddingCat(
            rope_dim,
            in_pixels=False,
            feat_shape=None,
            ref_feat_shape=None,
            temperature=rope_base,
        )
        self._cache: Optional[RoPE3DCache] = None

    @torch._dynamo.disable
    def get_embed(
        self,
        spatial_shape: Tuple[int, int, int], device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Get or rebuild cached 3D rotary embedding.

        Args:
            spatial_shape (Tuple[int, int, int]): spatial shape of the image.
            device (torch.device): device of the embedding.
            dtype (torch.dtype): dtype of the embedding.

        Returns:
            torch.Tensor: 3D rotary embedding with shape (seq_len, rope_dim)
        """
        if (self._cache is None or self._cache.spatial_shape != spatial_shape):
            embed = self.rope.get_embed(list(spatial_shape)).to(device=device, dtype=dtype)
            self._cache = RoPE3DCache(spatial_shape, dtype, device, embed)
        elif self._cache.device != device or self._cache.dtype != dtype:
            embed = self._cache.embed.to(device=device, dtype=dtype)
            self._cache = RoPE3DCache(spatial_shape, dtype, device, embed)
        return self._cache.embed

    @staticmethod
    def _gather_kept_embeddings(
        rope_embed: torch.Tensor,
        keep_indices: torch.Tensor,
    ) -> torch.Tensor:
        gathered = [
            rope_embed.index_select(0, keep_indices[i].to(rope_embed.device))
            for i in range(keep_indices.shape[0])
        ]
        return torch.stack(gathered, dim=0)

    @staticmethod
    @torch._dynamo.disable
    def update_cache(
        rope_embed: torch.Tensor,
        num_register_tokens: int,
        keep_indices: torch.Tensor | None,
        device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor | None:
        """
        Update the cache of the RoPE3D with
            - register tokens trick;
            - patch tokens trick;

        Args:
            rope_embed (torch.Tensor): rotary embedding tensor with shape (seq_len, rope_dim)
            num_register_tokens (int): number of register tokens, which do <NOT> need RoPE3D.
            keep_indices (torch.Tensor): keep indices with shape (batch_size, num_kept)
            device (torch.device): device of the embedding.
            dtype (torch.dtype): dtype of the embedding.

        Returns:
            torch.Tensor | None: updated RoPE3D embedding with shape
                - (seq_len, rope_dim) or
                - (num_register_tokens + seq_len, rope_dim) or
                - (batch_size, num_kept, rope_dim) or
                - (batch_size, num_register_tokens + num_kept, rope_dim)
        """
        if keep_indices is None:
            if num_register_tokens == 0:
                return None
            zeros_mask = torch.zeros(
                num_register_tokens,
                rope_embed.shape[-1],
                device=device,
                dtype=dtype,
            )
            return torch.cat((zeros_mask, rope_embed), dim=0)

        rope_embed_kept = RoPE3D._gather_kept_embeddings(rope_embed, keep_indices)
        if num_register_tokens == 0:
            return rope_embed_kept

        zeros_mask = torch.zeros(
            rope_embed_kept.shape[0],
            num_register_tokens,
            rope_embed_kept.shape[-1],
            device=device,
            dtype=dtype
        )
        return torch.cat((zeros_mask, rope_embed_kept), dim=1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    rope_embed: torch.Tensor,
    num_register_tokens: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embedding to Q and K.

    Args:
        q (torch.Tensor): query tensor with shape (batch_size, num_heads, seq_len, head_dim)
        k (torch.Tensor): key tensor with shape (batch_size, num_heads, seq_len, head_dim)
        rope_embed (torch.Tensor): rotary embedding tensor with shape
            (num_register_tokens + seq_len, rope_dim) or
            (batch_size, num_register_tokens + seq_len, rope_dim)
        num_register_tokens (int): number of register tokens, which do <NOT> need RoPE3D.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: tuple of query and key tensors with original shape.
    """
    rope_embed = rope_embed.to(device=q.device, dtype=q.dtype)
    rope_is_batched = rope_embed.ndim == 3

    if num_register_tokens == 0:
        rope = rope_embed.unsqueeze(1) if rope_is_batched else rope_embed
        q = apply_rot_embed_cat(q, rope)
        k = apply_rot_embed_cat(k, rope)
        return q, k

    q_prefix, q_main = torch.tensor_split(q, [num_register_tokens], dim=2)
    k_prefix, k_main = torch.tensor_split(k, [num_register_tokens], dim=2)

    if rope_is_batched:
        rope_main = rope_embed[:, num_register_tokens:, :].unsqueeze(1)
    else:
        rope_main = rope_embed[num_register_tokens:, :]

    q_main = apply_rot_embed_cat(q_main, rope_main)
    k_main = apply_rot_embed_cat(k_main, rope_main)
    q = torch.cat((q_prefix, q_main), dim=2)
    k = torch.cat((k_prefix, k_main), dim=2)
    return q, k
