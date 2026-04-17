import math

from einops import rearrange
import torch
from torch import nn
from timm.layers.patch_dropout import PatchDropoutWithIndices

from models.Backbone.blocks import EVABlock
from models.Backbone.config import BackboneConfig
from models.Backbone.decoder import LightDecoder
from models.Backbone.rope_3d import RoPE3D
from models.Backbone.tokenizer import Tokenizer3D
from models.Backbone.utils import LayerNormNd, restore_full_sequence

BackboneMaskOutput = tuple[torch.Tensor, torch.Tensor]
BackboneForwardOutput = torch.Tensor | BackboneMaskOutput
RestorationMask = torch.Tensor | None
KeepIndices = torch.Tensor | None


class _BackboneBase(nn.Module):
    """
    Shared EVA-02 style tokenization and transformer stack.
    """

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = Tokenizer3D(
            config.in_channels,
            config.embed_dim,
            config.patch_shape,
        )

        self.sequence_length = math.prod(config.tokens_spatial_shape)
        self.num_register_tokens = config.num_register_tokens

        self.pos_embed: nn.Parameter | None = None
        if config.use_learned_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.sequence_length, config.embed_dim)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.register_tokens: nn.Parameter | None = None
        if config.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, config.num_register_tokens, config.embed_dim)
            )
            nn.init.normal_(self.register_tokens, std=1e-6)

        self.patch_drop: PatchDropoutWithIndices | None = None
        self.mask_token: nn.Parameter | None = None
        if config.patch_drop_rate > 0.0:
            self.patch_drop = PatchDropoutWithIndices(
                config.patch_drop_rate,
                num_prefix_tokens=config.num_register_tokens,
            )
            self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        assert config.embed_dim % config.num_heads == 0, "embed_dim must be divisible by num_heads."
        assert (config.embed_dim // config.num_heads) % 1.5 == 0, "head_dim must be divisible by 1.5."
        rope_dim = int((config.embed_dim // config.num_heads) / 1.5)
        assert rope_dim % 4 == 0, "rope_dim must be divisible by 4."
        self.rope: RoPE3D | None = None
        if config.use_rotary_pos_embed:
            self.rope = RoPE3D(rope_dim, config.rope_base)

        dropout_prob_list = [
            value.item() for value in torch.linspace(0, config.drop_path_rate, config.depth)
        ]
        self.blocks = nn.ModuleList(
            [
                EVABlock(
                    dim=config.embed_dim,
                    num_heads=config.num_heads,
                    mlp_dropout=config.mlp_dropout,
                    attn_dropout=config.attn_dropout,
                    proj_dropout=config.proj_dropout,
                    drop_path=dropout_prob_list[index],
                    layer_scale_init_value=config.layer_scale_init_value,
                    num_register_tokens=config.num_register_tokens,
                )
                for index in range(config.depth)
            ]
        )
        self.after_trans_norm = nn.LayerNorm(config.embed_dim)

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        token_grid = self.tokenizer(x)
        return rearrange(token_grid, "b dim d h w -> b (d h w) dim")

    def _append_prefix_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed
        if self.register_tokens is not None:
            tokens = torch.cat(
                (self.register_tokens.expand(tokens.shape[0], -1, -1), tokens),
                dim=1,
            )
        return tokens

    def _apply_patch_drop(self, tokens: torch.Tensor) -> tuple[torch.Tensor, KeepIndices]:
        if self.patch_drop is None:
            return tokens, None
        return self.patch_drop(tokens)

    def _build_rope_embed(
        self,
        tokens: torch.Tensor,
        keep_indices: KeepIndices,
    ) -> torch.Tensor | None:
        if self.rope is None:
            return None
        # torch._dynamo.disable breaks method typing for Pylance here.
        rope_embed = self.rope.get_embed(  # type: ignore[attr-defined]
            spatial_shape=self.config.tokens_spatial_shape,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return self.rope.update_cache(  # type: ignore[attr-defined]
            rope_embed=rope_embed,
            num_register_tokens=self.num_register_tokens,
            keep_indices=keep_indices,
            device=tokens.device,
            dtype=tokens.dtype,
        )

    def _run_transformer(
        self,
        tokens: torch.Tensor,
        rope_embed: torch.Tensor | None,
    ) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(
                tokens,
                rope_embed=rope_embed,
                spatial_shape=self.config.tokens_spatial_shape,
            )
        return self.after_trans_norm(tokens)

    def _remove_register_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.num_register_tokens == 0:
            return tokens
        return tokens[:, self.num_register_tokens:, :]

    def _restore_sequence(
        self,
        patch_tokens: torch.Tensor,
        keep_indices: KeepIndices,
    ) -> tuple[torch.Tensor, RestorationMask]:
        if self.patch_drop is None:
            return patch_tokens, None

        if self.mask_token is None:
            raise RuntimeError("mask_token is required when patch dropout is enabled.")
        return restore_full_sequence(
            patch_tokens,
            self.mask_token,
            keep_indices,
            self.sequence_length,
        )

    def _tokens_to_grid(self, restored_tokens: torch.Tensor) -> torch.Tensor:
        return rearrange(
            restored_tokens,
            "b (d h w) c -> b c d h w",
            d=self.config.tokens_spatial_shape[0],
            h=self.config.tokens_spatial_shape[1],
            w=self.config.tokens_spatial_shape[2],
        )

    def _forward_features_with_restoration_mask(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, RestorationMask]:
        tokens = self._tokenize(x)
        tokens = self._append_prefix_tokens(tokens)
        tokens, keep_indices = self._apply_patch_drop(tokens)
        rope_embed = self._build_rope_embed(tokens, keep_indices)
        tokens = self._run_transformer(tokens, rope_embed)

        patch_tokens = self._remove_register_tokens(tokens)
        restored_tokens, restoration_mask = self._restore_sequence(
            patch_tokens,
            keep_indices,
        )
        feat = self._tokens_to_grid(restored_tokens)
        return feat, restoration_mask

    def _restore_mask_to_volume(
        self,
        restoration_mask: RestorationMask,
        batch_size: int,
    ) -> torch.Tensor | None:
        if restoration_mask is None:
            return None

        patch_mask = restoration_mask.view(
            batch_size,
            self.config.tokens_spatial_shape[0],
            self.config.tokens_spatial_shape[1],
            self.config.tokens_spatial_shape[2],
        )
        pz, py, px = self.config.patch_shape
        full_mask = patch_mask.repeat_interleave(pz, dim=1)
        full_mask = full_mask.repeat_interleave(py, dim=2)
        full_mask = full_mask.repeat_interleave(px, dim=3)
        return full_mask[:, None, ...]

    def _format_output(
        self,
        output: torch.Tensor,
        restoration_mask: RestorationMask,
        batch_size: int,
    ) -> BackboneForwardOutput:
        if not self.config.return_mask:
            return output

        full_mask = self._restore_mask_to_volume(restoration_mask, batch_size)
        if full_mask is None:
            return output
        return output, full_mask

    def _init_layer_norm(self, module: nn.Module | torch.Tensor) -> None:
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _init_stem(self) -> None:
        conv = self.tokenizer.proj
        nn.init.kaiming_normal_(conv.weight, a=1e-2)
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)

    def _init_transformer_blocks(self) -> None:
        for block in self.blocks:
            self._init_layer_norm(block.ln1)
            self._init_layer_norm(block.ln2)
            attn = block.attn

            # Pylance cannot fully resolve nested PyTorch module attribute types here;
            # qkv/proj/fc1_x/fc2 are runtime nn.Linear modules.
            nn.init.trunc_normal_(attn.qkv.weight, std=0.02)  # type: ignore[arg-type]
            if attn.qkv.bias is not None:  # type: ignore[arg-type]
                nn.init.zeros_(attn.qkv.bias)  # type: ignore[arg-type]
            nn.init.trunc_normal_(attn.proj.weight, std=0.02)  # type: ignore[arg-type]
            if attn.proj.bias is not None:  # type: ignore[arg-type]
                nn.init.zeros_(attn.proj.bias)  # type: ignore[arg-type]
            if isinstance(attn.post_attn_norm, nn.LayerNorm):  # type: ignore[arg-type]
                self._init_layer_norm(attn.post_attn_norm)  # type: ignore[arg-type]

            mlp = block.mlp
            nn.init.trunc_normal_(mlp.fc1_x.weight, std=0.02)  # type: ignore[arg-type]
            if mlp.fc1_x.bias is not None:  # type: ignore[arg-type]
                nn.init.zeros_(mlp.fc1_x.bias)  # type: ignore[arg-type]
            nn.init.trunc_normal_(mlp.fc2.weight, std=0.02)  # type: ignore[arg-type]
            if mlp.fc2.bias is not None:  # type: ignore[arg-type]
                nn.init.zeros_(mlp.fc2.bias)  # type: ignore[arg-type]

        self._init_layer_norm(self.after_trans_norm)
        self._rescale_transformer_projections()

    def _rescale_transformer_projections(self) -> None:
        with torch.no_grad():
            for layer_id, block in enumerate(self.blocks, start=1):
                scale = math.sqrt(2.0 * layer_id)
                block.attn.proj.weight.div_(scale)  # type: ignore[arg-type]
                block.mlp.fc2.weight.div_(scale)  # type: ignore[arg-type]

    def initialize(self) -> None:
        self._init_stem()
        self._init_transformer_blocks()


class ViT(_BackboneBase):
    """
    EVA-02 style backbone with:
        Tokenizer3D -> LPe -> UniEncoder (3D RoPE) -> LightDecoder
    """

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__(config)
        self.decoder = LightDecoder(
            config.embed_dim,
            config.out_channels,
            config.patch_shape,
        )
        self.initialize()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input volume into token-grid features before LightDecoder.
        """
        feat, _ = self._forward_features_with_restoration_mask(x)
        return feat

    def decode_features(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Decode token-grid features back to dense voxel predictions.
        """
        return self.decoder(feat)

    def forward(self, x: torch.Tensor) -> BackboneForwardOutput:
        """
        Forward pass of the EVA-02 Style ViT model.
            -> Tokenizer3D
            -> PositionalEmbedding (Learnable) -> RegisterTokens -> PatchDropout
            -> TransformerBlocks
            -> LightDecoder

        Args:
            x (torch.Tensor): tensor with shape (B, num_modals, D, H, W)

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            segmentation logits and, when enabled, the restored full-resolution mask.
        """
        feat, restoration_mask = self._forward_features_with_restoration_mask(x)
        seg = self.decode_features(feat)
        return self._format_output(seg, restoration_mask, x.shape[0])

    def _init_decoder(self) -> None:
        for module in self.decoder.decoder.modules():
            if isinstance(module, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(module.weight, a=1e-2)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, LayerNormNd):
                nn.init.ones_(module.norm.weight)
                nn.init.zeros_(module.norm.bias)

    def initialize(self) -> None:
        """
        Initialize the EVA-02 Style ViT model.
        """
        super().initialize()
        self._init_decoder()


class TokUniEncoder(_BackboneBase):
    """
    EVA-02 Backbone <WITHOUT> decoder, i.e.,
        Tokenizer3D -> LPe -> UniEncoder (3D RoPE)
    """

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__(config)
        self.initialize()

    def forward(self, x: torch.Tensor) -> BackboneForwardOutput:
        """
        Forward pass.
            -> Tokenizer3D
            -> PositionalEmbedding (Learnable) -> RegisterTokens -> PatchDropout
            -> TransformerBlocks

        Args:
            x (torch.Tensor): tensor with shape (B, num_modals, D, H, W)

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            token-grid features and, when enabled, the restored full-resolution mask.
        """
        feat, restoration_mask = self._forward_features_with_restoration_mask(x)
        return self._format_output(feat, restoration_mask, x.shape[0])
