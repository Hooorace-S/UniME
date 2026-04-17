import os
from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn

from models.Backbone.config import BackboneConfig
from models.UniME.configs import (
    LAYER_DECAY_DEFAULT,
    NanoScaleConfig,
    TinyScaleConfig,
    SmallScaleConfig,
    BaseScaleConfig,
    OriginalScaleConfig,
    get_scale_spec,
)
from models.UniME.decoder import CNNDecoder
from models.UniME.encoder import CNNEncoder
from models.UniME.mask import MaskModal
from models.UniME.regularizer import AuxiliaryRegularizer
from models.UniME.wrapper import UniEncoderWrapper


_PRETRAIN_ROOT = "log_pretrain"
_PRETRAIN_RUN = "BRATS2023-1105-Normal"


def _get_pretrained_checkpoint_path(encoder_name: str) -> str:
    return os.path.join(_PRETRAIN_ROOT, _PRETRAIN_RUN, encoder_name, "ema_best_checkpoint.pth")


def resolve_uni_encoder_pretrained_path(
    *,
    encoder_name: str,
    pretrained_path: Optional[str] = None,
) -> str:
    if pretrained_path is not None:
        return pretrained_path
    return _get_pretrained_checkpoint_path(encoder_name)


_DEFAULT_SCALE_SETTINGS = {
    "Base": {
        "config": BaseScaleConfig,
        "encoder_name": "UniEncoderBase",
    },
    "Small": {
        "config": SmallScaleConfig,
        "encoder_name": "UniEncoderSmall",
    },
    "Tiny": {
        "config": TinyScaleConfig,
        "encoder_name": "UniEncoderTiny",
    },
    "Nano": {
        "config": NanoScaleConfig,
        "encoder_name": "UniEncoderNano",
    },
    "Original": {
        "config": OriginalScaleConfig,
        "encoder_name": "UniEncoder",
    },
}


class UniMEModel(nn.Module):
    """
    Unified UniME segmentation model with scale-specific UniEncoder backbones.
    """
    def __init__(
        self,
        *,
        scale: str,
        num_classes: int = 4,
        num_modals: int = 4,
        mode: str = "finetune",
        layer_decay: Optional[float] = None,
        encoder_name: Optional[str] = None,
        pretrained_path: Optional[str] = None,
        checkpoint_config: Optional[Dict[str, bool]] = None,
    ) -> None:
        super().__init__()

        scale_key = scale.title()
        if scale_key not in _DEFAULT_SCALE_SETTINGS:
            raise ValueError(f"Unknown UniME scale '{scale}'. Supported scales: {sorted(_DEFAULT_SCALE_SETTINGS.keys())}.")

        scale_spec = get_scale_spec(scale_key)
        scale_settings = _DEFAULT_SCALE_SETTINGS[scale_key]
        config: BackboneConfig = scale_settings["config"]
        embed_dims = int(scale_spec["embed_dim"])
        basic_dims = int(scale_spec["basic_dims"])
        encoder_name = encoder_name or scale_settings["encoder_name"]
        resolved_pretrained_path = resolve_uni_encoder_pretrained_path(
            encoder_name=encoder_name,
            pretrained_path=pretrained_path,
        )

        checkpoint_flags = checkpoint_config
        if checkpoint_flags is None:
            from models import get_model_checkpoint_config

            checkpoint_flags = get_model_checkpoint_config(self.__class__.__name__)

        self.checkpoint_config = {
            "encoder": bool(checkpoint_flags.get("encoder", False)),
            "regularizer": bool(checkpoint_flags.get("regularizer", False)),
            "decoder": bool(checkpoint_flags.get("decoder", False)),
        }

        encoder_use_checkpoint = self.checkpoint_config["encoder"]
        regularizer_use_checkpoint = self.checkpoint_config["regularizer"]
        decoder_use_checkpoint = self.checkpoint_config["decoder"]

        self.flair_encoder = CNNEncoder(basic_dims=basic_dims, use_checkpoint=encoder_use_checkpoint)
        self.t1ce_encoder = CNNEncoder(basic_dims=basic_dims, use_checkpoint=encoder_use_checkpoint)
        self.t1_encoder = CNNEncoder(basic_dims=basic_dims, use_checkpoint=encoder_use_checkpoint)
        self.t2_encoder = CNNEncoder(basic_dims=basic_dims, use_checkpoint=encoder_use_checkpoint)

        self.uni_encoder = UniEncoderWrapper(
            mode=mode,
            config=config,
            pretrained_path=resolved_pretrained_path,
        )

        self.decoder_fuse = CNNDecoder(
            num_classes=num_classes,
            num_modals=num_modals,
            basic_dims=basic_dims,
            embed_dims=embed_dims,
            use_checkpoint=decoder_use_checkpoint,
        )

        self.regularizer = AuxiliaryRegularizer(
            num_classes=num_classes,
            basic_dims=basic_dims,
            use_checkpoint=regularizer_use_checkpoint,
        )

        self.is_training = False
        self.masker = MaskModal()
        self.default_layer_decay: float = LAYER_DECAY_DEFAULT
        self.layer_decay = layer_decay

    @property
    def layer_decay(self) -> Optional[float]:
        return getattr(self, "_layer_decay", None)

    @layer_decay.setter
    def layer_decay(self, value: Optional[float]) -> None:
        if value is None:
            self._layer_decay = None
            return

        decay = float(value)
        if decay <= 0:
            raise ValueError("layer_decay must be a positive float.")
        self._layer_decay = decay

    def configure_layer_decay(self, value: Optional[float]) -> None:
        self.layer_decay = value

    @property
    def layerwise_lr_decay_enabled(self) -> bool:
        return self.layer_decay is not None and self.uni_encoder.mode == "finetune"

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Union[torch.Tensor, Tuple]:
        flair_x1, flair_x2, flair_x3, flair_x4 = self.flair_encoder(x[:, 0:1, ...])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4 = self.t1ce_encoder(x[:, 1:2, ...])
        t1_x1, t1_x2, t1_x3, t1_x4 = self.t1_encoder(x[:, 2:3, ...])
        t2_x1, t2_x2, t2_x3, t2_x4 = self.t2_encoder(x[:, 3:4, ...])
        auxiliary_predictions: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None

        if self.is_training:
            auxiliary_predictions = (
                self.regularizer(flair_x1, flair_x2, flair_x3, flair_x4),
                self.regularizer(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4),
                self.regularizer(t1_x1, t1_x2, t1_x3, t1_x4),
                self.regularizer(t2_x1, t2_x2, t2_x3, t2_x4),
            )

        x_mask = self.masker(x=x.unsqueeze(2), mask=mask)
        fusion_feature = self.uni_encoder(x=x_mask)

        x1 = self.masker(torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1), mask)
        x2 = self.masker(torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1), mask)
        x3 = self.masker(torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1), mask)

        fused_pred, preds = self.decoder_fuse(x0=x1, x1=x2, x2=x3, x3=fusion_feature)

        if auxiliary_predictions is not None:
            return fused_pred, auxiliary_predictions, preds
        return fused_pred


class UniME(UniMEModel):
    def __init__(
        self,
        num_modals: int,
        num_classes: int,
        layer_decay: Optional[float] = LAYER_DECAY_DEFAULT,
        encoder_name: str = "UniEncoder",
        pretrained_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            scale="Original",
            num_modals=num_modals,
            num_classes=num_classes,
            mode="finetune",
            layer_decay=layer_decay,
            encoder_name=encoder_name,
            pretrained_path=pretrained_path,
        )


class UniMEBase(UniMEModel):
    def __init__(
        self,
        num_modals: int,
        num_classes: int,
        layer_decay: Optional[float] = LAYER_DECAY_DEFAULT,
        encoder_name: str = "UniEncoderBase",
        pretrained_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            scale="Base",
            num_modals=num_modals,
            num_classes=num_classes,
            mode="finetune",
            layer_decay=layer_decay,
            encoder_name=encoder_name,
            pretrained_path=pretrained_path,
        )


class UniMESmall(UniMEModel):
    def __init__(
        self,
        num_modals: int,
        num_classes: int,
        layer_decay: Optional[float] = LAYER_DECAY_DEFAULT,
        encoder_name: str = "UniEncoderSmall",
        pretrained_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            scale="Small",
            num_modals=num_modals,
            num_classes=num_classes,
            mode="finetune",
            layer_decay=layer_decay,
            encoder_name=encoder_name,
            pretrained_path=pretrained_path,
        )


class UniMETiny(UniMEModel):
    def __init__(
        self,
        num_modals: int,
        num_classes: int,
        layer_decay: Optional[float] = LAYER_DECAY_DEFAULT,
        encoder_name: str = "UniEncoderTiny",
        pretrained_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            scale="Tiny",
            num_modals=num_modals,
            num_classes=num_classes,
            mode="finetune",
            layer_decay=layer_decay,
            encoder_name=encoder_name,
            pretrained_path=pretrained_path,
        )


class UniMENano(UniMEModel):
    def __init__(
        self,
        num_modals: int,
        num_classes: int,
        layer_decay: Optional[float] = LAYER_DECAY_DEFAULT,
        encoder_name: str = "UniEncoderNano",
        pretrained_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            scale="Nano",
            num_modals=num_modals,
            num_classes=num_classes,
            mode="finetune",
            layer_decay=layer_decay,
            encoder_name=encoder_name,
            pretrained_path=pretrained_path,
        )
