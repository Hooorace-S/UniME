from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from models.Backbone.config import BackboneConfig
from models.Backbone.backbone import BackboneForwardOutput, TokUniEncoder


def resolve_checkpoint_path(
    checkpoint_path: str,
    *,
    allow_missing: bool = False
) -> tuple[str, bool]:
    """
    Resolve checkpoint path with respect to repository root and current working directory.
    """
    expanded = os.path.expanduser(checkpoint_path)
    path_obj = Path(expanded)

    candidates: list[Path] = []
    if path_obj.is_absolute():
        candidates.append(path_obj)
    else:
        repo_root = Path(__file__).resolve().parents[2]
        candidates.extend([
            repo_root / path_obj,
            Path.cwd() / path_obj,
            path_obj
        ])

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        key = str(candidate)
        if key not in seen:
            unique_candidates.append(candidate)
            seen.add(key)

    for candidate in unique_candidates:
        if candidate.is_file():
            return str(candidate), True

    searched = [str(c) for c in unique_candidates]
    if allow_missing:
        warnings.warn(
            f"Checkpoint not found at '{checkpoint_path}'. Tried: {searched}",
            RuntimeWarning,
            stacklevel=3
        )
        fallback = unique_candidates[0] if unique_candidates else path_obj
        return str(fallback), False

    raise FileNotFoundError(
        f"Checkpoint not found at '{checkpoint_path}'. Tried: {searched}"
    )


class UniEncoderWrapper(nn.Module):
    """
    Wrapper of Tokenizer3D & UniEncoder module.
    """
    def __init__(
        self,
        mode: str = "frozen",
        pretrained_path: str | None = None,
        config: BackboneConfig | None = None
    ) -> None:
        """
        Args:
            mode (str, optional): state of this module in UniME training, it can be 'scratch', 'frozen' or 'finetune'.
            pretrained_path (str, optional): optional path to pretrained UniEncoder weights.
            config (BackboneConfig, optional): configuration for Backbone module. Defaults to None.
        """
        super().__init__()
        self.mode = self._normalize_mode(mode)
        self.config = config if config is not None else BackboneConfig()
        self.tok_uniencoder = TokUniEncoder(self.config)

        self._pretrained_loaded: bool = False
        self._pretrained_source: str | None = None
        self._is_frozen: bool = False
        self.pretrained_path: str | None = None

        if pretrained_path is not None:
            resolved_path, exists = resolve_checkpoint_path(
                pretrained_path,
                allow_missing=True
            )
            self.pretrained_path = resolved_path
            if exists and self.mode in {"frozen", "finetune"}:
                self.load_pretrained(resolved_path, strict=False)

        self._apply_mode(self.mode)

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        normalized = mode.lower()
        if normalized == "forzen":
            warnings.warn(
                'Mode "forzen" detected; using "frozen" instead. '
                'Please update callers to use the correct spelling.',
                RuntimeWarning,
                stacklevel=3
            )
            normalized = "frozen"
        valid_modes = {"scratch", "frozen", "finetune"}
        if normalized not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}', choose from {sorted(valid_modes)}")
        return normalized

    def _apply_mode(self, mode: str) -> None:
        self._is_frozen = mode == "frozen"
        requires_grad = not self._is_frozen
        for param in self.parameters():
            param.requires_grad = requires_grad
        # Keep encoder deterministic when frozen to act as feature extractor
        self.tok_uniencoder.train(False if self._is_frozen else self.training)

    def set_mode(self, mode: str) -> None:
        """
        Update the operating mode of the wrapper.

        Args:
            mode (str): One of {'scratch', 'frozen', 'finetune'}.
        """
        normalized = self._normalize_mode(mode)
        self.mode = normalized
        self._apply_mode(self.mode)

    def freeze(self) -> None:
        """Alias for set_mode('frozen')."""
        self.set_mode("frozen")

    def unfreeze(self, finetune: bool = True) -> None:
        """
        Unfreeze the encoder parameters.

        Args:
            finetune (bool, optional): If True, set to 'finetune', otherwise 'scratch'.
        """
        self.set_mode("finetune" if finetune else "scratch")

    @property
    def is_frozen(self) -> bool:
        """Return True when encoder weights are frozen."""
        return self._is_frozen

    @staticmethod
    def _load_checkpoint(
        checkpoint_path: str,
        map_location: torch.device | str,
    ) -> Mapping[str, Any]:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint

    @staticmethod
    def _extract_state_dict(
        checkpoint: Mapping[str, Any],
        key_priority: tuple[str, ...],
    ) -> Mapping[str, Any]:
        if not isinstance(checkpoint, Mapping):
            raise TypeError("Checkpoint must be a mapping containing model weights.")
        for key in key_priority:
            if key in checkpoint:
                value = checkpoint[key]
                if isinstance(value, Mapping):
                    return value
                raise TypeError(f"Expected mapping for state dict under key '{key}', got {type(value)}.")
        return checkpoint

    @staticmethod
    def _strip_known_prefixes(name: str) -> str:
        prefixes = ("module.", "_orig_mod.", "backbone.", "tok_uniencoder.")
        stripped = name
        prefix_applied = True
        while prefix_applied:
            prefix_applied = False
            for prefix in prefixes:
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix):]
                    prefix_applied = True
        return stripped

    def _filter_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        expected_state = self.state_dict()
        filtered: dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        for key, value in state_dict.items():
            clean_key = self._strip_known_prefixes(key)
            if clean_key.startswith("decoder"):
                continue
            mapped_key = f"tok_uniencoder.{clean_key}"
            expected_value = expected_state.get(mapped_key)
            if expected_value is None:
                continue
            if tuple(expected_value.shape) != tuple(value.shape):
                skipped.append(
                    f"{key} -> {mapped_key}: checkpoint {tuple(value.shape)} != model {tuple(expected_value.shape)}"
                )
                continue
            filtered[mapped_key] = value
        return filtered, skipped

    def load_pretrained(
        self,
        checkpoint_path: str,
        *,
        strict: bool = False,
        map_location: torch.device | str = "cpu",
        state_dict_keys: tuple[str, ...] = ("model_state_dict", "state_dict")
    ) -> tuple[list[str], list[str]]:
        """
        Load pretrained weights into the Uni-Encoder.

        Args:
            checkpoint_path (str): Path to the saved checkpoint.
            strict (bool, optional): Forward to `load_state_dict`. Defaults to False.
            map_location (torch.device | str, optional): Device mapping for torch.load. Defaults to 'cpu'.
            state_dict_keys (Tuple[str, ...], optional): Keys to probe when extracting the state dict.

        Returns:
            tuple[list[str], list[str]]: missing keys, unexpected keys reported by `load_state_dict`.
        """
        resolved_path, _ = resolve_checkpoint_path(checkpoint_path)

        checkpoint = self._load_checkpoint(resolved_path, map_location)
        state_dict = self._extract_state_dict(checkpoint, state_dict_keys)
        filtered_state_dict, skipped_keys = self._filter_state_dict(state_dict)

        incompatible = self.load_state_dict(filtered_state_dict, strict=strict)
        missing_keys: list[str] = list(incompatible.missing_keys)
        unexpected_keys: list[str] = list(incompatible.unexpected_keys)
        if skipped_keys:
            warnings.warn(
                f"Skipped incompatible pretrained tensors: {skipped_keys}",
                RuntimeWarning,
                stacklevel=2
            )

        if missing_keys:
            warnings.warn(
                f"Missing keys when loading pretrained Uni-Encoder weights: {missing_keys}",
                RuntimeWarning,
                stacklevel=2
            )
        if unexpected_keys:
            warnings.warn(
                f"Unexpected keys ignored when loading pretrained Uni-Encoder weights: {unexpected_keys}",
                RuntimeWarning,
                stacklevel=2
            )

        self._pretrained_loaded = True
        self._pretrained_source = resolved_path
        self._apply_mode(self.mode)
        return missing_keys, unexpected_keys

    def forward(self, x: torch.Tensor) -> BackboneForwardOutput:
        if self._is_frozen:
            with torch.no_grad():
                return self.tok_uniencoder(x)
        return self.tok_uniencoder(x)

    def train(self, mode: bool = True) -> UniEncoderWrapper:
        super().train(mode)
        self.tok_uniencoder.train(False if self._is_frozen else mode)
        return self
