from collections.abc import Mapping
import os

import torch
from torch import nn


def require_cuda(context: str = "UniME-V2 runtime") -> torch.device:
    """
    Require a CUDA-capable runtime and return the CUDA device.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is required for {context}; CPU execution is no longer supported.")
    return torch.device("cuda")


def is_accelerate_launch() -> bool:
    """
    Detect whether the current process was launched under Accelerate/DDP control.
    """
    return (
        "LOCAL_RANK" in os.environ
        or "ACCELERATE_PROCESS_INDEX" in os.environ
        or int(os.environ.get("WORLD_SIZE", "1")) > 1
    )


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Peel off common runtime wrappers such as DDP and ``torch.compile``.
    """
    current = model
    visited: set[int] = set()

    while id(current) not in visited:
        visited.add(id(current))

        next_model = None
        if hasattr(current, "module") and isinstance(getattr(current, "module"), nn.Module):
            next_model = getattr(current, "module")
        elif hasattr(current, "_orig_mod") and isinstance(getattr(current, "_orig_mod"), nn.Module):
            next_model = getattr(current, "_orig_mod")

        if next_model is None:
            break
        current = next_model

    return current


def set_model_is_training(model: nn.Module, is_training: bool) -> None:
    """
    Apply the UniME forward-mode flag on both the runtime wrapper and the base module.
    """
    try:
        setattr(model, "is_training", is_training)
    except Exception:
        pass

    base_model = unwrap_model(model)
    if base_model is not model:
        setattr(base_model, "is_training", is_training)


def clone_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Clone tensors from a state dict so later in-place changes do not mutate saved checkpoints.
    """
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in state_dict.items()
    }


def strip_runtime_prefixes(name: str) -> str:
    """
    Remove common runtime prefixes introduced by DDP or ``torch.compile``.
    """
    stripped = name
    prefixes = ("module.", "_orig_mod.")

    prefix_applied = True
    while prefix_applied:
        prefix_applied = False
        for prefix in prefixes:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
                prefix_applied = True

    return stripped


def load_model_state_dict_compat(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    strict: bool = True
) -> object:
    """
    Load a checkpoint into the base model while tolerating runtime wrapper prefixes.
    """
    base_model = unwrap_model(model)
    normalized_state_dict = {
        strip_runtime_prefixes(name): tensor
        for name, tensor in state_dict.items()
    }
    return base_model.load_state_dict(normalized_state_dict, strict=strict)
