import importlib
from functools import lru_cache

from torch import nn


ModelClass = type[nn.Module]
CheckpointConfig = dict[str, bool]


_MODEL_PATHS: dict[str, str] = {
    "UniME": ".UniME.networks",
    "UniMEBase": ".UniME.networks",
    "UniMESmall": ".UniME.networks",
    "UniMETiny": ".UniME.networks",
    "UniMENano": ".UniME.networks",
}

_DEFAULT_CHECKPOINT_FLAGS: CheckpointConfig = {
    "encoder": False,
    "regularizer": False,
    "decoder": False,
}

MODEL_CHECKPOINT_CONFIGS: dict[str, CheckpointConfig] = {
    "UniME": {
        "encoder": False,
        "regularizer": False,
        "decoder": False,
    },
    "UniMEBase": {
        "encoder": True,
        "regularizer": True,
        "decoder": False,
    },
    "UniMESmall": {
        "encoder": False,
        "regularizer": True,
        "decoder": False,
    },
    "UniMETiny": {
        "encoder": False,
        "regularizer": False,
        "decoder": False,
    },
    "UniMENano": {
        "encoder": False,
        "regularizer": False,
        "decoder": False,
    }
}


def get_model_checkpoint_config(model_name: str) -> CheckpointConfig:
    config = MODEL_CHECKPOINT_CONFIGS.get(model_name, _DEFAULT_CHECKPOINT_FLAGS)
    merged: CheckpointConfig = dict(_DEFAULT_CHECKPOINT_FLAGS)
    merged.update(config)
    return merged


@lru_cache(maxsize=None)
def _import_class(model_name: str) -> ModelClass:
    try:
        module_path = _MODEL_PATHS.get(model_name)
        if not module_path:
            available_models = ", ".join(list(_MODEL_PATHS.keys()))
            raise ValueError(
                f"Model '{model_name}' is not registered.\n"
                f"Available models: {available_models}.\n"
                f"Total {len(_MODEL_PATHS)} models available."
            )

        module = importlib.import_module(module_path, package=__package__)
        return getattr(module, model_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Cannot import model {model_name}: {exc}") from exc


def get_model(name: str) -> ModelClass:
    return _import_class(name)


def get_available_models() -> list[str]:
    return list(_MODEL_PATHS.keys())


# Pylincer does not yet support dynamic construction of __all__
__all__ = list(_MODEL_PATHS.keys()) + [  # pyright: ignore[reportUnsupportedDunderAll]
    "get_model",
    "get_available_models",
    "MODEL_CHECKPOINT_CONFIGS",
    "get_model_checkpoint_config",
]
