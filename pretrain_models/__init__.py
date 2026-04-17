import importlib
from functools import lru_cache

from torch import nn

# Model path mapping
ModelClass = type[nn.Module]


_MODEL_PATHS: dict[str, str] = {
    "UniEncoder": ".UniEncoder.networks",
    "UniEncoderBase": ".UniEncoder.networks",
    "UniEncoderSmall": ".UniEncoder.networks",
    "UniEncoderTiny": ".UniEncoder.networks",
    "UniEncoderNano": ".UniEncoder.networks",
}


@lru_cache(maxsize=None)
def _import_class(model_name: str) -> ModelClass:
    """
    Dynamically import model class.

    Args:
        model_name: model name

    Returns:
        model_class: model class

    Raises:
        ValueError: When model is not registered or imported failed.
    """
    try:
        module_path = _MODEL_PATHS.get(model_name)
        if not module_path:
            available_models = ", ".join(list(_MODEL_PATHS.keys())[:10])
            raise ValueError(
                f"Model '{model_name}' is not registered.\n"
                f"Available models: {available_models}...\n"
                f"Total {len(_MODEL_PATHS)} models available."
            )

        module = importlib.import_module(module_path, package=__package__)
        return getattr(module, model_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Cannot import model {model_name}: {exc}") from exc


def get_model(name: str) -> ModelClass:
    """
    Get model class by name.

    Args:
        name(str): model name

    Returns:
        model_class(nn.Module): model class
    """
    return _import_class(name)


def get_available_models() -> list[str]:
    """
    Get all available model names.

    Returns:
        model_names(list): model names
    """
    return list(_MODEL_PATHS.keys())


# Export all model names
# Pylincer does not yet support dynamic construction of __all__
__all__ = list(_MODEL_PATHS.keys()) + [  # pyright: ignore[reportUnsupportedDunderAll]
    "get_model",
    "get_available_models"
]
