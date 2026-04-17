import os
import random

import torch
import numpy as np


def setup_seed(random_seed: int) -> None:
    """
    Set all random seeds strictly to ensure reproducibility.

    Args:
        random_seed (int): Random seed value

    Returns:
        None
    """
    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    # Set Python built-in random module
    random.seed(random_seed)

    # Set NumPy random seed
    np.random.seed(random_seed)

    # Set PyTorch CPU random seed
    torch.manual_seed(random_seed)

    # Set PyTorch GPU random seed (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    # Disable TF32, avoid non-determinism caused by mixed precision (if available)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    # else:
    #     torch.backends.cuda.matmul.allow_tf32 = True
    #     torch.backends.cudnn.allow_tf32 = True

    # cuDNN: enable deterministic and disable benchmark to avoid selecting non-deterministic operators
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Use deterministic algorithms
    # if hasattr(torch, "use_deterministic_algorithms"):
    #     torch.use_deterministic_algorithms(True)

    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    print(f"[Deterministic] Random seed and deterministic settings ready: {random_seed}")
