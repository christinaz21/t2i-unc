# src/utils/seed.py

from __future__ import annotations
import os
import random

def set_seed_global(seed: int) -> None:
    """
    Set global RNG seeds for Python, NumPy, and (optionally) PyTorch.

    Safe to call even if NumPy / Torch are not installed.
    """
    # Python's own RNG
    random.seed(seed)

    # Make hash-based things deterministic (dict iteration, etc.)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    # PyTorch
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Optional: more determinism (can slow things down)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
