import os
import random

import numpy as np
import torch

SEED = 32


def seed_everything(seed: int = SEED) -> None:
    """Set all relevant RNG seeds for full reproducibility.

    Call this once at the start of each top-level function (train_model,
    test_model, etc.) so every run begins from an identical random state,
    regardless of what happened during module imports.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn — seeds each worker deterministically.

    Pass as worker_init_fn=seed_worker to any DataLoader that uses
    num_workers > 0.  With num_workers=0 (the default) this is a no-op
    but kept for forward compatibility.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = SEED) -> torch.Generator:
    """Return a fresh CPU Generator seeded deterministically.

    Pass as generator=make_generator() to DataLoader so that shuffle
    order is identical across runs.
    """
    return torch.Generator().manual_seed(seed)
