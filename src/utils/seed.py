import random
from functools import partial
from typing import Callable

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic

    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(deterministic)


def make_worker_init_fn(seed: int) -> Callable[[int], None]:
    return partial(seed_worker, base_seed=seed)


def seed_worker(worker_id: int, base_seed: int) -> None:
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
