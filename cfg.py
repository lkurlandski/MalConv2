"""

"""

from pathlib import Path
import random
import typing as tp

import numpy as np
import torch

LOG_PATH = Path("./logs")


def print_config():
    print("-" * 88, f"{'-' * 40} CONFIG {'-' * 40}", "-" * 88, sep="\n")
    print(f"{seed=}")
    print(f"{device=}")
    print(f"{torch.backends.cudnn.enabled=}")


def init(device_: tp.Literal["cpu", "cuda:0"] = "cpu", seed_: int = 0, *, verbose: bool = True):
    global device
    global seed
    seed = seed_
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_)
    torch.backends.cudnn.enabled = False
    if verbose:
        print_config()


init("cpu", 0, verbose=False)
