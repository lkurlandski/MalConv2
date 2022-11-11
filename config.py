"""

"""

import random

import numpy as np
import torch


SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
torch.backends.cudnn.enabled = False


def print_configurations():
    print(f"CONFIG\n{'-' * 88}")
    print(f"{SEED=}")
    print(f"{device=}")
    print(f"{torch.backends.cudnn.enabled=}")


print_configurations()
