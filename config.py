"""

"""

import random

import numpy as np
import torch


SEED = 0
random.seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False

print(f"CONFIG\n{'-' * 88}")
print(f"{SEED=}")
print(f"{device=}")
print(f"{torch.backends.cudnn.enabled=}")
