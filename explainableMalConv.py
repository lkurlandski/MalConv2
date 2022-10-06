"""

"""

import multiprocessing as mp
from typing import Tuple

from captum import attr as capattr
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
from tqdm import tqdm
import torch
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from binaryLoader import BinaryDataset, RandomChunkSampler, pad_collate_func
from MalConvGCT_nocat import MalConvGCT
from MalConvML import MalConvML


class IdentityLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ExMalConvML(MalConvML):
    def __init__(
            self,
            out_size=2,
            channels=128,
            window_size=512,
            stride=512,
            layers=1,
            embd_size=8,
            log_stride=None
    ):
        super(ExMalConvML, self).__init__(
            out_size,
            channels,
            window_size,
            stride,
            layers,
            embd_size,
            log_stride
        )
        self.embd = IdentityLayer()

    def processRange(self, x):
        return super().processRange(self.x_embedded)

    def forward(self, x, x_embedded):
        #self.x_embedded = x_embedded
        return super().forward(x)


class ExMalConvGCT(MalConvGCT):
    def __init__(
            self,
            out_size=2,
            channels=128,
            window_size=512,
            stride=512,
            layers=1,
            embd_size=8,
            log_stride=None,
            low_mem=True,
    ):
        super().__init__(
            out_size,
            channels,
            window_size,
            stride,
            layers,
            embd_size,
            log_stride,
            low_mem,
        )
        self.embd = IdentityLayer()
        self.context_net = ExMalConvML(
            out_size=channels,
            channels=channels,
            window_size=window_size,
            stride=stride,
            layers=layers,
            embd_size=embd_size
        )

    def processRange(self, x, gct=None):
        return super().processRange(self.x_embedded, gct=gct)

    def forward(self, x, x_embedded):
        self.x_embedded = x_embedded
        #self.context_net.x_embedded = self.x_embedded
        return super().forward(x)
