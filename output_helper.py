"""

"""

from __future__ import annotations
from pathlib import Path
import typing as tp

from utils import Pathlike


class OutputHelper(Path):
    def __init__(
            self,
            output_root: Pathlike,
            classifier_helper: ClassifierOutputHelper,
            explainable_helper: ExplainableOutputHelper,
            modify_malware_helper: ModifyMalwareOutputHelper,
    ) -> None:
        self.output_root = Path(output_root)
        self.helpers = [classifier_helper, explainable_helper, modify_malware_helper]
        s = output_root.joinpath(*self.helpers)
        super().__init__(s)


class ClassifierOutputHelper(Path):
    def __init__(self, model_name: str, max_len: int):
        self.model_name = model_name
        self.max_len = max_len
        s = "/".join([self.model_name, self.max_len])
        super().__init__(s)


class ExplainableOutputHelper(Path):
    def __init__(self, algorithm: str, softmax: bool, layer: str, ):
        s = ""
        super().__init__(s)


class ModifyMalwareOutputHelper(Path):
    def __init__(self, toolkit: str, min_size: int = 0, max_size: int = None, n_benign: int = 16):
        self.toolkit = toolkit
        self.min_size = min_size
        self.max_size = max_size
        self.n_benign = n_benign
        s = [toolkit, min_size, max_size, n_benign]
        s = "/".join([str(i) for i in s])
        super().__init__(s)
