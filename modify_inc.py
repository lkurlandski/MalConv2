"""
Modify malware and record the classifier's response to the modified malware.
"""

from __future__ import annotations
from argparse import ArgumentParser
from configparser import ConfigParser, SectionProxy
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
import multiprocessing
import os
from pathlib import Path
from pprint import pformat, pprint
from random import shuffle
import sys
import time
import typing as tp

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import classifier as cl
import cfg
import executable_helper
import explain
from typing_ import ErrorMode, ExeToolkit, Pathlike
from utils import batch, ceil_divide, exception_info, get_outfile, raise_error


MAX_INC_SUB_BYTES = 2**24  # 2 ^ 16 = 64 KB, 2 ^ 20 = 1 MB, 2 ^ 24 = 16 MB
BENIGN_FILES = [
    cl.WINDOWS_TRAIN_PATH / f
    for f in [
        "f20a100e661a3179976ccf06ce4a773cbe8d19cd8f50f14e41c0a9e6.exe",  # 3.3748079e-06 malicious
        "09024e62ccab97df3b535e1d65025c54d2d8a684b9e6dcebba79786d.exe",  # 0.9886742 malicious
    ]
] + [
    cl.WINDOWS_TEST_PATH / f
    for f in [
        "05efe7acbe79a7f925c5bc763d11f9d5a1daa2055d297436d0325a1b.exe",  # 1.6685235e-06 malicious
        "256838fe2f037b8865a49d0485c568923f561fea082aa5fa656d6b2d.exe",  # 0.043622814 malicious
        "efe6c4f2299bdc4b89ea935c05c8ebc740937cc7ee4a3948ba552a92.exe",  # 4.975618e-05 malicious
        "701f928760a612a1e929551ca12363394922f30c7f8181f4df5b0ec0.exe",  # 9.903999e-06 malicious
    ]
]
GOOD_BENIGN_FILES = [
    # avg full_benign_corresponding .51 & avg_flipped_corresponding .45%
    cl.WINDOWS_TEST_PATH / "53e17b21d2ff8fa5732211eed9f74f591b9bff985e79f6ad6b85bb72.exe",
    # avg full_benign_corresponding .61 & avg_flipped_corresponding .35%
    cl.WINDOWS_TRAIN_PATH / "fedccb36656858a3aded2b756c7f8d2afa94236140f9defa1d51d1d7.exe",
]
INC_MODES = [
    "inc_baseline",
    "inc_random",
    "inc_benign_corresponding",
    "inc_benign_least",
]
FULL_MODES = [
    "full_baseline",
    "full_random",
    "full_benign_corresponding",
    "full_benign_least",
]
MULTI_FULL_MODES = ["multi_full_benign_corresponding", "multi_full_benign_least"]


@dataclass
class ModifyParams:
    mal_select_mode: str
    mal_replace_mode: str
    chunk_size: int
    toolkit: str = "pefile"
    ben_select_mode: str = None
    min_text_size: int = None
    max_text_size: int = None

    def __post_init__(self):
        if self.inc_params is not None and self.full_params is not None:
            raise ValueError("Cannot have both inc_params and full_params")
        if self.inc_params is None and self.full_params is None:
            raise ValueError("Must have either inc_params or full_params")


@dataclass
class ControlParams:
    output_root: Path
    start_idx: int = None
    stop_idx: int = None
    errors: str = "raise"

    def __post_init__(self):
        self.output_root = Path(self.output_root)


class OutputHelper:
    def __init__(
        self,
        output_root: Pathlike,
        model_name: cl.ModelName,
        chunk_size: int,
        toolkit: str,
        min_text_size: int,
        max_text_size: int,
        rep_source_mode: str,
        rep_target_mode: str,
    ) -> None:
        self.output_root = Path(output_root)
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.toolkit = toolkit
        self.min_text_size = min_text_size
        self.max_text_size = max_text_size
        self.rep_source_mode = rep_source_mode
        self.rep_target_mode = rep_target_mode
        self.output_path = self.output_root.joinpath(*self._get_components())

    def _get_components(self) -> tp.List[str]:
        components = ["__".join([k, str(v)]) for k, v in list(self.__dict__.items())]
        return components

    @classmethod
    def from_params(
        cls,
        control_params: ControlParams,
        modify_params: ModifyParams,
        model_params: cl.ModelParams,
    ) -> OutputHelper:
        return cls(
            control_params.output_root,
            model_params.model_name,
            modify_params.chunk_size,
            modify_params.toolkit,
            modify_params.min_text_size,
            modify_params.max_text_size,
            modify_params.ben_select_mode,
            modify_params.mal_select_mode,
        )


def get_offset_chunk_tensor(chunked_tensor: Tensor, chunk_size: int) -> int:
    first = chunked_tensor[0]
    for i in range(min(chunk_size, chunked_tensor.shape[0])):
        if chunked_tensor[i] != first:
            return i
    return 0


def get_least_suspicious_bounds(
    attributions: Tensor,
    block_size: int,
    suspicious_rank: int = 0,
) -> tp.Tuple[int, int]:
    """
    Get the upper and lower bounds of the least suspicious block of attributions.
    """
    if block_size > attributions.shape[0]:
        raise ValueError(
            f"The attribution vector is too short for the block size."
            f"Attribution vector length: {attributions.shape[0]}, block size: {block_size}"
        )
    # The sum of the attribution scores for a sliding window of size block_size
    block_scores = [
        torch.sum(attributions[i : i + block_size]).item()
        for i in range(attributions.shape[0] - block_size + 1)
    ]
    _, indices_sorted_block_scores = torch.sort(Tensor(block_scores))
    l = indices_sorted_block_scores[suspicious_rank]
    return l, l + block_size


def extend_tensor(
    x: Tensor,
    size: int,
    mode: tp.Literal["repeat", "pad"] = "repeat",
) -> Tensor:
    """
    Increase or decrease the size of the replacement tensor.
    """
    x = x.clone()

    if x.shape[0] < size:
        if mode == "pad":
            padding = torch.full((size - x.shape[0],), cl.PAD_VALUE)
            x = torch.cat((x, padding), 0)
        elif mode == "repeat":
            num_repeats = ceil_divide(size, x.shape[0])
            x = torch.cat([x for _ in range(num_repeats)], 0)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    return x[:size]


class IncSourceBytesSelector:
    def __init__(
        self,
        mode: str,
        chunk_size: int,
        rep_bytes: Tensor = None,
        rep_attribs: Tensor = None,
    ) -> None:
        self.mode = mode
        self.chunk_size = chunk_size
        self.rep_bytes = rep_bytes
        self.rep_attribs = rep_attribs
        self.least_idx = (
            self.sorted_chunked_tensor_indices(rep_attribs) if mode == "least" else None
        )
        self.iteration = -1

    def __call__(self, l_rep: int, u_rep: int, offset: int = None) -> Tensor:
        self.iteration += 1
        size = u_rep - l_rep
        if self.mode == "random":
            return self._random(size)
        elif self.mode == "pad":
            return self._pad(size)
        elif self.mode == "correspond":
            return self._correspond(l_rep, u_rep, offset)
        elif self.mode == "least":
            return self._least()
        else:
            raise ValueError()

    def _random(self, size) -> Tensor:
        return torch.full((self.size,), cl.BASELINE)

    def _pad(self, size) -> Tensor:
        return torch.randint(low=0, high=cl.NUM_EMBEDDINGS, size=(size,))

    def _correspond(self, l_rep: int, u_rep: int, offset: int) -> Tensor:
        return self.benign_replacement[l_rep - offset : u_rep - offset]

    def _least(self) -> Tensor:
        i = self.least_idx[self.iteration]
        return self.rep_bytes[i : i + self.chunk_size]

    @staticmethod
    def sorted_chunked_tensor_indices(chunked_tensor: Tensor, chunk_size: int) -> Tensor:
        offset = get_offset_chunk_tensor(chunked_tensor, chunk_size)
        start_of_chunks = torch.arange(offset, chunked_tensor.shape[0], chunk_size)
        if offset != 0:
            start_of_chunks = torch.cat([Tensor([0]).to(torch.int64), start_of_chunks], axis=0)
        _, indices = torch.sort(chunked_tensor[start_of_chunks])
        return start_of_chunks[indices]


class IncTargetBoundsSelector:
    def __init__(
        self,
        mode: tp.Literal["most", "random", "ordered"],
        l: int,
        u: int,
        chunk_size: int,
        attribs: tp.Optional[Tensor] = None,
        attrib_threshold: float = -float("inf"),
    ) -> None:
        self.mode = mode
        self.chunk_size = chunk_size
        self.length = ceil_divide(u - l, self.chunk_size)

        if self.mode == "most":
            lower_bounds = self._most(attribs, attrib_threshold)
        elif self.mode == "random":
            lower_bounds = self._random()
        elif self.mode == "ordered":
            lower_bounds = self._ordered()
        else:
            raise ValueError(f"Invalid replace mode: {self.replace_mode}")

        # Add the offset to the lower bounds and add the chunk size to the upper bounds
        lower_bounds = [min(u, l + lb) for lb in lower_bounds]
        upper_bounds = [min(u, lb + self.chunk_size) for lb in lower_bounds]

        self.bounds = [(lb, ub) for lb, ub in zip(lower_bounds, upper_bounds)]
        self.iteration = -1

    def __iter__(self) -> IncTargetBoundsSelector:
        return self

    def __len__(self) -> int:
        return len(self.bounds)

    def __next__(self) -> tp.Tuple[int, int]:
        self.iteration += 1
        if self.iteration < len(self.bounds):
            return self.bounds[self.iteration]
        raise StopIteration

    def _random(self) -> tp.List[int]:
        lower_bounds = list(range(0, self.length * self.chunk_size, self.chunk_size))
        shuffle(lower_bounds)
        return lower_bounds

    def _ordered(self) -> tp.List[int]:
        lower_bounds = list(range(0, self.length * self.chunk_size, self.chunk_size))
        return lower_bounds

    def _most(self, attribs: Tensor, attrib_threshold: float) -> tp.List[int]:
        lower_bounds = []
        attribs = attribs.clone()
        for i in range(self.length):
            # Stop if the most suspicious chunk is lower than some threshold
            if (max_attr := attribs.max()) <= attrib_threshold:
                break
            # Find the index of the maximum attribution score (torch version stable)
            max_attr_lower = (max_attr == attribs).nonzero()[0].item()
            # Set that region to a small value to avoid re-selecting it
            attribs[max_attr_lower : max_attr_lower + self.chunk_size] = -float("inf")
            # Get the lower and upper bounds and replacement value for the full input tensor
            lower_bounds.append(max_attr_lower)
        return lower_bounds


def inc_sub_and_eval(
    model: cl.MalConvLike,
    X: Tensor,
    l_text: int,
    source_replacer: IncSourceBytesSelector,
    target_bounds: IncTargetBoundsSelector,
    batch_size: int = 1,
) -> np.ndarray:
    X = X.clone()
    confs = []
    batch = [X]

    for i, (l_rep, u_rep) in enumerate(target_bounds):
        X = X.clone()
        X[l_rep:u_rep] = source_replacer(l_rep, u_rep, l_text)
        batch.append(X)
        if len(batch) == batch_size:
            c = cl.confidence_scores(model, batch).tolist()
            confs.extend(c)
            batch.clear()

    return np.array(confs)


def run(
    model_params: cl.ModelParams,
    exe_params: executable_helper.ExeParams,
    explain_params: explain.ExplainParams,
    mal_explain_oh: explain.OutputHelper,
    ben_explain_oh: explain.OutputHelper,
    modify_params: ModifyParams,
    control_params: ControlParams,
):
    oh = OutputHelper.from_params(modify_params, control_params)
    oh.output_path.mkdir(parents=True, exist_ok=True)

    mal_files = chain(cl.SOREL_TRAIN_PATH.iterdir(), cl.SOREL_TEST_PATH.iterdir())
    explained = set(p.name.strip(".pt") for p in mal_explain_oh.output_path.iterdir())
    mal_files = [p for p in mal_files if p.name in explained]

    bounds = executable_helper.get_bounds(exe_params.text_section_bounds_file)
    model = cl.get_model(model_params.model_name)

    # Optional parameters need default value of None
    rep_bytes, rep_attribs, attribs_text = None, None, None

    # Get the benign replacement bytes and the benign replacement's attributions
    if modify_params.rep_source_mode in {"correspond", "least"}:
        rep_bytes = Tensor(executable_helper.read_binary(modify_params.rep_file))
    if modify_params.rep_source_mode in {"least"}:
        l_text, u_text = bounds[modify_params.rep_file]
        attribs_file = ben_explain_oh.output_path / modify_params.rep_file
        rep_attribs = torch.load(attribs_file, map_location=cfg.device)[l_text:u_text]

    for f in mal_files:
        X = Tensor(executable_helper.read_binary(f))
        l_text, u_text = bounds[f.as_posix()]

        # Get attributions to identify the region of the malware to swap
        if modify_params.rep_target_mode in {"most"}:
            attribs_file = mal_explain_oh.output_path / f.name
            attribs_text = torch.load(attribs_file, map_location=cfg.device)[l_text:u_text]
            if (o := get_offset_chunk_tensor(attribs_text, explain_params.chunk_size)) != 0:
                raise ValueError(f"attributions have nonzero chunk offset {o=}")

        # Ensure the replacements are as large as the .text section to replace
        size = ceil_divide(u_text - l_text, explain_params.chunk_size) * explain_params.chunk_size
        rep_bytes = extend_tensor(rep_bytes, size) if rep_bytes is not None else rep_bytes
        rep_attribs = extend_tensor(rep_attribs, size) if rep_attribs is not None else rep_attribs

        # Get the bytes to replace with and the regions to replace at each iteration
        source_replacer = IncSourceBytesSelector(
            modify_params.rep_source_mode, explain_params.chunk_size, rep_bytes, rep_attribs
        )
        target_bounds = IncTargetBoundsSelector(
            modify_params.rep_target_mode, l_text, u_text, explain_params.chunk_size, attribs_text
        )

        # Compute the confidence at every iteration of the algorithm
        confs = inc_sub_and_eval(
            model,
            X,
            l_text,
            source_replacer,
            target_bounds,
            control_params.batch_size,
        )
        np.savetxt(oh.output_path / f.name, confs, delimiter="\n")


def parse_config(config: ConfigParser):
    p = config["MODEL"]
    model_params = cl.ModelParams(p.get("model_name"))
    p = config["DATA"]
    data_params = cl.DataParams(max_len=p.getint("max_len"), batch_size=p.getint("batch_size"))
    p = config["EXE"]
    exe_params = executable_helper.ExeParams(p.get("text_section_bounds_file"))

    p = config["MODIFY"]
    modify_params = ModifyParams(
        p.get("mal_select_mode"),
        p.get("mal_replace_mode"),
        p.getint("chunk_size"),
        p.get("toolkit"),
        p.get("ben_select_mode"),
        p.getint("min_text_size"),
        p.getint("max_text_size"),
    )

    p = config["CONTROL"]
    control_params = ControlParams(
        p.get("output_root"),
        p.getint("mal_start_idx"),
        p.getint("mal_stop_idx"),
        p.get("errors"),
    )

    p = config["EXPLAIN"]
    explain_config = ConfigParser()
    explain_config.read(p.get("config_file"))
    (
        exp_model_params,
        exp_data_params,
        exp_exe_params,
        exp_explain_params,
        exp_control_params,
    ) = explain.parse_config(explain_config)
    if model_params != exp_model_params:
        print(f"WARNING: parameters from explanation differ:\n{model_params=}\n{exp_model_params=}")
    if data_params != exp_data_params:
        print(f"WARNING: parameters from explanation differ:\n{data_params=}\n{exp_data_params=}")
    if exe_params != exp_exe_params:
        print(f"WARNING: parameters from explanation differ:\n{exe_params=}\n{exp_exe_params=}")

    mal_explain_oh = explain.OutputHelper.from_params(exp_explain_params, exp_control_params, "mal")
    ben_explain_oh = explain.OutputHelper.from_params(exp_explain_params, exp_control_params, "ben")

    return (
        model_params,
        exe_params,
        data_params,
        mal_explain_oh,
        ben_explain_oh,
        modify_params,
        control_params,
    )


def main(config: ConfigParser) -> None:
    cfg.init(config["CONTROL"].get("device"), config["CONTROL"].getint("seed"))
    run(*parse_config(config))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, default="config_files/modify_inc/default.ini")
    args = parser.parse_args()
    config = ConfigParser(allow_no_value=True)
    config.read(args.config_file)
    main(config)
