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
import explain as exp
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
    output_root: Pathlike
    start_idx: int = None
    stop_idx: int = None
    errors: str = "raise"
    num_workers: int = 1

    def __post_init__(self):
        self.output_root = Path(self.output_root)


class OutputHelper:

    def __init__(
        self,
        output_root: Pathlike,
        model_name: cl.ModelName,
        mal_select_mode: str,
        mal_replace_mode: str,
        chunk_size: int,
        toolkit: str,
        ben_select_mode: str,
        min_text_size: int,
        max_text_size: int,
    ) -> None:
        self.output_root = Path(output_root)
        self.model_name = model_name
        self.mal_select_mode = mal_select_mode
        self.mal_replace_mode = mal_replace_mode
        self.chunk_size = chunk_size
        self.toolkit = toolkit
        self.ben_select_mode = ben_select_mode
        self.min_text_size = min_text_size
        self.max_text_size = max_text_size
        self.output_path = self.output_root.joinpath(*self._get_components())

    def _get_components(self) -> tp.List[str]:
        components = ["__".join([k, str(v)]) for k, v in list(self.__dict__.items())]
        return components

    @classmethod
    def from_params(cls, control_params: ControlParams, modify_params: ModifyParams, model_params: cl.ModelParams) -> OutputHelper:
        return cls(
            control_params.output_root,
            model_params.model_name,
            modify_params.mal_select_mode,
            modify_params.mal_replace_mode,
            modify_params.chunk_size,
            modify_params.toolkit,
            modify_params.ben_select_mode,
            modify_params.min_text_size,
            modify_params.max_text_size,
        )


class GetLeastSuspiciousChunk:
    def __init__(
        self, X: Tensor, attributions: Tensor, chunk_size: int, suspicious_rank: int = 0
    ) -> None:
        """
        Acquire the least suspicious chunk from X using the corresponding attributions.
        """
        if X.shape != attributions.shape:
            raise ValueError(f"X.shape != attributions.shape: {X.shape} != {attributions.shape}")
        self.X = X
        self.attributions = attributions
        self.chunk_size = chunk_size
        self.suspicious_rank = suspicious_rank
        self.sorted_start_of_chunks = get_sorted_starts_of_chunks(attributions, chunk_size)

    def __call__(self, suspicious_rank=None) -> Tensor:
        if suspicious_rank is None:
            suspicious_rank = self.suspicious_rank
            self.suspicious_rank += 1
        offset = self.sorted_start_of_chunks[suspicious_rank]
        return self.X[offset : offset + self.chunk_size]


def get_sorted_starts_of_chunks(chunked_tensor: Tensor, chunk_size: int) -> Tensor:
    """Get the indices within a chunked tensor that correspond to the sorted chunk values."""
    offset = get_offset_chunk_tensor(chunked_tensor, chunk_size)
    start_of_chunks = torch.arange(offset, chunked_tensor.shape[0], chunk_size)
    if offset != 0:
        start_of_chunks = torch.cat([Tensor([0]).to(torch.int64), start_of_chunks], axis=0)
    _, indices = torch.sort(chunked_tensor[start_of_chunks])
    return start_of_chunks[indices]


def get_offset_chunk_tensor(chunked_tensor: Tensor, chunk_size: int) -> int:
    first = chunked_tensor[0]
    for i in range(min(chunk_size, chunked_tensor.shape[0])):
        if chunked_tensor[i] != first:
            return i
    return 0


def swaps_count(
    attributions: Tensor,
    chunk_size: int,
    attribution_threshold: float = 0.0,
) -> int:
    n_over_thresh = torch.sum(attributions > attribution_threshold)
    return ceil_divide(n_over_thresh, chunk_size)


def get_output_path(
    params: SectionProxy,
    *,
    output_root: Pathlike = None,
    model_name: cl.ModelName = None,
    max_len: int = None,
    softmax: bool = None,
    chunk_size: int = None,
) -> Path:
    output_root = output_root if output_root is not None else params.get("output_root")
    model_name = model_name if model_name is not None else params.get("model_name")
    max_len = max_len if max_len is not None else params.get("max_len")
    softmax = softmax if softmax is not None else params.getboolean("softmax")
    chunk_size = chunk_size if chunk_size is not None else params.get("chunk_size")
    output_path = (
        Path(output_root)
        / model_name
        / str(max_len)
        / "KernelShap"
        / str(softmax)
        / str(chunk_size)
        / "50"
        / "1"
    )
    return output_path


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


def slice_or_expand_tensor(
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


class IncrementalReplacementBoundsSelector:
    def __init__(
        self,
        replace_mode: tp.Literal["most_suspicious", "random", "ordered"],
        l: int,
        u: int,
        chunk_size: int,
        attributions: Tensor = None,
        attribution_threshold: float = None,
    ) -> None:
        self.replace_mode = replace_mode
        self.chunk_size = chunk_size
        self.length = ceil_divide(u - l, self.chunk_size)  # + 1 ?

        if self.replace_mode == "most_suspicious":
            lower_bounds = self._most_suspicious(attributions, attribution_threshold)
        elif self.replace_mode == "random":
            lower_bounds = self._random()
        elif self.replace_mode == "ordered":
            lower_bounds = self._ordered()
        else:
            raise ValueError(f"Invalid replace mode: {self.replace_mode}")

        # Add the offset to the lower bounds
        lower_bounds = [min(u, l + lb) for lb in lower_bounds]
        # Add the chunk size to the upper bounds
        upper_bounds = [min(u, lb + self.chunk_size) for lb in lower_bounds]

        self.bounds = [(lb, ub) for lb, ub in zip(lower_bounds, upper_bounds)]
        self.current = 0
        assert self.length == len(self.bounds)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.bounds)

    def __next__(self):
        current = self.current
        if current < len(self.bounds):
            self.current += 1
            return self.bounds[current]
        raise StopIteration

    def _most_suspicious(
        self, attributions: Tensor, attribution_threshold: float
    ) -> tp.List[tp.Tuple[int, int]]:
        lower_bounds = []
        attributions = attributions.clone()
        for i in range(self.length):
            # Stop if the most suspicious chunk is lower than some threshold
            if (max_attr := attributions.max()) <= attribution_threshold:
                break
            # Find the index of the maximum attribution score (torch version stable)
            max_attr_lower = (max_attr == attributions).nonzero()[0].item()
            # Set that region to a small value to avoid re-selecting it
            attributions[max_attr_lower : max_attr_lower + self.chunk_size] = -float("inf")
            # Get the lower and upper bounds and replacement value for the full input tensor
            lower_bounds.append(max_attr_lower)
        return lower_bounds

    def _random(self) -> tp.Tuple[int, int]:
        lower_bounds = list(range(0, self.length * self.chunk_size, self.chunk_size))
        shuffle(lower_bounds)
        return lower_bounds

    def _ordered(self) -> tp.Tuple[int, int]:
        lower_bounds = list(range(0, self.length * self.chunk_size, self.chunk_size))
        return lower_bounds


def incremental_substitute(
    run_flags: SectionProxy,
    model: cl.MalConvLike,
    attributions: Tensor,
    X: Tensor,
    l: int,
    u: int,
    chunk_size: int,
    replace_mode: tp.Literal["most_suspicious", "random", "ordered"],
    batch_size: int = 1,
    benign_replacement: tp.Optional[Tensor] = None,
    benign_attributions: tp.Optional[Tensor] = None,
    attribution_threshold: float = -float("inf"),
) -> tp.Tuple[tp.List[float]]:
    def update_input_batch_and_confs(X_: Tensor, batch: tp.List[Tensor], confs: tp.List[float]):
        X_[l_replace:u_replace] = r  # Update the input tensor
        batch.append(X_.clone())  # Update the batch of input tensors
        if len(batch) == batch_size or i == len(bounds) - 1:  # Every batch_size iterations (or end)
            batch_ = pad_sequence(batch, batch_first=True)  # Pad the batch (not needed bc same X)
            c = cl.confidence_scores(model, batch_).tolist()  # Compute the confidence scores
            confs.extend(c)  # Update the conf scores structure
            batch.clear()  # Clear the batch

    # Initial processes
    run_baseline, run_random, run_ben_corr, run_benign_least = [
        run_flags.getboolean("run_" + m) for m in INC_MODES
    ]
    if not any((run_baseline, run_random, run_ben_corr, run_benign_least)):
        return [], [], [], []
    # Ensure the size of the benign replacement is the same as the input's .text section
    if benign_replacement is not None or benign_attributions is not None:
        size = ceil_divide(u - l, chunk_size) * chunk_size
        if benign_replacement is not None:
            benign_replacement = slice_or_expand_tensor(benign_replacement, size)
        if benign_attributions is not None:
            benign_attributions = slice_or_expand_tensor(benign_attributions, size)
    # Set up the return data structures of confidence scores
    baseline_confs = []
    random_confs = []
    ben_corr_confs = []
    benign_least_confs = []
    # Populate with the original confidence score
    c = cl.confidence_scores(model, X).item()
    if run_baseline:
        baseline_confs = [c]
        baseline_X = X.clone()
        baseline_batch = []
    if run_random:
        random_confs = [c]
        random_X = X.clone()
        random_batch = []
    if run_ben_corr:
        ben_corr_confs = [c]
        ben_corr_X = X.clone()
        ben_corr_batch = []
    if run_benign_least:
        benign_least_confs = [c]
        benign_least_X = X.clone()
        benign_least_batch = []
        get_least_suspicious_chunk = GetLeastSuspiciousChunk(
            benign_replacement, benign_attributions, chunk_size
        )
    # Get an iterable of the bounds of the chunks to replace in the full input tensor
    bounds = IncrementalReplacementBoundsSelector(
        replace_mode, l, u, chunk_size, attributions, attribution_threshold
    )
    for i, (l_replace, u_replace) in enumerate(bounds):
        try:
            size = u_replace - l_replace
            # For every mode, get the replacement tensor and run the update subroutine
            if run_baseline:
                r = torch.full((size,), cl.BASELINE)
                update_input_batch_and_confs(baseline_X, baseline_batch, baseline_confs)
            if run_random:
                r = torch.randint(low=0, high=cl.NUM_EMBEDDINGS, size=(size,))
                update_input_batch_and_confs(random_X, random_batch, random_confs)
            if run_ben_corr:
                r = benign_replacement[l_replace - l : u_replace - l]
                update_input_batch_and_confs(ben_corr_X, ben_corr_batch, ben_corr_confs)
            if run_benign_least:
                r = get_least_suspicious_chunk()
                r = r[0:size]
                update_input_batch_and_confs(benign_least_X, benign_least_batch, benign_least_confs)
        except Exception as e:
            print(exception_info(e, locals()))
            raise_error(e, pre="incremental_substitute")

    return (
        baseline_confs,
        random_confs,
        ben_corr_confs,
        benign_least_confs,
    )


def full_benign_least_replacement(
    benign_replacement: Tensor,
    benign_attributions: Tensor,
    size: int,
    mode: tp.Literal["exact", "truncate", "repeat", "pad"],
) -> Tensor:
    benign_attributions = slice_or_expand_tensor(
        benign_attributions, max(size, benign_attributions.shape[0]), mode
    )
    benign_replacement = slice_or_expand_tensor(
        benign_replacement, max(size, benign_replacement.shape[0]), mode
    )
    l, u = get_least_suspicious_bounds(benign_attributions, size)
    return benign_replacement[l:u]


def full_substitute(
    run_flags: SectionProxy,
    model: cl.MalConvLike,
    X: Tensor,
    l: int,
    u: int,
    benign_replacement: tp.Optional[Tensor] = None,
    benign_attributions: tp.Optional[Tensor] = None,
    mode: tp.Literal["exact", "truncate", "repeat", "pad"] = "repeat",
) -> tp.Tuple[tp.List[float]]:
    # Initial processes
    run_baseline, run_random, run_benign_corresponding, run_benign_least = [
        run_flags.getboolean("run_" + m) for m in FULL_MODES
    ]
    if not any((run_baseline, run_random, run_benign_corresponding, run_benign_least)):
        return [], [], [], []
    # Set up the return data structures of confidence scores
    baseline_confs = []
    random_confs = []
    benign_corresponding_confs = []
    benign_least_confs = []
    # Size of the input's .text section and original confidence score
    size = u - l
    c = cl.confidence_scores(model, X).item()
    # Populate with the original confidence score
    if run_baseline:
        X_ = X.clone()
        X_[l:u] = torch.full((size,), cl.BASELINE)
        baseline_confs = [c, cl.confidence_scores(model, X_).item()]
    if run_random:
        X_ = X.clone()
        X_[l:u] = torch.randint(low=0, high=cl.NUM_EMBEDDINGS, size=(size,))
        random_confs = [c, cl.confidence_scores(model, X_).item()]
    if run_benign_corresponding:
        X_ = X.clone()
        X_[l:u] = slice_or_expand_tensor(benign_replacement, size, mode)
        benign_corresponding_confs = [c, cl.confidence_scores(model, X_).item()]
    if run_benign_least:
        X_ = X.clone()
        X_[l:u] = full_benign_least_replacement(benign_replacement, benign_attributions, size, mode)
        benign_least_confs = [c, cl.confidence_scores(model, X_).item()]

    return baseline_confs, random_confs, benign_corresponding_confs, benign_least_confs


# TODO: implement batched evaluation
# TODO: do not include the model's original confidence score?
def multi_full_substitute(
    run_flags: SectionProxy,
    model: cl.MalConvLike,
    X: Tensor,
    l: int,
    u: int,
    text_section_bounds: tp.Dict[str, tp.Tuple[int, int]],
    benign_files: tp.Iterable[Path],
    attributions_path: Path,
    mode: tp.Literal["exact", "truncate", "repeat", "pad"] = "repeat",
) -> tp.Tuple[tp.List[Path], tp.List[float], tp.List[float]]:
    # Initial processes
    run_benign_corresponding = run_flags.getboolean("run_multi_full_benign_corresponding")
    run_benign_least = run_flags.getboolean("run_multi_full_benign_least")
    if not any((run_benign_corresponding, run_benign_least)):
        return [], [], []
    # Size of the input's .text section and original confidence score
    size = u - l
    c = cl.confidence_scores(model, X).item()
    used_files = [None]
    benign_corresponding_confs = [c] if run_benign_corresponding else []
    benign_least_confs = [c] if run_benign_least else []

    for br_f in benign_files:
        br_l, br_u = text_section_bounds[br_f.as_posix()]
        used_files.append(br_f)
        br_X = Tensor(executable_helper.read_binary(br_f, l=br_l, u=br_u))
        br_A = get_text_section_attributions(attributions_path, br_f.name, br_l, br_u)
        if run_benign_corresponding:
            X_ = X.clone()
            X_[l:u] = slice_or_expand_tensor(br_X, size, mode)
            benign_corresponding_confs.append(cl.confidence_scores(model, X_).item())
        if run_benign_least:
            X_ = X.clone()
            X_[l:u] = full_benign_least_replacement(br_X, br_A, size, mode)
            benign_least_confs.append(cl.confidence_scores(model, X_).item())

    return used_files, benign_corresponding_confs, benign_least_confs


def get_text_section_attributions(
    attributions_path: Path,
    exe_name: str,
    l: int,
    u: int,
) -> Tensor:
    f_1 = attributions_path / "benign" / f"{exe_name}.pt"
    f_2 = attributions_path / "malicious" / f"{exe_name}.pt"
    if f_1.exists() and f_2.exists():
        raise ValueError(f"Attributions file in both benign/malicious directories: {exe_name=}")
    elif f_1.exists():
        f = f_1
    elif f_2.exists():
        f = f_2
    else:
        raise FileNotFoundError(
            f"Attributions file not in either benign/malicious directories: {exe_name=}"
        )

    return torch.load(f, map_location=cfg.device)[l:u]


def run_sample(
    run_flags: SectionProxy,
    chunk_size: int,
    model: cl.MalConvLike,
    attributions_path: Path,
    confidences_path: Path,
    text_section_bounds: tp.Dict[str, tp.Tuple[int, int]],
    inc_replace_modes: tp.Iterable[tp.Literal["most_suspicious", "random", "ordered"]],
    inc_batch_size: int,
    f: Path,
    benign_replacement: tp.Optional[Tensor] = None,
    benign_attributions: tp.Optional[Tensor] = None,
    benign_files: tp.Iterable[Path] = None,
    errors: ErrorMode = "warn",
) -> None:
    try:
        l, u = text_section_bounds[f.as_posix()]
        attributions = get_text_section_attributions(attributions_path, f.name, l, u)
        if get_offset_chunk_tensor(attributions, chunk_size) != 0:
            o = get_offset_chunk_tensor(attributions, chunk_size)
            raise ValueError(f"attributions have nonzero chunk offset {o=}")
        X = Tensor(executable_helper.read_binary(f))

        for inc_mode in inc_replace_modes:
            inc_values = incremental_substitute(
                run_flags,
                model,
                attributions,
                X,
                l,
                u,
                chunk_size,
                inc_mode,
                inc_batch_size,
                benign_replacement=benign_replacement,
                benign_attributions=benign_attributions,
            )
            for m, conf in zip(INC_MODES, inc_values):
                if conf:
                    p = confidences_path / m / inc_mode / f"{f.name}.txt"
                    p.parent.mkdir(exist_ok=True, parents=True)
                    np.savetxt(p, conf, delimiter="\n")

        full_values = full_substitute(
            run_flags,
            model,
            X,
            l,
            u,
            benign_replacement=benign_replacement,
            benign_attributions=benign_attributions,
        )
        for m, conf in zip(FULL_MODES, full_values):
            if conf:
                p = confidences_path / m / f"{f.name}.txt"
                p.parent.mkdir(exist_ok=True, parents=True)
                np.savetxt(p, conf, delimiter="\n")

        multi_full_values = multi_full_substitute(
            run_flags,
            model,
            X,
            l,
            u,
            text_section_bounds,
            benign_files,
            attributions_path,
        )

        if multi_full_values[1]:
            p = confidences_path / "multi_full_benign_corresponding" / f"{f.name}.csv"
            p.parent.mkdir(exist_ok=True, parents=True)
            pd.DataFrame(
                {"substitute": multi_full_values[0], "confidences": multi_full_values[1]}
            ).to_csv(p, index=False)
        if multi_full_values[2]:
            p = confidences_path / "multi_full_benign_least" / f"{f.name}.csv"
            p.parent.mkdir(exist_ok=True, parents=True)
            pd.DataFrame(
                {"substitute": multi_full_values[0], "confidences": multi_full_values[2]}
            ).to_csv(p, index=False)

    except Exception as e:
        if errors == "warn":
            locals_ = locals()
            locals_.pop("text_section_bounds")
            print(exception_info(e, locals_))
        elif errors == "ignore":
            pass
        else:
            raise e


def run_samples(
    mal_files: tp.Iterable[Path],
    run_flags: SectionProxy,
    chunk_size: int,
    model: cl.MalConvLike,
    attributions_path: Path,
    confidences_path: Path,
    text_section_bounds: tp.Dict[str, tp.Tuple[int, int]],
    inc_replace_modes: tp.Iterable[tp.Literal["most_suspicious", "random", "ordered"]],
    inc_batch_size: int,
    benign_replacement: tp.Optional[Tensor] = None,
    benign_attributions: tp.Optional[Tensor] = None,
    benign_files: tp.Iterable[Path] = None,
    errors: ErrorMode = "warn",
    verbose: bool = False,
):
    args = (
        (
            run_flags,
            chunk_size,
            model,
            attributions_path,
            confidences_path,
            text_section_bounds,
            inc_replace_modes,
            inc_batch_size,
            f,
            benign_replacement,
            benign_attributions,
            benign_files,
            errors,
        )
        for f in mal_files
    )
    for arg in tqdm(args, total=len(mal_files)):
        start = time.time()
        run_sample(*arg)
        if verbose:
            print(
                f"PID={os.getpid()} "
                f"DONE={arg[7].name} "
                f"@{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                f"in {time.time() - start:.2f}s "
            )


def run(
    model_params: cl.ModelParams,
    data_params: cl.DataParams,
    exe_params: executable_helper.ExeParams,
    mal_explain_oh: explain.OutputHelper,
    ben_explain_oh: explain.OutputHelper,
    modify_params: ModifyParams,
    control_params: ControlParams,
) -> None:
    oh = OutputHelper.from_params()

    attributions_path = output_path / "attributions"
    confidences_path = output_path / "confidences" / toolkit
    confidences_path.mkdir(parents=True, exist_ok=True)

    model = get_model(model_name)
    # Single benign replacement file for use with the incremental substitution methods
    br_f, br_l, br_u, br_d = next(stream_text_section_data(GOOD_BENIGN_FILES[0], toolkit, "torch"))
    br_a = get_text_section_attributions(attributions_path, br_f.name, br_l, br_u)

    # Full list of files to be used, which will be truncated shortly
    mal_files = list(chain(SOREL_TRAIN_PATH.iterdir(), SOREL_TEST_PATH.iterdir()))
    ben_files = list(chain(WINDOWS_TRAIN_PATH.iterdir(), WINDOWS_TEST_PATH.iterdir()))

    if text_section_bounds_file is None:
        text_section_bounds_file = (
            Path(params.get("output_root")) / f"text_section_bounds_{toolkit}.csv"
        )
        if not text_section_bounds_file.exists():
            generate_text_section_bounds_file(
                chain(ben_files, mal_files),
                toolkit,
                text_section_bounds_file,
                errors="ignore",
            )
    text_section_bounds = pd.read_csv(text_section_bounds_file, index_col="file").to_dict("index")
    text_section_bounds = {k: (v["lower"], v["upper"]) for k, v in text_section_bounds.items()}

    # Truncate the benign files based upon requested number to use
    ben_files = ben_files[:n_benign]
    # TODO: implement a more intelligent selection method for the benign files, eg:
    # df = pd.read_csv("/home/lk3591/Documents/MalConv2/output_model/gct/cum_results.csv")
    # ben_files = df.sort_values(by=["ts_confs"])["ts_files"].tolist()[0 : n_benign // 2]
    # ben_files += df.sort_values(by=["tr_confs"])["tr_files"].tolist()[0 : n_benign // 2]

    # Truncate the malicious files based upon their text section bounds
    def include_mal_file(f: str):
        return (
            f in text_section_bounds
            and ((text_section_bounds[f][0] >= min_size) if isinstance(min_size, int) else True)
            and ((text_section_bounds[f][1] <= max_size) if isinstance(max_size, int) else True)
        )

    def sort_mal_file(f: str):
        return text_section_bounds[f][1] - text_section_bounds[f][0]

    mal_files = [f for f in mal_files if include_mal_file(f.as_posix())]
    mal_files.sort(key=lambda f: sort_mal_file(f.as_posix()))

    # Slice the malicious files, skipping based upon the skip_idx and skip_val
    if skip_val is not None and isinstance(skip_val, str):
        skip_idx = [p.as_posix() for p in mal_files].index(skip_val)
    if skip_idx is not None and skip_idx > 0:
        mal_files = mal_files[skip_idx:]

    mal_files_splits = list(batch(mal_files, ceil_divide(len(mal_files), n_workers)))
    args = [
        (
            mal_files,
            run_flags,
            chunk_size,
            model,
            attributions_path,
            confidences_path,
            text_section_bounds,
            inc_replace_modes,
            inc_batch_size,
            br_d,
            br_a,
            ben_files,
            control.get("errors"),
            control.getboolean("verbose"),
        )
        for mal_files in mal_files_splits
    ]

    print(f"run START @{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start = time.time()

    # Single processor
    if n_workers == 1:
        run_samples(*args[0])
    # Concurrent (does not work very well)
    else:
        with multiprocessing.Pool(processes=n_workers) as pool:
            pool.starmap(run_samples, args, chunksize=1)

    # TODO: clean up run_swaps_count
    # if run_flags.getboolean("run_swaps_count"):
    #     attributions_file = attributions_path / f"{f.name}.pt"
    #     attributions = torch.load(attributions_file, map_location=device)[l:u]
    #     threshold = 0.0
    #     n_swaps = swaps_count(attributions, chunk_size, threshold)
    #     swaps_count_log = output_path / f"swaps_count_{threshold}.txt"
    #     with open(swaps_count_log, "a") as handle:
    #         handle.write(f"{f.as_posix()}, {n_swaps}\n")

    print(f"run DONE @{datetime.now()} in {time.time() - start:.2f}s")



def main(config: ConfigParser) -> None:
    p = config["MODEL"]
    model_params = cl.ModelParams(p.get("model_name"))
    p = config["EXE"]
    exe_params = executable_helper.ExeParams(p.get("text_section_bounds_file"))
    p = config["DATA"]
    data_params = cl.DataParams(max_len=p.getint("max_len"), batch_size=p.getint("batch_size"))
    p = config["EXPLAIN"]
    explain_oh = explain.OutputHelper.from_path_parent(p.get("output_path"))
    mal_explain_oh = explain_oh["mal"]
    ben_explain_oh = explain_oh["ben"]
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
    cfg.init(p.get("device"), p.getint("seed"))
    run(model_params, data_params, exe_params, mal_explain_oh, ben_explain_oh, modify_params, control_params)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, default="config_files/modify/default.ini")
    args = parser.parse_args()
    config = ConfigParser(allow_no_value=True)
    config.read(args.config_file)
    main(config)
