"""
Modify malware and record the classifier's response to the modified malware.

Run and append to existing log file:
python modify_inc.py --config_file=CONFIG_FILE >>LOG_FILE 2>&1 &

TODO:
    -
"""

from __future__ import annotations
from argparse import ArgumentParser
from configparser import ConfigParser
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
import os  # pylint: disable=unused-import
from pathlib import Path  # pylint: disable=unused-import
from pprint import pformat, pprint  # pylint: disable=unused-import
from random import shuffle
import sys  # pylint: disable=unused-import
import typing as tp

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

import classifier as cl
import cfg
import executable_helper
import explain
from typing_ import ErrorMode, Pathlike
from utils import ceil_divide, exception_info, get_offset_chunk_tensor, section_header


REP_SOURCE_MODES = ["correspond", "least", "pad", "random"]
REP_TARGET_MODES = ["most", "ordered", "random"]


@dataclass
class ModifyParams:
    rep_source_mode: str
    rep_target_mode: str
    exp_explain_params: explain.ExplainParams
    exp_control_params: explain.ControlParams
    rep_file: tp.Optional[Path] = None
    chunk_size: int = None

    def __post_init__(self):
        self.rep_file = cl.WINDOWS_PATH / self.rep_file if self.rep_file is not None else None
        self.chunk_size = self.exp_explain_params.attrib_params.feature_mask_size


@dataclass
class ControlParams:
    output_root: Path
    start_idx: int = None
    end_idx: int = None
    errors: ErrorMode = "warn"
    progress_bar: bool = True
    verbose: bool = False

    def __post_init__(self):
        self.output_root = Path(self.output_root)


class OutputHelper:
    def __init__(
        self,
        output_root: Pathlike,
        model_name: cl.ModelName,
        attribs_path: Path,
        rep_source_mode: str,
        rep_target_mode: str,
    ) -> None:
        self.output_root = Path(output_root)
        self.model_name = model_name
        self.rep_source_mode = rep_source_mode
        self.rep_target_mode = rep_target_mode
        self.output_path = (
            self.output_root
            / f"model_name__{self.model_name}"
            / attribs_path
            / f"rep_source_mode__{rep_source_mode}"
            / f"rep_target_mode__{rep_target_mode}"
        )

    @classmethod
    def from_params(
        cls,
        model_params: cl.ModelParams,
        exp_explain_params: explain.ExplainParams,
        exp_control_params: explain.ControlParams,
        control_params: ControlParams,
        modify_params: ModifyParams,
    ) -> OutputHelper:
        attribs_path = explain.OutputHelper.from_params(
            exp_explain_params, exp_control_params, split=""
        ).output_path.relative_to(exp_control_params.output_root)
        return cls(
            control_params.output_root,
            model_params.name,
            attribs_path,
            modify_params.rep_source_mode,
            modify_params.rep_target_mode,
        )


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
        mode: tp.Literal["random", "pad", "correspond", "least"],
        chunk_size: int,
        rep_bytes: Tensor = None,
        rep_attribs: Tensor = None,
    ) -> None:
        self.mode = mode
        self.chunk_size = chunk_size
        self.rep_bytes = rep_bytes
        self.rep_attribs = rep_attribs
        self.least_idx = None
        if mode == "least":
            self.least_idx = self.sorted_chunked_tensor_indices(rep_attribs, chunk_size)
        self.i = -1

    def __call__(self, l_rep: int, u_rep: int, offset: int = None) -> Tensor:
        self.i += 1
        size = u_rep - l_rep
        if self.mode == "random":
            return self._random(size)
        if self.mode == "pad":
            return self._pad(size)
        if self.mode == "correspond":
            return self._correspond(l_rep, u_rep, offset)
        if self.mode == "least":
            return self._least(size)

        raise ValueError(f"Invalid mode: {self.mode}")

    def _random(self, size) -> Tensor:
        return torch.full((size,), explain.BASELINE)

    def _pad(self, size) -> Tensor:
        return torch.randint(low=0, high=cl.NUM_EMBEDDINGS, size=(size,))

    def _correspond(self, l_rep: int, u_rep: int, offset: int) -> Tensor:
        return self.rep_bytes[l_rep - offset : u_rep - offset]

    def _least(self, size) -> Tensor:
        l = self.least_idx[self.i]
        u = l + min(self.chunk_size, size)
        return self.rep_bytes[l:u]

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
            raise ValueError(f"Invalid replace mode: {self.mode}")

        # Add the offset to the lower bounds and add the chunk size to the upper bounds
        lower_bounds = [min(u, l + lb) for lb in lower_bounds]
        upper_bounds = [min(u, lb + self.chunk_size) for lb in lower_bounds]

        self.bounds = list(zip(lower_bounds, upper_bounds))
        self.i = -1

    def __iter__(self) -> IncTargetBoundsSelector:
        return self

    def __len__(self) -> int:
        return len(self.bounds)

    def __next__(self) -> tp.Tuple[int, int]:
        self.i += 1
        if self.i < len(self.bounds):
            return self.bounds[self.i]
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
        for _ in range(self.length):
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
        if len(batch) == batch_size or i + 1 == len(target_bounds):
            c = cl.confidence_scores(model, torch.stack(batch)).tolist()
            confs.extend(c)
            batch.clear()

    return np.array(confs)


def run(
    model_params: cl.ModelParams,
    data_params: cl.DataParams,
    exe_params: executable_helper.ExeParams,
    modify_params: ModifyParams,
    control_params: ControlParams,
    output_path: Path,
    mal_attribs_path: Path,
    ben_attribs_path: Path,
):
    bounds = executable_helper.get_bounds(exe_params.text_section_bounds_file)
    model = cl.get_model(model_params.name)

    mal_files = chain(cl.SOREL_TRAIN_PATH.iterdir(), cl.SOREL_TEST_PATH.iterdir())
    explained = set(p.name.strip(".pt") for p in mal_attribs_path.iterdir())
    mal_files = [p for p in mal_files if p.name in explained]
    mal_files.sort(key=lambda p: bounds[p.as_posix()][1] - bounds[p.as_posix()][0])

    # Optional parameters need default value of None
    rep_bytes, rep_attribs, attribs_text = None, None, None

    # Get the benign replacement bytes and the benign replacement's attributions
    if modify_params.rep_source_mode in {"correspond", "least"}:
        rep_bytes = Tensor(executable_helper.read_binary(modify_params.rep_file))
    if modify_params.rep_source_mode in {"least"}:
        l_text, u_text = bounds[modify_params.rep_file.as_posix()]
        attribs_file = ben_attribs_path / (modify_params.rep_file.name + ".pt")
        rep_attribs = torch.load(attribs_file, map_location=cfg.device)[l_text:u_text]

    gen = mal_files
    total = len(mal_files) if control_params.end_idx is None else control_params.end_idx
    if control_params.progress_bar:
        gen = tqdm(gen, total=total)

    print(
        f"Starting modification: initial={0}, "
        f"start={control_params.start_idx}, "
        f"end={control_params.end_idx}, "
        f"@{datetime.now()}"
    )
    for i, f in enumerate(gen):
        if control_params.verbose:
            print(f"{i} / {total} = {100 * i // total}% @{datetime.now()}", flush=True)
        if control_params.start_idx is not None and i < control_params.start_idx:
            continue
        if control_params.end_idx is not None and i >= control_params.end_idx:
            break

        try:
            # Get the full piece of malware and the bounds of its .text section
            X = Tensor(executable_helper.read_binary(f, max_len=data_params.max_len))
            l_text, u_text = bounds[f.as_posix()]

            # Get attributions to identify the region of the malware to swap
            if modify_params.rep_target_mode in {"most"}:
                attribs_file = mal_attribs_path / (f.name + ".pt")
                attribs_text = torch.load(attribs_file, map_location=cfg.device)[l_text:u_text]
                if (o := get_offset_chunk_tensor(attribs_text, modify_params.chunk_size)) != 0:
                    raise ValueError(f"attributions have nonzero chunk offset {o=}")

            # Ensure the replacements are as large as the .text section to replace
            size = ceil_divide(u_text - l_text, modify_params.chunk_size) * modify_params.chunk_size
            rep_bytes = extend_tensor(rep_bytes, size) if rep_bytes is not None else rep_bytes
            rep_attribs = (
                extend_tensor(rep_attribs, size) if rep_attribs is not None else rep_attribs
            )

            # Get the bytes to replace with and the regions to replace at each iteration
            source_replacer = IncSourceBytesSelector(
                modify_params.rep_source_mode, modify_params.chunk_size, rep_bytes, rep_attribs
            )
            target_bounds = IncTargetBoundsSelector(
                modify_params.rep_target_mode,
                l_text,
                u_text,
                modify_params.chunk_size,
                attribs_text,
            )

            # Compute the confidence at every iteration of the algorithm
            confs = inc_sub_and_eval(
                model,
                X,
                l_text,
                source_replacer,
                target_bounds,
                data_params.batch_size,
            )
            np.savetxt(output_path / f.name, confs, delimiter="\n")

        except Exception as e:  # pylint: disable=broad-except
            if control_params.errors == "ignore":
                pass
            else:
                ignore = {"bounds", "mal_files", "gen", "explained"}
                locals_ = {k: v for k, v in locals().items() if k not in ignore}
                print(exception_info(e, locals_))
            if control_params.errors == "raise":
                raise e


def parse_config(
    config: ConfigParser,
) -> tp.Tuple[
    cl.ModelParams,
    cl.DataParams,
    executable_helper.ExeParams,
    ModifyParams,
    ControlParams,
    Path,
    Path,
    Path,
]:
    p = config["MODEL"]
    model_params = cl.ModelParams(p.get("model_name"))
    p = config["DATA"]
    data_params = cl.DataParams(max_len=p.getint("max_len"), batch_size=p.getint("batch_size"))
    p = config["EXE"]
    exe_params = executable_helper.ExeParams(p.get("text_section_bounds_file"))

    p = config["EXPLAIN"]
    explain_config = ConfigParser(allow_no_value=True)
    explain_config.read(p.get("config_file"))
    _, _, _, exp_explain_params, exp_control_params = explain.parse_config(explain_config)

    p = config["MODIFY"]
    modify_params = ModifyParams(
        p.get("rep_source_mode"),
        p.get("rep_target_mode"),
        exp_explain_params,
        exp_control_params,
        p.get("rep_file"),
    )

    p = config["CONTROL"]
    control_params = ControlParams(
        p.get("output_root"),
        p.getint("start_idx"),
        p.getint("end_idx"),
        p.get("errors"),
        p.getboolean("progress_bar"),
        p.getboolean("verbose"),
    )

    modify_output_path = OutputHelper.from_params(
        model_params, exp_explain_params, exp_control_params, control_params, modify_params
    ).output_path
    mal_attribs_path = explain.OutputHelper.from_params(
        exp_explain_params, exp_control_params, split="mal"
    ).output_path
    ben_attribs_path = explain.OutputHelper.from_params(
        exp_explain_params, exp_control_params, split="ben"
    ).output_path

    return (
        model_params,
        data_params,
        exe_params,
        modify_params,
        control_params,
        modify_output_path,
        mal_attribs_path,
        ben_attribs_path,
    )


def main(config: ConfigParser) -> None:
    cfg.init(config["CONTROL"].get("device"), config["CONTROL"].getint("seed"))
    configurations = parse_config(config)
    modify_output_path = configurations[5]
    modify_output_path.mkdir(parents=True, exist_ok=True)
    run(*configurations)


if __name__ == "__main__":
    print(section_header(f"START @{datetime.now()}"))
    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, default="config_files/modify_inc/default.ini")
    args = parser.parse_args()
    config = ConfigParser(allow_no_value=True)
    config.read(args.config_file)
    print(f"{args.config_file=}")
    main(config)
    print(section_header(f"END @{datetime.now()}"))
