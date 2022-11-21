"""
Explanation algorithms.

TODO:
    - Add a way to kickstart the explanation process from a specific:
        - Batch iteration (benign/malicious)
        - File index (benign/malicious)
        - File name (class agnostic)
"""

from __future__ import annotations
from argparse import ArgumentParser
from collections import OrderedDict
from configparser import ConfigParser
from dataclasses import asdict, dataclass
from inspect import signature
from itertools import chain
from pathlib import Path
from pprint import pformat, pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
import typing as tp

import captum.attr as ca
import pandas as pd
from tqdm import tqdm
import torch
from torch import Tensor

import classifier as cl
import cfg
import executable_helper
from utils import batch, ceil_divide, exception_info
from typing_ import ForwardFunction, Pathlike


BASELINE = cl.PAD_VALUE
TARGET = 1


# Not intended as a direct interface to the attribute method of an explanation algorithm.
@dataclass
class AttributeParams:
    baselines: int = BASELINE
    feature_mask_mode: tp.Literal["all", ".text"] = None
    feature_mask_size: int = None
    method: str = None
    n_steps: int = None
    perturbations_per_eval: int = None
    sliding_window_shapes_size: int = None
    strides: int = None
    target: int = TARGET

    def __post_init__(self) -> None:
        self.baselines = BASELINE if self.baselines is None else self.baselines
        self.target = TARGET if self.target is None else self.target

    def __dict__(self) -> OrderedDict:
        return OrderedDict((k, v) for k, v in sorted(self.__dict__))


@dataclass
class ExplainParams:
    softmax: bool
    layer: str
    alg: str
    attrib_params: AttributeParams


@dataclass
class ControlParams:
    output_root: Path
    ben_start_idx: int = None
    mal_start_idx: int = None
    ben_end_idx: int = None
    mal_end_idx: int = None
    ben_start_batch: int = None
    mal_start_batch: int = None
    ben_end_batch: int = None
    mal_end_batch: int = None
    errors: str = "warn"

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)
        idx = [self.ben_start_idx, self.mal_start_idx, self.ben_end_idx, self.mal_end_idx]
        batch_ = [
            self.ben_start_batch,
            self.mal_start_batch,
            self.ben_end_batch,
            self.mal_end_batch,
        ]
        using_idx = any(i is not None for i in idx)
        using_batch = any(b is not None for b in batch_)
        assert not (using_idx and using_batch), "Cannot specify both index and batch skipping"


class OutputHelper:
    def __init__(
        self,
        output_root: Pathlike,
        softmax: bool,
        layer: str,
        alg: str,
        attrib_params: AttributeParams,
        *,
        split: str = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.softmax = softmax
        self.layer = layer
        self.alg = alg
        self.attrib_params = attrib_params
        self.split = split
        self.dict = {"softmax": self.softmax, "layer": self.layer, "alg": self.alg}
        self.output_path = self.output_root.joinpath(*self._get_components())

    def _get_components(self) -> tp.List[str]:
        components = list(self.dict.items())
        components += list(asdict(self.attrib_params).items())
        components = ["__".join([k, str(v)]) for k, v in components]
        if self.split is not None:
            components.append(self.split)
        return components

    @classmethod
    def from_params(
        cls, explain_params: ExplainParams, control_params: ControlParams, *, split: str = None
    ) -> OutputHelper:
        return cls(
            control_params.output_root,
            explain_params.softmax,
            explain_params.layer,
            explain_params.alg,
            explain_params.attrib_params,
            split=split,
        )


class FeatureMask:
    def __init__(
        self,
        X: Tensor,
        size: int,
        mode: str,
        lowers: tp.List[int] = None,
        uppers: tp.List[int] = None,
    ) -> None:
        self.X = X if X.dim() == 2 else X.unsqueeze(0)
        self.size = size
        self.mode = mode
        self.lowers = lowers
        self.uppers = uppers
        if self.mode == "all":
            pass
        elif self.mode == "text":
            assert isinstance(self.lowers, list) and isinstance(self.uppers, list)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def __call__(self) -> Tensor:
        if self.mode == "all":
            mask = self.chunk_mask(self.X[0], self.size)
        elif self.mode == "text":
            masks = []
            for i, (l, u, x) in enumerate(zip(self.lowers, self.uppers, self.X)):
                u = min(u, x.shape[0])
                if l >= u:
                    raise ValueError(
                        f"The {i}th Tensor in this batch is invalid: {l=}, {u=}.\n"
                        "This should have been caught earlier."
                    )
                mask = self.chunk_mask(x[l:u], self.size)
                fill_val = ceil_divide(u - l, self.size)
                begin = torch.full((l,), fill_val, dtype=torch.int64)
                end = torch.full((x.shape[0] - u,), fill_val, dtype=torch.int64)
                mask = torch.cat([begin, mask, end])
                masks.append(mask)
            mask = torch.stack(masks).to(torch.int64)
        else:
            raise ValueError("Both or neither lower and upper bounds must be provided.")
        return mask

    @staticmethod
    def chunk_mask(x: Tensor, size: int) -> Tensor:
        length = x.shape[0]
        if length < size:
            return torch.full((length,), 0, dtype=torch.int64)
        q, r = divmod(length, size)
        mask = torch.cat([torch.full((size,), i) for i in range(q)])
        mask = torch.cat([mask, torch.full((r,), q)])
        return mask.type(torch.int64)


def explain_batch(
    explain_params: ExplainParams,
    model: cl.MalConvLike,
    forward_function: ForwardFunction,
    layer: torch.nn.Module,
    inputs: Tensor,
    lowers: tp.List[int] = None,
    uppers: tp.List[int] = None,
) -> Tensor:
    if explain_params.alg == "FeatureAblation":
        alg = ca.FeatureAblation(forward_function)
    elif explain_params.alg == "FeaturePermutation":
        alg = ca.FeaturePermutation(forward_function)
    elif explain_params.alg == "IntegratedGradients":
        alg = ca.IntegratedGradients(forward_function)
    elif explain_params.alg == "KernelShap":
        alg = ca.KernelShap(forward_function)
    elif explain_params.alg == "LayerActivation":
        alg = ca.LayerActivation(forward_function, getattr(model, layer))
    elif explain_params.alg == "LayerIntegratedGradients":
        alg = ca.LayerIntegratedGradients(forward_function, getattr(model, layer))
    elif explain_params.alg == "Occlusion":
        alg = ca.Occlusion(forward_function)
    elif explain_params.alg == "ShapleyValueSampling":
        alg = ca.ShapleyValueSampling(forward_function)
    else:
        raise ValueError(f"Unknown algorithm: {explain_params.alg}")

    # Collect valid keyword arguments for this particular algorithm's attribute method
    attrib_params = explain_params.attrib_params
    valid = set(signature(alg.attribute).parameters.keys())
    kwargs = {k: v for k, v in asdict(attrib_params).items() if k in valid and v is not None}

    # Get objects for the keyword arguments that cannot be contained in the AttributeParams
    if (
        "feature_mask" in valid
        and attrib_params.feature_mask_size is not None
        and attrib_params.feature_mask_mode is not None
    ):
        kwargs["feature_mask"] = FeatureMask(
            inputs, attrib_params.feature_mask_size, attrib_params.feature_mask_mode, lowers, uppers
        )().to(cfg.device)
    if "sliding_window_shapes" in valid and attrib_params.sliding_window_shapes_size is not None:
        kwargs["sliding_window_shapes"] = (attrib_params.sliding_window_shapes_size,)

    attribs = alg.attribute(inputs, **kwargs)
    return attribs


def filter_bounds(bounds: tp.Dict[str, tp.Dict[str, str]], max_len: int = None):
    return {
        k_1: {k_2: int(v_2) for k_2, v_2 in v_1.items()}
        for k_1, v_1 in bounds.items()
        if (
            v_1["lower"].isdigit()
            and v_1["upper"].isdigit()
            and int(v_1["size"]) >= int(v_1["upper"])
            and int(v_1["upper"]) > int(v_1["lower"])
            and (int(v_1["lower"]) < max_len if isinstance(max_len, int) else True)
        )
    }


def slice_files(
    files: tp.List[Pathlike],
    start_idx: int = None,
    end_idx: int = None,
    start_batch: int = None,
    end_batch: int = None,
    batch_size: int = None,
) -> tp.List[Pathlike]:
    if isinstance(start_idx, int):
        start = start_idx
    elif isinstance(start_batch, int):
        start = start_batch * batch_size
    else:
        start = None
    if isinstance(end_idx, int):
        end = end_idx
    elif isinstance(end_batch, int):
        end = end_batch * batch_size
    else:
        end = None
    return files[start:end]


def run(
    model_params: cl.ModelParams,
    data_params: cl.DataParams,
    exe_params: executable_helper.ExeParams,
    explain_params: ExplainParams,
    control_params: ControlParams,
) -> None:
    # Essential components for explanation
    model = cl.get_model(model_params.name)
    forward_function = cl.forward_function_malconv(model, explain_params.softmax)
    layer = None if explain_params.layer is None else getattr(model, explain_params.layer)

    # Set up the output structure
    ben_oh = OutputHelper.from_params(explain_params, control_params, split="ben")
    ben_oh.output_path.mkdir(parents=True, exist_ok=True)
    mal_oh = OutputHelper.from_params(explain_params, control_params, split="mal")
    mal_oh.output_path.mkdir(parents=True, exist_ok=True)

    # Benign and malicious files to explain
    ben_files = list(chain(cl.WINDOWS_TRAIN_PATH.iterdir(), cl.WINDOWS_TEST_PATH.iterdir()))
    mal_files = list(chain(cl.SOREL_TRAIN_PATH.iterdir(), cl.SOREL_TEST_PATH.iterdir()))

    # Lower and upper bounds for the text sections of all the files
    if explain_params.attrib_params.feature_mask_mode == "text":
        bounds = pd.read_csv(exe_params.text_section_bounds_file, index_col="file").to_dict("index")
        bounds = filter_bounds(bounds, max_len=data_params.max_len)
        ben_files = [p for p in ben_files if p.as_posix() in bounds]
        mal_files = [p for p in mal_files if p.as_posix() in bounds]

    # Slice the files to start/end at different points
    ben_files = slice_files(
        ben_files,
        control_params.ben_start_idx,
        control_params.ben_end_idx,
        control_params.ben_start_batch,
        control_params.ben_end_batch,
        data_params.batch_size,
    )
    mal_files = slice_files(
        mal_files,
        control_params.mal_start_idx,
        control_params.mal_end_idx,
        control_params.mal_start_batch,
        control_params.mal_end_batch,
        data_params.batch_size,
    )

    # Keep the benign and malicious data separate, so it can be placed in different directories
    ben_dataset, ben_loader = cl.get_dataset_and_loader(
        ben_files,
        None,
        max_len=data_params.max_len,
        batch_size=data_params.batch_size,
        shuffle_=False,
        sort_by_size=True,
    )
    mal_dataset, mal_loader = cl.get_dataset_and_loader(
        None,
        mal_files,
        max_len=data_params.max_len,
        batch_size=data_params.batch_size,
        shuffle_=False,
        sort_by_size=True,
    )

    # Conglomerate the different data structures
    data = [
        (mal_dataset, mal_loader, mal_oh),
        (ben_dataset, ben_loader, ben_oh),
    ]

    # Run the explanation algorithm on each dataset
    for dataset, loader, oh in data:
        files = batch([Path(e[0]) for e in dataset.all_files], data_params.batch_size)
        gen = tqdm(
            zip(loader, files),
            total=ceil_divide(len(dataset), data_params.batch_size),
            initial=0,
        )
        for i, ((inputs, targets), files) in enumerate(gen):
            try:
                inputs = inputs.to(cfg.device)
                targets = targets.to(cfg.device)

                if explain_params.attrib_params.feature_mask_mode == "text":
                    lowers = [bounds[f.as_posix()]["lower"] for f in files]
                    uppers = [bounds[f.as_posix()]["upper"] for f in files]
                else:
                    lowers = None
                    uppers = None

                attribs = explain_batch(
                    explain_params,
                    model,
                    forward_function,
                    layer,
                    inputs,
                    lowers,
                    uppers,
                )
                for attr, f in zip(attribs, files):
                    torch.save(attr, oh.output_path / (f.name + ".pt"))

            except Exception as e:
                if control_params.errors == "ignore":
                    pass
                else:
                    ignore = {"ben_files", "mal_files", "bounds"}
                    locals_ = {k: v for k, v in locals().items() if k not in ignore}
                    print(exception_info(e, locals_))
                if control_params.errors == "raise":
                    raise e


def main(config: ConfigParser) -> None:
    p = config["MODEL"]
    model_params = cl.ModelParams(p.get("model_name"))
    p = config["EXE"]
    exe_params = executable_helper.ExeParams(p.get("text_section_bounds_file"))
    p = config["DATA"]
    data_params = cl.DataParams(max_len=p.getint("max_len"), batch_size=p.getint("batch_size"))
    p = config[config.get("EXPLAIN", "alg")]
    attrib_params = AttributeParams(
        p.getint("baselines"),
        p.get("feature_mask_mode"),
        p.getint("feature_mask_size"),
        p.get("method"),
        p.getint("n_steps"),
        p.getint("perturbations_per_eval"),
        p.getint("sliding_window_shapes"),
        p.getint("strides"),
        p.getint("target"),
    )
    p = config["EXPLAIN"]
    explain_params = ExplainParams(
        p.getboolean("softmax"), p.get("layer"), p.get("alg"), attrib_params
    )
    p = config["CONTROL"]
    control_params = ControlParams(
        p.get("output_root"),
        p.getint("ben_start_idx"),
        p.getint("mal_start_idx"),
        p.getint("ben_stop_idx"),
        p.getint("mal_stop_idx"),
        p.getint("ben_start_batch"),
        p.getint("mal_start_batch"),
        p.getint("ben_stop_batch"),
        p.getint("mal_stop_batch"),
        p.get("errors"),
    )
    cfg.init(p.get("device"), p.getint("seed"))
    run(model_params, data_params, exe_params, explain_params, control_params)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, default="config_files/explain/default.ini")
    args = parser.parse_args()
    config = ConfigParser(allow_no_value=True)
    config.read(args.config_file)
    main(config)
