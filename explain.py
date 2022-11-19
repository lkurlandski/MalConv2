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
from collections import namedtuple, OrderedDict
from configparser import ConfigParser
from dataclasses import asdict, dataclass
from inspect import signature
from itertools import chain, islice
from pathlib import Path
from pprint import pformat
import sys
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
    ben_start_idx: int = 0
    ben_end_idx: int = None
    mal_start_idx: int = 0
    mal_end_idx: int = None
    errors: str = "warn"

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)


FilesAndBounds = namedtuple("FilesAndBounds", ["files", "lowers", "uppers"])


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


class BenignOutputHelper(OutputHelper):
    @classmethod
    def from_params(
        cls, explain_params: ExplainParams, control_params: ControlParams
    ) -> BenignOutputHelper:
        return super().from_params(explain_params, control_params, split="ben")


class MaliciousOutputHelper(OutputHelper):
    @classmethod
    def from_params(
        cls, explain_params: ExplainParams, control_params: ControlParams
    ) -> MaliciousOutputHelper:
        return OutputHelper.from_params(explain_params, control_params, split="mal")


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
            return self.chunk_mask(self.X[0], self.size)
        elif self.mode == "text":
            masks = []
            for l, u, x in zip(self.lowers, self.uppers, self.X):
                u = min(u, x.shape[0])
                if l > u:
                    masks.append(torch.zeros_like(x))
                    continue
                m = self.chunk_mask(x[l:u], self.size)
                fill_val = ceil_divide(u - l, self.size)
                b = torch.full((l,), fill_val, dtype=torch.int64)
                e = torch.full((x.shape[0] - u,), fill_val, dtype=torch.int64)
                mask = torch.cat([b, m, e])
                masks.append(mask)
            return torch.stack(masks)
        else:
            raise ValueError("Both or neither lower and upper bounds must be provided.")

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
        alg = ca.LayerActivation(forward_function, getattr(model, explain_params.layer))
    elif explain_params.alg == "LayerIntegratedGradients":
        alg = ca.LayerIntegratedGradients(forward_function, getattr(model, explain_params.layer))
    elif explain_params.alg == "Occlusion":
        alg = ca.Occlusion(forward_function)
    elif explain_params.alg == "ShapleyValueSampling":
        alg = ca.ShapleyValueSampling(forward_function)
    else:
        raise ValueError(f"Unknown algorithm: {explain_params.alg}")

    # Collect valid keyword arguments for this particular algorithm's attribute method
    ap = explain_params.attrib_params
    valid = set(signature(alg.attribute).parameters.keys())
    kwargs = {k: v for k, v in asdict(ap).items() if k in valid and v is not None}

    # Get objects for the keyword arguments that cannot be contained in the AttributeParams
    if (
        "feature_mask" in valid
        and ap.feature_mask_size is not None
        and ap.feature_mask_mode is not None
    ):
        kwargs["feature_mask"] = FeatureMask(
            inputs, ap.feature_mask_size, ap.feature_mask_mode, lowers, uppers
        )().to(cfg.device)
    if "sliding_window_shapes" in valid and ap.sliding_window_shapes_size is not None:
        kwargs["sliding_window_shapes"] = (ap.sliding_window_shapes_size,)

    attribs = alg.attribute(inputs, **kwargs)
    return attribs


def get_files_and_bounds(
    feature_mask_mode: str,
    text_section_bounds_file: Pathlike,
) -> tp.Tuple[FilesAndBounds, FilesAndBounds]:
    if feature_mask_mode == "all":
        ben_files = chain(cl.WINDOWS_TRAIN_PATH.iterdir(), cl.WINDOWS_TEST_PATH.iterdir())
        mal_files = chain(cl.SOREL_TRAIN_PATH.iterdir(), cl.SOREL_TEST_PATH.iterdir())
        ben_lowers = None
        mal_lowers = None
        ben_uppers = None
        mal_uppers = None
    elif feature_mask_mode == "text":
        bounds = pd.read_csv(text_section_bounds_file)
        ben_files = []
        mal_files = []
        ben_lowers = []
        mal_lowers = []
        ben_uppers = []
        mal_uppers = []
        for i, row in bounds.iterrows():
            if row["lower"] != "None" and row["upper"] != "None":
                if "Windows" in row["file"]:
                    ben_files.append(Path(row["file"]))
                    ben_lowers.append(int(row["lower"]))
                    ben_uppers.append(int(row["upper"]))
                elif "Sorel" in row["file"]:
                    mal_files.append(Path(row["file"]))
                    mal_lowers.append(int(row["lower"]))
                    mal_uppers.append(int(row["upper"]))
    else:
        raise ValueError(f"Unknown feature mask: {explain_params.attrib_params.feature_mask_mode}")

    ben = FilesAndBounds(ben_files, ben_lowers, ben_uppers)
    mal = FilesAndBounds(mal_files, mal_lowers, mal_uppers)
    return ben, mal


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
    ben_oh = BenignOutputHelper.from_params(explain_params, control_params)
    ben_oh.output_path.mkdir(parents=True, exist_ok=True)
    mal_oh = MaliciousOutputHelper.from_params(explain_params, control_params)
    mal_oh.output_path.mkdir(parents=True, exist_ok=True)

    ben_fab, mal_fab = get_files_and_bounds(
        explain_params.attrib_params.feature_mask_mode, exe_params.text_section_bounds_file
    )

    # Keep the benign and malicious data separate, so it can be placed in different directories
    ben_dataset, ben_loader = cl.get_dataset_and_loader(
        ben_fab.files,
        None,
        max_len=data_params.max_len,
        batch_size=data_params.batch_size,
        shuffle_=True,
        sort_by_size=True,
    )
    mal_dataset, mal_loader = cl.get_dataset_and_loader(
        None,
        mal_fab.files,
        max_len=data_params.max_len,
        batch_size=data_params.batch_size,
        shuffle_=True,
        sort_by_size=True,
    )

    # Conglomerate the different data structures
    data = [
        (mal_dataset, mal_loader, mal_oh, mal_fab.lowers, mal_fab.uppers),
        (ben_dataset, ben_loader, ben_oh, ben_fab.lowers, ben_fab.uppers),
    ]

    # Run the explanation algorithm on each dataset
    for dataset, loader, oh, lowers, uppers in data:
        files = batch([Path(e[0]) for e in dataset.all_files], data_params.batch_size)
        lowers = batch(lowers, data_params.batch_size) if lowers is not None else None
        uppers = batch(uppers, data_params.batch_size) if uppers is not None else None
        gen = tqdm(
            zip(loader, files, lowers, uppers),
            total=ceil_divide(len(dataset), data_params.batch_size),
            initial=0,
        )
        for i, ((inputs, targets), files, lowers, uppers) in enumerate(gen):
            try:
                inputs = inputs.to(cfg.device)
                attribs = explain_batch(
                    explain_params, model, forward_function, inputs, lowers, uppers
                )
                for i in range(len(files)):
                    torch.save(attribs[i], oh.output_path / (files[i].name + ".pt"))
            except Exception as e:
                if control_params.errors == "warn":
                    print(exception_info(e, locals()))
                elif control_params.errors == "ignore":
                    pass
                else:
                    raise e


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, default="config_files/explain/default.ini")
    args = parser.parse_args()

    config = ConfigParser(allow_no_value=True)
    config.read(args.config_file)

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
        p.get("ben_start_idx"),
        p.get("ben_stop_idx"),
        p.get("mal_start_idx"),
        p.get("mal_stop_idx"),
        p.get("errors"),
    )
    cfg.init(p.get("device"), p.getint("seed"))
    run(model_params, data_params, exe_params, explain_params, control_params)
