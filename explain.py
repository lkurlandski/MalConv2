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
from itertools import chain, islice
from pathlib import Path
from pprint import pformat
import sys
import typing as tp

import captum.attr as ca
from tqdm import tqdm
import torch
from torch import Tensor

import classifier as cl
import cfg
from utils import batch, ceil_divide, exception_info
from typing_ import ForwardFunction, Pathlike


BASELINE = cl.PAD_VALUE
TARGET = 1


# Caution: these parameters are not intended to be directly passed to the method.
@dataclass
class AttributeParams:
    baselines: int = BASELINE
    feature_mask: int = None
    method: str = None
    n_steps: int = None
    perturbations_per_eval: int = None
    sliding_window_shapes: int = None
    strides: int = None
    target: int = TARGET

    def __post_init__(self) -> None:
        self.baselines = BASELINE if self.baselines is None else self.baselines
        self.target = TARGET if self.target is None else self.target

    def __dict__(self) -> OrderedDict:
        return OrderedDict((k, v) for k, v in sorted(self.__dict__))

    def get_feature_mask(self, inputs) -> Tensor:
        if self.feature_mask is None:
            return None
        return get_feature_mask(inputs, self.feature_mask)

    def get_sliding_window_shapes(self) -> tp.Tuple[int]:
        if self.sliding_window_shapes is None:
            return None
        return (self.sliding_window_shapes,)


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


def get_feature_mask(
    inputs: torch.Tensor,
    mask_size: int,
    cat_across_batch: bool = False,
) -> torch.Tensor:
    if inputs.shape[1] < mask_size:
        return torch.full((inputs.shape[1],), 0).unsqueeze(0)

    q, r = divmod(inputs.shape[1], mask_size)
    feature_mask = torch.cat([torch.full((mask_size,), i) for i in range(q)])
    feature_mask = torch.cat([feature_mask, torch.full((r,), q)])
    if cat_across_batch:
        feature_mask = torch.cat([feature_mask.unsqueeze(0) for _ in range(inputs.shape[0])], 0)
    else:
        feature_mask = feature_mask.unsqueeze(0)
    feature_mask = feature_mask.type(torch.int64).to(cfg.device)
    return feature_mask


def explain_batch(
    explain_params: ExplainParams,
    model: cl.MalConvLike,
    forward_function: ForwardFunction,
    inputs: Tensor,
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

    ap = explain_params.attrib_params
    valid = set(signature(alg.attribute).parameters.keys())
    kwargs = {k: v for k, v in asdict(ap).items() if k in valid}

    # Convert certain arguments to the correct type
    if "feature_mask" in kwargs and ap.feature_mask is not None:
        kwargs["feature_mask"] = ap.get_feature_mask(inputs)
    if "sliding_window_shapes" in kwargs and ap.sliding_window_shapes is not None:
        kwargs["sliding_window_shapes"] = ap.get_sliding_window_shapes()

    attribs = alg.attribute(inputs, **kwargs)
    return attribs


def run(
    model_params: cl.ModelParams,
    data_params: cl.DataParams,
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

    # Keep the benign and malicious data separate, so it can be placed in different directories
    ben_dataset, ben_loader = cl.get_dataset_and_loader(
        chain(cl.WINDOWS_TRAIN_PATH.iterdir(), cl.WINDOWS_TEST_PATH.iterdir()),
        None,
        max_len=data_params.max_len,
        batch_size=data_params.batch_size,
        shuffle_=True,
        sort_by_size=True,
    )
    mal_dataset, mal_loader = cl.get_dataset_and_loader(
        None,
        chain(cl.SOREL_TRAIN_PATH.iterdir(), cl.SOREL_TEST_PATH.iterdir()),
        max_len=data_params.max_len,
        batch_size=data_params.batch_size,
        shuffle_=True,
        sort_by_size=True,
    )

    # Conglomerate the different data structures
    data = [
        (mal_dataset, mal_loader, mal_oh),
        (ben_dataset, ben_loader, ben_oh),
    ]

    # Run the explanation algorithm on each dataset
    for dataset, loader, oh in data:
        batched_files = batch([Path(e[0]) for e in dataset.all_files], data_params.batch_size)
        gen = zip(loader, batched_files)
        initial = 0
        total = ceil_divide(len(dataset), data_params.batch_size)
        for i, ((inputs, _), files) in enumerate(tqdm(gen, initial=initial, total=total)):
            try:
                inputs = inputs.to(cfg.device)
                attribs = explain_batch(explain_params, model, forward_function, inputs)
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
    p = config["DATA"]
    data_params = cl.DataParams(max_len=p.getint("max_len"), batch_size=p.getint("batch_size"))
    p = config[config.get("EXPLAIN", "alg")]
    attrib_params = AttributeParams(
        p.getint("baselines"),
        p.getint("feature_mask"),
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
    run(model_params, data_params, explain_params, control_params)
