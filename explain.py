"""
Explanation algorithms.

Run and append to existing log file:
python explain.py --run --analyze --config_file=CONFIG_FILE >>LOG_FILE 2>&1 &

TODO:
    - Use dict of tuples instead of dict of dicts for the bounds
    - Add documentation for the valid parameters in the config files
    - Alter the output path to use the default arguments passed to attribute
    - Refactor the OutputPath to contain the benign and native splits natively?
"""

from __future__ import annotations
from argparse import ArgumentParser
from collections import OrderedDict
from configparser import ConfigParser
from dataclasses import asdict, dataclass
from datetime import datetime
from inspect import signature
from itertools import chain
import logging
import os  # pylint: disable=unused-import
from pathlib import Path  # pylint: disable=unused-import
from pprint import pformat, pprint  # pylint: disable=unused-import
import shutil
import sys  # pylint: disable=unused-import
import typing as tp

import captum.attr as ca
from tqdm import tqdm
import torch
from torch import nn, Tensor

from argparse_ import LoggingArgumentParser
import classifier as cl
import cfg
import executable_helper
from logging_ import setup_logging
from typing_ import ForwardFunction, Pathlike
from utils import batch, ceil_divide, exception_info, get_offset_chunk_tensor, section_header


BASELINE = cl.PAD_VALUE
TARGET = 1


@dataclass
class AttributeParams:
    """
    Not intended as a direct interface to the attribute method of an explanation algorithm.
    Keeping the parameters in alphabetical order allows for easy access with the OutputHelper.
    """

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

    def cast_types(self) -> AttributeParams:
        try:
            self.baselines = int(self.baselines)
        except ValueError:
            self.baselines = None

        self.feature_mask_mode = str(self.feature_mask_mode)
        if self.feature_mask_mode == "None":
            self.feature_mask_mode = None

        try:
            self.feature_mask_size = int(self.feature_mask_size)
        except ValueError:
            self.feature_mask_size = None

        self.method = str(self.method)
        if self.method == "None":
            self.method = None

        try:
            self.n_steps = int(self.n_steps)
        except ValueError:
            self.n_steps = None

        try:
            self.perturbations_per_eval = int(self.perturbations_per_eval)
        except ValueError:
            self.perturbations_per_eval = None

        try:
            self.sliding_window_shapes_size = int(self.sliding_window_shapes_size)
        except ValueError:
            self.sliding_window_shapes_size = None

        try:
            self.strides = int(self.strides)
        except ValueError:
            self.strides = None

        try:
            self.target = int(self.target)
        except ValueError:
            self.target = None

        return self


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
    progress_bar: bool = True
    verbose: bool = False

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)
        idx = [
            self.ben_start_idx,
            self.mal_start_idx,
            self.ben_end_idx,
            self.mal_end_idx,
        ]
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
        split: str = "all",
    ) -> None:
        self.output_root = Path(output_root)
        self.softmax = softmax
        self.layer = layer
        self.alg = alg
        self.attrib_params = attrib_params
        self.split = split
        self.dict = {"softmax": self.softmax, "layer": self.layer, "alg": self.alg}
        self.base_path = self.output_root.joinpath(*self._get_components())
        self.output_path = self.base_path / "output"
        self.analysis_path = self.base_path / "analysis"
        self.summary_file = self.analysis_path / "summary.csv"

    def _get_components(self) -> tp.List[str]:
        components = list(self.dict.items())
        components += list(asdict(self.attrib_params).items())
        components = ["__".join([k, str(v)]) for k, v in components]
        components.append(self.split)
        return components

    @classmethod
    def from_params(
        cls,
        explain_params: ExplainParams,
        control_params: ControlParams,
        *,
        split: str = None,
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
            mask = self.chunk_mask(self.X[0], self.size).unsqueeze(0)
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
            mask = torch.stack(masks)
        else:
            raise ValueError("Both or neither lower and upper bounds must be provided.")
        return mask.to(torch.int64)

    @staticmethod
    def chunk_mask(x: Tensor, size: int) -> Tensor:
        length = x.shape[0]
        if length < size:
            return torch.full((length,), 0, dtype=torch.int64)
        q, r = divmod(length, size)
        mask = torch.cat([torch.full((size,), i) for i in range(q)])
        mask = torch.cat([mask, torch.full((r,), q)])
        return mask.to(torch.int64)


def get_explanation_algorithm(
    alg: str, forward_function: ForwardFunction, layer: nn.Module = None
) -> ca._utils.attribution.Attribution:
    if alg == "FeatureAblation":
        return ca.FeatureAblation(forward_function)
    if alg == "FeaturePermutation":
        return ca.FeaturePermutation(forward_function)
    if alg == "IntegratedGradients":
        return ca.IntegratedGradients(forward_function)
    if alg == "KernelShap":
        return ca.KernelShap(forward_function)
    if alg == "LayerActivation":
        return ca.LayerActivation(forward_function, layer)
    if alg == "LayerIntegratedGradients":
        return ca.LayerIntegratedGradients(forward_function, layer)
    if alg == "Occlusion":
        return ca.Occlusion(forward_function)
    if alg == "ShapleyValueSampling":
        return ca.ShapleyValueSampling(forward_function)
    raise ValueError(f"Unknown algorithm: {alg}")


def get_algorithm_kwargs(
    alg: ca._utils.attribution.Attribution,
    attrib_params: AttributeParams,
    inputs: Tensor,
    lowers: tp.List[int] = None,
    uppers: tp.List[int] = None,
) -> tp.Dict[str, tp.Any]:
    # Collect valid keyword arguments for this particular algorithm's attribute method
    valid = set(signature(alg.attribute).parameters.keys())
    kwargs = {k: v for k, v in asdict(attrib_params).items() if k in valid and v is not None}

    # Get objects for the keyword arguments that cannot be contained in the AttributeParams
    if (
        "feature_mask" in valid
        and attrib_params.feature_mask_size is not None
        and attrib_params.feature_mask_mode is not None
    ):
        kwargs["feature_mask"] = FeatureMask(
            inputs,
            attrib_params.feature_mask_size,
            attrib_params.feature_mask_mode,
            lowers,
            uppers,
        )().to(cfg.device)

    if "sliding_window_shapes" in valid and attrib_params.sliding_window_shapes_size is not None:
        kwargs["sliding_window_shapes"] = (attrib_params.sliding_window_shapes_size,)

    return kwargs


def setup_output_dir(path: Path, clean: bool, resume: bool) -> tp.Set[str]:
    if path.exists() and not list(path.iterdir()):
        if clean:
            shutil.rmtree(path)
        elif not resume:
            raise FileExistsError("Use --resume or --clean to continue or remove a previous experiment.")
    else:
        path.mkdir(exist_ok=True, parents=True)

    return set(p.stem for p in path.iterdir())


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
    alg = get_explanation_algorithm(explain_params.alg, forward_function, layer)
    
    ben_oh = OutputHelper.from_params(explain_params, control_params, split="ben")
    mal_oh = OutputHelper.from_params(explain_params, control_params, split="mal")
    ben_done = setup_output_dir(ben_oh.output_path, CLEAN, RESUME)
    mal_done = setup_output_dir(mal_oh.output_path, CLEAN, RESUME)

    # Benign and malicious files to explain
    ben_files = [p for p in chain(cl.WINDOWS_TRAIN_PATH.iterdir(), cl.WINDOWS_TEST_PATH.iterdir()) if p.stem not in ben_done]
    mal_files = [p for p in chain(cl.SOREL_TRAIN_PATH.iterdir(), cl.SOREL_TEST_PATH.iterdir()) if p.stem not in mal_done]

    logging.log(logging.INFO,
        f"{len(ben_done)=} -- {len(ben_files)=} -- len(WINDOWS)={len(list(chain(cl.WINDOWS_TRAIN_PATH.iterdir(), cl.WINDOWS_TEST_PATH.iterdir())))}"
        f"\n{len(mal_done)=} -- {len(mal_files)=} -- len(SOREL)={len(list(chain(cl.SOREL_TRAIN_PATH.iterdir(), cl.SOREL_TEST_PATH.iterdir())))}"
    )

    # Lower and upper bounds for the text sections of all the files
    if explain_params.attrib_params.feature_mask_mode == "text":
        # TODO: use dict of tuples instead of dict of dict
        bounds = executable_helper.get_bounds(
            exe_params.text_section_bounds_file, dict_of_dict=True
        )
        bounds = executable_helper.filter_bounds(
            bounds, max_len=data_params.max_len, dict_of_dict=True
        )
        ben_files = [p for p in ben_files if p.as_posix() in bounds]
        mal_files = [p for p in mal_files if p.as_posix() in bounds]

    # Keep the benign and malicious data separate, so it can be placed in different directories
    ben_dataset, ben_loader = cl.get_dataset_and_loader(
        ben_files,
        None,
        max_len=data_params.max_len,
        batch_size=data_params.batch_size,
        num_workers=data_params.num_workers,
        shuffle_=False,
        sort_by_size=True,
    )
    mal_dataset, mal_loader = cl.get_dataset_and_loader(
        None,
        mal_files,
        max_len=data_params.max_len,
        batch_size=data_params.batch_size,
        num_workers=data_params.num_workers,
        shuffle_=False,
        sort_by_size=True,
    )

    # Malicious start idx
    if control_params.mal_start_idx is not None:
        control_params.mal_start_batch = control_params.mal_start_idx // data_params.batch_size
    # Benign start idx
    if control_params.ben_start_idx is not None:
        control_params.ben_start_batch = control_params.ben_start_idx // data_params.batch_size
    # Malicious end idx
    if control_params.mal_end_idx is not None:
        control_params.mal_end_batch = control_params.mal_end_idx // data_params.batch_size
    # Benign end idx
    if control_params.ben_end_idx is not None:
        control_params.ben_end_batch = control_params.ben_end_idx // data_params.batch_size

    # Conglomerate the different data structures
    data = [
        (
            mal_dataset,
            mal_loader,
            mal_oh,
            control_params.mal_start_batch,
            control_params.mal_end_batch,
        ),
        (
            ben_dataset,
            ben_loader,
            ben_oh,
            control_params.ben_start_batch,
            control_params.ben_end_batch,
        ),
    ]

    # Run the explanation algorithm on each dataset
    for dataset, loader, oh, start, end in data:
        files = batch([Path(e[0]) for e in dataset.all_files], data_params.batch_size)
        gen = zip(loader, files)
        initial = 0
        total = ceil_divide(len(dataset), data_params.batch_size)
        gen = tqdm(gen, total=total, initial=initial) if control_params.progress_bar else gen
        logging.log(logging.INFO, f"Starting explanations: {initial=}, {start=}, {end=}, {total=} @{datetime.now()}")
        for i, ((inputs, targets), files) in enumerate(gen):
            try:
                logging.log(logging.INFO, f"{i} / {total} = {100 * i // total}% @{datetime.now()}")
                if (start is not None and i < start) or (end is not None and i > end):
                    continue
                inputs = inputs.to(cfg.device)
                targets = targets.to(cfg.device)

                lowers, uppers = None, None
                if explain_params.attrib_params.feature_mask_mode == "text":
                    lowers = [bounds[f.as_posix()]["lower"] for f in files]
                    uppers = [bounds[f.as_posix()]["upper"] for f in files]

                kwargs = get_algorithm_kwargs(
                    alg,
                    explain_params.attrib_params,
                    inputs,
                    lowers,
                    uppers,
                )
                attribs = alg.attribute(inputs, **kwargs)
                for attr, f in zip(attribs, files):
                    torch.save(attr, oh.output_path / (f.name + ".pt"))

            except Exception as e:
                if control_params.errors == "ignore":
                    pass
                else:
                    ignore = {"ben_files", "mal_files", "bounds"}
                    locals_ = {k: v for k, v in locals().items() if k not in ignore}
                    logging.log(logging.ERROR, exception_info(e, locals_))
                if control_params.errors == "raise":
                    raise e


####################################################################################################


def analyze(
    exe_params: executable_helper.ExeParams,
    explain_params: ExplainParams,
    control_params: ControlParams,
) -> None:
    chunk_size = explain_params.attrib_params.feature_mask_size
    assert isinstance(chunk_size, int), "Fixed-size chunk attribution method required."

    # Set up the output structure
    ben_oh = OutputHelper.from_params(explain_params, control_params, split="ben")
    mal_oh = OutputHelper.from_params(explain_params, control_params, split="mal")
    ben_done = setup_output_dir(ben_oh.analysis_path, CLEAN, RESUME)
    mal_done = setup_output_dir(mal_oh.analysis_path, CLEAN, RESUME)
    
    bounds = executable_helper.get_bounds(exe_params.text_section_bounds_file)
    bounds = executable_helper.filter_bounds(bounds)
    include = set(Path(p).name for p in bounds.keys()).intersection(
        set(p.name.strip(".pt") for p in ben_oh.output_path.iterdir())
        | set(p.name.strip(".pt") for p in mal_oh.output_path.iterdir())
    )

    ben_files = [
        p
        for p in chain(cl.WINDOWS_TRAIN_PATH.iterdir(), cl.WINDOWS_TEST_PATH.iterdir())
        if (p.name in include and p.stem not in ben_done)
    ]
    mal_files = [
        p
        for p in chain(cl.SOREL_TRAIN_PATH.iterdir(), cl.SOREL_TEST_PATH.iterdir())
        if (p.name in include and p.stem not in mal_done)
    ]

    data = [
        (ben_files, ben_oh.output_path, ben_oh.summary_file),
        (mal_files, mal_oh.output_path, mal_oh.summary_file),
    ]

    for files, attribs_path, summary_file in data:
        with open(summary_file, "w") as handle:
            handle.write("file,offset,attribution\n")

        for f in tqdm(files):
            l_text, u_text = bounds[f.as_posix()]
            attribs_text = torch.load(attribs_path / (f.name + ".pt"), map_location=cfg.device)[
                l_text:u_text
            ]

            # TODO: determine why this happens...
            if attribs_text.shape[0] == 0:
                logging.log(logging.WARNING, f"WARNING: skipping empty text attributions: {f.as_posix()}")
                continue

            if (o := get_offset_chunk_tensor(attribs_text, chunk_size)) != 0:
                logging.log(logging.WARNING, f"WARNING: attributions have nonzero chunk offset {o=}. Skipping {f.as_posix()}")

            chunk_offsets = [o for o in range(0, len(attribs_text), chunk_size)]
            chunk_attribs = attribs_text[chunk_offsets]
            with open(summary_file, "a") as handle:
                handle.writelines(
                    [
                        f"{f.as_posix()},{l_text + i * chunk_size},{a.item()}\n"
                        for i, a in enumerate(chunk_attribs)
                    ]
                )


def parse_config(
    config: ConfigParser,
) -> tp.Tuple[
    cl.ModelParams, cl.DataParams, executable_helper.ExeParams, ExplainParams, ControlParams
]:
    p = config["MODEL"]
    model_params = cl.ModelParams(p.get("model_name"))
    p = config["DATA"]
    data_params = cl.DataParams(
        max_len=p.getint("max_len"),
        batch_size=p.getint("batch_size"),
        num_workers=p.getint("num_workers"),
    )
    p = config["EXE"]
    exe_params = executable_helper.ExeParams(p.get("text_section_bounds_file"))
    p = config[config.get("EXPLAIN", "alg")]
    attrib_params = AttributeParams(
        p.getint("baselines"),
        p.get("feature_mask_mode"),
        p.getint("feature_mask_size"),
        p.get("method"),
        p.getint("n_steps"),
        p.getint("perturbations_per_eval"),
        p.getint("sliding_window_shapes_size"),
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
        p.getint("ben_end_idx"),
        p.getint("mal_end_idx"),
        p.getint("ben_start_batch"),
        p.getint("mal_start_batch"),
        p.getint("ben_end_batch"),
        p.getint("mal_end_batch"),
        p.get("errors"),
        p.getboolean("progress_bar"),
        p.getboolean("verbose"),
    )
    return model_params, data_params, exe_params, explain_params, control_params


def main(config: ConfigParser, run_: bool, analyze_: bool) -> None:
    cfg.init(config["CONTROL"].get("device"), config["CONTROL"].getint("seed"))
    configurations = parse_config(config)
    model_params = configurations[0]
    data_params = configurations[1]
    exe_params = configurations[2]
    explain_params = configurations[3]
    control_params = configurations[4]
    if run_:
        run(model_params, data_params, exe_params, explain_params, control_params)
    if analyze_:
        analyze(exe_params, explain_params, control_params)


if __name__ == "__main__":
    parser = LoggingArgumentParser()
    parser.add_argument("--config_file", type=str, default="config_files/explain/default.ini")
    parser.add_argument("--run", action="store_true", default=False)
    parser.add_argument("--analyze", action="store_true", default=False)
    parser.add_argument("--clean", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()
    config = ConfigParser(allow_no_value=True)
    config.read(args.config_file)
    setup_logging(args.log_filename, args.log_filemode, args.log_format, args.log_level)
    logging.log(logging.INFO, section_header(f"START @{datetime.now()}"))
    logging.log(logging.INFO, f"{config=}")
    CLEAN: bool = args.clean
    RESUME: bool = args.resume
    main(config, args.run, args.analyze)
    logging.log(logging.INFO, section_header(f"END @{datetime.now()}"))
