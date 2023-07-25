"""
It appears that the funciton boundary identification is completely wrong.

Note: this code memory leaks if run on a CPU.
No idea why, but make sure its run on GPU.
"""

from __future__ import annotations
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

from captum.attr import FeatureAblation, KernelShap
from tqdm import tqdm
import torch
from torch import Tensor

import classifier as cl
import cfg


BASELINE = 0
TARGET = 1
BATCH_SIZE = 1
MAX_LEN = 10**6


def lower_and_upper_bounds(boundary: list[tuple[int, int]]) -> tuple[int, int]:
    return boundary[0][0], boundary[-1][1]


def function_feature_mask(X: Tensor, bounds: list[list[tuple[int, int]]]) -> Tensor:
    masks = []
    for b, x in zip(bounds, X):
        mask = torch.zeros(x.shape)
        for fill_val, (lower, upper) in enumerate(b, 1):
            mask[lower:upper] = fill_val
        masks.append(mask)
    mask = torch.stack(masks)
    return mask.to(torch.int64)


class FunctionLevelExplainer:
    """
    Explains binaries at the function level.

    Output:
    |-- outpath
        |-- attribs
            |-- file1.pt
            |-- file2.pt
            ...
            |-- fileN.pt
        |-- regions.json
        |-- summary.json

    Note that the attributions are not for the entire binary,
        only for the specific region of interest. Therefore, the
        regions.json file is needed to map the attributions to
        the true locations in the binaries.
    """

    def __init__(
        self,
        outpath: Path,
        model: torch.nn.Module,
        files: list[Path],
        boundaries: dict[str, list[tuple[int, int]]],
        alg: KernelShap | FeatureAblation,
        batch_size: int = BATCH_SIZE,
        max_len: int = MAX_LEN,
    ) -> None:

        self.outpath = outpath
        self.attribs_path = outpath / "attribs"
        self.attribs_path.mkdir(exist_ok=True, parents=True)
        self.summary_path = outpath / "summary.json"
        self.regions_path = outpath / "regions.json"

        self.model = model
        boundaries = {f: l for f, l in boundaries.items() if l != []}
        self.files = [f for f in files if f.name in boundaries]
        names = set(f.name for f in self.files)
        self.boundaries = {f: l for f, l in boundaries.items() if f in names}

        self.alg = alg
        self.batch_size = batch_size
        self.max_len = max_len

    def get_regions(self) -> dict[str, tuple[int, int]]:
        regions = {}
        for f in self.files:
            regions[f.name] = lower_and_upper_bounds(self.boundaries[f.name])
        return regions

    def explain(self) -> None:
        print("Regioning...", flush=True)
        with open(self.regions_path, "w") as fp:
            json.dump(self.get_regions(), fp, indent=4)

        loader, batched_files = cl.get_loader_and_files(
            good=None,
            bad=self.files,
            max_len=self.max_len,
            batch_size=self.batch_size,
            group_by_size=True,
            largest_first=True,
        )

        print("Explaining...", flush=True)
        gen = tqdm(zip(loader, batched_files), total=len(batched_files))
        for (X, _), files in gen:
            boundaries = [self.boundaries[f.name] for f in files]
            lower_uppers = [lower_and_upper_bounds(b) for b in boundaries]
            mask = function_feature_mask(X, boundaries)
            attribs = self.alg.attribute(
                X.to(cfg.device),
                baselines=BASELINE,
                target=TARGET,
                feature_mask=mask.to(cfg.device),
            )
            for a, f, (l, u) in zip(attribs, files, lower_uppers):
                a = a[l:u].to(torch.float16).to("cpu")
                torch.save(a, self.attribs_path / (f.name + ".pt"))

    def analyze(self) -> None:
        lower_upper_attr = {}
        for f in list(self.attribs_path.iterdir()):
            attrib = torch.load(f)  # attrib is not for entire binary
            boundary = self.boundaries[f.name.strip(".pt")]  # true function offsets
            lower, upper = lower_and_upper_bounds(boundary)  # maps true to attrib
            l_u_a = []
            for l, u in boundary:
                a = attrib[lower - l : upper - u].sum()
                l_u_a.append((l, u, a.item()))
            lower_upper_attr[f.name] = l_u_a
        with open(self.summary_path, "w") as fp:
            json.dump(lower_upper_attr, fp, indent=4)


def main(
    outpath: Path,
    files: Path | list[Path],
    boundaries: Path | dict[dict[int, int]],
    batch_size: int = BATCH_SIZE,
    max_len: int = MAX_LEN,
) -> None:

    if isinstance(files, Path):
        files = list(files.iterdir())
    if isinstance(boundaries, Path):  # Assumes a FunctionBoundaryMapper...
        with open(boundaries) as fp:
            boundaries = json.load(fp)
        boundaries = {k: list(v.values()) for k, v in boundaries.items()}

    model = cl.get_model("gct")
    forward_function = cl.forward_function_malconv(model, False)
    alg = FeatureAblation(forward_function)

    explainer = FunctionLevelExplainer(
        outpath,
        model,
        files,
        boundaries,
        alg,
        batch_size,
        max_len,
    )

    explainer.explain()
    explainer.analyze()


def debug():
    main(
        Path("./tmp"),
        list(Path("../code/AssemblyStyleTransfer/data/filter/").iterdir())[0:64],
        Path("../code/AssemblyStyleTransfer/output/boundaries.json"),
        64,
        10**6,
    )
    sys.exit(0)


def cli():
    parser = ArgumentParser()
    parser.add_argument("--outpath", type=Path)
    parser.add_argument("--path_to_files", type=Path)
    parser.add_argument("--path_to_boundaries", type=Path)
    parser.add_argument("--alg", type=str, choices=["KernelShap", "FeatureAblation"])
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_len", type=int, default=MAX_LEN)
    args = parser.parse_args()

    main(
        args.outpath,
        args.path_to_files,
        args.path_to_boundaries,
        args.alg,
        args.batch_size,
        args.max_len,
    )


if __name__ == "__main__":
    cfg.init("cuda:0")
    debug()
    cli()
