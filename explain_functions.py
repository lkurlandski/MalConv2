"""
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
MAX_LEN = 10 ** 6


def function_feature_mask(X: Tensor, bounds: list[list[tuple[int, int]]]) -> Tensor:
    masks = []
    for b, x in zip(bounds, X):
        mask = torch.zeros(x.shape)
        for fill_val, (lower, upper) in enumerate(b, 1):
            mask[lower:upper] = fill_val            
        masks.append(mask)
    mask = torch.stack(masks)
    return mask.to(torch.int64)


def run(
    outpath: Path,
    files: list[Path],
    boundaries: dict[str, list[tuple[int, int]]],
    alg: str,
    batch_size: int = BATCH_SIZE,
    max_len: int = MAX_LEN,
) -> None:

    model = cl.get_model("gct")
    forward_function = cl.forward_function_malconv(model, False)
    if alg == "KernelShap":
        alg = KernelShap(forward_function)
    elif alg == "FeatureAblation":
        alg = FeatureAblation(forward_function)

    loader, batched_files = cl.get_loader_and_files(
        None,
        files,
        max_len=max_len,
        batch_size=batch_size,
    )

    pbar = tqdm(total=len(batched_files))
    for (X, y), file_batch in zip(loader, batched_files):
        pbar.write(f"{round(sum(f.stat().st_size / 1024 ** 2 for f in file_batch))} MB disk.")
        pbar.update(1)
        X = X.to(cfg.device)
        y = y.to(cfg.device)
        b = [boundaries[f.name] for f in file_batch]
        m = function_feature_mask(X, b) 
        attribs = alg.attribute(X, baselines=BASELINE, target=TARGET, feature_mask=m)
        attribs = attribs.to(torch.float16)
        for attr, f in zip(attribs, file_batch):
            torch.save(attr, outpath / (f.name + ".pt"))


def main(
    outpath: Path,
    files: Path | list[Path],
    boundaries: Path | dict[dict[int, int]],
    alg: str,
    batch_size: int = BATCH_SIZE,
    max_len: int = MAX_LEN,
) -> None:
    outpath.mkdir(parents=True, exist_ok=True)
    
    if isinstance(files, Path):
        files = list(files.iterdir())

    if isinstance(boundaries, Path):
        with open(boundaries) as fp:
            boundaries = json.load(fp)

    files = [f for f in files if f.name in boundaries]

    run(outpath, files, boundaries, alg, batch_size, max_len)


def debug():
    main(
        Path("./tmp"),
        Path("../code/AssemblyStyleTransfer/data/filter/"),
        Path("../code/AssemblyStyleTransfer/output/boundaries.json"),
        "FeatureAblation",
        16,
        10 ** 6,
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
    debug()
    cli()
