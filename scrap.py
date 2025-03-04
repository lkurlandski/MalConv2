"""

"""

from itertools import chain
from pathlib import Path
from pprint import pformat, pprint
import multiprocessing
import os
import time
import typing as tp

from captum.attr import GuidedGradCam, LayerGradCam
import torch
from torch import Tensor
from tqdm import tqdm

import classifier as cl
import cfg
from utils import section_header


def grad_cam_attribution(layer:bool = False):
    cfg.init("cpu", 0)
    model = cl.get_model("gct")
    if layer:
        gc = LayerGradCam(cl.ForwardFunctionMalConv(model, False), model.convs_share[-1])
    else:
        gc = GuidedGradCam(cl.ForwardFunctionMalConv(model, False), model.fc_2)
    dataset, dataloader = cl.get_dataset_and_loader(good=None, bad=cl.WINDOWS_TRAIN_PATH.iterdir())

    for X, _ in dataloader:
        X: Tensor = X.to(cfg.device)
        X: Tensor = X.to(torch.float32)
        X: Tensor = X.requires_grad_()
        attribs = gc.attribute(X, target=1)
        print(f"{attribs.shape=}\n{attribs}")
        break


def run_sample(i: int, j: int = None, verbose: bool = False):
    if verbose:
        print(f"run_sample: pid={os.getpid()} {i=} {j=}")


def run_samples(i: tp.Iterable[int], j: tp.Iterable[int] = None, verbose: bool = False):
    if verbose:
        print(f"run_samples: pid={os.getpid()} {i=} {j=}")
    for i_, j_ in zip(i, j):
        run_sample(i_, j_)


def run():
    print("run")
    n_workers = 8
    k = 100000
    i = list(range(k))
    j = list(range(k, 2 * k))

    # print("map")
    # iterable = i
    # print(f"iterable=\n{pformat(iterable)}")
    # with multiprocessing.Pool(processes=n_workers) as pool:
    #     pool.map(run_sample, iterable)
    # print("-" * 88)
    # iterable = (i_ for i_ in i)
    # print(f"iterable=\n{pformat(iterable)}")
    # with multiprocessing.Pool(processes=n_workers) as pool:
    #     pool.map(run_sample, iterable)
    # print("-" * 88)
    #
    # print("imap")
    #

    # The importance of chunksize
    # print("starmap")
    iterable = list(zip(i, j))
    for chunksize in (None, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        multi = []
        for iteration in range(5):
            s = time.time()
            with multiprocessing.Pool(processes=n_workers) as pool:
                pool.starmap(run_sample, iterable, chunksize)
            multi.append(time.time() - s)
        print(f"starmap {chunksize=}: {sum(multi) / len(multi)}")
    print("-" * 88)


    # iterable = [
    #     (i[:len(i) // 2], j[:len(i) // 2]),
    #     (i[len(i) // 2:], j[len(i) // 2:]),
    # ]
    # print(f"iterable=\n{pformat(iterable)}")
    # with multiprocessing.Pool(processes=n_workers) as pool:
    #     pool.starmap(run_samples, iterable)
    # print("-" * 88)

    iterable = zip(i, j)
    for chunksize in (None, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        multi = []
        for iteration in range(5):
            s = time.time()
            with multiprocessing.Pool(processes=n_workers) as pool:
                pool.starmap(run_sample, iterable, chunksize)
            multi.append(time.time() - s)
        print(f"starmap {chunksize=}: {sum(multi) / len(multi)}")
    print("-" * 88)


def move_output_files_around():
    benign = {p.stem for p in chain(SOREL_TRAIN_PATH.iterdir(), SOREL_TEST_PATH.iterdir())}
    mal = {p.stem for p in chain(WINDOWS_TRAIN_PATH.iterdir(), WINDOWS_TEST_PATH.iterdir())}
    attr_paths = (
        Path("outputs/gct/1000000/KernelShap/False/256/50/1/attributions/"),
        Path("outputs/two/1000000/KernelShap/False/256/50/1/attributions/"),
    )

    for attr_path in attr_paths:
        print(section_header(attr_path.as_posix()))
        benign_path = attr_path / "benign"
        mal_path = attr_path / "malicious"
        benign_path.mkdir()
        mal_path.mkdir()
        for f in attr_path.iterdir():
            print(f"\t{f.name}")
            if f.stem in benign and f.stem in mal:
                raise ValueError(f"File {f} is in both benign and malicious")
            if f.stem in benign:
                f.rename(benign_path / (f.stem + ".exe.pt"))
            if f.stem in mal:
                f.rename(mal_path / f.name)


def different_model_outputs():
    print("-" * 88)

    model = get_model("gct")

    X_sus = torch.load("most_suspicious.pt")
    X_rnd = torch.load("random.pt")

    print(f"{X_sus.shape=}")
    print(f"{X_rnd.shape=}")
    print(f"{torch.equal(X_sus, X_rnd)=}")

    c_sus = confidence_scores(model, X_sus).tolist()
    c_rnd = confidence_scores(model, X_rnd).tolist()

    print(f"{c_sus=}")
    print(f"{c_rnd=}")

    print("-" * 88)

    X_sus_0 = X_sus[0].unsqueeze(0)
    X_sus_1 = X_sus[1].unsqueeze(0)
    X_rnd_0 = X_rnd[0].unsqueeze(0)
    X_rnd_1 = X_rnd[1].unsqueeze(0)

    print(f"{torch.equal(X_sus_0, X_rnd_0)=}")
    print(f"{torch.equal(X_sus_0, X_rnd_1)=}")
    print(f"{torch.equal(X_sus_1, X_rnd_0)=}")
    print(f"{torch.equal(X_sus_1, X_rnd_1)=}")

    print()

    c_sus_0 = confidence_scores(model, X_sus_0).tolist()
    c_sus_1 = confidence_scores(model, X_sus_1).tolist()
    c_rnd_0 = confidence_scores(model, X_rnd_0).tolist()
    c_rnd_1 = confidence_scores(model, X_rnd_1).tolist()

    print(f"{c_sus_0=}")
    print(f"{c_sus_1=}")
    print(f"{c_rnd_0=}")
    print(f"{c_rnd_1=}")

    print("-" * 88)

    print(f"{torch.equal(X_sus[0], X_sus_0)=} {(c_sus[0]==c_sus_0[0])=}")
    print(f"{torch.equal(X_sus[1], X_sus_1)=} {(c_sus[1]==c_sus_1[0])=}")
    print(f"{torch.equal(X_rnd[0], X_rnd_0)=} {(c_rnd[0]==c_rnd_0[0])=}")
    print(f"{torch.equal(X_rnd[1], X_rnd_1)=} {(c_rnd[1]==c_rnd_1[0])=}")

    print("-" * 88)

    print(f"{torch.equal(X_sus[1].unsqueeze(0), X_rnd[1].unsqueeze(0))=}")
    c_sus_1 = confidence_scores(model, X_sus[1].unsqueeze(0)).tolist()
    c_rnd_1 = confidence_scores(model, X_rnd[1].unsqueeze(0)).tolist()
    print(f"{c_sus_1=}")
    print(f"{c_rnd_1=}")

    print("-" * 88)
    print("PROBLEM: c_sus[1] != c_sus_1[0]")

    os.environ["BATCH_OR_SINGLE"] = "batch"
    c_sus = confidence_scores(model, X_sus).tolist()

    os.environ["BATCH_OR_SINGLE"] = "single"
    c_sus_1 = confidence_scores(model, X_sus[1].unsqueeze(0)).tolist()
    print(f"{c_sus=}")
    print(f"{c_sus_1=}")
    second_tensor_as_batch = c_sus[1]
    second_tensor_as_single = c_sus_1[0]
    print(f"{second_tensor_as_batch=}")
    print(f"{second_tensor_as_single=}")
    print(f"{second_tensor_as_batch==second_tensor_as_single=}")


def move_files():
    paths = [
        Path(
            "/home/lk3591/Documents/code/MalConv2/outputs/explain/softmax__False/layer__None/alg__FeaturePermutation/baselines__0/feature_mask_mode__all/feature_mask_size__256/method__None/n_steps__None/perturbations_per_eval__None/sliding_window_shapes_size__None/strides__None/target__1"),
        Path(
            "/home/lk3591/Documents/code/MalConv2/outputs/explain/softmax__False/layer__None/alg__KernelShap/baselines__0/feature_mask_mode__all/feature_mask_size__256/method_None/n_steps__None/perturbations_per_eval__None/sliding_window_shapes_size__None/strides__None/target__1"),
        Path(
            "/home/lk3591/Documents/code/MalConv2/outputs/explain/softmax__False/layer__None/alg__KernelShap/baselines__0/feature_mask_mode__all/feature_mask_size__512/method_None/n_steps__None/perturbations_per_eval__None/sliding_window_shapes_size__None/strides__None/target__1"),
        Path(
            "/home/lk3591/Documents/code/MalConv2/outputs/explain/softmax__False/layer__None/alg__KernelShap/baselines__0/feature_mask_mode__text/feature_mask_size__64/method__None/n_steps__None/perturbations_per_eval__None/sliding_window_shapes_size__None/strides__None/target__1"),
    ]

    for path in paths:
        for m in ["mal", "ben"]:
            p = path / m
            p_ = p / "output"
            p_.mkdir(exist_ok=True)
            for f in p.iterdir():
                if f.suffix != ".pt":
                    continue
                f.rename(p_ / f.name)


if __name__ == "__main__":
    grad_cam_attribution()

