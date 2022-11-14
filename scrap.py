"""

"""

from itertools import chain
from pathlib import Path
from pprint import pformat, pprint
import multiprocessing
import os
import time
import typing as tp

from tqdm import tqdm

from classifier import SOREL_TRAIN_PATH, SOREL_TEST_PATH, WINDOWS_TRAIN_PATH, WINDOWS_TEST_PATH
from utils import section_header


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


if __name__ == "__main__":
    run()
