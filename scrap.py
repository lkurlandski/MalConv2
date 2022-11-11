"""

"""

from itertools import chain
from pathlib import Path
import multiprocessing

from classifier import SOREL_TRAIN_PATH, SOREL_TEST_PATH, WINDOWS_TRAIN_PATH, WINDOWS_TEST_PATH
from utils import section_header


def run_per_sample(i, j):
    print(f"run_per_sample: {i=} {j=}")


def run(n_workers=None):
    print("Run")
    iterable = [(i, i % 2) for i in range(10)]
    print(f"run: {iterable=}")
    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.starmap(run_per_sample, iterable)


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
    move_output_files_around()
