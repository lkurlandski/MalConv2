from argparse import ArgumentParser
import gzip
import json
from pathlib import Path
import shutil
import typing as tp

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm

from binaryLoader import pad_collate_func, BinaryDataset
from MalConvGCT_nocat import MalConvGCT

import cfg


def read_binary(
    file: Path,
    mode: str = "rb",
    max_len: int = None,
    l: int = None,
    u: int = None,
) -> np.ndarray:
    """
    Read a binary file into a numpy array with MalConv2 technique.
    """
    def read_handle(handle):
        n_bytes = -1 if max_len is None else max_len
        if l is not None:
            handle.seek(l)
            n_bytes = u - l if max_len is None else min(max_len, u - l)
        return handle.read(n_bytes)

    try:
        with gzip.open(file, mode) as handle:
            x = read_handle(handle)
    except OSError:
        with open(file, mode) as handle:
            x = read_handle(handle)

    x = np.frombuffer(x, dtype=np.uint8).astype(np.int16) + 1
    return x


def get_model(checkpoint_path: Path) -> MalConvGCT:
    """
    Load a pretrained MalConvGCT model.
    """
    model = MalConvGCT(channels=256, window_size=256, stride=64)
    state = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.to(cfg.device)
    return model


def get_dataset_and_loader(
    good: tp.Union[Path, tp.Iterable[Path]] = None,
    bad: tp.Union[Path, tp.Iterable[Path]] = None,
    max_len: int = 4000000,
    batch_size: int = 1,
    shuffle_: bool = False,
    sort_by_size: bool = True,
) -> tp.Tuple[BinaryDataset, DataLoader]:
    """
    Return a Dataset and a DataLoader for MalConv.
    """
    if good is None and bad is None:
        raise ValueError("No executables specified.")
    if shuffle_ and sort_by_size:
        raise ValueError("Specifying both shuffle_ and sort_by_size does not make sense.")
    dataset = BinaryDataset(good, bad, sort_by_size, max_len, shuffle_)
    loader = DataLoader(dataset, batch_size, collate_fn=pad_collate_func)
    return dataset, loader


def main():
    OUTPUT_PATH.mkdir()
    model = get_model(CHECKPOINT_PATH)
    dataset, loader = get_dataset_and_loader(GOODWARE, MALWARE, MAX_LEN, BATCH_SIZE, False, True)

    # Trackers to record information about the model's performance
    confs, truths = [], []
    for X, y in tqdm(loader):
        X, y = X.to(cfg.device), y.to(cfg.device)
        # Get model outputs
        outputs, _, _ = model(X)
        # The probabilities that each example in the batch is malicious
        conf = softmax(outputs, dim=-1).data[:, 1].detach().cpu().numpy().ravel()
        # Ground truth labels (0 or 1)
        truth = y.detach().cpu().numpy().ravel()
        # Update trackers
        confs.extend(conf.tolist())
        truths.extend(truth.tolist())

    decision = [int(i) for i in np.round(confs)]
    report = classification_report(
        truths, decision, zero_division=1, output_dict=True
    )
    with open(OUTPUT_PATH / "report.json", "w") as handle:
        json.dump(report, handle, indent=4) 

    results = pd.DataFrame({
        "file": [f for f, _, _ in dataset.all_files],
        "labels": truths,
        "confidence": confs,
        "decision": decision,
    })
    results.to_csv(OUTPUT_PATH / "results.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--output_path",
        help="Directory location to write results to."
    )
    parser.add_argument(
        "--checkpoint_path",
        default="./malconvGCT_nocat.checkpoint", 
        help="Location of the pretrained MalConvGCT model."
    )
    parser.add_argument(
        "--goodware",
        default=None,
        help="Directory containing benign executables."
    )
    parser.add_argument(
        "--malware",
        default=None,
        help="Directory containing malicious executables."
    )
    parser.add_argument(
        "--max_len",
        default=16000000,
        help="Number of bytes from the malware to use."
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        help="Number of examples per batch."
    )
    parser.add_argument(
        "--seed",
        default=0,
        help="Controls random number generation."
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="`cuda` for GPU else `cpu` for CPU."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output directory prior to running."
    )
    args = parser.parse_args()

    OUTPUT_PATH = Path(args.output_path)
    CHECKPOINT_PATH = Path(args.checkpoint_path)
    GOODWARE = Path(args.goodware) if args.goodware is not None else None
    MALWARE = Path(args.malware) if args.malware is not None else None
    MAX_LEN = int(args.max_len)
    BATCH_SIZE = int(args.batch_size)
    CLEAN = args.clean

    if GOODWARE is None and MALWARE is None:
        raise ValueError(
            "No executables were specified. "
            "Specify a path to executables for analysis using --malware or --goodware args."
        )

    if CLEAN and OUTPUT_PATH.exists():
        shutil.rmtree(OUTPUT_PATH)
    elif OUTPUT_PATH.exists():
        raise ValueError(
            "Output directory exists. "
            "Use the --clean flag or specify a different --output_path arg."
    )

    cfg.init(args.device, args.seed)
    main()
