"""

"""

from dataclasses import dataclass
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import typing as tp

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from binaryLoader import BinaryDataset, RandomChunkSampler, pad_collate_func
from MalConv import MalConv
from MalConvGCT_nocat import MalConvGCT

import cfg
from typing_ import Pathlike
from utils import batch, sorted_dict


DATASET_PATH = Path("/home/lk3591/Documents/datasets")
WINDOWS_PATH = DATASET_PATH / "Windows/processed"
WINDOWS_TRAIN_PATH = WINDOWS_PATH / "train"
WINDOWS_TEST_PATH = WINDOWS_PATH / "test"
SOREL_PATH = DATASET_PATH / "Sorel/processed"
SOREL_TRAIN_PATH = SOREL_PATH / "train"
SOREL_TEST_PATH = SOREL_PATH / "test"
MODELS_PATH = Path("./outputs/model")
MALCONV_GCT_PATH = MODELS_PATH / "gct" / "malconvGCT_nocat.checkpoint"
MALCONV_PATH = MODELS_PATH / "two" / "malconv.checkpoint"
BATCH_SIZE = 8
MAX_LEN = 16000000
PAD_VALUE = 0
NUM_EMBEDDINGS = 257
CONFIDENCE_THRESHOLD = 0.5

MalConvLike = tp.Union[MalConv, MalConvGCT]
ModelName = tp.Literal["two", "gct"]


@dataclass
class ModelParams:
    name: tp.Literal["two", "gct"]


@dataclass
class DataParams:
    max_len: int
    batch_size: int
    num_workers: tp.Optional[int] = None  # Use 0 when debugging
    good: tp.Optional[tp.Union[Pathlike, tp.Iterable[Pathlike]]] = None
    bad: tp.Optional[tp.Union[Pathlike, tp.Iterable[Pathlike]]] = None


class ForwardFunctionMalConv(torch.nn.Module):
    def __init__(self, model: MalConvLike, softmax: bool):
        super().__init__()
        self.model = model
        self.softmax = softmax

    def forward(self, X: Tensor):
        X = self.model(X)[0]
        if self.softmax:
            return F.softmax(X, dim=-1)
        return X


def forward_function_malconv(model: MalConvLike, softmax: bool) -> tp.Callable[[Tensor], Tensor]:
    if softmax:
        return lambda x: F.softmax(model(x)[0], dim=-1)
    else:
        return lambda x: model(x)[0]


def confidence_scores(model: tp.Union[MalConv, MalConvGCT], X: Tensor) -> np.ndarray:
    if X.dim() == 1:
        X = X.unsqueeze(0)
    with torch.no_grad():
        return F.softmax(model(X)[0], dim=-1).data[:, 1].detach().cpu().numpy().ravel()


def get_model(model_name: ModelName, verbose: bool = False) -> MalConvLike:
    torch.rand(4).to(cfg.device)  # This may help improve loading speed?
    if model_name == "gct":
        checkpoint_path = MALCONV_GCT_PATH
        model = MalConvGCT(channels=256, window_size=256, stride=64)
    elif model_name == "two":
        checkpoint_path = MALCONV_PATH
        model = MalConv()
    else:
        raise ValueError(f"Invalid model_name: {model_name}")
    state = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.to(cfg.device)
    model.eval()
    logging.log(logging.INFO, f"{model=}")
    return model


def get_loader_and_files(
    good: tp.Optional[tp.Union[Pathlike, tp.Iterable[Pathlike]]],
    bad: tp.Optional[tp.Union[Pathlike, tp.Iterable[Pathlike]]],
    max_len: int = MAX_LEN,
    batch_size: int = BATCH_SIZE,
    group_by_size: bool = True,
    largest_first: bool = False,
) -> tp.Tuple[BinaryDataset, DataLoader, tp.List[Path]]:
    """
    Return a data loader and the batched files that correspond to it.

    Args:
        group_by_size: if True, will group into batches of similar sized files
        largest_first: if True, the largest batch of files will be yielded first
            note, only the first largest batch will be yielded first,
            not every subsequent batch. After the largest batch is yielded first,
            subsequent batches will be yielded randomly.
    """
    dataset = BinaryDataset(good, bad, group_by_size, max_len)
    sampler = RandomChunkSampler(dataset, batch_size, False, largest_first)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=pad_collate_func,
        sampler=sampler,
        num_workers=len(os.sched_getaffinity(0)),
    )
    files = list(batch([Path(dataset.all_files[i][0]) for i in iter(sampler)], batch_size))
    return loader, files


def _get_datasets(
    train_good_dir: Pathlike,
    train_bad_dir: Pathlike,
    test_good_dir: Pathlike,
    test_bad_dir: Pathlike,
    max_len: int,
    n_train: int = None,
    n_test: int = None,
    verbose: bool = False,
) -> tp.Tuple[BinaryDataset, BinaryDataset]:
    train_dataset = BinaryDataset(
        train_good_dir, train_bad_dir, max_len=max_len, sort_by_size=False, shuffle=True
    )
    test_dataset = BinaryDataset(
        test_good_dir, test_bad_dir, max_len=max_len, sort_by_size=False, shuffle=True
    )
    # TODO: these features may be broken...
    if n_train is not None:
        idx = np.random.choice(len(train_dataset), n_train, replace=False)
        train_dataset = Subset(train_dataset, idx)
    if n_test is not None:
        idx = np.random.choice(len(test_dataset), n_test, replace=False)
        test_dataset = Subset(test_dataset, idx)
    if verbose:
        logging.log(logging.INFO,
            f"{len(train_dataset)=} -- train_dataset={train_dataset}"
            f"{len(test_dataset)=} -- test_dataset={test_dataset}"
        )
    return train_dataset, test_dataset


def _get_loaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    train_sampler: RandomChunkSampler = None,
    test_sampler: RandomChunkSampler = None,
    verbose: bool = False,
) -> tp.Tuple[DataLoader, DataLoader, RandomChunkSampler, RandomChunkSampler]:
    loader_threads = max(mp.cpu_count() - 4, mp.cpu_count() // 2 + 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=loader_threads,
        collate_fn=pad_collate_func,
        sampler=train_sampler,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=loader_threads,
        collate_fn=pad_collate_func,
        sampler=test_sampler,
    )
    logging.log(logging.INFO,
        f"{loader_threads=} -- train_loader={train_loader} -- test_loader={test_loader}"
    )
    return train_loader, test_loader


def get_data(
    train_good_dir: Pathlike = WINDOWS_TRAIN_PATH,
    train_bad_dir: Pathlike = SOREL_TRAIN_PATH,
    test_good_dir: Pathlike = WINDOWS_TEST_PATH,
    test_bad_dir: Pathlike = SOREL_TEST_PATH,
    max_len: int = MAX_LEN,
    batch_size: int = BATCH_SIZE,
    n_train: int = None,
    n_test: int = None,
    verbose: bool = False,
) -> tp.Tuple[
    BinaryDataset,
    BinaryDataset,
    DataLoader,
    DataLoader,
    tp.Optional[RandomChunkSampler],
    tp.Optional[RandomChunkSampler],
]:
    train_dataset, test_dataset = _get_datasets(
        train_good_dir,
        train_bad_dir,
        test_good_dir,
        test_bad_dir,
        max_len,
        n_train,
        n_test,
        verbose=verbose,
    )
    # If samplers are provided, the input order of the data is not consistent
    train_sampler = RandomChunkSampler(train_dataset, batch_size) if False else None
    test_sampler = RandomChunkSampler(test_dataset, batch_size) if False else None
    train_loader, test_loader = _get_loaders(
        train_dataset,
        test_dataset,
        batch_size,
        train_sampler,
        test_sampler,
        verbose=verbose,
    )
    return (
        train_dataset,
        test_dataset,
        train_loader,
        test_loader,
        train_sampler,
        test_sampler,
    )


def _evaluate_pretrain_malconv(
    model: MalConvLike,
    loader: DataLoader,
    files: tp.Optional[tp.List[Path]] = None,
) -> tp.Tuple[tp.List[float], tp.List[float], int, int]:
    # Yield a list of files corresponding to the inputs, or yield list of None
    files = [None] * len(loader) if files is None else files
    batched_files = batch(files, loader.batch_size)

    # Trackers to record information about the model's performance
    confs, truths = [], []
    n_correct, n_total = 0, 0
    for (
        (inputs, labels),
        f,
    ) in tqdm(zip(loader, batched_files), total=len(loader)):
        inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
        # Get model outputs
        outputs, penultimate_activ, conv_active = model(inputs)
        # The values of the maximally stimulated neuron and the indices of that neuron (0 or 1)
        values, pred = torch.max(outputs.data, 1)
        # The probabilities that each example in the batch is malicious
        conf = F.softmax(outputs, dim=-1).data[:, 1].detach().cpu().numpy().ravel()
        # Ground truth labels (0 or 1)
        truth = labels.detach().cpu().numpy().ravel()
        # Update trackers
        confs.extend(conf)
        truths.extend(truth)
        n_total += labels.size(0)
        n_correct += (pred == labels).sum().item()
        # Do something with the file
        if f is not None:
            pass

    return confs, truths, n_correct, n_total


def evaluate_pretrained_malconv(
    model_name: ModelName,
    n_test: int = None,  # FIXME: no idea if this will work
    n_train: int = None,  # FIXME: no idea if this will work
    verbose: bool = False,
) -> tp.Tuple[
    tp.List[float],
    tp.List[float],
    tp.List[int],
    tp.List[int],
    tp.List[Path],
    tp.List[Path],
    tp.Dict[str, tp.Any],
    tp.Dict[str, tp.Any],
    tp.Dict[str, tp.Any],
]:
    target_names = ["benign", "malicious"]
    model = get_model(model_name, verbose=verbose)
    tr_dataset, ts_dataset, tr_loader, ts_loader, _, _ = get_data(
        n_test=n_test, n_train=n_train, verbose=verbose
    )

    if n_test != 0:
        ts_files = [Path(e[0]) for e in ts_dataset.all_files]
        ts_confs, ts_truths, ts_n_correct, ts_n_total = _evaluate_pretrain_malconv(
            model, ts_loader, ts_files
        )
        ts_report = classification_report(
            ts_truths, np.round(ts_confs), target_names=target_names, output_dict=True
        )
        ts_report["auroc"] = (
            roc_auc_score(ts_truths, ts_confs) if len(ts_truths) - sum(ts_truths) != 0 else np.NaN
        )
    else:
        ts_confs, ts_truths, ts_files, ts_report = None, None, None, None

    if n_train != 0:
        tr_files = [Path(e[0]) for e in tr_dataset.all_files]
        tr_confs, tr_truths, tr_n_correct, tr_n_total = _evaluate_pretrain_malconv(
            model, tr_loader, tr_files
        )
        tr_report = classification_report(
            tr_truths, np.round(tr_confs), target_names=target_names, output_dict=True
        )
        tr_report["auroc"] = (
            roc_auc_score(tr_truths, tr_confs) if len(tr_truths) - sum(tr_truths) != 0 else np.NaN
        )
    else:
        tr_confs, tr_truths, tr_files, tr_report = None, None, None, None

    if n_test != 0 and n_train != 0:
        cum_report = {}
        for (tr_k, tr_v), (ts_k, ts_v) in zip(sorted_dict(tr_report), sorted_dict(ts_report)):
            if isinstance(tr_v, dict) and isinstance(ts_v, dict):
                tr_s = tr_v["support"]
                ts_s = ts_v["support"]
                cum_report[tr_k] = {}
                for m in ["precision", "recall", "f1-score"]:
                    cum_report[tr_k][m] = (tr_v[m] * tr_s + ts_v[m] * ts_s) / (tr_s + ts_s)
            elif isinstance(tr_v, float) and isinstance(ts_v, float):
                tr_s = tr_report["macro avg"]["support"]
                ts_s = ts_report["macro avg"]["support"]
                cum_report[tr_k] = (tr_v * tr_s + ts_v * ts_s) / (tr_s + ts_s)
    else:
        cum_report = None

    return (
        tr_confs,
        ts_confs,
        tr_truths,
        ts_truths,
        tr_files,
        ts_files,
        tr_report,
        ts_report,
        cum_report,
    )


def evaluate_pretrained_malconv_save_results() -> None:
    model_name = "gct"
    output = Path("outputs/model") / model_name
    output.mkdir(parents=True, exist_ok=True)
    (
        tr_confs,
        ts_confs,
        tr_truths,
        ts_truths,
        tr_files,
        ts_files,
        tr_report,
        ts_report,
        cum_report,
    ) = evaluate_pretrained_malconv(model_name)

    if all(tr_confs, tr_truths, tr_files, tr_report):
        with open(output / "tr_report.json", "w") as f:
            json.dump(tr_report, f, indent=4, sort_keys=True)
        pd.DataFrame(
            {
                "tr_files": [p.as_posix() for p in tr_files],
                "tr_confs": tr_confs,
                "tr_truths": tr_truths,
            }
        ).to_csv(output / "tr_results.csv", index=False)

    if all(ts_confs, ts_truths, ts_files, ts_report):
        with open(output / "ts_report.json", "w") as f:
            json.dump(ts_report, f, indent=4, sort_keys=True)
        pd.DataFrame(
            {
                "ts_files": [p.as_posix() for p in ts_files],
                "ts_confs": ts_confs,
                "ts_truths": ts_truths,
            }
        ).to_csv(output / "ts_results.csv", index=False)

    if cum_report is not None:
        with open(output / "cum_report.json", "w") as f:
            json.dump(cum_report, f, indent=4, sort_keys=True)
        pd.DataFrame(
            {
                "tr_files": [p.as_posix() for p in tr_files],
                "tr_confs": tr_confs,
                "tr_truths": tr_truths,
                "ts_files": [p.as_posix() for p in ts_files],
                "ts_confs": ts_confs,
                "ts_truths": ts_truths,
            }
        ).to_csv(output / "cum_results.csv", index=False)


if __name__ == "__main__":
    cfg.init("cuda:0", 0)
    evaluate_pretrained_malconv_save_results()
