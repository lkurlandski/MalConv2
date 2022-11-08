"""

"""

import multiprocessing as mp
from pathlib import Path
import typing as tp

import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from binaryLoader import BinaryDataset, RandomChunkSampler, pad_collate_func
from MalConv import MalConv
from MalConvGCT_nocat import MalConvGCT

from config import device
from typing_ import Pathlike


DATASET_PATH = Path("/home/lk3591/Documents/datasets")
WINDOWS_PATH = DATASET_PATH / "Windows/processed"
WINDOWS_TRAIN_PATH = WINDOWS_PATH / "train"
WINDOWS_TEST_PATH = WINDOWS_PATH / "test"
SOREL_PATH = DATASET_PATH / "Sorel/processed"
SOREL_TRAIN_PATH = SOREL_PATH / "train"
SOREL_TEST_PATH = SOREL_PATH / "test"
MODELS_PATH = Path("models")
MALCONV_GCT_PATH = MODELS_PATH / "malconvGCT_nocat.checkpoint"
MALCONV_PATH = MODELS_PATH / "malconv.checkpoint"
BATCH_SIZE = 8
MAX_LEN = 1000000  # 16000000 was used by the original authors
PAD_VALUE = 0
NUM_EMBEDDINGS = 257
CONFIDENCE_THRESHOLD = 0.5

MalConvLike = tp.Union[MalConv, MalConvGCT]
ModelName = tp.Literal["two", "gct"]


def forward_function_malconv(model, softmax: bool):
    if softmax:
        return lambda x: F.softmax(model(x)[0], dim=-1)
    else:
        return lambda x: model(x)[0]


def confidence_scores(model: tp.Union[MalConv, MalConvGCT], X: Tensor) -> Tensor:
    if X.dim() == 1:
        X = X.unsqueeze(0)
    return F.softmax(model(X)[0], dim=-1).data[:, 1].detach().cpu().numpy().ravel()


def get_model(model_name: ModelName, verbose: bool = False):
    torch.rand(4).to(device)  # This may help improve loading speed?
    if model_name == "gct":
        checkpoint_path = MALCONV_GCT_PATH
        model = MalConvGCT(channels=256, window_size=256, stride=64)
    elif model_name == "two":
        checkpoint_path = MALCONV_PATH
        model = MalConv()
    else:
        raise ValueError(f"Invalid model_name: {model_name}")
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    if verbose:
        print(f"{model=}")
    return model


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
        print(f"{len(train_dataset)=}")
        print(f"{train_dataset=}")
        print(f"{len(test_dataset)=}")
        print(f"{test_dataset=}")
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
    if verbose:
        print(f"{loader_threads=}")
        print(f"{train_loader=}")
        print(f"{test_loader=}")
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
) -> tp.Tuple[BinaryDataset, BinaryDataset, DataLoader, DataLoader]:
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


def evaluate_pretrained_malconv(
    model_name: ModelName, n_test: int = None, verbose: bool = False
) -> None:
    model = get_model(model_name, verbose=verbose)
    _, _, _, test_loader, _, _ = get_data(n_test=n_test, verbose=verbose)

    # Probability that the example is malicious and ground truths
    preds, truths = [], []
    eval_train_correct, eval_train_total = 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, penultimate_activ, conv_active = model(inputs)
            # The values of the maximally stimulated neuron and the indices of that neuron (0 or 1)
            values, predicted = torch.max(outputs.data, 1)
            # The probabilities that each example in the batch is malicious
            pred = F.softmax(outputs, dim=-1).data[:, 1].detach().cpu().numpy().ravel()
            # Ground truth labels (0 or 1)
            truth = labels.detach().cpu().numpy().ravel()
            # Update trackers
            preds.extend(pred)
            truths.extend(truth)
            eval_train_total += labels.size(0)
            eval_train_correct += (predicted == labels).sum().item()

    auc = roc_auc_score(truths, preds) if len(truths) - sum(truths) != 0 else np.NaN
    n_pos = sum(1 for i in truths if i == 1)
    n_neg = sum(1 for i in truths if i == 0)
    print(f"Observed Distribution: {dict(neg=n_neg, pos=n_pos)}")
    print(f"AUROC: {auc}")
    report = classification_report(truths, [int(round(i)) for i in preds])
    print(report)
