# 1:
# Fixes RuntimeError: cuDNN error: CUDNN_STATUS_MAPPING_ERROR?
# Source: https://discuss.pytorch.org/t/when-should-we-set-torch-backends
# -cudnn-enabled-to-false-especially-for-lstm/106571
import shutil
from collections import OrderedDict
from copy import deepcopy
import gzip
import multiprocessing as mp
from pathlib import Path
import pickle
from pprint import pformat, pprint
import random
import sys
import tempfile
import traceback
import typing as tp

from captum import attr as capattr
import lief
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report
import sys
from tqdm import tqdm
import torch
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from binaryLoader import BinaryDataset, RandomChunkSampler, pad_collate_func
from MalConv import MalConv
from MalConvGCT_nocat import MalConvGCT

import captum_explainability as ce
from utils import batch, section_header


# Globals
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

# Hyperparams
BATCH_SIZE = 8
MAX_LEN = int(16000000 / 16)

# device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


def integrated_gradients(
        model: MalConvGCT,
        inputs: torch.Tensor,
        target: int = 1,
        baseline: int = 256,
        m: int = 3,
) -> torch.Tensor:
    """Run the integrated gradients algorithm over a batch of inputs.

    Args:
        inputs: a batch of inputs
        target: the target class
        baseline: baseline to compute gradient w.r. to
        m: number of steps in the Riemman approximation (should be between 30 and 200)

    Returns:
        Integrated gradients for each feature corresponding to every input in inputs

    Notes:
        Inspired by code from
        https://towardsdatascience.com/integrated-gradients-from-scratch-b46311e4ab4
    """
    # Gradients corresponding to each example in inputs
    grads = []
    for inp in inputs:
        # Gets every scaled feature inside the call to F(.)
        scaled_features = torch.cat(
            [baseline + (k / m) * (inp.unsqueeze(0) - baseline) for k in range(m + 1)], dim=0
        ).requires_grad_()
        # Compute the outputs
        outputs = model(scaled_features)[0]
        # Probabilities that examples belongs to target class
        pred_probas = F.softmax(outputs, dim=-1)[:, target]
        # Compute the gradients
        g = grad(outputs=torch.unbind(pred_probas), inputs=scaled_features)
        if len(g) > 1:
            raise ValueError(f"Expected single tensor from grad, not {len(g)} tensors")
        else:
            g = g[0]
        # Summation
        g = g.sum(dim=0)
        # Multiply by factors outside the summation
        g = (1 / m) * (inp - baseline) * g
        # Add to list
        grads.append(g)
    # Group into tensor corresponding to inputs
    grads = torch.cat([t.unsqueeze(0) for t in grads])
    return grads


# def layer_integrated_gradients(
#         model: ExMalConvGCT,
#         inputs: torch.Tensor,
#         embeddings: torch.Tensor,
#         target: int = 1,
#         baseline: int = torch.Tensor, # embedding baseline for a single input
#         m: int = 3,
# ) -> torch.Tensor:
#     # Gradients corresponding to each example in inputs
#     grads = []
#     for inp, emb in zip(inputs, embeddings):
#         # Gets every scaled feature inside the call to F(.)
#         scaled_features = torch.cat(
#             [baseline + (k / m) * (emb.unsqueeze(0) - baseline) for k in range(m + 1)], dim=0
#         ).requires_grad_()
#         # Compute the outputs
#         outputs = model(inp, scaled_features)[0]
#         # Probabilities that examples belongs to target class
#         pred_probas = F.softmax(outputs, dim=-1)[:, target]
#         # Compute the gradients
#         g = grad(outputs=torch.unbind(pred_probas), inputs=scaled_features)
#         if len(g) > 1:
#             raise ValueError(f"Expected single tensor from grad, not {len(g)} tensors")
#         else:
#             g = g[0]
#         # Summation
#         g = g.sum(dim=0)
#         # Multiply by factors outside the summation
#         g = (1 / m) * (inp - baseline) * g
#         # Add to list
#         grads.append(g)
#     # Group into tensor corresponding to inputs
#     grads = torch.cat([t.unsqueeze(0) for t in grads])
#     return grads


def get_model(checkpoint_path: Path, verbose: bool = False):
    if checkpoint_path.name == MALCONV_GCT_PATH.name:
        model = MalConvGCT(channels=256, window_size=256, stride=64)
    elif checkpoint_path.name == MALCONV_PATH.name:
        model = MalConv()
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    if verbose:
        print(f"{model=}")
    return model


def get_datasets(
        n_train: int = None,
        n_test: int = None,
        verbose: bool = False
) -> tp.Tuple[BinaryDataset, BinaryDataset]:
    train_dataset = BinaryDataset(
        WINDOWS_TRAIN_PATH,
        SOREL_TRAIN_PATH,
        max_len=MAX_LEN,
        sort_by_size=False,
        shuffle=True
    )
    test_dataset = BinaryDataset(
        WINDOWS_TEST_PATH,
        SOREL_TEST_PATH,
        max_len=MAX_LEN,
        sort_by_size=False,
        shuffle=True
    )
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


def get_loaders(
        train_dataset: Dataset,
        test_dataset: Dataset,
        train_sampler: RandomChunkSampler = None,
        test_sampler: RandomChunkSampler = None,
        verbose: bool = False,
) -> tp.Tuple[DataLoader, DataLoader, RandomChunkSampler, RandomChunkSampler]:
    loader_threads = max(mp.cpu_count() - 4, mp.cpu_count() // 2 + 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=loader_threads,
        collate_fn=pad_collate_func,
        sampler=train_sampler,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
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
        n_train: int = None,
        n_test: int = None,
        verbose: bool = False,
) -> tp.Tuple[BinaryDataset, BinaryDataset, DataLoader, DataLoader]:
    train_dataset, test_dataset = get_datasets(n_train, n_test, verbose=verbose)
    # train_sampler = RandomChunkSampler(train_dataset, BATCH_SIZE)
    # test_sampler = RandomChunkSampler(test_dataset, BATCH_SIZE)
    train_sampler = None
    test_sampler = None
    train_loader, test_loader = get_loaders(
        train_dataset,
        test_dataset,
        train_sampler,
        test_sampler,
        verbose=verbose,
    )
    return train_dataset, test_dataset, train_loader, test_loader, train_sampler, test_sampler


def evaluate_pretrained_malconv(
        checkpoint_path: Path,
        n_test: int = None,
        verbose: bool = False
) -> None:
    model = get_model(checkpoint_path, verbose=verbose)
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


def forward_function_malconv(model, softmax: bool):
    if softmax:
        return lambda x: F.softmax(model(x)[0], dim=-1)
    else:
        return lambda x: model(x)[0]


def explain_pretrained_malconv(
        checkpoint_path: Path,
        run_integrated_gradients: bool = False,
        run_layer_integrated_gradients: bool = False,
        run_layer_activation: bool = False,
        run_feature_permutation: bool = False,
        run_feature_ablation: bool = False,
        run_occlusion: bool = False,
        run_shapley_value_sampling: bool = False,
        run_kernel_shap: bool = False,
        output_path: Path = None,
        verbose: bool = False,
) -> None:
    print(section_header("Model"))
    model = get_model(checkpoint_path, verbose=verbose)

    print(section_header("Data"))
    train_dataset, _, train_loader, _, train_sampler, _ = get_data(verbose=verbose)
    # loader = iter(train_loader)
    # sampler = list(range(len(train_dataset)))
    # X, y = next(loader)
    # i = 0
    # file_indices = sampler[i : i+BATCH_SIZE]
    # if BATCH_SIZE >= 8: # We can get pos/neg examples in each batch
    #     while torch.equal(y, torch.ones_like(y)) or torch.equal(y, torch.zeros_like(y)):
    #         i += BATCH_SIZE
    #         X, y = next(loader)# [Path(train_dataset.all_files[i][0]).parents[2].name for i in file_indices]
    #         file_indices = sampler[i : i+BATCH_SIZE]
    #         print(" ".join([str(i.item()) for i in y]))
    #         print(" ".join([Path(train_dataset.all_files[i][0]).parents[2].name[0] for i in file_indices]))
    # else: # Just look for malware, i.e., pos examples
    #     while not torch.equal(y, torch.ones_like(y)):
    #         X, y = next(loader)
    #         file_indices = [next(sampler) for _ in range(BATCH_SIZE)]
    #
    # X:torch.Tensor = X.to(device)
    # y:torch.Tensor = y.to(device)
    # if verbose:
    #     print(f"{X.shape=}")
    #     print(f"{X=}")
    #     print(f"{y.shape=}")
    #     print(f"{y=}")
    # if output_path is not None:
    #     torch.save(X, output_path / "X.pt")
    #     torch.save(y, output_path / "y.pt")
    #     files = [train_dataset.all_files[i][0] + "\n" for i in file_indices]
    #     with open(output_path / "files.txt", "w") as f:
    #         f.writelines(files)

    print(section_header("Captum"))
    forward_functions = {
        f"{softmax=}": forward_function_malconv(model, softmax)
        for softmax in (False,)
    }
    layers = ["fc_2"] # ["embd", "conv_1", "conv_2", "fc_1","fc_2"]
    print(pformat(f"forward_functions={list(forward_functions.keys())}"))
    print(pformat(f"{layers=}"))

    files = [Path(e[0]) for e in train_dataset.all_files]
    for (X, _), f in tqdm(zip(train_loader, batch(files, BATCH_SIZE))):
        X = X.to(device)
        explain_batch(
            model,
            X,
            f,
            forward_functions,
            layers,
            run_integrated_gradients,
            run_layer_integrated_gradients,
            run_layer_activation,
            run_feature_permutation,
            run_feature_ablation,
            run_occlusion,
            run_shapley_value_sampling,
            run_kernel_shap,
            output_path,
            False,
        )


def explain_batch(
        model: nn.Module,
        X: torch.tensor,
        files: tp.List[Path],
        forward_functions: tp.Dict[str, tp.Callable],
        layers: tp.List[nn.Module],
        run_integrated_gradients: bool = False,
        run_layer_integrated_gradients: bool = False,
        run_layer_activation: bool = False,
        run_feature_permutation: bool = False,
        run_feature_ablation: bool = False,
        run_occlusion: bool = False,
        run_shapley_value_sampling: bool = False,
        run_kernel_shap: bool = False,
        output_path: Path = None,
        verbose: bool = False,
) -> None:

    if run_integrated_gradients:
        print(section_header("IntegratedGradients"))
        for name, forward_func in forward_functions.items():
            print(f"forward_func_name={name}")
            output_path_ = output_path / "IntegratedGradients" / name
            output_path_.mkdir(exist_ok=True, parents=True)
            ce.integrated_gradients(forward_func, X, files, output_path_, verbose)

    if run_layer_integrated_gradients:
        print(section_header("LayerIntegratedGradients"))
        for name, forward_func in forward_functions.items():
            print(f"forward_func_name={name}")
            for layer in layers:
                output_path_ = output_path / "LayerIntegratedGradients" / name / layer
                output_path_.mkdir(exist_ok=True, parents=True)
                ce.layer_integrated_gradients(forward_func, getattr(model, layer), X, files, output_path_, verbose)

    if run_layer_activation:
        print(section_header("LayerActivation"))
        for name, forward_func in forward_functions.items():
            print(f"forward_func_name={name}")
            for layer in layers:
                output_path_ = output_path / "LayerActivation" / name / layer
                output_path_.mkdir(exist_ok=True, parents=True)
                ce.layer_activation(forward_func, getattr(model, layer), X, files, output_path_, verbose)

    if run_feature_permutation:
        print(section_header("FeaturePermutation"))
        for name, forward_func in forward_functions.items():
            print(f"forward_func_name={name}")
            output_path_ = output_path / "FeaturePermutation" / name
            output_path_.mkdir(exist_ok=True, parents=True)
            ce.feature_permutation(forward_func, X, files, output_path_, verbose)

    if run_feature_ablation:
        print(section_header("FeatureAblation"))
        for name, forward_func in forward_functions.items():
            print(f"forward_func_name={name}")
            output_path_ = output_path / "FeatureAblation" / name
            output_path_.mkdir(exist_ok=True, parents=True)
            ce.captum_feature_ablation(forward_func, X, files, output_path_, verbose)

    if run_occlusion:
        print(section_header("Occlusion"))
        for name, forward_func in forward_functions.items():
            print(f"forward_func_name={name}")
            output_path_ = output_path / "Occlusion" / name
            output_path_.mkdir(exist_ok=True, parents=True)
            ce.occlusion(forward_func, X, files, output_path_, verbose)

    if run_shapley_value_sampling:
        print(section_header("ShapleyValueSampling"))
        for name, forward_func in forward_functions.items():
            print(f"forward_func_name={name}")
            output_path_ = output_path / "ShapleyValueSampling" / name
            output_path_.mkdir(exist_ok=True, parents=True)
            ce.shapley_value_sampling(forward_func, X, files, output_path_, verbose)

    if run_kernel_shap:
        print(section_header("KernelShap"))
        for name, forward_func in forward_functions.items():
            print(f"forward_func_name={name}")
            output_path_ = output_path / "KernelShap" / name
            output_path_.mkdir(exist_ok=True, parents=True)
            ce.kernel_shap(forward_func, X, files, output_path_, verbose)


def analyze_explainability_data(output_path: Path, methods: tp.Iterable[str] = None):
    with open(output_path / "files.txt", "r") as file:
        files = file.readlines()
    files = [Path(f.strip("\n")).name for f in files]
    for attr_file in output_path.rglob("attributions.pt"):
        if methods and not any(m in [p.name for p in attr_file.parents] for m in methods):
            continue
        local_path = attr_file.parent
        attrs = torch.load(attr_file, map_location=torch.device("cpu"))
        sorted_, indices = torch.sort(attrs, descending=True)
        sorted_ = sorted_.cpu().numpy()
        indices = indices.cpu().numpy()
        for s, i, f in zip(sorted_, indices, files):
            fig, ax = plt.subplots()
            ax.plot(x=i, y=s)
            fig.savefig()
            val_ind = np.stack((s, i), 1)
            np.savetxt((local_path / f).with_suffix(".txt"), val_ind, fmt=["%.18e", "%i"])


def read_binary(file: Path, mode: str = "rb"):
    try:
        with gzip.open(file, mode) as f:
            x = f.read(MAX_LEN)
    except OSError:
        with open(file, mode) as f:
            x = f.read(MAX_LEN)
    x = np.frombuffer(x, dtype=np.uint8).astype(np.int16) + 1
    return x


def build_corpora(checkpoint_path: Path, output_path: Path):
    print(section_header("Building Corpora"))

    model = get_model(checkpoint_path)
    train_dataset, _, train_loader, _, train_sampler, _ = get_data()
    forward_func = forward_function_malconv(model, False)
    alg = capattr.KernelShap(forward_func)
    mask_size = 512
    max_hash_evasion = 10

    divisions = ("malicious", "benign")
    for d in divisions:
        (output_path / d).mkdir(exist_ok=True)

    skip_until_file = "e0fc92b75a4ef2c0003d8a0bc4c9f2dae6dbf6d20ce700ea21fd75d3"
    skip = False

    for i, (X, y) in enumerate(tqdm(train_loader)):
        X, y = X.to(device), y.to(device)
        file_indices = list(range(i * BATCH_SIZE, (i + 1) * BATCH_SIZE))[0:X.shape[0]]
        files = [Path(train_dataset.all_files[i][0]) for i in file_indices]
        feature_mask = ce.get_feature_mask(X, mask_size=mask_size)[0].unsqueeze(0)
        for f, x in zip(files, X):
            if f.stem == skip_until_file:
                skip = False
            if skip:
                continue
            # Read the file's bytes
            binary = read_binary(f)
            # Get the attributions
            attributions = alg.attribute(
                inputs=x.unsqueeze(0),
                baselines=ce.baseline,
                target=1,
                feature_mask=feature_mask
            )[0]
            # Divide into malicious and benign regions
            malicious_regions, benign_regions = [], []
            for j in range(0, (binary.shape[0] // mask_size) * mask_size + 1, mask_size):
                b = j
                e = min(j + mask_size, binary.shape[0])
                if e != b:
                    if attributions[j].item() > 0:
                        malicious_regions.append((b, e))
                    elif attributions[j].item() < 0:
                        benign_regions.append((b, e))
            # Write the malicious and benign snippets to files
            for r, d in zip((malicious_regions, benign_regions), divisions):
                for j, (b, e) in enumerate(r):
                    binary_section = (binary[b:e] - 1).astype(np.uint8)
                    # Switch directories to avoid hashing collisions
                    for i in range(max_hash_evasion):
                        path = output_path / d / str(i) / f"{f.name}_{j}.bin"
                        path.parent.mkdir(exist_ok=True)
                        try:
                            binary_section.tofile(path)
                            break
                        except OSError as error:
                            continue
                        print(f"Failed to place: {path.name}")
                        print(error)


def code_section_offset_bounds(f: Path):
    binary = lief.parse(f.as_posix())

    try:
        section = binary.get_section(".text")
    except lief.not_found as e:
        print(f"No .text section found for {f.as_posix()}")
        print("Sections found:")
        pprint([s.name for s in binary.sections])
        raise e

    return section.offset, section.offset + section.size


def build_corpora_fixed_chunk_size(
        output_path: Path,
        attributions_path: Path,
        corpora_path: Path,
        chunk_size: int,
):
    print(section_header("Building Corpora"))

    train_dataset, _, train_loader, _, train_sampler, _ = get_data()
    max_hash_evasion = 10

    hyperparam_path = attributions_path.relative_to(output_path)
    corpora_path = corpora_path / hyperparam_path

    binary_files = [Path(e[0]) for e in train_dataset.all_files]
    for f in binary_files: #tqdm(binary_files):
        if f.stat().st_size == 0:   # Ignore empty files
            continue
        # Saved attributions tensor
        attributions = torch.load(
            (attributions_path / f.name).with_suffix(".pt"),
            map_location=torch.device("cpu")
        )
        # Byte-view of the binary
        binary = read_binary(f)
        # .text section bounds to produce the snippets from
        try:
            lower, upper = code_section_offset_bounds(f)
        except lief.not_found:  # code section could not be located
            continue
        print(f.as_posix())
        # Identify malicious and benign regions, save snippets to file
        for j in range(0, (binary.shape[0] // chunk_size) * chunk_size + 1, chunk_size):
            # Beginning and end regions to consider
            b = j
            e = min(j + chunk_size, binary.shape[0])
            # Skip if outside the binary's code section or zero-length slice
            if b == e or b < lower or e > upper:
                continue
            # Section of binary corresponding to the snippet
            binary_section = (binary[b:e] - 1).astype(np.uint8)
            # Whether snippet is malicious-looking or benign-looking
            if attributions[j].item() > 0:
                division = "malicious"
            elif attributions[j].item() < 0:
                division = "benign"
            # Save the snippet
            for i in range(max_hash_evasion):
                path = corpora_path / division / str(i) / f"{f.name}_{b}_{e}.bin"
                path.parent.mkdir(exist_ok=True, parents=True)
                try:
                    binary_section.tofile(path)
                    break
                except OSError:
                    continue
                print(f"Failed to place: {path.name}")
                print(error)


if __name__ == "__main__":

    random.seed(0)
    print(section_header("Pytorch", False))
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.enabled = False
    print(f"{device=}")
    print(f"{torch.backends.cudnn.enabled=}")

    output_path = Path("outputs/7")
    # explain_pretrained_malconv(
    #     MALCONV_PATH,
    #     # run_integrated_gradients=True,
    #     # run_layer_integrated_gradients=True,
    #     # run_layer_activation=True,
    #     # run_feature_permutation=True,
    #     # run_feature_ablation=True,
    #     # run_occlusion=True,
    #     # run_shapley_value_sampling=True,
    #     run_kernel_shap=True,
    #     output_path=output_path,
    #     verbose=True,
    # )

    attributions_path = Path("outputs/7/KernelShap/softmax=False/<class 'torch.Tensor'>/50/1/attributions")
    corpora_path = Path("/home/lk3591/Documents/datasets/MachineCodeTranslation/")
    build_corpora_fixed_chunk_size(
        output_path,
        attributions_path,
        corpora_path,
        256,
    )
