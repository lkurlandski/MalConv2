# 1:
# Fixes RuntimeError: cuDNN error: CUDNN_STATUS_MAPPING_ERROR?
# Source: https://discuss.pytorch.org/t/when-should-we-set-torch-backends
# -cudnn-enabled-to-false-especially-for-lstm/106571

from collections import OrderedDict
from copy import deepcopy
import multiprocessing as mp
from pathlib import Path
import sys
import typing as tp

from captum import attr as capattr
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
from tqdm import tqdm
import torch
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from binaryLoader import BinaryDataset, RandomChunkSampler, pad_collate_func
from MalConv import MalConv
from MalConvGCT_nocat import MalConvGCT


ForwardFunction = tp.Callable[[torch.Tensor], torch.Tensor]


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
OUTPUT_PATH = Path("output")

# Hyperparams
BATCH_SIZE = 2
MAX_LEN = int(16000000 / 16)


def print_section_header(
        name: str,
        start_with_newline: bool = True,
        underline_length: int = 88
) -> None:
    print(("\n" if start_with_newline else "") + f"{name}\n{'-' * underline_length}")


# Pytorch
print_section_header("Pytorch", False)
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
print(f"{device=}")
print(f"{torch.backends.cudnn.enabled=}")


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
    train_dataset = BinaryDataset(WINDOWS_TRAIN_PATH, SOREL_TRAIN_PATH, max_len=MAX_LEN)
    test_dataset = BinaryDataset(WINDOWS_TEST_PATH, SOREL_TEST_PATH, max_len=MAX_LEN)
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
        train_dataset: Dataset = None,
        test_dataset: Dataset = None,
        verbose: bool = False
):
    loader_threads = max(mp.cpu_count() - 4, mp.cpu_count() // 2 + 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=loader_threads,
        collate_fn=pad_collate_func,
        sampler=RandomChunkSampler(train_dataset, BATCH_SIZE)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=loader_threads,
        collate_fn=pad_collate_func,
        sampler=RandomChunkSampler(test_dataset, BATCH_SIZE)
    )
    if verbose:
        print(f"{loader_threads=}")
        print(f"{train_loader=}")
        print(f"{test_loader=}")
    return train_loader, test_loader


def get_dataset_and_loaders(
        n_train: int = None,
        n_test: int = None,
        verbose: bool = False
) -> tp.Tuple[BinaryDataset, BinaryDataset, DataLoader, DataLoader]:
    train_dataset, test_dataset = get_datasets(n_train, n_test, verbose=verbose)
    train_loader, test_loader = get_loaders(train_dataset, test_dataset, verbose=verbose)
    return train_dataset, test_dataset, train_loader, test_loader


def evaluate_pretrained_malconv(
        checkpoint_path: Path,
        n_test: int = None,
        verbose: bool = False
) -> None:
    model = get_model(checkpoint_path, verbose=verbose)
    _, _, _, test_loader = get_dataset_and_loaders(n_test=n_test, verbose=verbose)

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


def captum_integrated_gradients(

):
    ...


def captum_layer_integrated_gradients(

):
    ...


def captum_layer_activation(
    forward_func: ForwardFunction,
    layer: nn.Module,
    inputs: torch.Tensor,
    verbose: bool = False
):
    print_section_header("LayerActivation")
    alg = capattr.LayerActivation(forward_func, layer)
    attrs = alg.attribute(inputs=inputs)
    if verbose:
        print(f"{alg=}")
        print(f"{attrs.shape=}, {attrs=}")
    return attrs


def captum_feature_permutation(

):
    ...


def captum_feature_ablation(

):
    ...


def captum_occlusion(

):
    ...


def captum_shaply_value_sampling(
    forward_function,
    inputs,
    **kwargs
):
    svs = capattr.ShapleyValueSampling(forward_function)
    attributions = svs.attribute(inputs, **kwargs)
    return attributions


def explain_pretrained_malconv(
        checkpoint_path: Path,
        ig: bool = False,
        lig: bool = False,
        layer_act: bool = False,
        feature_perm: bool = False,
        feature_ablt: bool = False,
        occ: bool = False,
        my_ig: bool = False,
        my_lig: bool = False,
        layer: str = "fc_2",
        verbose: bool = False,
) -> dict:
    print_section_header("Model & Data")
    model = get_model(checkpoint_path, verbose=verbose)
    _, _, train_loader, _ = get_dataset_and_loaders(verbose=verbose)

    X, y = iter(train_loader).next()
    X:torch.Tensor = X.to(device)
    y:torch.Tensor = y.to(device)
    if verbose:
        print(f"{X.shape=}")
        print(f"{X=}")
        print(f"{y.shape=}")
        print(f"{y=}")

    print_section_header("Captum")
    baselines = 256
    target = 1
    n_steps = 3
    use_softmax = False
    forward_function = forward_function_malconv(model, use_softmax)
    if verbose:
        print(f"{baselines=}")
        print(f"{target=}")
        print(f"{n_steps=}")
        print(f"{use_softmax=}")

    if ig:
        algorithm = capattr.IntegratedGradients(forward_function)
        print_section_header(type(algorithm).__name__)
        attributions, delta = algorithm.attribute(
            inputs=X,
            baselines=baselines,
            target=target,
            n_steps=n_steps,
            return_convergence_delta=True,
        )
        if verbose:
            print(f"{algorithm=}")
            print(f"{delta.shape=}, {delta=}")
            print(f"{attributions.shape=}, {attributions=}")

    if lig:
        algorithm = capattr.LayerIntegratedGradients(forward_function, getattr(model, layer))
        print_section_header(type(algorithm).__name__)
        attributions, delta = algorithm.attribute(
            inputs=X,
            baselines=baselines,
            target=target,
            n_steps=n_steps,
            return_convergence_delta=True,
        )
        if verbose:
            print(f"{algorithm=}")
            print(f"{delta.shape=}, {delta=}")
            print(f"{attributions.shape=}, {attributions=}")

    if layer_act:
        captum_layer_activation()

    if feature_perm:
        algorithm = capattr.FeaturePermutation(forward_function)
        print_section_header(type(algorithm).__name__)
        attributions = algorithm.attribute(
            inputs=X,
            target=1,
            perturbations_per_eval=1,
            show_progress=verbose,
        )
        if verbose:
            print(f"{algorithm=}")
            print(f"{len(attributions)} {attributions[0].shape=}, {attributions=}")

    if feature_ablt:
        algorithm = capattr.FeatureAblation(forward_function)
        print_section_header(type(algorithm).__name__)
        attributions = algorithm.attribute(
            inputs=X,
            baselines=baselines,
            target=1,
            perturbations_per_eval=1,
            show_progress=verbose,
        )
        if verbose:
            print(f"{algorithm=}")
            print(f"{len(attributions)} {attributions[0].shape=}, {attributions=}")

    if my_ig:
        print_section_header("Custom Integrated Gradients")
        grads = integrated_gradients(model, X, 1, 256, 3)
        print(f"{grads.shape=}")
        print(f"{grads=}")

    if occ:
        algorithm = capattr.Occlusion(forward_function)
        print_section_header(type(algorithm).__name__)
        attributions = algorithm.attribute(
            inputs=X,
            sliding_window_shapes=(10000,),
            baselines=baselines,
            target=1,
            perturbations_per_eval=1,
            show_progress=verbose,
        )
        if verbose:
            print(f"{algorithm=}")
            print(f"{len(attributions)} {attributions[0].shape=}, {attributions=}")



    if my_lig:
        pass
        # print_section_header("Custom Layer Integrated Gradients")
        # X_embedded = model.embd(X.type(torch.int64))
        # print(f"{X_embedded.shape=}")
        # print(f"{X_embedded=}")
        # baselines = model.embd(256 * torch.ones_like(X[0]).type(torch.int64))
        # print(f"{baselines.shape=}")
        # print(f"{baselines=}")
        # model = ExMalConvGCT(channels=256, window_size=256, stride=64)
        # model.load_state_dict(state['model_state_dict'], strict=False)
        # model.to(device)
        # model.eval()
        # grads = layer_integrated_gradients(model, X, X_embedded, 1, baselines, 3)
        # print(f"{grads.shape=}")
        # print(f"{grads=}")


def main():

    # evaluate_pretrained_malconv(MALCONV_GCT_PATH, 100, verbose=True)
    # evaluate_pretrained_malconv(MALCONV_PATH, 100, verbose=True)

    malconv_layers = [
        "embd", # 0
        "conv_1", # 1
        "conv_2", # 2
        "fc_1", # 3
        "fc_2" # 4
    ]

    malconv_gct_layers = [
        "embd", # 0
        "context_net", # 1
        "convs", # 2
        "linear_atn", # 3
        "convs_share", # 4
        "fc_1", # 5
        "fc_2", # 6
    ]

    explain_pretrained_malconv(
        MALCONV_PATH,
        ig=False,
        lig=False,
        layer_act=False,
        feature_perm=False,
        feature_ablt=False,
        occ=True,
        my_ig=False,
        my_lig=False,
        layer=malconv_layers[0],
        verbose=True,
    )



if __name__ == "__main__":
    main()