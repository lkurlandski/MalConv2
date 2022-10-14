# 1:
# Fixes RuntimeError: cuDNN error: CUDNN_STATUS_MAPPING_ERROR?
# Source: https://discuss.pytorch.org/t/when-should-we-set-torch-backends
# -cudnn-enabled-to-false-especially-for-lstm/106571

from collections import OrderedDict
from copy import deepcopy
import multiprocessing as mp
from pathlib import Path
import pickle
import random
import sys
import typing as tp

from captum import attr as capattr
import numpy as np
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


random.seed(0)


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

# Hyperparams
BATCH_SIZE = 8
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
    train_dataset = BinaryDataset(WINDOWS_TRAIN_PATH, SOREL_TRAIN_PATH, max_len=MAX_LEN, sort_by_size=False)
    test_dataset = BinaryDataset(WINDOWS_TEST_PATH, SOREL_TEST_PATH, max_len=MAX_LEN, sort_by_size=False)
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
        shuffle: bool = True,
        verbose: bool = False,
) -> tp.Tuple[BinaryDataset, BinaryDataset, DataLoader, DataLoader]:
    train_dataset, test_dataset = get_datasets(n_train, n_test, verbose=verbose)
    train_sampler = RandomChunkSampler(train_dataset, BATCH_SIZE, random=shuffle)
    test_sampler = RandomChunkSampler(test_dataset, BATCH_SIZE, random=shuffle)
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


def output_captum_results(
        output_path: Path = None,
        filenames_tensors: dict = None,
        filenames_objects: dict = None,
        verbose: bool = False,
) -> None:
    filenames_tensors = {} if filenames_tensors is None else filenames_tensors
    filenames_objects = {} if filenames_objects is None else filenames_objects
    if verbose:
        for filename, tensor in filenames_tensors.items():
            print(f"{Path(filename).name}.shape={tensor.shape}")
            print(f"{Path(filename).name}={tensor}")
        for filename, obj in filenames_objects.items():
            print(f"{Path(filename).name}={obj}")
    if output_path is not None:
        output_path.mkdir(parents=False, exist_ok=True)
        for filename, tensor in filenames_tensors.items():
            torch.save(tensor, output_path / filename)
        for filename, obj in filenames_objects.items():
            with open(output_path / filename, "wb") as handle:
                pickle.dump(obj, handle)


def captum_integrated_gradients(
    forward_func: ForwardFunction,
    inputs: torch.Tensor,
    baselines: int,
    target: int,
    n_steps: int = 50,
    output_path: Path = None,
    verbose: bool = False,
    **kwargs,
) -> tp.Tuple[torch.tensor, torch.tensor]:
    alg = capattr.IntegratedGradients(forward_func)
    print_section_header(f"{type(alg).__name__}")
    attributions, delta = alg.attribute(
        inputs=inputs,
        baselines=baselines,
        target=target,
        n_steps=n_steps,
        return_convergence_delta=True,
        **kwargs,
    )
    output_captum_results(
        output_path / f"{type(alg).__name__}",
        {"attributions.pt": attributions, "delta.pt": delta},
        # {"alg.pickle": alg},
        verbose=verbose,
    )
    return attributions, delta

def captum_layer_integrated_gradients(
        forward_func: ForwardFunction,
        layer: nn.Module,
        inputs: torch.Tensor,
        baselines: int,
        target: int,
        n_steps: int = 50,
        output_path: Path = None,
        verbose: bool = False,
        **kwargs,
):
    alg = capattr.LayerIntegratedGradients(forward_func, layer)
    print_section_header(f"{type(alg).__name__}")
    attributions, delta = alg.attribute(
        inputs=inputs,
        baselines=baselines,
        target=target,
        n_steps=n_steps,
        return_convergence_delta=True,
        **kwargs,
    )
    output_captum_results(
        output_path / f"{type(alg).__name__}",
        {"attributions.pt": attributions, "delta.pt": delta},
        # {"alg.pickle": alg},
        verbose=verbose,
    )
    return attributions, delta


def captum_layer_activation(
        forward_func: ForwardFunction,
        layer: nn.Module,
        inputs: torch.Tensor,
        output_path: Path = None,
        verbose: bool = False,
        **kwargs,
):
    alg = capattr.LayerActivation(forward_func, layer)
    print_section_header(f"{type(alg).__name__}")
    attributions = alg.attribute(inputs=inputs, **kwargs)
    output_captum_results(
        output_path / f"{type(alg).__name__}",
        {"attributions.pt": attributions},
        # {"alg.pickle": alg},
        verbose=verbose,
    )
    return attributions


def captum_feature_permutation(
        forward_func: ForwardFunction,
        inputs: torch.Tensor,
        target: int,
        perturbations_per_eval: int = 1,
        output_path: Path = None,
        verbose: bool = False,
        **kwargs,
):
    alg = capattr.FeaturePermutation(forward_func)
    print_section_header(type(alg).__name__)
    attributions = alg.attribute(
        inputs=inputs,
        target=target,
        perturbations_per_eval=perturbations_per_eval,
        show_progress=verbose,
        **kwargs,
    )
    output_captum_results(
        output_path / f"{type(alg).__name__}",
        {"attributions.pt": attributions},
        # {"alg.pickle": alg},
        verbose=verbose
    )
    return attributions


def captum_feature_ablation(
        forward_func: ForwardFunction,
        inputs: torch.Tensor,
        baselines: int,
        target: int,
        perturbations_per_eval: int = 1,
        output_path: Path = None,
        verbose: bool = False,
        **kwargs,
):
    alg = capattr.FeatureAblation(forward_func)
    print_section_header(type(alg).__name__)
    attributions = alg.attribute(
        inputs=inputs,
        target=target,
        baselines=baselines,
        perturbations_per_eval=perturbations_per_eval,
        show_progress=verbose,
        **kwargs,
    )
    output_captum_results(
        output_path / f"{type(alg).__name__}",
        {"attributions.pt": attributions},
        # {"alg.pickle": alg},
        verbose=verbose
    )
    return attributions


def captum_occlusion(
        forward_func: ForwardFunction,
        inputs: torch.Tensor,
        baselines: int,
        target: int,
        perturbations_per_eval: int = 1,
        sliding_window_shapes=(10000,),
        output_path: Path = None,
        verbose: bool = False,
        **kwargs,
):
    alg = capattr.Occlusion(forward_func)
    print_section_header(type(alg).__name__)
    attributions = alg.attribute(
        inputs=inputs,
        sliding_window_shapes=sliding_window_shapes,
        baselines=baselines,
        target=target,
        perturbations_per_eval=perturbations_per_eval,
        show_progress=verbose,
        **kwargs,
    )
    output_captum_results(
        output_path / f"{type(alg).__name__}",
        {"attributions.pt": attributions},
        # {"alg.pickle": alg},
        verbose=verbose
    )
    return attributions


def captum_shapley_value_sampling(
        forward_func: ForwardFunction,
        inputs: torch.Tensor,
        baselines: int,
        target: int,
        n_samples: int = 25,
        perturbations_per_eval: int = 1,
        output_path: Path = None,
        verbose: bool = False,
        **kwargs,
):
    alg = capattr.ShapleyValueSampling(forward_func)
    print_section_header(type(alg).__name__)
    attributions = alg.attribute(
        inputs=inputs,
        baselines=baselines,
        target=target,
        n_samples=n_samples,
        perturbations_per_eval=perturbations_per_eval,
        show_progress=verbose,
        **kwargs,
    )
    output_captum_results(
        output_path / f"{type(alg).__name__}",
        {"attributions.pt": attributions},
        # {"alg.pickle": alg},
        verbose=verbose
    )
    return attributions


def explain_pretrained_malconv(
        checkpoint_path: Path,
        run_integrated_gradients: bool = False,
        run_layer_integrated_gradients: bool = False,
        run_layer_activation: bool = False,
        run_feature_permutation: bool = False,
        run_feature_ablation: bool = False,
        run_occlusion: bool = False,
        run_shapley_value_sampling: bool = False,
        layer: str = "fc_2",
        output_path: Path = None,
        verbose: bool = False,
) -> dict:
    print_section_header("Model & Data")
    model = get_model(checkpoint_path, verbose=verbose)
    train_dataset, _, train_loader, _, train_sampler, _ = get_data(shuffle=True, verbose=verbose)

    loader = iter(train_loader)
    sampler = iter(train_sampler)
    X, y = next(loader)
    file_indices = [next(sampler) for _ in range(BATCH_SIZE)]

    if BATCH_SIZE >= 8: # We can get pos/neg examples in each batch
        while torch.equal(y, torch.ones_like(y)) or torch.equal(y, torch.zeros_like(y)):
            X, y = next(loader)
            file_indices = [next(sampler) for _ in range(BATCH_SIZE)]
    else: # Just look for malware, i.e., pos examples
        while not torch.equal(y, torch.ones_like(y)):
            X, y = next(loader)
            file_indices = [next(sampler) for _ in range(BATCH_SIZE)]

    X:torch.Tensor = X.to(device)
    y:torch.Tensor = y.to(device)
    if verbose:
        print(f"{X.shape=}")
        print(f"{X=}")
        print(f"{y.shape=}")
        print(f"{y=}")
    if output_path is not None:
        torch.save(X, output_path / "X.pt")
        torch.save(y, output_path / "y.pt")
        files = [train_dataset.all_files[i][0] + "\n" for i in file_indices]
        with open(output_path / "files.txt", "w") as f:
            f.writelines(files)

    print_section_header("Captum")
    baselines = 256
    target = 1
    n_steps = 3
    use_softmax = False
    layer = getattr(model, layer)
    forward_func = forward_function_malconv(model, use_softmax)
    if verbose:
        print(f"{baselines=}")
        print(f"{target=}")
        print(f"{n_steps=}")
        print(f"{use_softmax=}")
        print(f"{layer=}")

    if run_integrated_gradients:
        try:
            captum_integrated_gradients(
                forward_func,
                X,
                baselines,
                target,
                n_steps=3,
                output_path=output_path,
                verbose=verbose
            )
        except Exception as e:
            print("-" * 40 + " ERROR " + "-" * 40)
            print(e)

    if run_layer_integrated_gradients:
        try:
            captum_layer_integrated_gradients(
                forward_func,
                layer,
                X,
                baselines,
                target,
                n_steps=3,
                output_path=output_path,
                verbose=verbose
            )
        except Exception as e:
            print("-" * 40 + " ERROR " + "-" * 40)
            print(e)

    if run_layer_activation:
        try:
            captum_layer_activation(
                forward_func,
                model.fc_2,
                X,
                output_path=output_path,
                verbose=verbose
            )
        except Exception as e:
            print("-" * 40 + " ERROR " + "-" * 40)
            print(e)

    if run_feature_permutation:
        try:
            captum_feature_permutation(
                forward_func,
                X,
                target,
                perturbations_per_eval=1,
                output_path=output_path,
                verbose=verbose,
            )
        except Exception as e:
            print("-" * 40 + " ERROR " + "-" * 40)
            print(e)

    if run_feature_ablation:
        try:
            mask_size = 100
            q, r = divmod(X.shape[1], mask_size)
            feature_mask = torch.cat([torch.full((mask_size,), i) for i in range(q)])
            feature_mask = torch.cat([feature_mask, torch.full((r,), q)])
            feature_mask = torch.cat([feature_mask.unsqueeze(0) for _ in range(BATCH_SIZE)], 0)
            feature_mask = feature_mask.type(torch.int64).to(device)
            softmax = False
            (output_path / f"{softmax=}").mkdir(exist_ok=True)
            captum_feature_ablation(
                forward_function_malconv(model, softmax),
                X,
                baselines,
                target,
                perturbations_per_eval=1,
                output_path=(output_path / f"{softmax=}"),
                verbose=verbose,
                feature_mask=feature_mask,
            )
            softmax = True
            (output_path / f"{softmax=}").mkdir(exist_ok=True)
            captum_feature_ablation(
                forward_function_malconv(model, softmax),
                X,
                baselines,
                target,
                perturbations_per_eval=1,
                output_path=(output_path / f"{softmax=}"),
                verbose=verbose,
                feature_mask=feature_mask,
            )
        except Exception as e:
            print("-" * 40 + " ERROR " + "-" * 40)
            print(e)
            raise e

    if run_occlusion:
        try:
            captum_occlusion(
                forward_func,
                X,
                baselines,
                target,
                perturbations_per_eval=1,
                sliding_window_shapes=(10000,),
                output_path=output_path,
                verbose=verbose,
            )
        except Exception as e:
            print("-" * 40 + " ERROR " + "-" * 40)
            print(e)

    if run_shapley_value_sampling:
        try:
            captum_shapley_value_sampling(
                forward_func,
                X,
                baselines,
                target,
                n_samples=25,
                perturbations_per_eval=1,
                output_path=output_path,
                verbose=verbose,
            )
        except Exception as e:
            print("-" * 40 + " ERROR " + "-" * 40)
            print(e)

    # if my_ig:
    #     print_section_header("Custom Integrated Gradients")
    #     grads = integrated_gradients(model, X, 1, 256, 3)
    #     print(f"{grads.shape=}")
    #     print(f"{grads=}")

    # if my_lig:
    #     pass
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


def process_attributions(
        attributions: tp.Union[Path, torch.Tensor],
        verbose: bool = False,
):
    if isinstance(attributions, (str, Path)):
        attributions = torch.load(attributions)

    if verbose:
        print(f"{attributions.shape=}")



def main():

    # evaluate_pretrained_malconv(MALCONV_GCT_PATH, 100, verbose=True)
    # evaluate_pretrained_malconv(MALCONV_PATH, 100, verbose=True)

    # process_attributions("outputs/1/FeaturePermutation/attributions.pt", verbose=True)
    # sys.exit()

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
        run_integrated_gradients=True,
        run_layer_integrated_gradients=True,
        run_layer_activation=True,
        # run_feature_permutation=True,
        run_feature_ablation=True,
        # run_occlusion=True,
        # run_shapley_value_sampling=True,
        layer=malconv_layers[4],
        output_path=Path("outputs/3"),
        verbose=True,
    )


if __name__ == "__main__":
    main()