"""

"""

from itertools import chain
from pathlib import Path
from pprint import pformat
import traceback
import typing as tp

from captum.attr import (
    FeatureAblation,
    FeaturePermutation,
    IntegratedGradients,
    KernelShap,
    LayerActivation,
    LayerIntegratedGradients,
    Occlusion,
    ShapleyValueSampling,
)
from tqdm import tqdm
import torch
import torch.nn as nn

from classifier import (
    get_model,
    get_dataset_and_loader,
    forward_function_malconv,
    ModelName,
    SOREL_TEST_PATH,
    SOREL_TRAIN_PATH,
    PAD_VALUE,
    WINDOWS_TEST_PATH,
    WINDOWS_TRAIN_PATH,
)
from config import device
from utils import batch, error_line, section_header
from typing_ import ForwardFunction, Pathlike


BASELINE = PAD_VALUE


# Order of how the paths are created
attribution_parameters = {
    "FeatureAblation": ("mask_size", "perturbations_per_eval"),
    "FeaturePermutation": ("mask_size", "perturbations_per_eval"),
    "IntegratedGradients": ("method", "n_steps"),
    "KernelShap": ("mask_size", "n_samples", "perturbations_per_eval"),
    "LayerActivation": ("layer_name",),
    "LayerIntegratedGradients": ("layer_name", "method", "n_steps"),
    "Occlusion": ("sliding_window_shapes", "strides", "perturbations_per_eval"),
    "ShapleyValueSampling": ("mask_size", "n_samples", "perturbations_per_eval"),
}


def get_feature_mask(
    inputs: torch.Tensor,
    mask_size: int = 256,
    copy_for_each_input: bool = True,
) -> torch.Tensor:
    if inputs.shape[1] < mask_size:
        return torch.full((inputs.shape[1],), 0).unsqueeze(0)

    q, r = divmod(inputs.shape[1], mask_size)
    feature_mask = torch.cat([torch.full((mask_size,), i) for i in range(q)])
    feature_mask = torch.cat([feature_mask, torch.full((r,), q)])
    if copy_for_each_input:
        feature_mask = torch.cat([feature_mask.unsqueeze(0) for _ in range(inputs.shape[0])], 0)
    else:
        feature_mask = feature_mask.unsqueeze(0)
    feature_mask = feature_mask.type(torch.int64).to(device)
    return feature_mask


def output_captum_attributions(
    output_root: Path,
    filenames: tp.List[Path],
    attributions: torch.Tensor,
    output_path: Pathlike = None,
    delta: tp.Optional[torch.Tensor] = None,
) -> None:
    if output_root is None or filenames is None:
        return

    if len(attributions.shape) == 1 and len(filenames) != 1:
        raise ValueError()
    elif len(attributions.shape) > 1 and attributions.shape[0] != len(filenames):
        raise ValueError()

    path = output_root / "attributions"
    path = path / output_path if output_path is not None else output_root
    path.mkdir(exist_ok=True, parents=True)
    for t, f in zip(attributions, filenames):
        p = path / (f.name + ".pt")
        torch.save(t, p)

    if delta is not None:
        path = output_root / "delta"
        path = path / output_path if output_path is not None else output_root
        path.mkdir(exist_ok=True, parents=True)
        for t, f in zip(attributions, filenames):
            p = (path / f.name).with_suffix(".pt")
            torch.save(t, p)


def feature_ablation(
    forward_func: ForwardFunction,
    inputs: torch.Tensor,
    filenames: tp.List[Path] = None,
    output_root: Path = None,
    output_path_attributions: Path = None,
    verbose: bool = False,
) -> None:
    if verbose:
        print(section_header("", start_with_newline=False))
    output_root.mkdir(exist_ok=True)
    alg = FeatureAblation(forward_func)
    for mask_size in [256]:
        feature_mask = get_feature_mask(inputs, mask_size)
        for perturbations_per_eval in [1]:
            if verbose:
                print(f"{mask_size=}")
                print(f"{perturbations_per_eval=}")
            output_path = output_root / str(mask_size) / str(perturbations_per_eval)
            output_path.mkdir(exist_ok=True, parents=True)
            try:
                attributions = alg.attribute(
                    inputs=inputs,
                    baselines=BASELINE,
                    target=1,
                    feature_mask=feature_mask,
                    perturbations_per_eval=perturbations_per_eval,
                    show_progress=verbose,
                )
                output_captum_attributions(
                    output_path, filenames, attributions, output_path=output_path_attributions
                )
            except Exception as e:
                print(error_line())
                print(e)
                traceback.print_exc()
                print(error_line())


def feature_permutation(
    forward_func: ForwardFunction,
    inputs: torch.Tensor,
    filenames: tp.List[Path] = None,
    output_root: Path = None,
    output_path_attributions: Path = None,
    verbose: bool = False,
) -> None:
    if verbose:
        print(section_header("", start_with_newline=False))
    output_root.mkdir(exist_ok=True)
    alg = FeaturePermutation(forward_func)
    for mask_size in [256]:
        feature_mask = get_feature_mask(inputs, mask_size, copy_for_each_input=False)
        for perturbations_per_eval in [1]:
            if verbose:
                print(f"{mask_size=}")
                print(f"{perturbations_per_eval=}")
            output_path = output_root / str(mask_size) / str(perturbations_per_eval)
            output_path.mkdir(exist_ok=True, parents=True)
            try:
                attributions = alg.attribute(
                    inputs=inputs,
                    target=1,
                    feature_mask=feature_mask,
                    perturbations_per_eval=perturbations_per_eval,
                    show_progress=verbose,
                )
                output_captum_attributions(
                    output_path, filenames, attributions, output_path=output_path_attributions
                )
            except Exception as e:
                print(error_line())
                print(e)
                traceback.print_exc()
                print(error_line())


def integrated_gradients(
    forward_func: ForwardFunction,
    inputs: torch.Tensor,
    filenames: tp.List[Path] = None,
    output_root: Path = None,
    output_path_attributions: Path = None,
    verbose: bool = False,
) -> None:
    if verbose:
        print(section_header("", start_with_newline=False))
    output_root.mkdir(exist_ok=True)
    alg = IntegratedGradients(forward_func)
    for method in ["gausslegendre"]:
        for n_steps in [50]:
            if verbose:
                print(f"{method=}")
                print(f"{n_steps=}")
            output_path = output_root / method / str(n_steps)
            output_path.mkdir(exist_ok=True, parents=True)
            try:
                attributions, delta = alg.attribute(
                    inputs=inputs,
                    baselines=BASELINE,
                    target=1,
                    n_steps=n_steps,
                    method=method,
                    return_convergence_delta=True,
                )
                output_captum_attributions(
                    output_path,
                    filenames,
                    attributions,
                    delta,
                    output_path=output_path_attributions,
                )
            except Exception as e:
                print(error_line())
                print(e)
                traceback.print_exc()
                print(error_line())


def kernel_shap(
    forward_func: ForwardFunction,
    inputs: torch.Tensor,
    filenames: tp.List[Path] = None,
    output_root: Path = None,
    output_path_attributions: Path = None,
    verbose: bool = False,
) -> None:
    if verbose:
        print(section_header("", start_with_newline=False))
    output_root.mkdir(exist_ok=True)
    alg = KernelShap(forward_func)
    for mask_size in [256]:
        feature_mask = get_feature_mask(inputs, mask_size)[0]
        for n_samples in [50]:
            for perturbations_per_eval in [1]:
                if verbose:
                    print(f"{mask_size=}")
                    print(f"{n_samples=}")
                    print(f"{perturbations_per_eval=}")
                output_path = (
                    output_root / str(mask_size) / str(n_samples) / str(perturbations_per_eval)
                )
                output_path.mkdir(exist_ok=True, parents=True)
                # It is recommended to only provide a single example as input
                for i in range(inputs.shape[0]):
                    try:
                        attributions = alg.attribute(
                            inputs=inputs[i].unsqueeze(0),
                            baselines=BASELINE,
                            target=1,
                            feature_mask=feature_mask,
                            n_samples=n_samples,
                            perturbations_per_eval=perturbations_per_eval,
                            show_progress=verbose,
                        )
                        output_captum_attributions(
                            output_path,
                            [filenames[i]],
                            attributions,
                            output_path=output_path_attributions,
                        )
                    except Exception as e:
                        print(error_line())
                        print(e)
                        traceback.print_exc()
                        print(error_line())


def layer_activation(
    forward_func: ForwardFunction,
    layer: nn.Module,
    inputs: torch.Tensor,
    filenames: tp.List[Path] = None,
    output_root: Path = None,
    output_path_attributions: Path = None,
    verbose: bool = False,
) -> None:
    if verbose:
        print(section_header("", start_with_newline=False))
    output_root.mkdir(exist_ok=True)
    alg = LayerActivation(forward_func, layer)
    try:
        attributions = alg.attribute(inputs=inputs)
        output_captum_attributions(
            output_root, filenames, attributions, output_path=output_path_attributions
        )
    except Exception as e:
        print(error_line())
        print(e)
        traceback.print_exc()
        print(error_line())


def layer_integrated_gradients(
    forward_func: ForwardFunction,
    layer: nn.Module,
    inputs: torch.Tensor,
    filenames: tp.List[Path] = None,
    output_root: Path = None,
    output_path_attributions: Path = None,
    verbose: bool = False,
) -> None:
    if verbose:
        print(section_header("", start_with_newline=False))
    output_root.mkdir(exist_ok=True)
    alg = LayerIntegratedGradients(forward_func, layer)
    for method in ["gausslegendre"]:
        for n_steps in [50]:
            if verbose:
                print(f"{method=}")
                print(f"{n_steps=}")
            output_path = output_root / method / str(n_steps)
            output_path.mkdir(exist_ok=True, parents=True)
            try:
                attributions, delta = alg.attribute(
                    inputs=inputs,
                    baselines=BASELINE,
                    target=1,
                    n_steps=n_steps,
                    method=method,
                    return_convergence_delta=True,
                )
                output_captum_attributions(
                    output_path,
                    filenames,
                    attributions,
                    delta=delta,
                    output_path=output_path_attributions,
                )
            except Exception as e:
                print(error_line())
                print(e)
                traceback.print_exc()
                print(error_line())


def occlusion(
    forward_func: ForwardFunction,
    inputs: torch.Tensor,
    filenames: tp.List[Path] = None,
    output_root: Path = None,
    output_path_attributions: Path = None,
    verbose: bool = False,
) -> None:
    if verbose:
        print(section_header("", start_with_newline=False))
    output_root.mkdir(exist_ok=True)
    alg = Occlusion(forward_func)
    for sliding_window_shapes in [(10000,)]:
        for strides in [1]:
            for perturbations_per_eval in [1]:
                if verbose:
                    print(f"{sliding_window_shapes=}")
                    print(f"{strides=}")
                    print(f"{perturbations_per_eval=}")
                output_path = (
                    output_root
                    / str(sliding_window_shapes)
                    / str(strides)
                    / str(perturbations_per_eval)
                )
                output_path.mkdir(exist_ok=True, parents=True)
                try:
                    attributions = alg.attribute(
                        inputs=inputs,
                        sliding_window_shapes=sliding_window_shapes,
                        strides=strides,
                        baselines=BASELINE,
                        target=1,
                        perturbations_per_eval=perturbations_per_eval,
                        show_progress=verbose,
                    )
                    output_captum_attributions(
                        output_path, filenames, attributions, output_path=output_path_attributions
                    )
                except Exception as e:
                    print(error_line())
                    print(e)
                    traceback.print_exc()
                    print(error_line())


def shapley_value_sampling(
    forward_func: ForwardFunction,
    inputs: torch.Tensor,
    filenames: tp.List[Path] = None,
    output_root: Path = None,
    output_path_attributions: Path = None,
    verbose: bool = False,
) -> None:
    if verbose:
        print(section_header("", start_with_newline=False))
    output_root.mkdir(exist_ok=True)
    alg = ShapleyValueSampling(forward_func)
    for mask_size in [256]:
        feature_mask = get_feature_mask(inputs, mask_size)
        for n_samples in [25]:
            for perturbations_per_eval in [1]:
                if verbose:
                    print(f"{mask_size=}")
                    print(f"{n_samples=}")
                    print(f"{perturbations_per_eval=}")
                output_path = (
                    output_root / str(mask_size) / str(n_samples) / str(perturbations_per_eval)
                )
                output_path.mkdir(exist_ok=True, parents=True)
                try:
                    attributions = alg.attribute(
                        inputs=inputs,
                        baselines=BASELINE,
                        target=1,
                        feature_mask=feature_mask,
                        n_samples=n_samples,
                        perturbations_per_eval=perturbations_per_eval,
                        show_progress=verbose,
                    )
                    output_captum_attributions(
                        output_path, filenames, attributions, output_path=output_path_attributions
                    )
                except Exception as e:
                    print(error_line())
                    print(e)
                    traceback.print_exc()
                    print(error_line())


def explain_batch(
    model: nn.Module,
    X: torch.tensor,
    files: tp.List[Path],
    forward_functions: tp.Dict[str, tp.Callable],
    layers: tp.List[str],
    run_feature_ablation: bool = False,
    run_feature_permutation: bool = False,
    run_integrated_gradients: bool = False,
    run_kernel_shap: bool = False,
    run_layer_activation: bool = False,
    run_layer_integrated_gradients: bool = False,
    run_occlusion: bool = False,
    run_shapley_value_sampling: bool = False,
    output_root: Path = None,
    output_path_attributions: Pathlike = None,
    verbose: bool = False,
) -> None:

    if run_feature_ablation:
        if verbose:
            print(section_header("FeatureAblation"))
        for softmax, forward_func in forward_functions.items():
            if verbose:
                print(f"{softmax=}")
            output_path = output_root / "FeatureAblation" / str(softmax)
            output_path.mkdir(exist_ok=True, parents=True)
            feature_ablation(
                forward_func,
                X,
                files,
                output_root=output_path,
                output_path_attributions=output_path_attributions,
                verbose=verbose,
            )

    if run_feature_permutation:
        if verbose:
            print(section_header("FeaturePermutation"))
        for softmax, forward_func in forward_functions.items():
            if verbose:
                print(f"{softmax=}")
            output_path = output_root / "FeaturePermutation" / str(softmax)
            output_path.mkdir(exist_ok=True, parents=True)
            feature_permutation(
                forward_func,
                X,
                files,
                output_root=output_path,
                output_path_attributions=output_path_attributions,
                verbose=verbose,
            )

    if run_integrated_gradients:
        if verbose:
            print(section_header("IntegratedGradients"))
        for softmax, forward_func in forward_functions.items():
            if verbose:
                print(f"{softmax=}")
            output_path = output_root / "IntegratedGradients" / str(softmax)
            output_path.mkdir(exist_ok=True, parents=True)
            integrated_gradients(
                forward_func,
                X,
                files,
                output_root=output_path,
                output_path_attributions=output_path_attributions,
                verbose=verbose,
            )

    if run_kernel_shap:
        if verbose:
            print(section_header("KernelShap"))
        for softmax, forward_func in forward_functions.items():
            if verbose:
                print(f"{softmax=}")
            output_path = output_root / "KernelShap" / str(softmax)
            output_path.mkdir(exist_ok=True, parents=True)
            kernel_shap(
                forward_func,
                X,
                files,
                output_root=output_path,
                output_path_attributions=output_path_attributions,
                verbose=verbose,
            )

    if run_layer_activation:
        if verbose:
            print(section_header("LayerActivation"))
        for softmax, forward_func in forward_functions.items():
            if verbose:
                print(f"{softmax=}")
            for layer in layers:
                output_path = output_root / "LayerActivation" / str(softmax) / layer
                output_path.mkdir(exist_ok=True, parents=True)
                layer_activation(
                    forward_func,
                    getattr(model, layer),
                    X,
                    files,
                    output_root=output_path,
                    output_path_attributions=output_path_attributions,
                    verbose=verbose,
                )

    if run_layer_integrated_gradients:
        if verbose:
            print(section_header("LayerIntegratedGradients"))
        for softmax, forward_func in forward_functions.items():
            if verbose:
                print(f"{softmax=}")
            for layer in layers:
                output_path = output_root / "LayerIntegratedGradients" / str(softmax) / layer
                output_path.mkdir(exist_ok=True, parents=True)
                layer_integrated_gradients(
                    forward_func,
                    getattr(model, layer),
                    X,
                    files,
                    output_root=output_path,
                    output_path_attributions=output_path_attributions,
                    verbose=verbose,
                )

    if run_occlusion:
        if verbose:
            print(section_header("Occlusion"))
        for softmax, forward_func in forward_functions.items():
            if verbose:
                print(f"{softmax=}")
            output_path = output_root / "Occlusion" / str(softmax)
            output_path.mkdir(exist_ok=True, parents=True)
            occlusion(
                forward_func,
                X,
                files,
                output_path=output_path,
                output_path_attributions=output_path_attributions,
                verbose=verbose,
            )

    if run_shapley_value_sampling:
        if verbose:
            print(section_header("ShapleyValueSampling"))
        for softmax, forward_func in forward_functions.items():
            if verbose:
                print(f"{softmax=}")
            output_path = output_root / "ShapleyValueSampling" / str(softmax)
            output_path.mkdir(exist_ok=True, parents=True)
            shapley_value_sampling(
                forward_func,
                X,
                files,
                output_path=output_path,
                output_path_attributions=output_path_attributions,
                verbose=verbose,
            )


def explain_pretrained_malconv(
    model_name: ModelName,
    max_len: int,
    run_feature_ablation: bool = False,
    run_feature_permutation: bool = False,
    run_integrated_gradients: bool = False,
    run_kernel_shap: bool = False,
    run_layer_activation: bool = False,
    run_layer_integrated_gradients: bool = False,
    run_occlusion: bool = False,
    run_shapley_value_sampling: bool = False,
    output_root: Path = None,
    verbose: bool = False,
) -> None:
    print(section_header("Model"))
    model = get_model(model_name, verbose=verbose)

    print(section_header("Data"))
    batch_size = 1
    benign_dataset, benign_loader = get_dataset_and_loader(
        chain(WINDOWS_TRAIN_PATH.iterdir(), WINDOWS_TEST_PATH.iterdir()),
        None,
        max_len=max_len,
        batch_size=batch_size,
    )
    malicious_dataset, malicious_loader = get_dataset_and_loader(
        None,
        chain(SOREL_TRAIN_PATH.iterdir(), SOREL_TEST_PATH.iterdir()),
        max_len=max_len,
        batch_size=batch_size,
    )
    data = (
        (benign_dataset, benign_loader, "benign"),
        (malicious_dataset, malicious_loader, "malicious"),
    )

    print(section_header("Captum"))
    # We do not want to apply the softmax layer to the forward function, I think
    forward_functions = {
        softmax: forward_function_malconv(model, softmax) for softmax in (False,)  # (False, True)
    }
    layers = ["fc_2"]  # ["embd", "conv_1", "conv_2", "fc_1","fc_2"]
    print(pformat(f"forward_functions={list(forward_functions.keys())}"))
    print(pformat(f"{layers=}"))

    for d, l, c in data:
        print(section_header(f"Working on Attributions for class {c}"))
        batched_files = batch([Path(e[0]) for e in d.all_files], l.batch_size)
        for (X, _), files in tqdm(zip(l, batched_files), total=len(d) / batch_size):
            try:
                explain_batch(
                    model,
                    X.to(device),
                    files,
                    forward_functions,
                    layers,
                    run_feature_ablation,
                    run_feature_permutation,
                    run_integrated_gradients,
                    run_kernel_shap,
                    run_layer_activation,
                    run_layer_integrated_gradients,
                    run_occlusion,
                    run_shapley_value_sampling,
                    output_root=output_root,
                    output_path_attributions=c,
                    verbose=verbose,
                )
            except Exception as e:
                print(error_line())
                print(f"f={pformat(files)}")
                print(f"{X.shape=}")
                print(f"{X=}")
                print(f"{e=}")
                traceback.print_exc()
                print(error_line())


if __name__ == "__main__":
    model_name = "gct"
    max_len = 4000000
    explain_pretrained_malconv(
        "gct",
        max_len,
        run_feature_ablation=False,
        run_feature_permutation=False,
        run_integrated_gradients=False,
        run_kernel_shap=True,
        run_layer_activation=False,
        run_layer_integrated_gradients=False,
        run_occlusion=False,
        run_shapley_value_sampling=False,
        output_root=Path(f"outputs/{model_name}/{max_len}"),
        verbose=False,
    )
