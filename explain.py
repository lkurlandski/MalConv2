"""

"""

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
    get_data,
    forward_function_malconv,
    ModelName,
    PAD_VALUE,
)
from config import device
from utils import batch, error_line, section_header
from typing_ import ForwardFunction


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
        feature_mask = torch.cat(
            [feature_mask.unsqueeze(0) for _ in range(inputs.shape[0])], 0
        )
    else:
        feature_mask = feature_mask.unsqueeze(0)
    feature_mask = feature_mask.type(torch.int64).to(device)
    return feature_mask


def output_captum_attributions(
    output_path: Path,
    filenames: tp.List[Path],
    attributions: torch.Tensor,
    delta: tp.Optional[torch.Tensor] = None,
) -> None:
    if output_path is None or filenames is None:
        return

    if len(attributions.shape) == 1 and len(filenames) != 1:
        raise ValueError()
    elif len(attributions.shape) > 1 and attributions.shape[0] != len(filenames):
        raise ValueError()

    path = output_path / "attributions"
    path.mkdir(exist_ok=True)
    for t, f in zip(attributions, filenames):
        p = (path / f.name).with_suffix(".pt")
        torch.save(t, p)

    if delta is not None:
        path = output_path / "delta"
        path.mkdir(exist_ok=True)
        for t, f in zip(attributions, filenames):
            p = (path / f.name).with_suffix(".pt")
            torch.save(t, p)


def feature_ablation(
    forward_func: ForwardFunction,
    inputs: torch.Tensor,
    filenames: tp.List[Path] = None,
    output_path: Path = None,
    verbose: bool = False,
) -> None:
    if verbose:
        print(section_header("", start_with_newline=False))
    output_path.mkdir(exist_ok=True)
    alg = FeatureAblation(forward_func)
    for mask_size in [256]:
        feature_mask = get_feature_mask(inputs, mask_size)
        for perturbations_per_eval in [1]:
            if verbose:
                print(f"{mask_size=}")
                print(f"{perturbations_per_eval=}")
            output_path_ = output_path / str(mask_size) / str(perturbations_per_eval)
            output_path_.mkdir(exist_ok=True, parents=True)
            try:
                attributions = alg.attribute(
                    inputs=inputs,
                    baselines=BASELINE,
                    target=1,
                    feature_mask=feature_mask,
                    perturbations_per_eval=perturbations_per_eval,
                    show_progress=verbose,
                )
                output_captum_attributions(output_path_, filenames, attributions)
            except Exception as e:
                print(error_line())
                print(e)
                traceback.print_exc()
                print(error_line())


def feature_permutation(
    forward_func: ForwardFunction,
    inputs: torch.Tensor,
    filenames: tp.List[Path] = None,
    output_path: Path = None,
    verbose: bool = False,
) -> None:
    if verbose:
        print(section_header("", start_with_newline=False))
    output_path.mkdir(exist_ok=True)
    alg = FeaturePermutation(forward_func)
    for mask_size in [256]:
        feature_mask = get_feature_mask(inputs, mask_size, copy_for_each_input=False)
        for perturbations_per_eval in [1]:
            if verbose:
                print(f"{mask_size=}")
                print(f"{perturbations_per_eval=}")
            output_path_ = output_path / str(mask_size) / str(perturbations_per_eval)
            output_path_.mkdir(exist_ok=True, parents=True)
            try:
                attributions = alg.attribute(
                    inputs=inputs,
                    target=1,
                    feature_mask=feature_mask,
                    perturbations_per_eval=perturbations_per_eval,
                    show_progress=verbose,
                )
                output_captum_attributions(output_path_, filenames, attributions)
            except Exception as e:
                print(error_line())
                print(e)
                traceback.print_exc()
                print(error_line())


def integrated_gradients(
    forward_func: ForwardFunction,
    inputs: torch.Tensor,
    filenames: tp.List[Path] = None,
    output_path: Path = None,
    verbose: bool = False,
) -> None:
    if verbose:
        print(section_header("", start_with_newline=False))
    output_path.mkdir(exist_ok=True)
    alg = IntegratedGradients(forward_func)
    for method in ["gausslegendre"]:
        for n_steps in [50]:
            if verbose:
                print(f"{method=}")
                print(f"{n_steps=}")
            output_path_ = output_path / method / str(n_steps)
            output_path_.mkdir(exist_ok=True, parents=True)
            try:
                attributions, delta = alg.attribute(
                    inputs=inputs,
                    baselines=BASELINE,
                    target=1,
                    n_steps=n_steps,
                    method=method,
                    return_convergence_delta=True,
                )
                output_captum_attributions(output_path_, filenames, attributions, delta)
            except Exception as e:
                print(error_line())
                print(e)
                traceback.print_exc()
                print(error_line())


def kernel_shap(
    forward_func: ForwardFunction,
    inputs: torch.Tensor,
    filenames: tp.List[Path] = None,
    output_path: Path = None,
    verbose: bool = False,
) -> None:
    if verbose:
        print(section_header("", start_with_newline=False))
    output_path.mkdir(exist_ok=True)
    alg = KernelShap(forward_func)
    for mask_size in [256]:
        feature_mask = get_feature_mask(inputs, mask_size)[0]
        for n_samples in [50]:
            for perturbations_per_eval in [1]:
                if verbose:
                    print(f"{mask_size=}")
                    print(f"{n_samples=}")
                    print(f"{perturbations_per_eval=}")
                output_path_ = (
                    output_path
                    / str(mask_size)
                    / str(n_samples)
                    / str(perturbations_per_eval)
                )
                output_path_.mkdir(exist_ok=True, parents=True)
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
                            output_path_, [filenames[i]], attributions
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
    output_path: Path = None,
    verbose: bool = False,
) -> None:
    if verbose:
        print(section_header("", start_with_newline=False))
    output_path.mkdir(exist_ok=True)
    alg = LayerActivation(forward_func, layer)
    try:
        attributions = alg.attribute(inputs=inputs)
        output_captum_attributions(output_path, filenames, attributions)
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
    output_path: Path = None,
    verbose: bool = False,
) -> None:
    if verbose:
        print(section_header("", start_with_newline=False))
    output_path.mkdir(exist_ok=True)
    alg = LayerIntegratedGradients(forward_func, layer)
    for method in ["gausslegendre"]:
        for n_steps in [50]:
            if verbose:
                print(f"{method=}")
                print(f"{n_steps=}")
            output_path_ = output_path / method / str(n_steps)
            output_path_.mkdir(exist_ok=True, parents=True)
            try:
                attributions, delta = alg.attribute(
                    inputs=inputs,
                    baselines=BASELINE,
                    target=1,
                    n_steps=n_steps,
                    method=method,
                    return_convergence_delta=True,
                )
                output_captum_attributions(output_path_, filenames, attributions, delta)
            except Exception as e:
                print(error_line())
                print(e)
                traceback.print_exc()
                print(error_line())


def occlusion(
    forward_func: ForwardFunction,
    inputs: torch.Tensor,
    filenames: tp.List[Path] = None,
    output_path: Path = None,
    verbose: bool = False,
) -> None:
    if verbose:
        print(section_header("", start_with_newline=False))
    output_path.mkdir(exist_ok=True)
    alg = Occlusion(forward_func)
    for sliding_window_shapes in [(10000,)]:
        for strides in [1]:
            for perturbations_per_eval in [1]:
                if verbose:
                    print(f"{sliding_window_shapes=}")
                    print(f"{strides=}")
                    print(f"{perturbations_per_eval=}")
                output_path_ = (
                    output_path
                    / str(sliding_window_shapes)
                    / str(strides)
                    / str(perturbations_per_eval)
                )
                output_path_.mkdir(exist_ok=True, parents=True)
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
                    output_captum_attributions(output_path_, filenames, attributions)
                except Exception as e:
                    print(error_line())
                    print(e)
                    traceback.print_exc()
                    print(error_line())


def shapley_value_sampling(
    forward_func: ForwardFunction,
    inputs: torch.Tensor,
    filenames: tp.List[Path] = None,
    output_path: Path = None,
    verbose: bool = False,
) -> None:
    if verbose:
        print(section_header("", start_with_newline=False))
    output_path.mkdir(exist_ok=True)
    alg = ShapleyValueSampling(forward_func)
    for mask_size in [256]:
        feature_mask = get_feature_mask(inputs, mask_size)
        for n_samples in [25]:
            for perturbations_per_eval in [1]:
                if verbose:
                    print(f"{mask_size=}")
                    print(f"{n_samples=}")
                    print(f"{perturbations_per_eval=}")
                output_path_ = (
                    output_path
                    / str(mask_size)
                    / str(n_samples)
                    / str(perturbations_per_eval)
                )
                output_path_.mkdir(exist_ok=True, parents=True)
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
                    output_captum_attributions(output_path_, filenames, attributions)
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
    output_path: Path = None,
    verbose: bool = False,
) -> None:

    if run_feature_ablation:
        if verbose:
            print(section_header("FeatureAblation"))
        for name, forward_func in forward_functions.items():
            if verbose:
                print(f"forward_func_name={name}")
            output_path_ = output_path / "FeatureAblation" / name
            output_path_.mkdir(exist_ok=True, parents=True)
            feature_ablation(forward_func, X, files, output_path_, verbose)

    if run_feature_permutation:
        if verbose:
            print(section_header("FeaturePermutation"))
        for name, forward_func in forward_functions.items():
            if verbose:
                print(f"forward_func_name={name}")
            output_path_ = output_path / "FeaturePermutation" / name
            output_path_.mkdir(exist_ok=True, parents=True)
            feature_permutation(forward_func, X, files, output_path_, verbose)

    if run_integrated_gradients:
        if verbose:
            print(section_header("IntegratedGradients"))
        for name, forward_func in forward_functions.items():
            if verbose:
                print(f"forward_func_name={name}")
            output_path_ = output_path / "IntegratedGradients" / name
            output_path_.mkdir(exist_ok=True, parents=True)
            integrated_gradients(forward_func, X, files, output_path_, verbose)

    if run_kernel_shap:
        if verbose:
            print(section_header("KernelShap"))
        for name, forward_func in forward_functions.items():
            if verbose:
                print(f"forward_func_name={name}")
            output_path_ = output_path / "KernelShap" / name
            output_path_.mkdir(exist_ok=True, parents=True)
            kernel_shap(forward_func, X, files, output_path_, verbose)

    if run_layer_activation:
        if verbose:
            print(section_header("LayerActivation"))
        for name, forward_func in forward_functions.items():
            if verbose:
                print(f"forward_func_name={name}")
            for layer in layers:
                output_path_ = output_path / "LayerActivation" / name / layer
                output_path_.mkdir(exist_ok=True, parents=True)
                layer_activation(
                    forward_func, getattr(model, layer), X, files, output_path_, verbose
                )

    if run_layer_integrated_gradients:
        if verbose:
            print(section_header("LayerIntegratedGradients"))
        for name, forward_func in forward_functions.items():
            if verbose:
                print(f"forward_func_name={name}")
            for layer in layers:
                output_path_ = output_path / "LayerIntegratedGradients" / name / layer
                output_path_.mkdir(exist_ok=True, parents=True)
                layer_integrated_gradients(
                    forward_func, getattr(model, layer), X, files, output_path_, verbose
                )

    if run_occlusion:
        if verbose:
            print(section_header("Occlusion"))
        for name, forward_func in forward_functions.items():
            if verbose:
                print(f"forward_func_name={name}")
            output_path_ = output_path / "Occlusion" / name
            output_path_.mkdir(exist_ok=True, parents=True)
            occlusion(forward_func, X, files, output_path_, verbose)

    if run_shapley_value_sampling:
        if verbose:
            print(section_header("ShapleyValueSampling"))
        for name, forward_func in forward_functions.items():
            if verbose:
                print(f"forward_func_name={name}")
            output_path_ = output_path / "ShapleyValueSampling" / name
            output_path_.mkdir(exist_ok=True, parents=True)
            shapley_value_sampling(forward_func, X, files, output_path_, verbose)


def explain_pretrained_malconv(
    model_name: ModelName,
    run_feature_ablation: bool = False,
    run_feature_permutation: bool = False,
    run_integrated_gradients: bool = False,
    run_kernel_shap: bool = False,
    run_layer_activation: bool = False,
    run_layer_integrated_gradients: bool = False,
    run_occlusion: bool = False,
    run_shapley_value_sampling: bool = False,
    output_path: Path = None,
    verbose: bool = False,
) -> None:
    print(section_header("Model"))
    model = get_model(model_name, verbose=verbose)

    print(section_header("Data"))
    # TODO: mind the change in API of get_data
    train_dataset, _, train_loader, _, train_sampler, _ = get_data(
        max_len=1000000, batch_size=1
    )

    print(section_header("Captum"))
    # Generally speaking we do not want to apply the softmax layer to the forward function, I think
    forward_functions = {
        f"{softmax=}": forward_function_malconv(model, softmax) for softmax in (False,)
    }
    layers = ["fc_2"]  # ["embd", "conv_1", "conv_2", "fc_1","fc_2"]
    print(pformat(f"forward_functions={list(forward_functions.keys())}"))
    print(pformat(f"{layers=}"))

    batched_files = batch(
        [Path(e[0]) for e in train_dataset.all_files], train_loader.batch_size
    )
    total = len(train_dataset) / train_loader.batch_size
    for (X, _), f in tqdm(zip(train_loader, batched_files), total=total):
        X = X.to(device)
        # TODO: this can be caused by empty files...which should be removed?
        if X.shape[1] == 0:
            continue
        try:
            explain_batch(
                model,
                X,
                f,
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
                output_path,
                verbose,
            )
        except Exception as e:
            print(error_line())
            print(f"f={pformat(f)}")
            print(f"{X.shape=}")
            print(f"{X=}")
            print(f"{e=}")
            traceback.print_exc()
            print(error_line())


if __name__ == "__main__":
    output_path = Path("outputs/gct")
    explain_pretrained_malconv(
        "gct",
        run_feature_ablation=False,
        run_feature_permutation=False,
        run_integrated_gradients=False,
        run_kernel_shap=True,
        run_layer_activation=False,
        run_layer_integrated_gradients=False,
        run_occlusion=False,
        run_shapley_value_sampling=False,
        output_path=output_path,
        verbose=False,
    )
