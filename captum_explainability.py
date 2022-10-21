"""

"""

from pathlib import Path
import pickle
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
import torch
import torch.nn as nn

from utils import section_header, error_line


ForwardFunction = tp.Callable[[torch.Tensor], torch.Tensor]


device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
baseline = 0


def get_feature_mask(
        inputs: torch.Tensor,
        mask_size: int = 256,
        copy_for_each_input: bool = True,
) -> torch.Tensor:
    q, r = divmod(inputs.shape[1], mask_size)
    feature_mask = torch.cat([torch.full((mask_size,), i) for i in range(q)])
    feature_mask = torch.cat([feature_mask, torch.full((r,), q)])
    if copy_for_each_input:
        feature_mask = torch.cat([feature_mask.unsqueeze(0) for _ in range(inputs.shape[0])], 0)
    else:
        feature_mask = feature_mask.unsqueeze(0)
    feature_mask = feature_mask.type(torch.int64).to(device)
    return feature_mask


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


def integrated_gradients(
        forward_func: ForwardFunction,
        inputs: torch.Tensor,
        filenames: tp.List[Path] = None,
        output_path: Path = None,
        verbose: bool = False,
) -> None:
    print(section_header("", start_with_newline=False))
    # output_path = output_path / "IntegratedGradients"
    output_path.mkdir(exist_ok=True)
    alg = IntegratedGradients(forward_func)
    for method in ["gausslegendre"]:
        for n_steps in [50]:
            print(f"{method=}")
            print(f"{n_steps=}")
            output_path_ = output_path / method / str(n_steps)
            output_path_.mkdir(exist_ok=True, parents=True)
            try:
                attributions, delta = alg.attribute(
                    inputs=inputs,
                    baselines=baseline,
                    target=1,
                    n_steps=n_steps,
                    method=method,
                    return_convergence_delta=True
                )
                # output_captum_results(
                #     output_path_,
                #     {"attributions.pt": attributions, "delta.pt": delta},
                #     verbose=verbose,
                # )
                output_captum_attributions(output_path_, filenames, attributions, delta)
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
    print(section_header("", start_with_newline=False))
    # output_path = output_path / "LayerIntegratedGradients"
    output_path.mkdir(exist_ok=True)
    alg = LayerIntegratedGradients(forward_func, layer)
    for method in ["gausslegendre"]:
        for n_steps in [50]:
            print(f"{method=}")
            print(f"{n_steps=}")
            output_path_ = output_path / method / str(n_steps)
            output_path_.mkdir(exist_ok=True, parents=True)
            try:
                attributions, delta = alg.attribute(
                    inputs=inputs,
                    baselines=baseline,
                    target=1,
                    n_steps=n_steps,
                    method=method,
                    return_convergence_delta=True
                )
                # output_captum_results(
                #     output_path_,
                #     {"attributions.pt": attributions, "delta.pt": delta},
                #     verbose=verbose,
                # )
                output_captum_attributions(output_path_, filenames, attributions, delta)
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
    print(section_header("", start_with_newline=False))
    # output_path = output_path / "LayerActivation"
    output_path.mkdir(exist_ok=True)
    alg = LayerActivation(forward_func, layer)
    try:
        attributions = alg.attribute(inputs=inputs)
        # output_captum_results(
        #     output_path,
        #     {"attributions.pt": attributions},
        #     verbose=verbose,
        # )
        output_captum_attributions(output_path, filenames, attributions)
    except Exception as e:
        print(error_line())
        print(e)
        traceback.print_exc()
        print(error_line())


def captum_feature_ablation(
        forward_func: ForwardFunction,
        inputs: torch.Tensor,
        filenames: tp.List[Path] = None,
        output_path: Path = None,
        verbose: bool = False,
) -> None:
    print(section_header("", start_with_newline=False))
    # output_path = output_path / "FeatureAblation"
    output_path.mkdir(exist_ok=True)
    alg = FeatureAblation(forward_func)
    for mask_size in [256]:
        feature_mask = get_feature_mask(inputs, mask_size)
        for perturbations_per_eval in [1]:
            print(f"{mask_size=}")
            print(f"{perturbations_per_eval=}")
            output_path_ = output_path / str(mask_size) / str(perturbations_per_eval)
            output_path_.mkdir(exist_ok=True, parents=True)
            try:
                attributions = alg.attribute(
                    inputs=inputs,
                    baselines=baseline,
                    target=1,
                    feature_mask=feature_mask,
                    perturbations_per_eval=perturbations_per_eval,
                    show_progress=verbose,
                )
                # output_captum_results(
                #     output_path_,
                #     {"attributions.pt": attributions},
                #     verbose=verbose,
                # )
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
    print(section_header("", start_with_newline=False))
    # output_path = output_path / "FeaturePermutation"
    output_path.mkdir(exist_ok=True)
    alg = FeaturePermutation(forward_func)
    for mask_size in [256]:
        feature_mask = get_feature_mask(inputs, mask_size, copy_for_each_input=False)
        for perturbations_per_eval in [1]:
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
                # output_captum_results(
                #     output_path_,
                #     {"attributions.pt": attributions},
                #     verbose=verbose,
                # )
                output_captum_attributions(output_path_, filenames, attributions)
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
    print(section_header("", start_with_newline=False))
    # output_path = output_path / "Occlusion"
    output_path.mkdir(exist_ok=True)
    alg = Occlusion(forward_func)
    for sliding_window_shapes in [(10000,)]:
        for strides in [1]:
            for perturbations_per_eval in [1]:
                print(f"{sliding_window_shapes=}")
                print(f"{strides=}")
                print(f"{perturbations_per_eval=}")
                output_path_ = output_path / str(sliding_window_shapes) / str(strides) / str(perturbations_per_eval)
                output_path_.mkdir(exist_ok=True, parents=True)
                try:
                    attributions = alg.attribute(
                        inputs=inputs,
                        sliding_window_shapes=sliding_window_shapes,
                        strides=strides,
                        baselines=baseline,
                        target=1,
                        perturbations_per_eval=perturbations_per_eval,
                        show_progress=verbose,
                    )
                    # output_captum_results(
                    #     output_path_,
                    #     {"attributions.pt": attributions},
                    #     verbose=verbose,
                    # )
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
    print(section_header("", start_with_newline=False))
    # output_path = output_path / "ShapleyValueSampling"
    output_path.mkdir(exist_ok=True)
    alg = ShapleyValueSampling(forward_func)
    for mask_size in [256]:
        feature_mask = get_feature_mask(inputs, mask_size)
        for n_samples in [25]:
            for perturbations_per_eval in [1]:
                print(f"{mask_size=}")
                print(f"{n_samples=}")
                print(f"{perturbations_per_eval=}")
                output_path_ = output_path / str(mask_size) / str(n_samples) / str(perturbations_per_eval)
                output_path_.mkdir(exist_ok=True, parents=True)
                try:
                    attributions = alg.attribute(
                        inputs=inputs,
                        baselines=baseline,
                        target=1,
                        feature_mask=feature_mask,
                        n_samples=n_samples,
                        perturbations_per_eval=perturbations_per_eval,
                        show_progress=verbose,
                    )
                    # output_captum_results(
                    #     output_path_,
                    #     {"attributions.pt": attributions},
                    #     verbose=verbose,
                    # )
                    output_captum_attributions(output_path_, filenames, attributions)
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
    print(section_header("", start_with_newline=False))
    # output_path = output_path / "KernelShap"
    output_path.mkdir(exist_ok=True)
    alg = KernelShap(forward_func)
    for mask_size in [256]:
        feature_mask = get_feature_mask(inputs, mask_size)[0]
        for n_samples in [50]:
            for perturbations_per_eval in [1]:
                print(f"{mask_size=}")
                print(f"{n_samples=}")
                print(f"{perturbations_per_eval=}")
                output_path_ = output_path / str(mask_size) / str(n_samples) / str(perturbations_per_eval)
                output_path_.mkdir(exist_ok=True, parents=True)
                # It is recommended to only provide a single example as input
                for i in range(inputs.shape[0]):
                    try:
                        attributions = alg.attribute(
                            inputs=inputs[i].unsqueeze(0),
                            baselines=baseline,
                            target=1,
                            feature_mask=feature_mask,
                            n_samples=n_samples,
                            perturbations_per_eval=perturbations_per_eval,
                            show_progress=verbose,
                        )
                        # output_captum_results(
                        #     output_path_,
                        #     {file_name: attributions},
                        #     verbose=verbose,
                        # )
                        output_captum_attributions(output_path_, [filenames[i]], attributions)
                    except Exception as e:
                        print(error_line())
                        print(e)
                        traceback.print_exc()
                        print(error_line())
