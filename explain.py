"""

"""

from pathlib import Path
import pickle
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
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F

from MalConvGCT_nocat import MalConvGCT

from classifier import get_model, get_data, forward_function_malconv, MALCONV_PATH
from config import device
from utils import batch, error_line, section_header


ForwardFunction = tp.Callable[[torch.Tensor], torch.Tensor]
BASELINE = 0


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
                    baselines=BASELINE,
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
                    baselines=BASELINE,
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
                    baselines=BASELINE,
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
                        baselines=BASELINE,
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
                        baselines=BASELINE,
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
                            baselines=BASELINE,
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


def integrated_gradients(
        model: MalConvGCT,
        inputs: torch.Tensor,
        target: int = 1,
        baseline: int = BASELINE,
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
            integrated_gradients(forward_func, X, files, output_path_, verbose)

    if run_layer_integrated_gradients:
        print(section_header("LayerIntegratedGradients"))
        for name, forward_func in forward_functions.items():
            print(f"forward_func_name={name}")
            for layer in layers:
                output_path_ = output_path / "LayerIntegratedGradients" / name / layer
                output_path_.mkdir(exist_ok=True, parents=True)
                layer_integrated_gradients(forward_func, getattr(model, layer), X, files, output_path_, verbose)

    if run_layer_activation:
        print(section_header("LayerActivation"))
        for name, forward_func in forward_functions.items():
            print(f"forward_func_name={name}")
            for layer in layers:
                output_path_ = output_path / "LayerActivation" / name / layer
                output_path_.mkdir(exist_ok=True, parents=True)
                layer_activation(forward_func, getattr(model, layer), X, files, output_path_, verbose)

    if run_feature_permutation:
        print(section_header("FeaturePermutation"))
        for name, forward_func in forward_functions.items():
            print(f"forward_func_name={name}")
            output_path_ = output_path / "FeaturePermutation" / name
            output_path_.mkdir(exist_ok=True, parents=True)
            feature_permutation(forward_func, X, files, output_path_, verbose)

    if run_feature_ablation:
        print(section_header("FeatureAblation"))
        for name, forward_func in forward_functions.items():
            print(f"forward_func_name={name}")
            output_path_ = output_path / "FeatureAblation" / name
            output_path_.mkdir(exist_ok=True, parents=True)
            captum_feature_ablation(forward_func, X, files, output_path_, verbose)

    if run_occlusion:
        print(section_header("Occlusion"))
        for name, forward_func in forward_functions.items():
            print(f"forward_func_name={name}")
            output_path_ = output_path / "Occlusion" / name
            output_path_.mkdir(exist_ok=True, parents=True)
            occlusion(forward_func, X, files, output_path_, verbose)

    if run_shapley_value_sampling:
        print(section_header("ShapleyValueSampling"))
        for name, forward_func in forward_functions.items():
            print(f"forward_func_name={name}")
            output_path_ = output_path / "ShapleyValueSampling" / name
            output_path_.mkdir(exist_ok=True, parents=True)
            shapley_value_sampling(forward_func, X, files, output_path_, verbose)

    if run_kernel_shap:
        print(section_header("KernelShap"))
        for name, forward_func in forward_functions.items():
            print(f"forward_func_name={name}")
            output_path_ = output_path / "KernelShap" / name
            output_path_.mkdir(exist_ok=True, parents=True)
            kernel_shap(forward_func, X, files, output_path_, verbose)


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

    print(section_header("Captum"))
    forward_functions = {
        f"{softmax=}": forward_function_malconv(model, softmax)
        for softmax in (False,)
    }
    layers = ["fc_2"] # ["embd", "conv_1", "conv_2", "fc_1","fc_2"]
    print(pformat(f"forward_functions={list(forward_functions.keys())}"))
    print(pformat(f"{layers=}"))

    files = [Path(e[0]) for e in train_dataset.all_files]
    for (X, _), f in tqdm(zip(train_loader, batch(files, train_loader.batch_size))):
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


if __name__ == "__main__":
    output_path = Path("outputs/8")
    explain_pretrained_malconv(
        MALCONV_PATH,
        # run_integrated_gradients=True,
        # run_layer_integrated_gradients=True,
        # run_layer_activation=True,
        # run_feature_permutation=True,
        # run_feature_ablation=True,
        # run_occlusion=True,
        # run_shapley_value_sampling=True,
        run_kernel_shap=True,
        output_path=output_path,
        verbose=True,
    )
