"""

"""

from argparse import ArgumentParser
from collections import defaultdict
from configparser import ConfigParser, SectionProxy
from pathlib import Path
from pprint import pformat, pprint
import typing as tp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from classifier import CONFIDENCE_THRESHOLD
import cfg
from utils import get_outfile

def plot_proportion_negative_vs_swap_absolute(
    data: tp.Dict[str, np.ndarray],
    chunk_size: int,
    mode: tp.Literal["last", "any"],
    **outfile_kwargs,
) -> tp.Tuple[np.ndarray, np.ndarray]:
    lengths = {k: len(confs) for k, confs in data.items()}
    max_iterations = max(lengths.values())
    n_neg_clfs = np.zeros(max_iterations)
    for itr in range(max_iterations):
        for k, confs in sorted(data.items(), key=lambda x: x[0]):
            if mode == "any":
                if any(c < CONFIDENCE_THRESHOLD for c in confs[:itr]):
                    n_neg_clfs[itr] += 1
            elif mode == "last":
                c = confs[itr] if itr < lengths[k] else confs[-1]
                if c < CONFIDENCE_THRESHOLD:
                    n_neg_clfs[itr] += 1
            else:
                raise ValueError(f"Invalid mode: {mode}")

    p_neg_clfs = n_neg_clfs / len(data)

    # Plot one
    fig, ax = plt.subplots()
    x = chunk_size * np.arange(max_iterations)
    y = 100 * p_neg_clfs
    ax.plot(x, y)
    ax.set_ylim([0, 100])
    ax.set_title("False Negative Classification vs Bytes Swapped")
    ax.set_xlabel("Bytes Swapped in .text Section")
    ax.set_ylabel("Percent Classified as Benign")

    if "outfile_stem" not in outfile_kwargs:
        outfile_kwargs["outfile_stem"] = f"proportion_negative_vs_swap_absolute_{mode}"
    if "outfile_suffix" not in outfile_kwargs:
        outfile_kwargs["outfile_suffix"] = ".png"
    fig.savefig(get_outfile(**outfile_kwargs), dpi=400)
    plt.close(fig)

    return x, y


def plot_proportion_negative_vs_swap_proportion(
    data: tp.Dict[str, np.ndarray],
    mode: tp.Literal["last", "any"],
    granularity: int = 100,
    **outfile_kwargs,
) -> tp.Tuple[np.ndarray, np.ndarray]:
    n_neg_clfs = np.zeros(granularity)

    for p in range(granularity):
        for k, confs in sorted(data.items(), key=lambda x: x[0]):
            itr = p * len(confs) // granularity
            if mode == "any":
                if any(c < CONFIDENCE_THRESHOLD for c in confs[:itr]):
                    n_neg_clfs[p] += 1
            elif mode == "last":
                c = confs[itr]
                if c < CONFIDENCE_THRESHOLD:
                    n_neg_clfs[p] += 1
            else:
                raise ValueError(f"Invalid mode: {mode}")

    p_neg_clfs = n_neg_clfs / len(data)

    # Plot one
    fig, ax = plt.subplots()
    x = np.arange(granularity)
    y = 100 * p_neg_clfs
    ax.plot(x, y)
    ax.set_ylim([0, 100])
    ax.set_title("False Negative Classification vs Bytes Swapped")
    ax.set_xlabel("Percent Bytes Swapped in .text Section")
    ax.set_ylabel("Percent Classified as Benign")

    if "outfile_stem" not in outfile_kwargs:
        outfile_kwargs["outfile_stem"] = f"proportion_negative_vs_swap_proportion_{mode}"
    if "outfile_suffix" not in outfile_kwargs:
        outfile_kwargs["outfile_suffix"] = ".png"
    fig.savefig(get_outfile(**outfile_kwargs), dpi=400)
    plt.close(fig)

    return x, y


def analyze_inc(
    confidences_path: Path,
    analysis_path: Path,
    run_modes: tp.List[str],
    replace_modes: tp.List[str],
    chunk_size: int,
    only_common_files: bool = False,
) -> None:
    # Check if the different experiments are using the same files and get common files if needed
    files = set(f.name for f in (confidences_path / run_modes[0] / replace_modes[0]).iterdir())
    common_files = set(files) if only_common_files else None
    for m in run_modes:
        for r in replace_modes:
            files_ = set(f.name for f in (confidences_path / m / r).iterdir())
            if files != files_:
                print(
                    "WARNING: files sets are not the same across modes and replace modes."
                    f"\n\t{only_common_files=}."
                    f"\n\t{m}/{r} has {len(files_)} files."
                )
            common_files = common_files.intersection(files_) if only_common_files else None

    for m in tqdm(run_modes):
        # The x and y axes for the individual plots
        x, y = defaultdict(lambda: defaultdict(dict)), defaultdict(lambda: defaultdict(dict))

        for r in tqdm(replace_modes, leave=False):
            output_path = analysis_path / m / r
            output_path.mkdir(parents=True, exist_ok=True)
            # Load data into memory
            data = {}
            files = (confidences_path / m / r).iterdir()
            files = list(f for f in files if not common_files or f.name in common_files)
            for f in tqdm(files, leave=False):
                confs = np.loadtxt(f, delimiter="\n")
                confs = np.expand_dims(confs, axis=0) if not confs.shape else confs
                data[f.stem] = confs
            # Compute and save statistics
            init_tp, init_fp, avg_conf_diff, prop_flipped = 0, 0, 0, 0
            for confs in data.values():
                if len(confs) > 1 and confs[0] > CONFIDENCE_THRESHOLD:
                    avg_conf_diff += confs[-1] - confs[0]
                    init_tp += 1
                    if confs[-1] < CONFIDENCE_THRESHOLD:
                        prop_flipped += 1
                else:
                    init_fp += 1
            pd.DataFrame(
                {
                    "init_tp": [init_tp],
                    "init_fp": [init_fp],
                    "avg_conf_diff": [avg_conf_diff],
                    "prop_flipped": [prop_flipped],
                }
            ).to_csv(output_path / "statistics.csv")
            # Generate plots and collect the x and y values for cumulative plotting
            for abs_or_pro in ["absolute", "proportion"]:
                for any_or_last in ["any", "last"]:
                    if abs_or_pro == "absolute":
                        x_, y_ = plot_proportion_negative_vs_swap_absolute(
                            data,
                            chunk_size,
                            any_or_last,
                            outfile_parent=output_path,
                        )
                    elif abs_or_pro == "proportion":
                        x_, y_ = plot_proportion_negative_vs_swap_proportion(
                            data,
                            any_or_last,
                            outfile_parent=output_path,
                        )
                    x[r][abs_or_pro][any_or_last] = x_
                    y[r][abs_or_pro][any_or_last] = y_

        # Generate the cumulative statistics and plots
        output_path = analysis_path / m / "all"
        output_path.mkdir(parents=True, exist_ok=True)
        for abs_or_pro in ["absolute", "proportion"]:
            for any_or_last in ["any", "last"]:
                fig, ax = plt.subplots()
                for r in replace_modes:
                    ax.plot(x[r][abs_or_pro][any_or_last], y[r][abs_or_pro][any_or_last], label=r)
                ax.set_title("False Negative Classification vs Bytes Swapped")
                if abs_or_pro == "absolute":
                    xlabel = "Bytes Swapped in .text Section"
                elif abs_or_pro == "proportion":
                    xlabel = "Percent Bytes Swapped in .text Section"
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Percent Classified as Benign")
                ax.set_ylim([0, 100])
                ax.legend()
                outfile_stem = f"proportion_negative_vs_swap_{abs_or_pro}_{any_or_last}"
                outfile = get_outfile(
                    outfile_parent=output_path, outfile_stem=outfile_stem, outfile_suffix=".png"
                )
                fig.savefig(outfile, dpi=400)
                plt.close(fig)


def analyze_multi_full(
    analysis_path: Path,
    modes_paths: tp.Dict[str, Path],
):
    for m, p in tqdm(list(modes_paths.items())):
        # mal_files = list(p.iterdir())
        # sub_files = pd.read_csv(mal_files[0])["substitute"].tolist()
        # confs = [pd.read_csv(f)["confidences"].numpy() for f in mal_files]
        # # Each row corresponds to a different substitute file
        # confs = np.stack(confs, axis=0)
        outpath = analysis_path / m
        outpath.mkdir(exist_ok=True)

        cum_df, flipped_df = None, None
        n_total, n_cls_pos = 0, 0
        for f in tqdm(list(p.iterdir()), leave=False):
            n_total += 1
            df = pd.read_csv(f, index_col=0)  # Uses the filenames as the index
            cum_df = df if cum_df is None else cum_df + df
            # The NaN index corresponds to the initial confidence
            if df.loc[df.index.isnull()]["confidences"].item() > CONFIDENCE_THRESHOLD:
                n_cls_pos += 1
                cls_neg_df = -1 * (df.round().astype(np.int16) - 1)
                flipped_df = cls_neg_df if flipped_df is None else flipped_df + cls_neg_df

        avg_df = cum_df / n_total
        if avg_df.isnull().values.any():
            raise ValueError("NaNs in the average dataframe!")
        avg_df.sort_values(by=["confidences"]).to_csv(outpath / "average.csv")

        print(flipped_df)
        avg_flipped_df = flipped_df / n_cls_pos
        if avg_flipped_df.isnull().values.any():
            raise ValueError("NaNs in the average dataframe!")
        avg_flipped_df.sort_values(by=["confidences"]).to_csv(outpath / "average_flipped.csv")


def analyze(run_flags: SectionProxy, params: SectionProxy) -> None:
    output_path = get_output_path(params)
    toolkit = params.get("toolkit")
    chunk_size = params.getint("chunk_size")

    inc_run_modes = [m for m in INC_MODES if run_flags.getboolean("run_" + m, False)]
    inc_replace_modes = params.get("inc_replace_modes").split(",")

    confidences_path = output_path / "confidences" / toolkit
    analysis_path = confidences_path / "analysis"
    analysis_path.mkdir(exist_ok=True)

    analyze_inc(
        confidences_path,
        analysis_path,
        inc_run_modes,
        inc_replace_modes,
        chunk_size,
        only_common_files=True,
    )

    modes_paths = {
        m: confidences_path / m for m in MULTI_FULL_MODES if run_flags.getboolean("run_" + m, False)
    }
    if modes_paths:
        for m in modes_paths:
            (analysis_path / m).mkdir(exist_ok=True)
        analyze_multi_full(analysis_path, modes_paths, chunk_size)


def _verify_inc_and_full_same():
    # Only differences should be in the random method
    root = Path("outputs/7/KernelShap/False/256/50/1/confidences/pefile")
    paths = {
        "baseline": (root / "inc_baseline", root / "full_baseline"),
        "random": (root / "inc_random", root / "full_random"),
        "benign": (
            root / "inc_benign_corresponding",
            root / "full_benign_corresponding",
        ),
    }
    data = {k: [] for k in paths}
    for k, (inc_path, full_path) in paths.items():
        for f_inc in inc_path.iterdir():
            f_full = full_path / f_inc.name
            inc = np.loadtxt(f_inc, delimiter="\n")
            full = np.loadtxt(f_full, delimiter="\n")
            if not inc.shape:
                print(f_inc)
            if not full.shape:
                print(f_full)
            inc = np.expand_dims(inc, 0) if not inc.shape else inc
            full = np.expand_dims(full, 0) if not inc.shape else inc
            if inc[-1] != full[-1]:
                data[k].append((f_inc.name, inc, full))
    return data


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, default="config_files/modify_malware/default.ini"
    )
    args = parser.parse_args()

    config = ConfigParser(allow_no_value=True)
    config.read(args.config_file)

    run_flags = config["RUN_FLAGS"]
    control = config["CONTROL"]
    params = config["PARAMS"]

    p = config["CONTROL"]
    cfg.init(p.get("device"), p.getint("seed"))

    if run_flags.getboolean("run"):
        run(run_flags, params, control)
    if run_flags.getboolean("analyze"):
        results = analyze(run_flags, params)
        pprint(results)
