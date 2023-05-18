"""
Analyze the outcomes of incremental malware substitutions.

TODO:
    - Incorporate OutputHelpers instead of using raw paths
"""

from __future__ import annotations
from collections import defaultdict, OrderedDict
import json
from pathlib import Path
from pprint import pformat, pprint  # pylint: disable=unused-import
import typing as tp

import matplotlib.pyplot as plt
if tp.TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
import numpy as np

from classifier import CONFIDENCE_THRESHOLD
from modify_inc import REP_SOURCE_MODES, REP_TARGET_MODES


class Processor:
    path: tp.Optional[Path]
    files: Path
    data: tp.Dict[str, np.ndarray]

    def __init__(
        self, path: Path = None, files: tp.Iterable[Path] = None
    ) -> None:
        if bool(path) == bool(files):
            raise ValueError("Specify one or the other.")
        self.path = path
        self.files = files
        if self.files is None:
            self.files = self.path.iterdir()
        self.data = None

    def __call__(self) -> Processor:
        self.data = {f.name: np.loadtxt(f, delimiter="\n") for f in self.files}
        return self

    def to_json(self, output_file) -> Processor:
        data = OrderedDict(sorted(self.data.items(), key=lambda x: len(x[1])))
        with open(output_file, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=4)
        return self


class Analyzer:
    processed_file: tp.Optional[Path]
    data: tp.Dict[str, np.ndarray]
    chunk_size: int
    granularity: int
    fn_rates: tp.Dict[str, tp.Dict[str, tp.Tuple[np.ndarray, np.ndarray]]]
    fn_tn_summary: tp.Dict[str, tp.Tuple[float, float]]

    def __init__(self, processed_file: Path = None, data: tp.Dict[str, np.ndarray] = None, chunk_size: int = 1, granularity: int = 100) -> None:
        if bool(processed_file) == bool(data):
            raise ValueError("Specify one or the other.")
        self.processed_file = processed_file
        self.data = data
        if self.data is None:
            with open(self.processed_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            self.data = {f: np.array(v) for f, v in data.items()}
        self.chunk_size = chunk_size
        self.granularity = granularity
        self.fn_rates = None
        self.fn_tn_summary = None

    def __call__(self) -> Analyzer:
        error = "ERROR: last should always achieve a lower/equal fn rate than any"

        self.fn_tn_summary = {}
        for m in ("initial", "last", "any"):
            self.fn_tn_summary[m] = self.compute_negative_rates(self.data, m)
        assert self.fn_tn_summary["initial"][0] <= self.fn_tn_summary["last"][0], error
        assert self.fn_tn_summary["last"][0] <= self.fn_tn_summary["any"][0], error

        self.fn_rates = defaultdict(dict)
        for m in ("last", "any"):
            self.fn_rates["num"][m] = self.fn_rate_vs_num_bytes(self.data, self.chunk_size, m)
            self.fn_rates["pro"][m] = self.fn_rate_vs_pro_bytes(self.data, self.granularity, m)

        assert all(
            l <= a
            for l, a in
            zip(self.fn_rates["num"]["last"][1], self.fn_rates["num"]["any"][1])
        ), error
        assert all(
            l <= a
            for l, a in
            zip(self.fn_rates["pro"]["last"][1], self.fn_rates["pro"]["any"][1])
        ), error

        return self

    @staticmethod
    def compute_negative_rates(
        data: tp.Dict[str, np.ndarray],
        mode: tp.Literal["first", "last", "any"],
    ) -> tp.Tuple[float, float]:
        if mode == "initial":
            fn = sum(1 for confs in data.values() if confs[0] < CONFIDENCE_THRESHOLD)
        elif mode == "any":
            fn = sum(1 for confs in data.values() if (confs < CONFIDENCE_THRESHOLD).any())
        elif mode == "last":
            fn = sum(1 for confs in data.values() if confs[-1] < CONFIDENCE_THRESHOLD)
        else:
            raise ValueError(f"Invalid mode, {mode=}")
        tn = len(data) - fn
        return fn, tn

    @staticmethod
    def fn_rate_vs_pro_bytes(
        data: tp.Dict[str, np.ndarray],
        granularity: float = 100,
        mode: tp.Literal["last", "any"] = "any",
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        n_neg_clfs = np.zeros(granularity)
        for p in range(granularity):
            for confs in data.values():
                itr = p * len(confs) // granularity
                if mode == "any":
                    if any(c < CONFIDENCE_THRESHOLD for c in confs[:itr+1]):
                        n_neg_clfs[p] += 1
                elif mode == "last":
                    if confs[itr] < CONFIDENCE_THRESHOLD:
                        n_neg_clfs[p] += 1
                else:
                    raise ValueError(f"Invalid mode: {mode}")

        pro_bytes = np.arange(granularity) / granularity
        p_neg_clfs = n_neg_clfs / len(data)
        assert len(pro_bytes) == len(p_neg_clfs)
        return pro_bytes, p_neg_clfs

    @staticmethod
    def fn_rate_vs_num_bytes(
        data: tp.Dict[str, np.ndarray],
        chunk_size: int = 1,
        mode: tp.Literal["last", "any"] = "any",
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        lengths = {k: len(confs) for k, confs in data.items()}
        max_iterations = max(lengths.values())
        n_neg_clfs = np.zeros(max_iterations)
        for itr in range(max_iterations):
            for k, confs in data.items():
                if mode == "any":
                    if any(c < CONFIDENCE_THRESHOLD for c in confs[:itr + 1]):
                        n_neg_clfs[itr] += 1
                elif mode == "last":
                    c = confs[itr] if itr < lengths[k] else confs[-1]
                    if c < CONFIDENCE_THRESHOLD:
                        n_neg_clfs[itr] += 1
                else:
                    raise ValueError(f"Invalid mode: {mode}")

        num_bytes = chunk_size * np.arange(max_iterations)
        p_neg_clfs = n_neg_clfs / len(data)
        assert len(num_bytes) == len(p_neg_clfs)
        return num_bytes, p_neg_clfs


class Plotter:
    fig: Figure
    ax: Axes

    def __init__(self) -> None:
        self.fig, self.ax = plt.subplots()

    def __call__(self, x: tp.Iterable[float], y: tp.Iterable[float], **kwargs) -> Plotter:
        self.ax.plot(x, y, **kwargs)
        return self

    def markup(
        self,
        title: str = "FN Rate vs Bytes Swapped",
        x_label: str = "Bytes Swapped",
        y_label: str = "FN Rate",
        y_lim_l: int = 0,
        y_lim_u: int = 1,
    ) -> Plotter:
        self.ax.set_title(title)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.set_ylim([y_lim_l, y_lim_u])
        
        self.ax.legend()
        self.ax.grid()
        
        self.ax.set_yticks(ticks=self.ax.get_yticks()[1:-1], labels=[f"{int(x)}%" for x in self.ax.get_yticks()[1:-1]])
        self.ax.set_xticks(ticks=self.ax.get_yticks()[1:-1], labels=[f"{int(x)}%" for x in self.ax.get_xticks()[1:-1]])
    
        
        return self

    def save(self, outfile: Path = "FN_Rate_VS_Bytes_Swapped.png", **kwargs) -> Plotter:
        self.fig.savefig(outfile, **kwargs)
        return self

    def close(self) -> Plotter:
        plt.close(self.fig)
        return self


class BoxPlotter(Plotter):

    def __call__(self, x: tp.Iterable[float], y: tp.Iterable[float], **kwargs) -> Plotter:
        return self


def compare(
    paths: tp.Iterable[Path], labels: tp.Iterable[str], chunk_size: int
) -> tp.Tuple[tp.List[Processor], tp.List[Analyzer], tp.Dict[str, tp.Dict[str, Plotter]], tp.Dict[str, tp.Dict[str, BoxPlotter]]]:
    plotters = defaultdict(lambda: defaultdict(Plotter))
    boxplotters = defaultdict(lambda: defaultdict(BoxPlotter))
    processors = []
    analyzers = []
    for path, label in zip(paths, labels):
        processor = Processor(path=path)()
        analyzer = Analyzer(data=processor.data, chunk_size=chunk_size)()
        for m_1 in analyzer.fn_rates.keys():
            for m_2 in analyzer.fn_rates[m_1].keys():
                x, y = analyzer.fn_rates[m_1][m_2]
                plotters[m_1][m_2](x, y, label=label)
                boxplotters[m_1][m_2](x, y, label=label)
        analyzers.append(analyzer)
        processors.append(processor)

    return processors, analyzers, plotters, boxplotters


def save_plotters(output_dir: Path, plotters: tp.Dict[str, tp.Dict[str, Plotter]]) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)
    for m_1 in plotters.keys():
        for m_2 in plotters[m_1].keys():
            # title = f"FN Rate vs {m_1.capitalize()} Bytes Swapped ({m_2.capitalize()})"
            # x_label = f"{m_1.capitalize()} Bytes Swapped"
            stem = f"{m_1}_{m_2}_box" if isinstance(plotters[m_1][m_2], BoxPlotter) else f"{m_1}_{m_2}_plt"
            outfile = (output_dir / stem).with_suffix(".svg")
            plotters[m_1][m_2].markup(title="", x_label="", y_label="")
            plotters[m_1][m_2].save(outfile)
            plotters[m_1][m_2].close()


def run(output_root: Path, chunk_size: int) -> None:

    print(f"Working on experiment located in {output_root=}")

    print(f"Holding source mode constant. Comparing {REP_TARGET_MODES=}")
    for s_m in REP_SOURCE_MODES:
        paths = [
            output_root / f"rep_source_mode__{s_m}" / f"rep_target_mode__{t_m}"
            for t_m in REP_TARGET_MODES
        ]
        paths = [p for p in paths if p.exists()]
        if not paths:
            continue
        _, _, plotters, boxplotters = compare(paths, REP_TARGET_MODES, chunk_size)
        save_plotters(output_root / "plots" / "target_modes" / s_m, plotters)
        save_plotters(output_root / "plots" / "target_modes" / s_m, boxplotters)

    print(f"Holding target mode constant. Comparing {REP_SOURCE_MODES=}")
    for t_m in REP_TARGET_MODES:
        paths = [
            output_root / f"rep_source_mode__{s_m}" / f"rep_target_mode__{t_m}"
            for s_m in REP_SOURCE_MODES
        ]
        paths = [p for p in paths if p.exists()]
        if not paths:
            continue
        _, _, plotters, boxplotters = compare(paths, REP_SOURCE_MODES, chunk_size)
        save_plotters(output_root / "plots" / "source_modes" / t_m, plotters)
        save_plotters(output_root / "plots" / "target_modes" / s_m, boxplotters)


def main():
    output_root = Path(
        "outputs/modify/model_name__gct/softmax__False/layer__None/"
        "alg__FeatureAblation/baselines__0/feature_mask_mode__text/"
        f"feature_mask_size__256/method__None/n_steps__None/"
        "perturbations_per_eval__None/sliding_window_shapes_size__None/"
        "strides__None/target__1"
    )
    run(output_root, 256)

    output_root = Path(
        "outputs/modify/model_name__gct/softmax__False/layer__None/"
        "alg__KernelShap/baselines__0/feature_mask_mode__text/"
        f"feature_mask_size__64/method__None/n_steps__None/"
        "perturbations_per_eval__None/sliding_window_shapes_size__None/"
        "strides__None/target__1"
    )
    run(output_root, 64)


if __name__ == "__main__":
    main()
