"""

"""

import multiprocessing.pool as mpp
from pathlib import Path
from pprint import pformat
import traceback
import typing as tp

from torch import Tensor

Pathlike = tp.Union[str, Path]


def section_header(name: str, start_with_newline: bool = True, underline_length: int = 88) -> None:
    s = "\n" if start_with_newline else ""
    s += "-" * underline_length + "\n"
    s += " " + "-" * int(((underline_length - len(name)) // 2)) + " "
    s += f"{name}"
    s += "-" * int(((underline_length - len(name)) / 2)) + "\n"
    s += "-" * underline_length
    return s


def error_line(l_r_buffer: int = 40) -> None:
    return "-" * l_r_buffer + " ERROR " + "-" * l_r_buffer


# TODO: adjust for lazy inputs
def batch(iterable: tp.Iterable, n: int = 1, l: int = None) -> tp.Iterable[tp.List]:
    l = len(iterable) if l is None else l
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def ceil_divide(a: float, b: float) -> int:
    return int(-(a // -b))


def get_outfile(
    outfile: Pathlike = None,
    outfile_parent: Pathlike = ".",
    outfile_stem: str = "outfile",
    outfile_prefix: str = "",
    outfile_postfix: str = "",
    outfile_suffix: str = ".out",
) -> Path:
    if outfile is not None:
        return Path(outfile)
    outfile_prefix = outfile_prefix + "_" if outfile_prefix else ""
    outfile_postfix = "_" + outfile_postfix if outfile_postfix else ""
    return (Path(outfile_parent) / f"{outfile_prefix}{outfile_stem}{outfile_postfix}").with_suffix(
        outfile_suffix
    )


def sorted_dict(d: tp.Dict[str, float]) -> tp.List[tp.Tuple[str, float]]:
    return sorted(d.items(), key=lambda x: x[0])


def istarmap(self, func: tp.Callable, iterable: tp.Iterable, chunksize: int = 1) -> tp.Generator:
    # For Python >= 3.8
    # https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm
    self._check_running()
    # if chunksize < 1:
    #     raise ValueError("Chunksize must be 1+, not {0:n}".format(chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job, mpp.starmapstar, task_batches),
            result._set_length,
        )
    )
    return (item for chunk in result for item in chunk)


def istarmap_(self, func: tp.Callable, iterable: tp.Iterable, chunksize: int = 1) -> tp.Generator:
    # For Python <= 3.7
    # https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm
    if self._state != mpp.RUN:
        raise ValueError("Pool not running")

    if chunksize < 1:
        raise ValueError("Chunksize must be 1+, not {0:n}".format(chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self._cache)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job, mpp.starmapstar, task_batches),
            result._set_length,
        )
    )
    return (item for chunk in result for item in chunk)


# TODO: replace with itertools.dropwhile
def consume_until(
    gen: tp.Generator,
    skip_idx: int = None,
    skip_val: str = None,
    process_element: tp.Callable = None,
) -> None:
    if skip_idx is None and (skip_val is None or process_element is None):
        raise ValueError("Either skip_idx or skip_val and process_element must be specified.")

    for i, v in enumerate(gen):
        if i == skip_idx:
            return
        if process_element is not None and process_element(v) == skip_val:
            return


def exception_info(e: Exception, locals_: tp.Dict[str, tp.Any]) -> str:
    locals_.update(
        **{f"k.shape": v.shape for k, v in locals_.items() if hasattr(v, "shape")},
        **{f"k.dtype": v.dtype for k, v in locals_.items() if hasattr(v, "dtype")},
        **{f"len(k)": len(v) for k, v in locals_.items() if hasattr(v, "__len__")},
    )
    s = "\n".join(
        [
            error_line(),
            str(e),
            traceback.format_exc(),
            f"locals:\n{pformat(locals_)}",
            error_line(),
        ]
    )
    return s


def raise_error(e: Exception, pre: str = "", post: str = "") -> None:
    raise type(e)(pre + e.message + post)


def str_type_cast(iterable: tp.Iterable[str]) -> tp.Generator[tp.Any, None, None]:
    for i in iterable:
        if i == "None":
            yield None
        elif i == "True":
            yield True
        elif i == "False":
            yield False
        else:
            yield i


def get_offset_chunk_tensor(chunked_tensor: Tensor, chunk_size: int) -> int:
    first = chunked_tensor[0]
    for i in range(min(chunk_size, chunked_tensor.shape[0])):
        if chunked_tensor[i] != first:
            return i
    return 0
