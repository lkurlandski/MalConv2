"""

"""
from pathlib import Path
import typing as tp


Pathlike = tp.Union[str, Path]


def section_header(
    name: str, start_with_newline: bool = True, underline_length: int = 88
) -> None:
    return ("\n" if start_with_newline else "") + f"{name}\n{'-' * underline_length}"


def error_line(l_r_buffer: int = 40) -> None:
    return "-" * l_r_buffer + " ERROR " + "-" * l_r_buffer


def batch(iterable: tp.Iterable, n: int = 1) -> tp.Iterable[tp.List]:
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def ceil_divide(a: float, b: float) -> int:
    return int(-(a // -b))
