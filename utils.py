"""

"""

import typing as tp


def section_header(
        name: str,
        start_with_newline: bool = True,
        underline_length: int = 88
) -> None:
    return ("\n" if start_with_newline else "") + f"{name}\n{'-' * underline_length}"


def error_line(l_r_buffer: int = 40):
    return "-" * l_r_buffer + " ERROR " + "-" * l_r_buffer


def batch(iterable: tp.Iterable, n: int = 1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
