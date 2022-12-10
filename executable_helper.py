"""

- Use the BaseOfCode and SizeOfCode instead of the .text section
https://blog.kowalczyk.info/articles/pefileformat.html
"""

from collections import OrderedDict
from dataclasses import dataclass
import gzip
from pathlib import Path
from pprint import pformat, pprint
import typing as tp

import lief
import numpy as np
import pandas as pd
import pefile
from torch import Tensor

from utils import Pathlike
from typing_ import ErrorMode, ExeToolkit


class BadBinaryError(Exception):
    def __init__(self, f: Pathlike):
        super().__init__(f"Bad file: {f}")


class ToolkitError(BadBinaryError):
    def __init__(self, f: Pathlike, toolkit_error: Exception):
        super(Exception, self).__init__(f"Bad file: {f}\nToolkit error: {toolkit_error}")


class EmptyFileError(BadBinaryError):
    ...


class SectionUpperLargerThanFileError(BadBinaryError):
    ...


class EmptySectionError(BadBinaryError):
    ...


class TextSectionError(BadBinaryError):
    ...


class NoTextSectionError(TextSectionError):
    ...


class EmptyTextSectionError(TextSectionError, EmptySectionError):
    ...


class TextSectionUpperLargerThanFileError(TextSectionError, SectionUpperLargerThanFileError):
    ...


@dataclass
class ExeParams:
    text_section_bounds_file: Pathlike

    def __post_init__(self):
        self.text_section_bounds_file = Path(self.text_section_bounds_file)


def get_overlapping_bounds(bounds: tp.Dict[str, tp.Tuple[int, int]]) -> tp.Dict[str, bool]:
    overlapping = {k: False for k in bounds}
    l_, u_ = None, None  # Previous bounds
    for n, (l, u) in bounds.items():  # Sort by lower bound
        if (l_ and u_) and ((l < u_) or u < l):
            overlapping[n] = True
        l_, u_ = l, u
    return overlapping


def get_section_bounds_pefile(pe: pefile.PE) -> tp.Dict[str, tp.Tuple[int, int]]:
    bounds = {}
    for section in pe.sections:
        n = section.Name.decode("utf-8", errors="ignore")
        l = section.PointerToRawData
        u = l + section.SizeOfRawData
        bounds[n] = (l, u)
    bounds = OrderedDict(sorted(bounds.items(), key=lambda x: x[1][0]))
    return bounds


def check_file_length(f: Pathlike, min_bytes: int = 1) -> int:
    """
    Check that a file is not empty and get its length.
    """
    length = Path(f).stat().st_size
    if length < min_bytes:
        raise EmptyFileError(f)
    return length


def read_binary(
    file: Path,
    mode: str = "rb",
    max_len: int = None,
    l: int = None,
    u: int = None,
) -> np.ndarray:
    """
    Read a binary file into a numpy array with MalConv2 technique.
    """

    def read_handle(handle):
        n_bytes = -1 if max_len is None else max_len
        if l is not None:
            handle.seek(l)
            n_bytes = u - l if max_len is None else min(max_len, u - l)
        return handle.read(n_bytes)

    try:
        with gzip.open(file, mode) as handle:
            x = read_handle(handle)
    except OSError:
        with open(file, mode) as handle:
            x = read_handle(handle)

    x = np.frombuffer(x, dtype=np.uint8).astype(np.int16) + 1
    return x


def _text_section_bounds_pefile(f: Pathlike) -> tp.Tuple[int, int]:
    """
    Get the lower/upper bounds of the .text section from a PE file using pefile.

    Does not perform any checks on the bounds.
    """
    try:
        pe = pefile.PE(f)
    except pefile.PEFormatError as e:
        raise ToolkitError(f, e)

    for section in pe.sections:
        if ".text" in section.Name.decode("utf-8", errors="ignore"):
            lower = section.PointerToRawData
            upper = lower + section.SizeOfRawData
            return lower, upper

    raise NoTextSectionError(f)


def _text_section_bounds_lief(f: Pathlike) -> tp.Tuple[int, int]:
    """
    Get the lower/upper bounds of the .text section from a PE file using LIEF.

    Does not perform any checks on the bounds.
    """
    f = Path(f).as_posix()
    binary = lief.parse(f)
    if binary is None:
        raise BadBinaryError(f)

    try:
        section = binary.get_section(".text")
    except lief.not_found as e:
        raise ToolkitError(f, e)

    if section is None:
        raise NoTextSectionError(f)

    lower = section.offset
    upper = lower + section.size
    return lower, upper


def _text_section_bounds(
    f: Pathlike,
    toolkit: ExeToolkit,
    errors: ErrorMode,
) -> tp.Tuple[int, int]:
    """
    Get the lower/upper bounds of the .text section from a PE file.

    Checks the bounds.
    """
    length = check_file_length(f)

    try:
        if toolkit == "pefile":
            lower, upper = _text_section_bounds_pefile(f)
        elif toolkit == "lief":
            lower, upper = _text_section_bounds_lief(f)
        else:
            raise ValueError(f"Unknown toolkit: {toolkit}")
    except BadBinaryError as e:
        if errors == "replace":
            return None, None
        else:
            raise e

    try:
        if upper > length:
            raise TextSectionUpperLargerThanFileError(f)
        if lower == upper:
            raise EmptyTextSectionError(f)
    except BadBinaryError as e:
        if errors != "replace":
            raise e

    return lower, upper


def stream_text_section_bounds(
    files: tp.Union[tp.Iterable[Pathlike], Pathlike],
    toolkit: ExeToolkit,
    min_size: tp.Optional[int] = None,
    max_size: tp.Optional[int] = None,
    errors: ErrorMode = "raise",
    verbose: bool = False,
) -> tp.Generator[tp.Tuple[Path, int, int], None, None]:
    """
    Get the lower and upper bounds of the .text section from non-problematic PE files.
    """
    files = [files] if isinstance(files, (str, Path)) else files
    if toolkit == "lief":  # LIEF is fucking annoying
        lief.logging.disable()

    count = 0
    for i, f in enumerate(map(Path, files), 1):
        try:
            lower, upper = _text_section_bounds(f, toolkit, errors)
        except BadBinaryError as e:
            if errors == "raise":
                raise e
            elif errors == "warn":
                print(f"ERROR: {f.as_posix()}\n{type(e).__name__}: {e}")
            elif errors == "ignore":
                pass
            else:
                raise ValueError(f"Invalid value for errors: {errors}")
        except FileNotFoundError as e:
            if errors == "raise":
                raise e
            elif errors == "warn":
                print(f"ERROR: {f.as_posix()}\n{type(e).__name__}: {e}")
            elif errors == "ignore":  # This is weird, so should print anyway
                print(f"ERROR: {f.as_posix()}\n{type(e).__name__}: {e}")
            else:
                raise ValueError(f"Invalid value for errors: {errors}")
        else:
            if min_size is not None and upper - lower < min_size:
                continue
            if max_size is not None and upper - lower > max_size:
                continue

            count += 1
            yield f, lower, upper

    if verbose:
        print(f"Found {count} / {i} good files.")


def stream_text_section_data(
    files: tp.Union[tp.Iterable[Pathlike], Pathlike],
    toolkit: ExeToolkit,
    datatype: tp.Literal["bytes", "numpy", "torch"],
    min_size: tp.Optional[int] = None,
    max_size: tp.Optional[int] = None,
    errors: ErrorMode = "raise",
    verbose: bool = False,
) -> tp.Generator[tp.Tuple[Path, int, int, tp.Union[str, np.ndarray, Tensor]], None, None]:
    for f, l, u in stream_text_section_bounds(files, toolkit, min_size, max_size, errors, verbose):
        if datatype == "bytes":
            raise NotImplementedError()
        elif datatype == "numpy":
            x = read_binary(f)
        elif datatype == "torch":
            x = Tensor(read_binary(f))
        else:
            raise ValueError(f"Invalid value for datatype: {datatype}")

        x = x[l:u]
        yield f, l, u, x


def generate_text_section_bounds_file(
    files: tp.Union[Pathlike, tp.Iterable[Pathlike]],
    toolkit: ExeToolkit,
    outfile: Pathlike,
    min_size: tp.Optional[int] = None,
    max_size: tp.Optional[int] = None,
    errors: ErrorMode = "raise",
    verbose: bool = False,
) -> None:
    """
    Generate a file containing the lower and upper bounds of the .text section from non-problematic PE files.
    """
    with open(outfile, "w") as handle:
        handle.write("file,lower,upper,size\n")
        for f, l, u in stream_text_section_bounds(
            files, toolkit, min_size, max_size, errors, verbose
        ):
            s = f.stat().st_size
            handle.write(f"{f.as_posix()},{l},{u},{s}\n")


def _test(compare=False, analyze=True):
    import numpy as np
    import pandas as pd

    from classifier import SOREL_TRAIN_PATH

    if compare:
        total = len(list(Path(SOREL_TRAIN_PATH).iterdir()))

        print("Testing LIEF Functions")
        gen = stream_text_section_bounds(
            files=Path(SOREL_TRAIN_PATH).iterdir(),
            toolkit="lief",
            errors="warn",
        )
        lief_files = {}
        for i, (f, l, u) in enumerate(gen):
            # print(f.name, l, u)
            lief_files[f.name] = (l, u)

        print("Testing pefile Functions")
        gen = stream_text_section_bounds(
            files=Path(SOREL_TRAIN_PATH).iterdir(),
            toolkit="pefile",
            errors="warn",
        )
        pefile_files = dict()
        for i, (f, l, u) in enumerate(gen):
            # print(f.name, l, u)
            pefile_files[f.name] = (l, u)

        print(f"pefile processed: {len(pefile_files)} / {total} files")
        x = set(pefile_files.keys()) - set(lief_files.keys())
        print(f"pefile successful where LIEF failed:\n{pformat(x)}")
        print(f"LIEF processed: {len(lief_files)} / {total} files")
        x = set(lief_files.keys()) - set(pefile_files.keys())
        print(f"LIEF successful where pefile failed:\n{pformat(x)}")

        files = [p.name for p in SOREL_TRAIN_PATH.iterdir()]
        results = pd.DataFrame(
            {
                "File": files,
                "lower-lief": [lief_files[f][0] if f in lief_files else np.NaN for f in files],
                "lower-pefile": [
                    pefile_files[f][0] if f in pefile_files else np.NaN for f in files
                ],
                "upper-lief": [lief_files[f][1] if f in lief_files else np.NaN for f in files],
                "upper-pefile": [
                    pefile_files[f][1] if f in pefile_files else np.NaN for f in files
                ],
            }
        )
        results.to_csv("results.txt", index=False)

    if analyze:
        results = pd.read_csv("results.txt")
        lower_equal = (results["lower-lief"] == results["lower-pefile"]) | (
            results["lower-lief"].isna() & results["lower-pefile"].isna()
        )
        upper_equal = (results["upper-lief"] == results["upper-pefile"]) | (
            results["upper-lief"].isna() & results["upper-pefile"].isna()
        )
        equal = lower_equal & upper_equal
        results["equal"] = equal
        results.to_csv("results.txt", index=False)


if __name__ == "__main__":
    import classifier as cl

    for toolkit in ["pefile", "lief"]:
        for errors in ["ignore", "replace"]:
            print(toolkit, errors)
            outfile = Path(f"outputs/dataset/text_section_bounds_{toolkit}_{errors}.csv")
            if outfile.exists():
                continue
            generate_text_section_bounds_file(
                (
                    list(cl.SOREL_TEST_PATH.iterdir())
                    + list(cl.SOREL_TRAIN_PATH.iterdir())
                    + list(cl.WINDOWS_TEST_PATH.iterdir())
                    + list(cl.WINDOWS_TRAIN_PATH.iterdir())
                ),
                toolkit,
                outfile,
                errors=errors,
            )


# TODO: use dict of tuples instead of dict of dict
# FIXME: buggy
def filter_bounds(
    bounds: tp.Dict[str, tp.Dict[str, int]],
    max_len: int = None,
    dict_of_dict: bool = False,
) -> tp.Dict[str, tp.Dict[str, int]]:
    if dict_of_dict:
        return {
            k_1: {k_2: v_2 for k_2, v_2 in v_1.items()}
            for k_1, v_1 in bounds.items()
            if (
                v_1["lower"] is not None
                and v_1["upper"] is not None
                and v_1["size"] >= v_1["upper"]
                and v_1["upper"] > v_1["lower"]
                and (v_1["lower"] < max_len if isinstance(max_len, int) else True)
            )
        }
    return {
        k: v
        for k, v in bounds.items()
        if (
            v["lower"] is not None
            and v["upper"] is not None
            and v["size"] >= v["upper"]
            and v["upper"] > v["lower"]
            and (v["lower"] < max_len if isinstance(max_len, int) else True)
        )
    }


# TODO: use dict of tuples instead of dict of dict
# FIXME: buggy
def get_bounds(
    text_section_bounds_file: Pathlike,
    dict_of_dict: bool = False,
) -> tp.Dict[str, tp.Dict[str, int]]:
    d = pd.read_csv(text_section_bounds_file, index_col="file").to_dict("index")
    d = {
        k_1: {k_2: int(v_2) if str(v_2).isdigit() else None for k_2, v_2 in v_1.items()}
        for k_1, v_1 in d.items()
    }
    if dict_of_dict:
        return d
    return {k: (v["lower"], v["upper"]) for k, v in d.items()}
