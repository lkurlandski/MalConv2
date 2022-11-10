"""

"""

import gzip
from pathlib import Path
from pprint import pformat, pprint
import typing as tp

import lief
import numpy as np
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


def check_file_length(f: Pathlike, min_bytes: int = 1) -> int:
    """
    Check that a file is not empty and get its length.
    """
    length = Path(f).stat().st_size
    if length < min_bytes:
        raise EmptyFileError(f)
    return length


def read_binary(file: Path, mode: str = "rb", max_len: int = None) -> np.ndarray:
    """
    Read a binary file into a numpy array with MalConv2 technique.
    """
    try:
        with gzip.open(file, mode) as f:
            x = f.read(max_len)
    except OSError:
        with open(file, mode) as f:
            x = f.read(max_len)
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
) -> tp.Tuple[int, int]:
    """
    Get the lower/upper bounds of the .text section from a PE file.

    Checks the bounds.
    """
    length = check_file_length(f)

    if toolkit == "pefile":
        lower, upper = _text_section_bounds_pefile(f)
    elif toolkit == "lief":
        lower, upper = _text_section_bounds_lief(f)
    else:
        raise ValueError(f"Unknown toolkit: {toolkit}")

    if upper > length:
        raise TextSectionUpperLargerThanFileError(f)
    if lower == upper:
        raise EmptyTextSectionError(f)

    return lower, upper


def text_section_bounds(
    files: tp.Union[tp.Iterable[Pathlike], Pathlike],
    toolkit: ExeToolkit,
    min_size: tp.Optional[int] = None,
    max_size: tp.Optional[int] = None,
    errors: ErrorMode = "raise",
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
            lower, upper = _text_section_bounds(f, toolkit)
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

    print(f"Found {count} / {i} good files.")


def text_section_data(
    files: tp.Union[tp.Iterable[Pathlike], Pathlike],
    toolkit: ExeToolkit,
    datatype: tp.Literal["bytes", "numpy", "torch"],
    min_size: tp.Optional[int] = None,
    max_size: tp.Optional[int] = None,
    errors: ErrorMode = "raise",
) -> tp.Generator[tp.Tuple[Path, int, int, tp.Union[str, np.ndarray, Tensor]], None, None]:
    for f, l, u in text_section_bounds(files, toolkit, min_size, max_size, errors):
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


def _test(compare=False, analyze=True):
    import numpy as np
    import pandas as pd

    from classifier import SOREL_TRAIN_PATH

    if compare:
        total = len(list(Path(SOREL_TRAIN_PATH).iterdir()))

        print("Testing LIEF Functions")
        gen = text_section_bounds(
            files=Path(SOREL_TRAIN_PATH).iterdir(),
            toolkit="lief",
            errors="warn",
        )
        lief_files = {}
        for i, (f, l, u) in enumerate(gen):
            # print(f.name, l, u)
            lief_files[f.name] = (l, u)

        print("Testing pefile Functions")
        gen = text_section_bounds(
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
    _test(True, True)
