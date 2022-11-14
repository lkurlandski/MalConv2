"""

"""

from contextlib import redirect_stdout, redirect_stderr
import pefile
import gzip
import os
from pathlib import Path
from pprint import pprint
import sys
import typing as tp

import capstone as cs
import lief
import numpy as np
import torch

from utils import section_header

from classifier import get_data
from executable_helper import read_binary


# TODO: remove this asset, as it has been replaced by executable_helper.stream_text_section_bounds
def code_section_offset_bounds(f: Path, errors: str = "raise", silence_lief: bool = True):
    """
    Returns the lower and upper bounds of the .text section in the PE file.

    # TODO: use lief characteristics to identify code sections, instead of the name ".text"
    """
    if f.stat().st_size == 0:  # Ignore empty files
        if errors == "raise":
            raise ValueError(f"File is Empty: {f.name}")
        elif errors == "warn":
            print(f"Warning: file is empty: {f.name} ")
            return None, None
        elif errors == "ignore":
            return None, None

    if silence_lief:  # Doesn't work, unfortunately
        with open(os.devnull, "w") as redirect:
            with redirect_stdout(redirect), redirect_stderr(redirect):
                binary = lief.parse(f.as_posix())
    else:
        binary = lief.parse(f.as_posix())

    try:
        section = binary.get_section(".text")
    except lief.not_found as e:
        if errors == "raise":
            raise e
        elif errors == "warn":
            print(f"Warning: no .text section found: {f.name}")
            return None, None
        elif errors == "ignore":
            return None, None

    if section is None:
        if errors == "raise":
            raise lief.not_found(f"Could not parse: {f.name}")
        elif errors == "warn":
            print(f"Warning: could not parse: {f.name}")
            return None, None
        elif errors == "ignore":
            return None, None

    if section.size == 0:
        if errors == "raise":
            raise ValueError(f".text section has size 0 : {f.name}")
        elif errors == "warn":
            print(f".text section has size 0 : {f.name}")
            return None, None
        elif errors == "ignore":
            return None, None

    lower = section.offset
    upper = lower + section.size
    return lower, upper


def build_corpora_fixed_chunk_size(
    output_path: Path,
    attributions_path: Path,
    corpora_path: Path,
    chunk_size: int,
):
    print(section_header("Building Corpora"))
    # TODO: mind the change in API of get_data
    train_dataset, _, train_loader, _, train_sampler, _ = get_data()
    max_hash_evasion = 10

    hyperparam_path = attributions_path.relative_to(output_path)
    corpora_path = corpora_path / hyperparam_path

    binary_files = [Path(e[0]) for e in train_dataset.all_files]
    for f in binary_files:  # tqdm(binary_files):
        if f.stat().st_size == 0:  # Ignore empty files
            continue
        # Saved attributions tensor
        attributions = torch.load(
            (attributions_path / f.name).with_suffix(".pt"),
            map_location=torch.device("cpu"),
        )
        # Byte-view of the binary
        binary = read_binary(f)
        # .text section bounds to produce the snippets from
        try:
            lower, upper = code_section_offset_bounds(f)
        except lief.not_found:  # code section could not be located
            continue
        print(f.as_posix())
        # Identify malicious and benign regions, save snippets to file
        for j in range(0, (binary.shape[0] // chunk_size) * chunk_size + 1, chunk_size):
            # Beginning and end regions to consider
            b = j
            e = min(j + chunk_size, binary.shape[0])
            # Skip if outside the binary's code section or zero-length slice
            if b == e or b < lower or e > upper:
                continue
            # Section of binary corresponding to the snippet
            binary_section = (binary[b:e] - 1).astype(np.uint8)
            # Whether snippet is malicious-looking or benign-looking
            if attributions[j].item() > 0:
                division = "malicious"
            elif attributions[j].item() < 0:
                division = "benign"
            # Save the snippet
            for i in range(max_hash_evasion):
                path = corpora_path / division / str(i) / f"{f.name}_{b}_{e}.bin"
                path.parent.mkdir(exist_ok=True, parents=True)
                try:
                    binary_section.tofile(path)
                    break
                except OSError:
                    continue
                print(f"Failed to place: {path.name}")
                print(error)


def move_files_based_upon_parsing_needs(corpora_path: Path):
    """
    Separate files into distinct directories based upon the
    presumed architecture and mode they were compiled in.
    """

    def process_single_file(f):
        if f.suffix != ".bin":
            return
        with open(f, "rb") as handle:
            binary = handle.read()
        for arch, modes in arch_modes.items():
            for mode in modes:
                instructions = list(cs.Cs(arch, mode).disasm_lite(binary, 0x0))
                if instructions:
                    output_path = path / f"arch{arch}" / f"mode{mode}" / f.name
                    output_path.parent.mkdir(exist_ok=True, parents=True)
                    binary_file.rename(output_path)
                    break  # File parsed successfully, so move on to next file
            else:
                continue  # File not parsed successfully, so move on to next mode
            break  # Triggered only if inner loop breaks
        else:
            # File could not be parsed, so move it to the unknown directory
            f.rename(unknown_path / f.name)

    print(section_header("Moving Files Based Upon Parsing Needs"))

    # Ordered manner in which the capstone architectures and their corresponding
    # modes are used in attempts to parse the PE snippets
    arch_modes = {
        cs.CS_ARCH_X86: (cs.CS_MODE_32, cs.CS_MODE_64, cs.CS_MODE_16),
        cs.CS_ARCH_ARM: (cs.CS_MODE_ARM, cs.CS_MODE_THUMB),
        cs.CS_ARCH_ARM64: (cs.CS_MODE_ARM,),
        cs.CS_ARCH_MIPS: (cs.CS_MODE_MIPS32, cs.CS_MODE_MIPS64),
        cs.CS_ARCH_PPC: (cs.CS_MODE_32, cs.CS_MODE_64),
    }

    for division in ("malicious", "benign"):
        path = corpora_path / division
        unknown_path = path / "archUnknown" / "modeUnknown"
        unknown_path.mkdir(exist_ok=True, parents=True)
        for collection in range(10):  # Corresponds to the max_hash_evasion
            collection_path = path / str(collection)
            if not collection_path.exists():
                break
            for binary_file in collection_path.iterdir():
                process_single_file(binary_file)
            collection_path.rmdir()


if __name__ == "__main__":
    corpora_path = Path(
        "/home/lk3591/Documents/datasets/MachineCodeTranslation/KernelShap/softmaxFalse/256/50/1/attributions/"
    )
    move_files_based_upon_parsing_needs(corpora_path)
