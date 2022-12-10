"""
Build a corpus of benign-looking and malicious-looking snippets.
"""

from pathlib import Path

import capstone as cs


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
    pass
