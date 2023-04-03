
import os
from pathlib import Path
from pprint import pprint
import shutil
import subprocess
import typing as tp

import lief
import torch
from torch import tensor

os.chdir("/home/lk3591/Documents/code/MalConv2")

from classifier import confidence_scores, get_model
from executable_helper import read_binary, _text_section_bounds_pefile, _text_section_bounds_lief

WINDOWS = False

compiler = "g++" if WINDOWS else "i686-w64-mingw32-g++"
compiler = "g++"
source_code_files = [
	Path("./proof_of_concept/one.cpp"),
	Path("./proof_of_concept/two.cpp"),
	Path("./proof_of_concept/three.cpp"),
]
compiler_flags = [
    ["-O1"],
    ["-O2"],
    ["-O3"],
]
executables = []
for _, f in enumerate(source_code_files):
	for j, c in enumerate(compiler_flags):
		out = f.with_name(f"{f.stem}_{j}.exe")
		executables.append(out)
		args = [compiler] + c + [f.as_posix(), "-o", out.as_posix()]
		print(args)
		result = subprocess.run(args, capture_output=True, text=True)
		print(result.stdout)
		print(result.stderr)
		if WINDOWS:
			subprocess.run(f"./{out}")


model = get_model("gct")

def get_confidence_scores_from_files(files: tp.List[Path]) -> tp.List[float]:
	confidences = []
	for f in files:
		x = read_binary(f)
		x = tensor(x, dtype=torch.int64)
		c = confidence_scores(model, x)
		confidences.append(c[0])
	return confidences


confidences = get_confidence_scores_from_files(executables)
initial_confidences = {f : c for f, c in zip(executables, confidences)}
pprint(initial_confidences)

def swap_text_sections(f_source: Path, f_replace: Path) -> Path:
    source = lief.parse(f_source.as_posix())
    replace = lief.parse(f_replace.as_posix())
    source_text = source.get_section(".text")
    replace_text = replace.get_section(".text")
    source.remove(source_text)
    source.add_section(replace_text)
    
    builder = lief.PE.Builder(source)
    builder.build_imports(True)
    builder.build()
    f_out = f_source.parent / (f_source.stem + f_replace.stem + ".exe")
    builder.write(f_out.as_posix())

    return f_out

for f_source in executables:
    for f_replace in [f for f in executables if f != f_source]:
        f_out = swap_text_sections(f_source, f_replace)
        if WINDOWS:
            subprocess.run(f"./{f_out}")


