"""

"""

from pathlib import Path
import typing as tp

from torch import Tensor


Pathlike = tp.Union[Path, str]
ErrorMode = tp.Literal["raise", "warn", "ignore"]
ExeToolkit = tp.Literal["lief", "pefile"]
ForwardFunction = tp.Callable[[Tensor], Tensor]
