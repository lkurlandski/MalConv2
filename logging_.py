"""

"""

import typing as tp
from logging import basicConfig, getLogger


def setup_logging(filename: str, filemode: str, format: str, level:tp.Union[str, int]):
    basicConfig(filename=filename, filemode=filemode, format=format, level=level)
    log = getLogger()
    print(f"{log=} - {log.name=} - {log.level=}")
