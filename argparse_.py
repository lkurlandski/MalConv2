"""

"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, MetavarTypeHelpFormatter
from pathlib import Path
import sys
import typing as tp

import cfg


class CustomArgumentParser(ArgumentParser):
    default_help: str = "-"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(formatter_class=DefaultAndTypeFormatter, *args, **kwargs)

    def add_argument(self, *args, **kwargs) -> tp.Any:
        if "help" not in kwargs:
            kwargs["help"] = self.default_help
        return super().add_argument(*args, **kwargs)


class LoggingArgumentParser(CustomArgumentParser):

    def __init__(self, filename: tp.Union[Path, str] = cfg.LOG_PATH / (Path(sys.argv[0]).stem + ".log"), filemode: str = "a", format: str = "%(asctime)s - %(levelname)s - %(message)s", level: str = "INFO", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        filename = Path(filename)
        self.add_argument("--log_filename", type=Path, default=filename)
        self.add_argument("--log_filemode", type=str, default=filemode)
        self.add_argument("--log_format", type=str, default=format)
        self.add_argument("--log_level", type=str, default=level,
            help="One of `CRITICAL`, `ERROR`, `WARNING`, `INFO`, `DEBUG`, `NOTSET`"
        )


class TorchTrainingArgumentParser(CustomArgumentParser):

    def __init__(self, device: str = "cpu", batch_size: int = 1, epochs: int = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_argument("--device", type=str, default=device)
        self.add_argument("--batch_size", type=int, default=batch_size)
        self.add_argument("--epochs", type=int, default=epochs)


class TorchTrainingLoggingArgumentParser(LoggingArgumentParser, TorchTrainingArgumentParser):
    ...


class DefaultAndTypeFormatter(ArgumentDefaultsHelpFormatter, MetavarTypeHelpFormatter):
    ...
