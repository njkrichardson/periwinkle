import argparse
import logging
import os
from typing import Optional

from utils import get_log_dir, get_now_str

# --- type aliases
namespace: type = argparse.Namespace


def level_from_args(args: namespace) -> int:
    level: int = (
        logging.DEBUG if args.debug else logging.INFO if args.verbose else logging.WARN
    )
    return level

def setup_logger(name: str, level: int = logging.INFO, custom_handle: str=None) -> logging.Logger:
    # --- create the entry point logger
    logger: logging.Logger = logging.getLogger(name)

    if not getattr(logger, "handler_set", None):
        # --- add the file handler
        log_file: Path = custom_handle if custom_handle is not None else os.path.join(get_log_dir(), get_now_str() + ".out")
        file_handler = logging.FileHandler(log_file)

        # --- format the file handler
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        file_handler.setFormatter(fmt)

        # --- setup stream handler 
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(fmt)

        # --- configure the logger
        logger.addHandler(file_handler)
        logger.addHandler(console)
        logger.setLevel(level)

        # --- don't add more handlers next time
        logger.handler_set = True
        logger.propagate = False

    return logger
