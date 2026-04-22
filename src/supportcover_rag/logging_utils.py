from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

from supportcover_rag.io_utils import ensure_dir


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
    )


@contextmanager
def attach_run_log(path: str | Path):
    target = Path(path)
    ensure_dir(target.parent)
    handler = logging.FileHandler(target, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        yield target
    finally:
        root_logger.removeHandler(handler)
        handler.close()
