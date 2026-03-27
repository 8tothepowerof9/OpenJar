import logging
import sys

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_LEVEL = logging.INFO


def setup_logging(level: int = DEFAULT_LEVEL, fmt: str = DEFAULT_FORMAT) -> None:
    """Configure the root logger with a consistent format and level."""
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )


def get_logger(name: str, setup: bool = False) -> logging.Logger:
    """Return a named logger, initialising root config on first call."""
    if not logging.root.handlers and setup:
        setup_logging()
    return logging.getLogger(name)
