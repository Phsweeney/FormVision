"""Logging setup.

Analysis runs in a background task, which means when something goes wrong the
log is the only evidence available — there is no HTTP response to inspect. So
every stage of the pipeline logs its timing and outcome, and configuration
happens once at application startup.
"""

from __future__ import annotations

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-34s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Third-party libraries that are noisy at INFO and tell us nothing useful.
_NOISY_LOGGERS = ("multipart", "python_multipart", "urllib3", "matplotlib")


def configure_logging(level: str = "INFO") -> None:
    """Install a single stdout handler on the root logger.

    Uses ``force=True`` so that running under uvicorn — which installs its own
    handlers — does not result in every line being emitted twice.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        stream=sys.stdout,
        force=True,
    )

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger. Use ``get_logger(__name__)``."""
    return logging.getLogger(name)
