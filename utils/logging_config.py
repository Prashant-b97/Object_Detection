"""Central logging configuration for CLI scripts."""

import logging
import os
import sys
from typing import Optional

DEFAULT_LOG_PATH = os.path.join("logs", "app.log")


def setup_logging(log_path: Optional[str] = None) -> str:
    """Configure root logging handlers once and return the active log file path."""
    active_log_path = log_path or DEFAULT_LOG_PATH

    if logging.getLogger().handlers:
        return active_log_path

    os.makedirs(os.path.dirname(active_log_path), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(active_log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("Logging configured. Writing to %s", active_log_path)
    return active_log_path

__all__ = ["setup_logging", "DEFAULT_LOG_PATH"]
