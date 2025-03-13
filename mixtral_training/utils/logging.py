"""Logging utilities for Mixtral training framework."""

import os
import sys
import logging
import json
from typing import Any, Dict, Optional, Union


def configure_root_logger(level: int = logging.INFO) -> logging.Logger:
    """
    Configure the root logger.

    Args:
        level: Logging level

    Returns:
        logging.Logger: Root logger
    """
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure root logger
    root_logger.setLevel(level)

    # Create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create formatter and add it to the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    root_logger.addHandler(console_handler)

    return root_logger


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name
        level: Optional logging level

    Returns:
        logging.Logger: Logger instance
    """
    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)

    return logger


def log_dict(
    logger: logging.Logger,
    data: Dict[str, Any],
    message: Optional[str] = None,
    level: int = logging.INFO,
) -> None:
    """
    Log a dictionary with indentation for readability.

    Args:
        logger: Logger to use
        data: Dictionary to log
        message: Optional message to include
        level: Logging level
    """
    if message:
        logger.log(level, message)

    formatted_data = json.dumps(data, indent=2, default=str)
    for line in formatted_data.splitlines():
        logger.log(level, line)


def setup_file_logging(
    logger: logging.Logger, file_path: str, level: int = logging.DEBUG
) -> None:
    """
    Set up file logging for a specific logger.

    Args:
        logger: Logger to configure
        file_path: Path to log file
        level: Logging level for file handler
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    # Create file handler
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setLevel(level)

    # Create formatter and add it to the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)


def configure_wandb_logging(root_logger: logging.Logger) -> None:
    """
    Configure logging to Weights & Biases.

    Args:
        root_logger: Root logger
    """
    try:
        import wandb

        class WandbHandler(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.INFO:
                    msg = self.format(record)
                    # Skip noisy debug logs
                    if not any(
                        skip_term in msg
                        for skip_term in [
                            "[tensor]",
                            "gradient norm is",
                            "learning rate:",
                        ]
                    ):
                        wandb.log(
                            {
                                "log/level": record.levelname,
                                "log/message": msg,
                            },
                            commit=False,
                        )

        # Add wandb handler
        handler = WandbHandler()
        handler.setFormatter(
            logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        )
        handler.setLevel(logging.INFO)
        root_logger.addHandler(handler)

    except ImportError:
        root_logger.warning("wandb not available, skipping wandb logging configuration")
