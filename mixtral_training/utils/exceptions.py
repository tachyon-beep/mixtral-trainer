"""Custom exceptions for the Mixtral training framework."""

import logging
from typing import Optional, Type


class MixtralTrainingError(Exception):
    """Base exception for all Mixtral training errors."""

    def __init__(self, message: str, *args):
        self.message = message
        super().__init__(message, *args)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.message}"


class ConfigError(MixtralTrainingError):
    """Exception raised for configuration errors."""

    pass


class ModelError(MixtralTrainingError):
    """Exception raised for model-related errors."""

    pass


class DatasetError(MixtralTrainingError):
    """Exception raised for dataset-related errors."""

    pass


class TrainingError(MixtralTrainingError):
    """Exception raised for training-related errors."""

    pass


class EvaluationError(MixtralTrainingError):
    """Exception raised for evaluation-related errors."""

    pass


class MemoryError(MixtralTrainingError):
    """Exception raised for memory-related errors."""

    pass


class StorageError(MixtralTrainingError):
    """Exception raised for storage-related errors."""

    pass


class SecurityError(MixtralTrainingError):
    """Exception raised for security-related errors."""

    pass


def log_exception(
    logger: logging.Logger,
    exc: Exception,
    prefix: str = "Error",
    level: int = logging.ERROR,
) -> None:
    """
    Log an exception with appropriate level and formatting.

    Args:
        logger: Logger to use
        exc: Exception to log
        prefix: Prefix for log message
        level: Logging level
    """
    if isinstance(exc, MixtralTrainingError):
        logger.log(level, f"{prefix}: {exc}")
    else:
        logger.log(level, f"{prefix}: {type(exc).__name__}: {exc}")

    # Log traceback at debug level
    if level >= logging.ERROR:
        logger.debug("Exception details:", exc_info=True)
