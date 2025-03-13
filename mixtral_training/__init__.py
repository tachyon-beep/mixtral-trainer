"""Mixtral-8x7B R1 Reasoning Training Framework."""

from mixtral_training.version import __version__
from mixtral_training.utils.logging import get_logger

# Import main components for easy access
try:
    from mixtral_training.config import TrainingConfig, ModelConfig, DatasetConfig
    from mixtral_training.train import train_model, resume_training, predict_batch
except ImportError:
    # Handle circular imports or missing modules during package setup
    pass

__all__ = [
    "__version__",
    "TrainingConfig",
    "ModelConfig",
    "DatasetConfig",
    "train_model",
    "resume_training",
    "predict_batch",
    "get_logger",
]
