"""Checkpoint management utilities for Mixtral training."""

import os
import re
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from mixtral_training.utils.storage import ensure_directory_exists, copy_file
from mixtral_training.utils.exceptions import StorageError

# Get module logger
logger = logging.getLogger(__name__)


def create_checkpoint_dir(output_dir: Union[str, Path], step: int) -> str:
    """
    Create a checkpoint directory.

    Args:
        output_dir: Base output directory
        step: Training step

    Returns:
        str: Path to checkpoint directory
    """
    # Ensure output directory exists
    ensure_directory_exists(output_dir)

    # Create checkpoint directory
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")

    if os.path.exists(checkpoint_dir):
        logger.warning(f"Checkpoint directory already exists: {checkpoint_dir}")
    else:
        ensure_directory_exists(checkpoint_dir)
        logger.info(f"Created checkpoint directory: {checkpoint_dir}")

    return checkpoint_dir


def find_latest_checkpoint(directory: Union[str, Path]) -> Optional[str]:
    """
    Find the latest checkpoint in a directory.

    Args:
        directory: Directory to search

    Returns:
        Optional[str]: Path to latest checkpoint directory or None if not found
    """
    if not os.path.exists(directory):
        logger.warning(f"Directory does not exist: {directory}")
        return None

    checkpoint_dirs = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isdir(path) and name.startswith("checkpoint-"):
            try:
                # Extract step number from directory name
                step = int(name.split("-")[-1])
                checkpoint_dirs.append((step, path))
            except ValueError:
                # Skip if not a valid checkpoint directory
                continue

    if not checkpoint_dirs:
        logger.warning(f"No checkpoints found in {directory}")
        return None

    # Sort by step number (descending) and return the path for the highest step
    checkpoint_dirs.sort(reverse=True)
    _, latest_path = checkpoint_dirs[0]

    logger.info(f"Found latest checkpoint: {latest_path}")
    return latest_path


def create_checkpoint_state(
    checkpoint_dir: str, state: Dict[str, Any], filename: str = "training_state.json"
) -> bool:
    """
    Create a checkpoint state file.

    Args:
        checkpoint_dir: Checkpoint directory
        state: Training state
        filename: State filename

    Returns:
        bool: True if successful
    """
    from mixtral_training.utils.storage import safe_write_json

    # Add timestamp to state
    state["timestamp"] = datetime.datetime.now().isoformat()

    # Write state to file
    state_path = os.path.join(checkpoint_dir, filename)
    return safe_write_json(state, state_path)


def load_checkpoint_state(
    checkpoint_dir: str, filename: str = "training_state.json"
) -> Optional[Dict[str, Any]]:
    """
    Load checkpoint state.

    Args:
        checkpoint_dir: Checkpoint directory
        filename: State filename

    Returns:
        Optional[Dict[str, Any]]: Checkpoint state or None if error
    """
    from mixtral_training.utils.storage import safe_read_json

    # Read state from file
    state_path = os.path.join(checkpoint_dir, filename)
    return safe_read_json(state_path)


def verify_checkpoint_is_complete(checkpoint_dir: str) -> bool:
    """
    Verify that a checkpoint is complete.

    Args:
        checkpoint_dir: Checkpoint directory

    Returns:
        bool: True if complete
    """
    # Check if checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        logger.warning(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return False

    # Check for standard files expected in a checkpoint
    config_file = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(config_file):
        logger.warning(f"Missing config.json in checkpoint: {checkpoint_dir}")
        return False

    # For LoRA models, check for adapter config
    lora_config = os.path.join(checkpoint_dir, "adapter_config.json")
    if os.path.exists(lora_config):
        logger.info(f"Found LoRA adapter config in checkpoint: {checkpoint_dir}")
        return True

    # For full models, check for model weights
    model_file = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.exists(model_file) or any(
        f.endswith(".bin") for f in os.listdir(checkpoint_dir)
    ):
        logger.info(f"Found model weights in checkpoint: {checkpoint_dir}")
        return True

    # Check for safetensors format
    if any(f.endswith(".safetensors") for f in os.listdir(checkpoint_dir)):
        logger.info(f"Found safetensors weights in checkpoint: {checkpoint_dir}")
        return True

    logger.warning(f"Checkpoint appears incomplete: {checkpoint_dir}")
    return False


def save_checkpoint_config(checkpoint_dir: str, config: Dict[str, Any]) -> bool:
    """
    Save training configuration to checkpoint.

    Args:
        checkpoint_dir: Checkpoint directory
        config: Training configuration

    Returns:
        bool: True if successful
    """
    from mixtral_training.utils.storage import safe_write_json

    # Write config to file
    config_path = os.path.join(checkpoint_dir, "training_config.json")
    return safe_write_json(config, config_path)


def get_checkpoint_step(checkpoint_path: str) -> int:
    """
    Extract step number from checkpoint path.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        int: Step number or 0 if unable to extract
    """
    try:
        # Extract step number from directory name
        checkpoint_dir = os.path.basename(checkpoint_path)
        match = re.match(r"checkpoint-(\d+)", checkpoint_dir)
        if match:
            return int(match.group(1))
    except Exception as e:
        logger.warning(f"Error extracting step from checkpoint path: {e}")

    return 0
