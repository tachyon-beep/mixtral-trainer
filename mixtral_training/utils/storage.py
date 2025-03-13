"""Storage utilities for Mixtral training framework."""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Get module logger
logger = logging.getLogger(__name__)


def ensure_directory_exists(directory: Union[str, Path]) -> bool:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory: Directory path

    Returns:
        bool: True if successful
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {e}")
        return False


def safe_write_json(data: Any, path: Union[str, Path], indent: int = 2) -> bool:
    """
    Safely write data to a JSON file.

    Args:
        data: Data to write
        path: File path
        indent: JSON indentation

    Returns:
        bool: True if successful
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # Write to temporary file first
        temp_path = f"{path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

        # Rename to final path (atomic operation)
        shutil.move(temp_path, path)

        logger.debug(f"Successfully wrote data to {path}")
        return True

    except Exception as e:
        logger.error(f"Error writing JSON to {path}: {e}")

        # Clean up temporary file if it exists
        try:
            if os.path.exists(f"{path}.tmp"):
                os.remove(f"{path}.tmp")
        except Exception:
            pass

        return False


def safe_read_json(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Safely read data from a JSON file.

    Args:
        path: File path

    Returns:
        Optional[Dict[str, Any]]: Loaded data or None if error
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.debug(f"Successfully read data from {path}")
        return data

    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        return None

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {path}: {e}")
        return None

    except Exception as e:
        logger.error(f"Error reading JSON from {path}: {e}")
        return None


def get_available_disk_space_gb(path: Union[str, Path] = ".") -> float:
    """
    Get available disk space in GB.

    Args:
        path: Path to check

    Returns:
        float: Available space in GB
    """
    try:
        stats = shutil.disk_usage(path)
        return stats.free / (1024**3)  # Convert to GB

    except Exception as e:
        logger.error(f"Error getting disk space for {path}: {e}")
        return 0.0


def cleanup_checkpoints(directory: Union[str, Path], max_to_keep: int = 3) -> bool:
    """
    Clean up old checkpoints, keeping only the most recent ones.

    Args:
        directory: Directory containing checkpoints
        max_to_keep: Maximum number of checkpoints to keep

    Returns:
        bool: True if successful
    """
    try:
        directory = Path(directory)

        # Check if directory exists
        if not directory.exists():
            logger.warning(f"Checkpoint directory {directory} does not exist")
            return False

        # Find all checkpoint directories
        checkpoint_dirs = []
        for item in directory.glob("checkpoint-*"):
            if item.is_dir():
                try:
                    # Extract step number from directory name
                    step = int(item.name.split("-")[-1])
                    checkpoint_dirs.append((step, item))
                except ValueError:
                    # Skip if not a valid checkpoint directory
                    continue

        # Sort by step number (descending)
        checkpoint_dirs.sort(reverse=True)

        # Keep the most recent checkpoints
        to_keep = checkpoint_dirs[:max_to_keep]
        to_delete = checkpoint_dirs[max_to_keep:]

        # Delete old checkpoints
        for _, checkpoint_dir in to_delete:
            logger.info(f"Removing old checkpoint: {checkpoint_dir}")
            shutil.rmtree(checkpoint_dir)

        logger.info(f"Kept {len(to_keep)} checkpoints, deleted {len(to_delete)}")
        return True

    except Exception as e:
        logger.error(f"Error cleaning up checkpoints in {directory}: {e}")
        return False


def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> bool:
    """
    Copy a file from source to destination.

    Args:
        src: Source path
        dst: Destination path

    Returns:
        bool: True if successful
    """
    try:
        # Ensure destination directory exists
        dst_dir = os.path.dirname(dst)
        if dst_dir:
            os.makedirs(dst_dir, exist_ok=True)

        # Copy file
        shutil.copy2(src, dst)

        logger.debug(f"Copied {src} to {dst}")
        return True

    except Exception as e:
        logger.error(f"Error copying {src} to {dst}: {e}")
        return False
