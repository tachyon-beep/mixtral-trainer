"""Memory management utilities for Mixtral training."""

import os
import math
import logging
from typing import Dict, List, Optional, Tuple, Union

# Get module logger
logger = logging.getLogger(__name__)

# Constants
GB = 1024**3  # 1 GB in bytes
MB = 1024**2  # 1 MB in bytes


def get_total_gpu_memory_gb() -> float:
    """
    Get the total GPU memory available across all GPUs, in GB.

    Returns:
        float: Total memory in GB
    """
    try:
        import torch

        if not torch.cuda.is_available():
            logger.warning("CUDA is not available, returning 0 for GPU memory")
            return 0.0

        total_memory = 0.0
        for i in range(torch.cuda.device_count()):
            device_memory = torch.cuda.get_device_properties(i).total_memory
            total_memory += device_memory

        return total_memory / GB

    except ImportError:
        logger.warning("PyTorch not installed, returning 0 for GPU memory")
        return 0.0

    except Exception as e:
        logger.warning(f"Error getting GPU memory: {e}")
        return 0.0


def get_per_gpu_memory_gb() -> float:
    """
    Get the memory of a single GPU in GB (assuming all GPUs are the same).

    Returns:
        float: Per-GPU memory in GB
    """
    try:
        import torch

        if not torch.cuda.is_available():
            logger.warning("CUDA is not available, returning 0 for GPU memory")
            return 0.0

        if torch.cuda.device_count() == 0:
            logger.warning("No GPUs detected, returning 0 for GPU memory")
            return 0.0

        # Get memory of the first GPU
        device_memory = torch.cuda.get_device_properties(0).total_memory

        return device_memory / GB

    except ImportError:
        logger.warning("PyTorch not installed, returning 0 for GPU memory")
        return 0.0

    except Exception as e:
        logger.warning(f"Error getting GPU memory: {e}")
        return 0.0


def log_memory_stats(prefix: str = "Memory stats") -> Dict[str, float]:
    """
    Log current memory stats.

    Args:
        prefix: Prefix for log message

    Returns:
        Dict[str, float]: Memory statistics
    """
    try:
        import torch
        import psutil

        process = psutil.Process(os.getpid())

        # System memory
        system_memory = psutil.virtual_memory()

        # Process memory
        process_memory = process.memory_info().rss

        # GPU memory
        gpu_stats = {}

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / MB
                reserved = torch.cuda.memory_reserved(i) / MB
                gpu_stats[f"gpu_{i}_allocated_mb"] = allocated
                gpu_stats[f"gpu_{i}_reserved_mb"] = reserved

        # Combine all stats
        stats = {
            "system_memory_used_percent": system_memory.percent,
            "system_memory_available_gb": system_memory.available / GB,
            "process_memory_gb": process_memory / GB,
            **gpu_stats,
        }

        # Log stats
        stats_str = ", ".join(f"{k}={v:.2f}" for k, v in stats.items())
        logger.info(f"{prefix}: {stats_str}")

        return stats

    except ImportError:
        logger.warning("PyTorch or psutil not installed, skipping memory stats")
        return {}

    except Exception as e:
        logger.warning(f"Error logging memory stats: {e}")
        return {}


def set_memory_fraction(fraction: float = 0.95) -> None:
    """
    Set the fraction of GPU memory to use.

    Args:
        fraction: Fraction of memory to use (0.0-1.0)
    """
    try:
        import torch

        if not torch.cuda.is_available():
            logger.warning("CUDA is not available, skipping memory fraction setting")
            return

        # Apply to all devices
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(fraction, i)
            logger.info(f"Set GPU {i} memory fraction to {fraction}")

    except ImportError:
        logger.warning("PyTorch not installed, skipping memory fraction setting")

    except Exception as e:
        logger.warning(f"Error setting memory fraction: {e}")


def calculate_model_parameters(model) -> int:
    """
    Calculate number of parameters in the model dynamically.

    Args:
        model: PyTorch model

    Returns:
        int: Number of parameters in the model
    """
    try:
        # Count parameters directly from model
        return int(sum(p.numel() for p in model.parameters()))
    except Exception as e:
        logger.warning(f"Error calculating model parameters: {e}")
        # Fallback to default for Mixtral-8x7B
        return int(12.9 * 10**9)  # Convert to int


def calculate_optimal_batch_size(
    sequence_length: int,
    gpu_memory_gb: float,
    gpu_count: int,
    precision_bits: int = 4,
    min_batch_size: int = 1,
    safety_factor: float = 0.8,
    model=None,
) -> int:
    """
    Calculate optimal batch size based on model and GPU parameters.

    Args:
        sequence_length: Length of input sequences
        gpu_memory_gb: Available memory per GPU in GB
        gpu_count: Number of GPUs
        precision_bits: Precision in bits (4, 8, 16, 32)
        min_batch_size: Minimum batch size
        safety_factor: Fraction of calculated memory to use (for overhead)
        model: Optional model instance for dynamic parameter calculation

    Returns:
        int: Estimated optimal batch size
    """
    if gpu_memory_gb <= 0 or gpu_count <= 0:
        logger.warning("No GPU memory available, using minimum batch size")
        return min_batch_size

    # Calculate model parameters dynamically if model is provided
    if model is not None:
        model_params = calculate_model_parameters(model)
        logger.info(f"Calculated model parameters dynamically: {model_params/1e9:.2f}B")
    else:
        # Mixtral 8x7B is approximately 46.7B parameters in total
        # But when using MoE, we only load the active experts
        # The base model with experts has 46.7B parameters
        # The base model without experts has 12.9B parameters
        model_params = 12.9 * 10**9  # Base model size estimate
        logger.info(f"Using default model parameter estimate: {model_params/1e9:.2f}B")

    bytes_per_param = max(precision_bits / 8, 0.5)  # Minimum 0.5 bytes per parameter

    # Calculate memory for model
    model_memory_bytes = model_params * bytes_per_param

    # Activation memory formula is an approximation:
    # 6 * sequence_length * sqrt(model_parameters) * bytes_per_param
    # This accounts for KV cache and gradient accumulation
    activation_bytes_per_sample = (
        6 * sequence_length * math.sqrt(model_params) * bytes_per_param
    )

    # Total available memory across all GPUs
    total_memory_bytes = gpu_memory_gb * GB * gpu_count * safety_factor

    # Subtract model memory to get memory available for activations
    activations_memory = total_memory_bytes - model_memory_bytes

    # Calculate maximum batch size
    max_batch_size = max(1, int(activations_memory / activation_bytes_per_sample))

    # Apply minimum and log
    batch_size = max(min_batch_size, max_batch_size)
    logger.info(
        f"Calculated batch size: {batch_size} with sequence length: {sequence_length}"
    )
    logger.info(
        f"Memory estimate: model={model_memory_bytes/GB:.2f}GB, activations={activation_bytes_per_sample*batch_size/GB:.2f}GB"
    )

    return batch_size
