"""Security utilities for Mixtral training framework."""

import os
import logging
import getpass
from typing import Dict, Optional

from mixtral_training.utils.exceptions import SecurityError

# Get module logger
logger = logging.getLogger(__name__)


def prompt_for_api_key(
    service_name: str, env_var_name: Optional[str] = None
) -> Optional[str]:
    """
    Prompt the user for an API key, checking environment variables first.

    Args:
        service_name: Name of the service (for display)
        env_var_name: Environment variable name to check

    Returns:
        Optional[str]: API key or None
    """
    # First check environment variable
    if env_var_name and env_var_name in os.environ:
        logger.info(
            f"Using {service_name} API key from environment variable {env_var_name}"
        )
        return os.environ[env_var_name]

    # Prompt user for API key
    try:
        print(f"\nPlease enter your {service_name} API key (input will be hidden)")
        print(
            f"You can also set the {env_var_name} environment variable to avoid this prompt"
        )
        api_key = getpass.getpass("API Key: ")

        if not api_key.strip():
            logger.warning(f"No {service_name} API key provided")
            return None

        return api_key.strip()

    except (KeyboardInterrupt, EOFError):
        logger.warning(f"{service_name} API key input cancelled")
        return None

    except Exception as e:
        logger.error(f"Error getting {service_name} API key: {e}")
        return None


def mask_sensitive_data(
    data: Dict, sensitive_keys: list = ["token", "api_key", "password", "secret"]
) -> Dict:
    """
    Mask sensitive data in a dictionary.

    Args:
        data: Dictionary to mask
        sensitive_keys: List of sensitive key patterns

    Returns:
        Dict: Masked dictionary
    """
    # Create a shallow copy to avoid modifying original
    result = dict(data)

    for key, value in data.items():
        # Check if this is a sensitive key
        is_sensitive = any(
            sensitive_key in key.lower() for sensitive_key in sensitive_keys
        )

        if is_sensitive and value and isinstance(value, str):
            # Mask the value, showing just the first 3 characters
            visible_chars = min(3, len(value))
            masked_value = value[:visible_chars] + "******"
            result[key] = masked_value
        elif isinstance(value, dict):
            # Recursively mask nested dictionaries
            result[key] = mask_sensitive_data(value, sensitive_keys)
        elif isinstance(value, list):
            # Handle lists of dictionaries
            if value and isinstance(value[0], dict):
                result[key] = [
                    (
                        mask_sensitive_data(item, sensitive_keys)
                        if isinstance(item, dict)
                        else item
                    )
                    for item in value
                ]

    return result


def verify_path_safety(path: str) -> bool:
    """
    Verify that a file path is safe to use.

    Args:
        path: File path to verify

    Returns:
        bool: True if safe

    Raises:
        SecurityError: If path is unsafe
    """
    # Convert to absolute path
    abs_path = os.path.abspath(path)

    # Check for path traversal
    if ".." in path:
        raise SecurityError(f"Path contains directory traversal: {path}")

    # Check for suspicious characters
    suspicious_chars = ["*", "?", "|", ">", "<", ";", "&", "`"]
    for char in suspicious_chars:
        if char in path:
            raise SecurityError(f"Path contains suspicious character '{char}': {path}")

    # Check if path is a regular file or directory
    if os.path.exists(abs_path) and not (
        os.path.isfile(abs_path) or os.path.isdir(abs_path)
    ):
        raise SecurityError(f"Path is not a regular file or directory: {abs_path}")

    return True
