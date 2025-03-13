"""Dataset loading and preparation for Mixtral training."""

import os
import logging
import random
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

from mixtral_training.config import TrainingConfig
from mixtral_training.utils.logging import get_logger
from mixtral_training.utils.exceptions import DatasetError, log_exception

# Get module logger
logger = get_logger(__name__)


def format_instruction(row: Dict[str, Any], template: str) -> str:
    """
    Format an instruction from a dataset row.

    Args:
        row: Dataset row
        template: Template string with {instruction} placeholder

    Returns:
        str: Formatted instruction
    """
    # Handle different formats of instruction datasets
    if "instruction" in row:
        instruction = row["instruction"]
    elif "prompt" in row:
        instruction = row["prompt"]
    elif "input" in row:
        instruction = row["input"]
    elif "question" in row:
        instruction = row["question"]
    else:
        # Default to empty if no recognized field
        instruction = ""

    # Apply template
    return template.format(instruction=instruction)


def format_response(row: Dict[str, Any], template: str) -> str:
    """
    Format a response from a dataset row.

    Args:
        row: Dataset row
        template: Template string with {response} placeholder

    Returns:
        str: Formatted response
    """
    # Handle different formats of instruction datasets
    if "output" in row:
        response = row["output"]
    elif "response" in row:
        response = row["response"]
    elif "answer" in row:
        response = row["answer"]
    elif "completion" in row:
        response = row["completion"]
    else:
        # Default to empty if no recognized field
        response = ""

    # Apply template
    return template.format(response=response)


def create_prompt_completion_pairs(
    dataset, tokenizer, config: TrainingConfig
) -> List[Dict[str, str]]:
    """
    Create prompt-completion pairs from dataset.

    Args:
        dataset: Hugging Face dataset
        tokenizer: Tokenizer
        config: Training configuration

    Returns:
        List[Dict[str, str]]: List of prompt-completion pairs
    """
    pairs = []

    for i, row in enumerate(dataset):
        try:
            # Format instruction and response
            instruction = format_instruction(row, config.dataset.prompt_template)
            response = format_response(row, config.dataset.response_template)

            # Skip empty entries
            if not instruction or not response:
                continue

            # Add to pairs
            pairs.append({"prompt": instruction, "completion": response})

            # Limit to max samples if specified
            if config.dataset.max_samples and len(pairs) >= config.dataset.max_samples:
                break

        except Exception as e:
            logger.warning(f"Error processing dataset row {i}: {e}")

    return pairs


def prepare_training_data(dataset, tokenizer, config: TrainingConfig) -> Dict[str, Any]:
    """
    Prepare dataset for training.

    Args:
        dataset: Hugging Face dataset
        tokenizer: Tokenizer
        config: Training configuration

    Returns:
        Dict[str, Any]: Dictionary with train_dataset and eval_dataset
    """
    # Create prompt-completion pairs
    logger.info("Creating prompt-completion pairs")
    pairs = create_prompt_completion_pairs(dataset, tokenizer, config)

    # Shuffle pairs
    random.shuffle(pairs)

    # Split into train and eval
    eval_size = min(int(len(pairs) * 0.1), 100)  # 10% of data or 100 examples
    train_pairs = pairs[:-eval_size] if eval_size > 0 else pairs
    eval_pairs = pairs[-eval_size:] if eval_size > 0 else []

    logger.info(
        f"Created {len(train_pairs)} training examples and {len(eval_pairs)} evaluation examples"
    )

    # Create data collator
    data_collator = create_data_collator(tokenizer, config)

    # Create tokenization function
    def tokenize_function(examples: Dict[str, List[str]]) -> Dict[str, List[int]]:
        """Tokenize prompts and completions."""
        # Concatenate prompt and completion
        texts = [
            prompt + completion
            for prompt, completion in zip(examples["prompt"], examples["completion"])
        ]

        # Tokenize
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=config.model.max_length,
            return_tensors="pt",
        )

        return tokenized

    # Create datasets
    try:
        from datasets import Dataset

        train_dataset = Dataset.from_dict(
            {
                "prompt": [p["prompt"] for p in train_pairs],
                "completion": [p["completion"] for p in train_pairs],
            }
        )

        eval_dataset = Dataset.from_dict(
            {
                "prompt": [p["prompt"] for p in eval_pairs],
                "completion": [p["completion"] for p in eval_pairs],
            }
        )

        # Apply tokenization
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["prompt", "completion"],
        )

        eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["prompt", "completion"],
        )

        return {
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "data_collator": data_collator,
        }

    except Exception as e:
        log_exception(logger, e, "Error preparing datasets")
        raise DatasetError(f"Error preparing datasets: {e}")


def create_data_collator(tokenizer, config: TrainingConfig) -> Callable:
    """
    Create a data collator function.

    Args:
        tokenizer: Tokenizer
        config: Training configuration

    Returns:
        Callable: Data collator function
    """
    try:
        from transformers.data.data_collator import DataCollatorForLanguageModeling

        return DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Not using masked language modeling
        )

    except ImportError:
        logger.warning(
            "Could not import DataCollatorForLanguageModeling, using default data collator"
        )
        from transformers import default_data_collator

        return default_data_collator


def load_dataset(config: TrainingConfig, tokenizer) -> Dict[str, Any]:
    """
    Load and prepare dataset.

    Args:
        config: Training configuration
        tokenizer: Tokenizer

    Returns:
        Dict[str, Any]: Dictionary with train_dataset and eval_dataset

    Raises:
        DatasetError: If dataset loading fails
    """
    try:
        from datasets import load_dataset as hf_load_dataset

        logger.info(f"Loading dataset: {config.dataset.dataset_name}")

        # Load dataset
        dataset = hf_load_dataset(config.dataset.dataset_name)

        # Get train and validation splits
        train_data = dataset[config.dataset.train_split]

        # Handle datasets without validation split
        if config.dataset.eval_split in dataset:
            eval_data = dataset[config.dataset.eval_split]
        else:
            logger.warning(
                f"Validation split '{config.dataset.eval_split}' not found, "
                "using 10% of training data for evaluation"
            )
            # We'll create eval set from train set later
            eval_data = None

        # Prepare data for training
        return prepare_training_data(train_data, tokenizer, config)

    except Exception as e:
        log_exception(logger, e, "Error loading dataset")
        raise DatasetError(f"Error loading dataset: {e}")


def load_evaluation_dataset(
    dataset_name: str, tokenizer, config: TrainingConfig
) -> Dict[str, Any]:
    """
    Load and prepare dataset for evaluation.

    Args:
        dataset_name: Dataset name or path
        tokenizer: Tokenizer
        config: Training configuration

    Returns:
        Dict[str, Any]: Dictionary with eval_dataset

    Raises:
        DatasetError: If dataset loading fails
    """
    try:
        from datasets import load_dataset as hf_load_dataset

        logger.info(f"Loading evaluation dataset: {dataset_name}")

        # Create a copy of config with updated dataset name
        eval_config = TrainingConfig(
            dataset=config.dataset,
            model=config.model,
            output_dir=config.output_dir,
        )
        eval_config.dataset.dataset_name = dataset_name

        # Load dataset
        dataset = hf_load_dataset(dataset_name)

        # Get evaluation split
        if config.dataset.eval_split in dataset:
            eval_data = dataset[config.dataset.eval_split]
        else:
            logger.warning(
                f"Validation split '{config.dataset.eval_split}' not found, "
                "using first available split"
            )
            # Use whatever split is available
            eval_data = dataset[list(dataset.keys())[0]]

        # Prepare data for evaluation
        result = prepare_training_data(eval_data, tokenizer, eval_config)

        return result["eval_dataset"]

    except Exception as e:
        log_exception(logger, e, "Error loading evaluation dataset")
        raise DatasetError(f"Error loading evaluation dataset: {e}")


def save_dataset(dataset: Dict[str, Any], output_dir: str) -> bool:
    """
    Save dataset to disk.

    Args:
        dataset: Dataset dictionary
        output_dir: Output directory

    Returns:
        bool: True if successful
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Save train dataset
        train_dataset = dataset.get("train_dataset")
        if train_dataset:
            train_path = os.path.join(output_dir, "train_dataset")
            train_dataset.save_to_disk(train_path)
            logger.info(f"Saved train dataset to {train_path}")

        # Save eval dataset
        eval_dataset = dataset.get("eval_dataset")
        if eval_dataset:
            eval_path = os.path.join(output_dir, "eval_dataset")
            eval_dataset.save_to_disk(eval_path)
            logger.info(f"Saved eval dataset to {eval_path}")

        return True

    except Exception as e:
        log_exception(logger, e, "Error saving dataset")
        return False


def load_saved_dataset(dataset_dir: str) -> Dict[str, Any]:
    """
    Load dataset from disk.

    Args:
        dataset_dir: Dataset directory

    Returns:
        Dict[str, Any]: Dictionary with train_dataset and eval_dataset

    Raises:
        DatasetError: If dataset loading fails
    """
    try:
        from datasets import Dataset
        from transformers import default_data_collator

        result = {}

        # Load train dataset
        train_path = os.path.join(dataset_dir, "train_dataset")
        if os.path.exists(train_path):
            train_dataset = Dataset.load_from_disk(train_path)
            result["train_dataset"] = train_dataset
            logger.info(f"Loaded train dataset from {train_path}")

        # Load eval dataset
        eval_path = os.path.join(dataset_dir, "eval_dataset")
        if os.path.exists(eval_path):
            eval_dataset = Dataset.load_from_disk(eval_path)
            result["eval_dataset"] = eval_dataset
            logger.info(f"Loaded eval dataset from {eval_path}")

        # Use default data collator
        result["data_collator"] = default_data_collator

        return result

    except Exception as e:
        log_exception(logger, e, "Error loading saved dataset")
        raise DatasetError(f"Error loading saved dataset: {e}")
