"""Main entry point for Mixtral training framework."""

import os
import sys
import uuid
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import mixtral_training
from mixtral_training.config import TrainingConfig, ModelConfig, DatasetConfig
from mixtral_training.utils.logging import configure_root_logger, get_logger
from mixtral_training.utils.memory import get_total_gpu_memory_gb, set_memory_fraction
from mixtral_training.utils.storage import safe_read_json, safe_write_json
from mixtral_training.utils.exceptions import log_exception, ConfigError

# Configure root logger
logger = configure_root_logger()
module_logger = get_logger(__name__)


def setup_command(args: argparse.Namespace) -> int:
    """
    Handle setup command to create a configuration file.

    Args:
        args: Command-line arguments

    Returns:
        int: Exit code
    """
    try:
        # Create default config
        config = TrainingConfig()

        # Generate a unique run ID if not provided
        config.run_id = str(uuid.uuid4())[:8]

        # Set up model config
        config.model.model_name = args.model_name or config.model.model_name

        # Determine LoRA settings based on available GPU memory
        if args.use_lora is not None:
            config.model.lora_enabled = args.use_lora
        else:
            # Auto-detect if we should use LoRA based on GPU memory
            total_gpu_memory = get_total_gpu_memory_gb()
            if total_gpu_memory < 40:  # Less than 40GB total GPU memory
                module_logger.info(
                    f"Auto-enabling LoRA due to limited GPU memory ({total_gpu_memory:.1f}GB)"
                )
                config.model.lora_enabled = True

        # Set up dataset config
        if args.dataset_name:
            config.dataset.dataset_name = args.dataset_name

        # Set output directory
        if args.output_dir:
            config.output_dir = args.output_dir

        # Set DeepSpeed config
        if args.use_deepspeed is not None:
            config.deepspeed.enabled = args.use_deepspeed

        # Set Weights & Biases config
        if args.wandb is not None:
            config.wandb.enabled = args.wandb

        # Set debug mode
        if args.debug:
            config.is_debug = True

        # Save configuration to specified path
        output_path = args.output or "mixtral_config.json"
        if safe_write_json(config.to_dict(), output_path):
            module_logger.info(f"Configuration saved to {output_path}")
            return 0
        else:
            module_logger.error(f"Failed to save configuration to {output_path}")
            return 1

    except Exception as e:
        log_exception(module_logger, e, "Error setting up configuration")
        return 1


def train_command(args: argparse.Namespace) -> int:
    """
    Handle train command.

    Args:
        args: Command-line arguments

    Returns:
        int: Exit code
    """
    try:
        # Load configuration
        if args.config:
            config_path = args.config
            module_logger.info(f"Loading configuration from {config_path}")
            config_dict = safe_read_json(config_path)
            if not config_dict:
                raise ConfigError(f"Failed to load configuration from {config_path}")

            config = TrainingConfig.from_dict(config_dict)
        else:
            # Create config from command-line arguments
            module_logger.info("Creating configuration from command-line arguments")
            config = TrainingConfig(
                run_id=str(uuid.uuid4())[:8],
                output_dir=args.output_dir or "outputs",
                is_debug=args.debug or False,
                model=ModelConfig(
                    model_name=args.model_name or "mistralai/Mixtral-8x7B-v0.1",
                    lora_enabled=args.use_lora or False,
                ),
                dataset=DatasetConfig(
                    dataset_name=args.dataset_name
                    or "timdettmers/openassistant-guanaco"
                ),
            )

            if args.use_deepspeed:
                config.deepspeed.enabled = True

            if args.wandb:
                config.wandb.enabled = True

        # Override config with command-line arguments
        if args.model_name:
            config.model.model_name = args.model_name

        if args.dataset_name:
            config.dataset.dataset_name = args.dataset_name

        if args.output_dir:
            config.output_dir = args.output_dir

        if args.use_lora is not None:
            config.model.lora_enabled = args.use_lora

        if args.use_deepspeed is not None:
            config.deepspeed.enabled = args.use_deepspeed

        if args.wandb is not None:
            config.wandb.enabled = args.wandb

        if args.debug:
            config.is_debug = True

        # Log configuration
        module_logger.info(f"Using model: {config.model.model_name}")
        module_logger.info(f"Using dataset: {config.dataset.dataset_name}")
        module_logger.info(f"Output directory: {config.output_dir}")
        module_logger.info(f"LoRA enabled: {config.model.lora_enabled}")
        module_logger.info(f"DeepSpeed enabled: {config.deepspeed.enabled}")
        module_logger.info(f"Weights & Biases enabled: {config.wandb.enabled}")

        # Import here to avoid circular dependencies
        from mixtral_training.model import load_model_and_tokenizer
        from mixtral_training.data import load_dataset
        from mixtral_training.train import train_model

        # Load model and tokenizer
        module_logger.info("Loading model and tokenizer")
        model, tokenizer = load_model_and_tokenizer(config)

        # Load dataset
        module_logger.info("Loading dataset")
        dataset = load_dataset(config, tokenizer)

        # Train model
        module_logger.info("Starting training")
        success = train_model(model, tokenizer, dataset, config)

        if success:
            module_logger.info("Training completed successfully")
            return 0
        else:
            module_logger.error("Training failed")
            return 1

    except Exception as e:
        log_exception(module_logger, e, "Error during training")
        return 1


def evaluate_command(args: argparse.Namespace) -> int:
    """
    Handle evaluate command.

    Args:
        args: Command-line arguments

    Returns:
        int: Exit code
    """
    try:
        # Import here to avoid circular dependencies
        from mixtral_training.evaluate import evaluate_model
        from mixtral_training.model import load_model_from_checkpoint

        # Use provided config or create a new one
        if args.config:
            config_path = args.config
            module_logger.info(f"Loading configuration from {config_path}")
            config_dict = safe_read_json(config_path)
            if not config_dict:
                raise ConfigError(f"Failed to load configuration from {config_path}")

            config = TrainingConfig.from_dict(config_dict)
        else:
            # Create minimal config for evaluation
            config = TrainingConfig(output_dir=args.output_dir or "evaluation_results")

        # Override config with command-line arguments
        if args.output_dir:
            config.output_dir = args.output_dir

        # Ensure checkpoint path is provided
        if not args.checkpoint:
            raise ConfigError("Checkpoint path is required for evaluation")

        # Load model and tokenizer from checkpoint
        module_logger.info(f"Loading model from checkpoint: {args.checkpoint}")
        model, tokenizer = load_model_from_checkpoint(args.checkpoint, config)

        # Load evaluation dataset
        from mixtral_training.data import load_evaluation_dataset

        dataset_name = args.dataset or config.dataset.dataset_name
        if not dataset_name:
            raise ConfigError("Dataset name is required for evaluation")

        module_logger.info(f"Loading evaluation dataset: {dataset_name}")
        eval_dataset = load_evaluation_dataset(dataset_name, tokenizer, config)

        # Evaluate model
        module_logger.info("Starting evaluation")
        results = evaluate_model(model, tokenizer, eval_dataset, config)

        # Save results
        from mixtral_training.evaluate import save_evaluation_results

        results_file = save_evaluation_results(results, config.output_dir)

        if results_file:
            module_logger.info(f"Evaluation results saved to {results_file}")
            return 0
        else:
            module_logger.error("Failed to save evaluation results")
            return 1

    except Exception as e:
        log_exception(module_logger, e, "Error during evaluation")
        return 1


def main() -> int:
    """
    Main entry point.

    Returns:
        int: Exit code
    """
    parser = argparse.ArgumentParser(description="Mixtral-8x7B R1 Reasoning Training")

    # Check if any arguments were provided
    if len(sys.argv) <= 1:
        parser.print_help()
        return 0

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup training configuration")
    setup_parser.add_argument(
        "--output", type=str, help="Path to save configuration file"
    )
    setup_parser.add_argument(
        "--model_name", type=str, help="HuggingFace model name or path"
    )
    setup_parser.add_argument("--dataset_name", type=str, help="Dataset name or path")
    setup_parser.add_argument(
        "--output_dir", type=str, help="Directory to save outputs"
    )
    setup_parser.add_argument(
        "--use_lora", action="store_true", help="Use LoRA for efficient fine-tuning"
    )
    setup_parser.add_argument(
        "--use_deepspeed", action="store_true", help="Use DeepSpeed for training"
    )
    setup_parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    setup_parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", type=str, help="Path to configuration file")
    train_parser.add_argument(
        "--model_name", type=str, help="HuggingFace model name or path"
    )
    train_parser.add_argument("--dataset_name", type=str, help="Dataset name or path")
    train_parser.add_argument(
        "--output_dir", type=str, help="Directory to save outputs"
    )
    train_parser.add_argument(
        "--use_lora", action="store_true", help="Use LoRA for efficient fine-tuning"
    )
    train_parser.add_argument(
        "--use_deepspeed", action="store_true", help="Use DeepSpeed for training"
    )
    train_parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    train_parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    eval_parser.add_argument(
        "--dataset", type=str, help="Dataset name or path for evaluation"
    )
    eval_parser.add_argument(
        "--output_dir", type=str, help="Directory to save evaluation results"
    )
    eval_parser.add_argument("--config", type=str, help="Path to configuration file")

    # Parse arguments
    args = parser.parse_args()

    # Print banner
    print("=" * 60)
    print(f"Mixtral-8x7B R1 Reasoning Training (v{mixtral_training.__version__})")
    print("=" * 60)

    # Set memory fraction for GPU
    set_memory_fraction(0.95)

    # Dispatch command
    if args.command == "setup":
        return setup_command(args)
    elif args.command == "train":
        return train_command(args)
    elif args.command == "evaluate":
        return evaluate_command(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
