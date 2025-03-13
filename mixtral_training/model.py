"""Model loading and configuration for Mixtral training."""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple, Union

from mixtral_training.config import TrainingConfig, PrecisionType
from mixtral_training.utils.logging import get_logger
from mixtral_training.utils.memory import log_memory_stats
from mixtral_training.utils.exceptions import ModelError, log_exception

# Get module logger
logger = get_logger(__name__)


def load_model_and_tokenizer(config: TrainingConfig) -> Tuple[Any, Any]:
    """
    Load model and tokenizer based on configuration.

    Args:
        config: Training configuration

    Returns:
        Tuple[Any, Any]: (model, tokenizer)

    Raises:
        ModelError: If model loading fails
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft.utils.other import prepare_model_for_kbit_training
        from peft.tuners.lora import LoraConfig
        from peft.mapping import get_peft_model

        # Log memory stats before loading model
        log_memory_stats("Before model loading")

        # Setup quantization config for low-bit precision
        quantization_config = None
        if config.model.precision_type == PrecisionType.INT8:
            logger.info("Using 8-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        elif config.model.precision_type == PrecisionType.INT4:
            logger.info("Using 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        # Determine device to use
        device_map = "auto"
        if config.gpu.gpu_count == 0:
            # CPU only
            device_map = "cpu"
            logger.warning("No GPUs detected, using CPU only")

        # Load tokenizer
        logger.info(f"Loading tokenizer for {config.model.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.model_name,
            use_fast=True,
            padding_side="right",
            legacy=False,
        )

        # Add padding token if needed
        if tokenizer.pad_token is None:
            logger.info("Adding padding token to tokenizer")
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with quantization if enabled
        logger.info(f"Loading model {config.model.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=get_torch_dtype(config.model.precision_type),
            trust_remote_code=True,
            cache_dir=(
                os.path.join(config.output_dir, "model_cache")
                if config.output_dir
                else None
            ),
        )
        model.config.pad_token_id = tokenizer.pad_token_id

        # Log memory stats after loading base model
        log_memory_stats("After model loading")

        # Configure LoRA if enabled
        if config.model.lora_enabled:
            logger.info(
                f"Setting up LoRA with rank={config.model.lora_r}, alpha={config.model.lora_alpha}"
            )
            if quantization_config is not None:
                model = prepare_model_for_kbit_training(model)

            lora_config = LoraConfig(
                r=config.model.lora_r,
                lora_alpha=config.model.lora_alpha,
                target_modules=config.model.lora_target_modules,
                lora_dropout=config.model.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )

            # Apply LoRA
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

            # Log memory stats after LoRA setup
            log_memory_stats("After LoRA setup")

        return model, tokenizer

    except Exception as e:
        log_exception(logger, e, "Error loading model")
        raise ModelError(f"Failed to load model: {e}")


def load_model_from_checkpoint(
    checkpoint_path: str, config: TrainingConfig
) -> Tuple[Any, Any]:
    """
    Load model and tokenizer from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory
        config: Training configuration

    Returns:
        Tuple[Any, Any]: (model, tokenizer)

    Raises:
        ModelError: If model loading fails
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft.peft_model import PeftModel
        from peft.config import PeftConfig

        logger.info(f"Loading model from checkpoint: {checkpoint_path}")

        # Check if this is a LoRA checkpoint
        peft_config_path = os.path.join(checkpoint_path, "adapter_config.json")
        is_lora_checkpoint = os.path.exists(peft_config_path)

        # Load tokenizer
        logger.info("Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        if is_lora_checkpoint:
            # Load base model and apply LoRA adapter
            logger.info("Loading LoRA model")
            peft_config = PeftConfig.from_pretrained(checkpoint_path)

            # Load base model
            if (
                hasattr(peft_config, "base_model_name_or_path")
                and peft_config.base_model_name_or_path
            ):
                base_model_path = peft_config.base_model_name_or_path
            else:
                # Fallback to default model
                logger.warning(
                    "No base model path found in adapter config, using default model"
                )
                base_model_path = "mistralai/Mixtral-8x7B-v0.1"

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=get_torch_dtype(config.model.precision_type),
                device_map="auto",
                trust_remote_code=True,
            )

            # Load LoRA adapter
            model = PeftModel.from_pretrained(base_model, checkpoint_path)
        else:
            # Load regular model
            logger.info("Loading full model")
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=get_torch_dtype(config.model.precision_type),
                device_map="auto",
                trust_remote_code=True,
            )

        return model, tokenizer

    except Exception as e:
        log_exception(logger, e, "Error loading model from checkpoint")
        raise ModelError(f"Failed to load model from checkpoint: {e}")


def get_torch_dtype(precision_type: PrecisionType) -> Any:
    """
    Get torch dtype based on precision type.

    Args:
        precision_type: Precision type

    Returns:
        Any: Torch dtype
    """
    import torch

    if precision_type == PrecisionType.FP16:
        return torch.float16
    elif precision_type == PrecisionType.BF16:
        return torch.bfloat16
    else:
        return torch.float32  # Default to float32 for other precision types


def save_trained_model(
    model: Any, tokenizer: Any, output_dir: str, is_lora: bool = False
) -> str:
    """
    Save trained model and tokenizer.

    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Output directory
        is_lora: Whether this is a LoRA model

    Returns:
        str: Path to saved model

    Raises:
        ModelError: If saving fails
    """
    try:
        final_dir = os.path.join(output_dir, "final_model")
        os.makedirs(final_dir, exist_ok=True)

        logger.info(f"Saving model to {final_dir}")

        if is_lora:
            # Save LoRA adapter
            model.save_pretrained(final_dir)
        else:
            # Save full model
            model.save_pretrained(final_dir, safe_serialization=True)

        # Save tokenizer
        tokenizer.save_pretrained(final_dir)

        logger.info(f"Model and tokenizer saved to {final_dir}")
        return final_dir

    except Exception as e:
        log_exception(logger, e, "Error saving model")
        raise ModelError(f"Failed to save model: {e}")
