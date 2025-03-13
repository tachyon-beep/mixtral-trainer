"""Training module for Mixtral models."""

import os
import time
import math
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import torch
from torch import Tensor

from mixtral_training.config import TrainingConfig
from mixtral_training.utils.logging import get_logger
from mixtral_training.utils.memory import log_memory_stats
from mixtral_training.utils.storage import ensure_directory_exists
from mixtral_training.utils.checkpoint import (
    create_checkpoint_dir,
    save_checkpoint_config,
    create_checkpoint_state,
    find_latest_checkpoint,
)
from mixtral_training.utils.exceptions import TrainingError, log_exception
from mixtral_training.routing import (
    calculate_router_loss,
    analyze_router,
    extract_routing_logits,
    extract_reasoning_operation_masks,
    save_router_analysis,
    visualize_expert_utilization,
    RouterAnalysisResult,
)

# Get module logger
logger = get_logger(__name__)


class MixtralTrainer:
    """Custom trainer with MoE optimization for Mixtral models."""

    def __init__(self, trainer, model, tokenizer, config: TrainingConfig):
        """
        Initialize MixtralTrainer.

        Args:
            trainer: Hugging Face Trainer instance
            model: Model to train
            tokenizer: Tokenizer
            config: Training configuration
        """
        self.trainer = trainer
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.router_analyses = []  # Store router analyses

        # Set up custom training step if router optimization is enabled
        if config.model.router.enable_router_optimization:
            self._setup_custom_training_step()

        # Set up router analysis logging
        if (
            config.model.router.track_expert_assignments
            or config.model.expert_collaboration.enable_collaboration_tracking
        ):
            self._setup_analysis_callbacks()

    def _setup_custom_training_step(self):
        """Set up custom training step with router optimization."""
        original_training_step = self.trainer.training_step

        def custom_training_step(model, inputs):
            """Custom training step with router optimization."""
            # Forward pass to compute loss and routing logits
            outputs = model(**inputs)
            loss = outputs.loss

            # Extract routing logits
            routing_logits = extract_routing_logits(model, inputs)

            # Calculate router loss if we have routing logits
            if routing_logits:
                router_loss = calculate_router_loss(routing_logits, self.config)
                # Combine with main loss
                loss = loss + router_loss

            return loss

        # Replace training step
        self.trainer.training_step = custom_training_step

    def _setup_analysis_callbacks(self):
        """Set up callbacks for router analysis."""
        # Store original state dict to avoid circular reference
        original_state_dict = self.trainer.state.__dict__.copy()

        def on_step_end(args, state, control, **kwargs):
            """Analyze router on step end."""
            # Check if we should do analysis based on logging steps
            if (
                state.global_step
                % self.config.model.router.expert_assignment_logging_steps
                == 0
            ):
                self._perform_router_analysis(state.global_step)
            return control

        # Add callback
        self.trainer.add_callback({"on_step_end": on_step_end})

        # Restore state to avoid circular reference
        self.trainer.state.__dict__ = original_state_dict

    def _perform_router_analysis(self, step: int):
        """
        Perform router analysis and save results.

        Args:
            step: Current training step
        """
        # Get a batch from evaluation dataset
        if self.trainer.eval_dataset is not None:
            from torch.utils.data import DataLoader

            eval_dataloader = DataLoader(
                self.trainer.eval_dataset,
                batch_size=4,  # Small batch for analysis
                shuffle=False,
                collate_fn=self.trainer.data_collator,
            )

            # Get a single batch
            for batch in eval_dataloader:
                # Move to appropriate device
                batch = {
                    k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Extract reasoning operation masks if configured
                reasoning_op_masks = None
                if self.config.model.router.track_reasoning_operations:
                    reasoning_ops = self.config.model.router.reasoning_operations
                    input_ids = batch.get("input_ids")
                    attention_mask = batch.get("attention_mask")

                    if input_ids is not None and attention_mask is not None:
                        reasoning_op_masks = extract_reasoning_operation_masks(
                            input_ids, attention_mask, self.tokenizer, reasoning_ops
                        )

                # Analyze router
                analysis = analyze_router(
                    self.model,
                    batch,
                    self.config,
                    reasoning_op_masks=reasoning_op_masks,
                )

                # Save analysis
                self.router_analyses.append(analysis)

                # Save to file
                save_path = save_router_analysis(analysis, self.config.output_dir, step)

                # Log summary
                analysis.log_summary()

                # Only process one batch
                break

    def save_visualizations(self):
        """Save router analysis visualizations."""
        if self.router_analyses:
            # Save expert utilization visualization
            visualize_expert_utilization(self.router_analyses, self.config.output_dir)

    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, float]:
        """
        Train the model.

        Args:
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Dict[str, float]: Training metrics
        """
        # Run training
        output = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save analysis visualizations
        self.save_visualizations()

        return output

    def save_model(self, output_dir: Optional[str] = None):
        """
        Save the model.

        Args:
            output_dir: Output directory (uses config.output_dir if None)
        """
        dir_to_use = output_dir or self.config.output_dir
        self.trainer.save_model(dir_to_use)
        self.tokenizer.save_pretrained(dir_to_use)


def setup_training(model, tokenizer, dataset, config: TrainingConfig) -> Dict[str, Any]:
    """
    Set up training components.

    Args:
        model: Model to train
        tokenizer: Tokenizer
        dataset: Dataset dict with train_dataset, eval_dataset
        config: Training configuration

    Returns:
        Dict[str, Any]: Training components
    """
    try:
        import torch
        from transformers import Trainer, TrainingArguments, default_data_collator

        # Ensure output directory exists
        ensure_directory_exists(config.output_dir)

        # Setup data collator if not provided
        data_collator = dataset.get("data_collator", default_data_collator)

        # Create training arguments
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            overwrite_output_dir=True,
            # Training parameters
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps,
            # Schedule
            max_steps=config.max_steps,
            save_steps=config.save_steps,
            eval_steps=config.eval_steps,
            logging_steps=config.logging_steps,
            save_total_limit=3,  # Keep only last 3 checkpoints
            # Mixed precision
            fp16=config.model.precision_type.is_float16(),
            bf16=config.model.precision_type.is_bfloat16()
            and torch.cuda.get_device_capability()[0] >= 8,
            # Eval and metrics
            evaluation_strategy="steps",
            logging_strategy="steps",
            save_strategy="steps",
            # Seeds
            seed=config.seed,
            data_seed=config.seed,
            # Misc
            disable_tqdm=False,
            group_by_length=True,  # Group samples of similar lengths for efficiency
            report_to=["wandb"] if config.wandb.enabled else [],
            run_name=config.run_name,
            ddp_find_unused_parameters=False,
        )

        # Setup DeepSpeed if enabled
        if config.deepspeed.enabled:
            training_args.deepspeed = config.deepspeed.to_dict(
                batch_size=config.batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                gpu_count=config.gpu.gpu_count,
            )
            logger.info(
                f"DeepSpeed enabled with ZeRO-{config.deepspeed.zero_stage.value}"
            )

        # Create a custom Trainer class if router optimization is enabled
        if config.model.router.enable_router_optimization:

            class MixtralRouterTrainer(Trainer):
                """Custom Trainer with router optimization."""

                def __init__(self, *args, config: TrainingConfig, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.config = config

                def compute_loss(
                    self, model, inputs, return_outputs=False, num_items_in_batch=None
                ):
                    """
                    Custom compute_loss method that adds router optimization losses.

                    Args:
                        model: Model
                        inputs: Model inputs
                        return_outputs: Whether to return outputs
                        num_items_in_batch: Number of items in batch (for gradient accumulation)

                    Returns:
                        Union[torch.Tensor, Tuple[torch.Tensor, Any]]: Loss or (loss, outputs)
                    """
                    # Forward pass
                    outputs = model(**inputs)
                    loss = outputs.loss

                    # Add router optimization losses
                    routing_logits = extract_routing_logits(model, inputs)
                    if routing_logits:
                        # Calculate and add router loss
                        router_loss = calculate_router_loss(routing_logits, self.config)
                        loss = loss + router_loss

                    return (loss, outputs) if return_outputs else loss

            # Use custom trainer
            trainer = MixtralRouterTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset["train_dataset"],
                eval_dataset=dataset["eval_dataset"],
                data_collator=data_collator,
                config=config,
            )
        else:
            # Use standard trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset["train_dataset"],
                eval_dataset=dataset["eval_dataset"],
                data_collator=data_collator,
            )

        # Store tokenizer in trainer's state for later use
        trainer.tokenizer = tokenizer

        # Create MixtralTrainer for MoE-specific optimizations
        mixtral_trainer = MixtralTrainer(trainer, model, tokenizer, config)

        # Save full config
        save_checkpoint_config(config.output_dir, config.to_dict())

        return {
            "trainer": trainer,
            "mixtral_trainer": mixtral_trainer,
            "training_args": training_args,
            "model": model,
            "tokenizer": tokenizer,
        }

    except Exception as e:
        log_exception(logger, e, "Failed to setup training")
        raise TrainingError(f"Failed to setup training: {e}")


def train_model(model, tokenizer, dataset, config: TrainingConfig) -> bool:
    """
    Train the model.

    Args:
        model: Model to train
        tokenizer: Tokenizer
        dataset: Dataset dict with train_dataset, eval_dataset
        config: Training configuration

    Returns:
        bool: True if successful
    """
    start_time = time.time()

    try:
        # Setup training components
        components = setup_training(model, tokenizer, dataset, config)
        mixtral_trainer = components["mixtral_trainer"]

        # Log starting message
        logger.info(f"Starting training with {config.model.model_name}")
        logger.info(f"Batch size: {config.batch_size}, Steps: {config.max_steps}")
        if config.model.lora_enabled:
            logger.info(
                f"Using LoRA with r={config.model.lora_r}, alpha={config.model.lora_alpha}"
            )

        # Log memory stats before training
        log_memory_stats("Before training")

        # Train model with MixtralTrainer
        mixtral_trainer.train()

        # Save final model
        logger.info("Saving final model")
        mixtral_trainer.save_model(config.output_dir)

        # Calculate training time
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)

        logger.info(f"Training completed in {hours}h {minutes}m {seconds}s")
        logger.info(f"Final model saved to {config.output_dir}")

        return True

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return False

    except Exception as e:
        log_exception(logger, e, "Training error")
        return False


def predict_batch(
    model, tokenizer, input_texts: List[str], config: TrainingConfig
) -> List[str]:
    """
    Generate predictions for a batch of input texts.

    Args:
        model: Model to use
        tokenizer: Tokenizer
        input_texts: List of input texts
        config: Configuration

    Returns:
        List[str]: Generated texts
    """
    try:
        import torch

        # Set model to evaluation mode
        model.eval()

        # Tokenize input texts
        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.model.max_length,
        )

        # Move to appropriate device
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate predictions
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=512,  # Generate up to 512 new tokens
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        # Decode outputs
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return generated_texts

    except Exception as e:
        log_exception(logger, e, "Prediction error")
        return [f"Error: {e}"] * len(input_texts)


def resume_training(
    checkpoint_path: str, model, tokenizer, dataset, config: TrainingConfig
) -> bool:
    """
    Resume training from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model: Model
        tokenizer: Tokenizer
        dataset: Dataset
        config: Configuration

    Returns:
        bool: True if successful
    """
    try:
        # Setup training components
        components = setup_training(model, tokenizer, dataset, config)
        mixtral_trainer = components["mixtral_trainer"]

        # Resume training
        logger.info(f"Resuming training from {checkpoint_path}")
        mixtral_trainer.train(resume_from_checkpoint=checkpoint_path)

        # Save final model
        logger.info("Saving final model")
        mixtral_trainer.save_model(config.output_dir)

        return True

    except Exception as e:
        log_exception(logger, e, "Resume training error")
        return False
