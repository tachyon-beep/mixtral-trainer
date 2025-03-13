"""Evaluation module for Mixtral models."""

import os
import json
import time
import logging
import datetime
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

from mixtral_training.config import TrainingConfig
from mixtral_training.utils.logging import get_logger
from mixtral_training.utils.storage import safe_write_json, ensure_directory_exists
from mixtral_training.utils.exceptions import EvaluationError, log_exception

# Get module logger
logger = get_logger(__name__)


def calculate_perplexity(
    logits: Any, labels: Any, ignore_token_id: int = -100
) -> float:
    """
    Calculate perplexity from logits and labels.

    Args:
        logits: Model logits
        labels: True labels
        ignore_token_id: Token ID to ignore

    Returns:
        float: Perplexity score
    """
    try:
        import torch
        import numpy as np
        from torch.nn import CrossEntropyLoss

        # Create loss function
        loss_fn = CrossEntropyLoss(reduction="none")

        # Calculate loss for each token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten tensors
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        # Calculate cross-entropy loss
        losses = loss_fn(shift_logits, shift_labels)

        # Create mask for tokens to include
        mask = (shift_labels != ignore_token_id).float()

        # Filter losses based on mask
        masked_losses = losses * mask

        # Calculate mean loss for non-ignored tokens
        total_loss = masked_losses.sum()
        non_ignored_tokens = mask.sum()

        # Handle edge case of all tokens being ignored
        if non_ignored_tokens == 0:
            return float("inf")

        # Calculate mean loss
        mean_loss = total_loss / non_ignored_tokens

        # Convert to perplexity
        perplexity = torch.exp(mean_loss).item()

        return perplexity

    except Exception as e:
        logger.error(f"Error calculating perplexity: {e}")
        return float("inf")


def calculate_exact_match(
    generated_texts: List[str], reference_texts: List[str]
) -> float:
    """
    Calculate exact match score.

    Args:
        generated_texts: List of generated texts
        reference_texts: List of reference texts

    Returns:
        float: Exact match score [0-1]
    """
    if not generated_texts or not reference_texts:
        return 0.0

    # Normalize texts for comparison
    normalize = lambda text: text.strip().lower()
    normalized_generated = [normalize(text) for text in generated_texts]
    normalized_reference = [normalize(text) for text in reference_texts]

    # Count exact matches
    matches = sum(
        1 for gen, ref in zip(normalized_generated, normalized_reference) if gen == ref
    )

    # Calculate score
    return matches / len(generated_texts)


def calculate_metrics(eval_predictions: Any, tokenizer: Any) -> Dict[str, float]:
    """
    Calculate evaluation metrics.

    Args:
        eval_predictions: Model predictions
        tokenizer: Tokenizer

    Returns:
        Dict[str, float]: Metrics dictionary
    """
    try:
        logits, labels = eval_predictions

        # Calculate perplexity
        perplexity = calculate_perplexity(logits, labels)

        # Create metric dictionary
        metrics = {
            "perplexity": perplexity,
        }

        return metrics

    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {"perplexity": float("inf")}


def decode_prediction_batch(
    logits: Any, tokenizer: Any, skip_special_tokens: bool = True
) -> List[str]:
    """
    Decode prediction batch to text.

    Args:
        logits: Model logits
        tokenizer: Tokenizer
        skip_special_tokens: Whether to skip special tokens

    Returns:
        List[str]: Decoded texts
    """
    try:
        import torch

        # Get predicted token IDs
        predicted_token_ids = torch.argmax(logits, dim=-1)

        # Decode to text
        decoded_texts = tokenizer.batch_decode(
            predicted_token_ids, skip_special_tokens=skip_special_tokens
        )

        return decoded_texts

    except Exception as e:
        logger.error(f"Error decoding predictions: {e}")
        return []


def sample_evaluation_examples(
    dataset: Any, num_samples: int = 10
) -> List[Dict[str, Any]]:
    """
    Sample examples from evaluation dataset.

    Args:
        dataset: Evaluation dataset
        num_samples: Number of samples to select

    Returns:
        List[Dict[str, Any]]: Sampled examples
    """
    try:
        import random

        # Get dataset length
        dataset_length = len(dataset)

        # Adjust num_samples if needed
        num_samples = min(num_samples, dataset_length)

        # Randomly select indices
        indices = random.sample(range(dataset_length), num_samples)

        # Extract examples
        examples = [dataset[idx] for idx in indices]

        return examples

    except Exception as e:
        logger.error(f"Error sampling evaluation examples: {e}")
        return []


def prepare_sample_prompts(
    examples: List[Dict[str, Any]], tokenizer: Any, config: TrainingConfig
) -> List[str]:
    """
    Prepare sample prompts for generation.

    Args:
        examples: List of examples
        tokenizer: Tokenizer
        config: Training configuration

    Returns:
        List[str]: Sample prompts
    """
    from mixtral_training.data import format_instruction

    prompts = []

    # Extract prompts from examples
    for example in examples:
        try:
            # Format prompt using instruction template
            prompt = format_instruction(example, config.dataset.prompt_template)
            prompts.append(prompt)
        except Exception as e:
            logger.warning(f"Error preparing sample prompt: {e}")

    return prompts


def generate_sample_texts(
    model: Any, tokenizer: Any, prompts: List[str], config: TrainingConfig
) -> List[str]:
    """
    Generate sample texts from prompts.

    Args:
        model: Model to use
        tokenizer: Tokenizer
        prompts: List of prompts
        config: Training configuration

    Returns:
        List[str]: Generated texts
    """
    try:
        import torch

        # Set model to evaluation mode
        model.eval()

        # Tokenize prompts
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.model.max_length // 2,  # Use half for prompt
        )

        # Move inputs to device
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=min(
                    512, config.model.max_length // 2
                ),  # Use half for generation
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        # Decode outputs
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return generated_texts

    except Exception as e:
        logger.error(f"Error generating sample texts: {e}")
        return [f"Error: {e}"] * len(prompts)


def evaluate_model(
    model: Any,
    tokenizer: Any,
    eval_dataset: Any,
    config: Optional[TrainingConfig] = None,
    prompt_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a model.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        eval_dataset: Evaluation dataset
        config: Training configuration
        prompt_file: Path to prompt file (optional)

    Returns:
        Dict[str, Any]: Evaluation results
    """
    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import default_data_collator

        logger.info("Starting model evaluation")
        start_time = time.time()

        # Use default config if not provided
        if config is None:
            from mixtral_training.config import TrainingConfig

            config = TrainingConfig()

        # Create evaluation results dictionary
        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": {},
            "samples": [],
        }

        # Set model to evaluation mode
        model.eval()

        # Load prompts from file if provided
        custom_prompts = []
        if prompt_file and os.path.exists(prompt_file):
            try:
                with open(prompt_file, "r", encoding="utf-8") as f:
                    for line in f:
                        prompt = line.strip()
                        if prompt:
                            custom_prompts.append(prompt)
                logger.info(f"Loaded {len(custom_prompts)} prompts from {prompt_file}")
            except Exception as e:
                logger.error(f"Error loading prompts from {prompt_file}: {e}")

        # Create dataloader
        batch_size = min(
            8, config.batch_size * 2
        )  # Use larger batch size for evaluation
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            collate_fn=default_data_collator,
            shuffle=False,
        )

        # Evaluate on dataset
        total_loss = 0.0
        num_batches = 0

        logger.info("Calculating perplexity...")

        for batch in eval_dataloader:
            # Move batch to device
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}

            # Forward pass
            with torch.no_grad():
                outputs = model(**batch)

            # Calculate loss
            loss = outputs.loss.item()
            total_loss += loss
            num_batches += 1

        # Calculate average loss and perplexity
        avg_loss = total_loss / max(1, num_batches)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        # Store metrics
        results["metrics"]["loss"] = avg_loss
        results["metrics"]["perplexity"] = perplexity

        # Generate sample texts
        logger.info("Generating samples...")

        # Sample examples for generation
        examples = []
        if custom_prompts:
            # Use custom prompts
            examples = [{"instruction": prompt} for prompt in custom_prompts]
        elif eval_dataset is not None:
            # Sample from dataset
            examples = sample_evaluation_examples(eval_dataset)

        # Prepare prompts
        prompts = prepare_sample_prompts(examples, tokenizer, config)

        # Generate texts
        if prompts:
            generated_texts = generate_sample_texts(model, tokenizer, prompts, config)

            # Store samples
            for prompt, generated_text in zip(prompts, generated_texts):
                results["samples"].append(
                    {
                        "prompt": prompt,
                        "generated_text": generated_text,
                    }
                )

        # Calculate total time
        total_time = time.time() - start_time
        results["time_seconds"] = total_time

        logger.info(f"Evaluation completed in {total_time:.2f} seconds")
        logger.info(f"Perplexity: {perplexity:.4f}")

        return results

    except Exception as e:
        log_exception(logger, e, "Evaluation error")
        raise EvaluationError(f"Evaluation failed: {e}")


def save_evaluation_results(results: Dict[str, Any], output_dir: str) -> str:
    """
    Save evaluation results to file.

    Args:
        results: Evaluation results
        output_dir: Output directory

    Returns:
        str: Path to results file
    """
    try:
        ensure_directory_exists(output_dir)

        # Create timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        results_file = os.path.join(output_dir, f"eval_results_{timestamp}.json")

        # Save results
        if safe_write_json(results, results_file):
            logger.info(f"Evaluation results saved to {results_file}")
            return results_file
        else:
            logger.error(f"Failed to save evaluation results to {results_file}")
            return ""

    except Exception as e:
        logger.error(f"Error saving evaluation results: {e}")
        return ""
