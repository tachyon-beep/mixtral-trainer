"""Router analysis and optimization for Mixtral MoE models."""

import os
import math
import logging
import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mixtral_training.config import TrainingConfig, RouterConfig, ReasoningOperationType
from mixtral_training.utils.logging import get_logger
from mixtral_training.utils.storage import safe_write_json, ensure_directory_exists

# Get module logger
logger = get_logger(__name__)


@dataclass
class RouterAnalysisResult:
    """Results from router analysis."""

    # Expert utilization metrics
    expert_loads: List[float]  # Fraction of tokens assigned to each expert
    expert_importance: List[float]  # Router importance scores
    gini_coefficient: (
        float  # Measure of load imbalance (0=perfect balance, 1=all load to one expert)
    )
    entropy: float  # Entropy of load distribution (higher=more balanced)

    # Reasoning operation analysis
    reasoning_op_assignments: Dict[str, List[float]]  # Ops to expert assignments
    reasoning_op_confidence: Dict[str, float]  # Routing confidence per operation

    # Expert co-activation metrics (which experts tend to work together)
    co_activation_matrix: Optional[np.ndarray] = None
    expert_transition_matrix: Optional[np.ndarray] = None

    # Expert specialization metrics
    specialization_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "expert_loads": self.expert_loads,
            "expert_importance": self.expert_importance,
            "gini_coefficient": self.gini_coefficient,
            "entropy": self.entropy,
            "reasoning_op_assignments": self.reasoning_op_assignments,
            "reasoning_op_confidence": self.reasoning_op_confidence,
        }

        if self.co_activation_matrix is not None:
            result["co_activation_matrix"] = self.co_activation_matrix.tolist()

        if self.expert_transition_matrix is not None:
            result["expert_transition_matrix"] = self.expert_transition_matrix.tolist()

        if self.specialization_scores is not None:
            result["specialization_scores"] = self.specialization_scores

        return result

    def save(self, path: str) -> bool:
        """Save analysis results to file."""
        ensure_directory_exists(os.path.dirname(path))
        return safe_write_json(self.to_dict(), path)

    def log_summary(self) -> None:
        """Log a summary of the analysis results."""
        logger.info("Router Analysis Summary:")
        logger.info(
            f"  Expert Load Balance (Gini): {self.gini_coefficient:.4f} (0=balanced, 1=imbalanced)"
        )
        logger.info(
            f"  Expert Load Distribution (Entropy): {self.entropy:.4f} (higher=more balanced)"
        )

        # Log top expert for each reasoning operation
        logger.info("  Reasoning Operation Routing:")
        for op_name, expert_probs in self.reasoning_op_assignments.items():
            top_expert_idx = np.argmax(expert_probs)
            top_expert_prob = expert_probs[top_expert_idx]
            logger.info(
                f"    {op_name}: Expert {top_expert_idx} ({top_expert_prob:.2f})"
            )

        # Log specialization scores if available
        if self.specialization_scores:
            logger.info("  Expert Specialization Scores:")
            for metric, score in self.specialization_scores.items():
                logger.info(f"    {metric}: {score:.4f}")


def extract_routing_logits(model: nn.Module, batch: Dict[str, Tensor]) -> List[Tensor]:
    """
    Extract routing logits from model forward pass.

    Args:
        model: Mixtral model
        batch: Input batch

    Returns:
        List[Tensor]: List of routing logits tensors
    """
    # Run forward pass with hook to capture routing logits
    routing_logits = []

    def hook_fn(module, input, output):
        # Capture router logits
        # This is specific to Mixtral model architecture
        # For a standard implementation, this would capture the logits from each router
        routing_logits.append(
            output[1]
        )  # Assuming output[1] contains the routing logits

    # Find router modules and register hooks
    router_modules = []
    for name, module in model.named_modules():
        # Look for router modules in Mixtral architecture
        # This pattern will need to be adjusted based on exact model implementation
        if "router" in name.lower() or "gate" in name.lower():
            router_modules.append(module)
            module.register_forward_hook(hook_fn)

    # Run forward pass to trigger hooks
    with torch.no_grad():
        _ = model(**batch)

    # Remove hooks to avoid memory leaks
    for module in router_modules:
        module._forward_hooks.clear()

    return routing_logits


def get_expert_assignments(
    routing_logits: List[Tensor], top_k: int = 2
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Get expert assignments from routing logits.

    Args:
        routing_logits: List of routing logits tensors
        top_k: Number of experts selected per token

    Returns:
        Tuple[List[Tensor], List[Tensor]]: Expert indices and weights
    """
    expert_indices = []
    expert_weights = []

    for logits in routing_logits:
        # Get top-k experts and their weights
        weights, indices = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
        expert_indices.append(indices)
        expert_weights.append(weights)

    return expert_indices, expert_weights


def calculate_load_balancing_loss(
    routing_logits: List[Tensor], config: RouterConfig
) -> Tensor:
    """
    Calculate load balancing auxiliary loss for router.

    Args:
        routing_logits: List of routing logits tensors
        config: Router configuration

    Returns:
        Tensor: Load balancing loss
    """
    aux_loss = torch.tensor(0.0, device=routing_logits[0].device)

    for logits in routing_logits:
        # Calculate router probability
        router_probs = torch.softmax(
            logits, dim=-1
        )  # [batch_size, seq_len, num_experts]

        # Calculate mean routing probability
        router_probs_mean = router_probs.mean(dim=[0, 1])  # [num_experts]

        # Number of experts
        num_experts = router_probs.size(-1)

        # Compute auxiliary loss
        # This loss encourages uniform expert assignment
        aux_loss_layer = num_experts * torch.sum(router_probs_mean * router_probs_mean)
        aux_loss = aux_loss + aux_loss_layer

    # Average across all layers
    if len(routing_logits) > 0:
        aux_loss = aux_loss / len(routing_logits)

    return aux_loss


def calculate_z_loss(routing_logits: List[Tensor], coefficient: float = 1e-3) -> Tensor:
    """
    Calculate router z-loss to penalize large logits.

    The z-loss encourages router logits to stay small, preventing
    the router from becoming too confident (saturating).

    Args:
        routing_logits: List of routing logits tensors
        coefficient: Z-loss coefficient

    Returns:
        Tensor: Z-loss value
    """
    z_loss = torch.tensor(0.0, device=routing_logits[0].device)

    for logits in routing_logits:
        # Calculate log of sum of exponentials
        log_z = torch.logsumexp(logits, dim=-1)  # [batch_size, seq_len]

        # Calculate z-loss: torch.square(log_z)
        z_loss_layer = torch.mean(torch.square(log_z))
        z_loss = z_loss + z_loss_layer

    # Average across all layers and apply coefficient
    if len(routing_logits) > 0:
        z_loss = coefficient * (z_loss / len(routing_logits))

    return z_loss


def calculate_expert_importance_loss(
    routing_logits: List[Tensor], coefficient: float = 0.01
) -> Tensor:
    """
    Calculate expert importance loss.

    This loss encourages all experts to be used by at least some tokens.

    Args:
        routing_logits: List of routing logits tensors
        coefficient: Importance loss coefficient

    Returns:
        Tensor: Importance loss value
    """
    importance_loss = torch.tensor(0.0, device=routing_logits[0].device)

    for logits in routing_logits:
        # Calculate router probability
        router_probs = torch.softmax(
            logits, dim=-1
        )  # [batch_size, seq_len, num_experts]

        # Calculate fraction of tokens routing to each expert
        expert_usage = router_probs.mean(dim=[0, 1])  # [num_experts]

        # Expert importance loss: encourage all experts to be used
        # Each expert should process at least 1/num_experts of the tokens
        num_experts = router_probs.size(-1)
        target_usage = 1.0 / num_experts

        # Penalize experts that are used less than target
        # MSE between expert usage and target usage, only counting experts below target
        below_target = torch.relu(target_usage - expert_usage)
        importance_loss_layer = torch.mean(torch.square(below_target))
        importance_loss = importance_loss + importance_loss_layer

    # Average across all layers and apply coefficient
    if len(routing_logits) > 0:
        importance_loss = coefficient * (importance_loss / len(routing_logits))

    return importance_loss


def calculate_expert_specialization_metrics(
    routing_logits: List[Tensor], token_features: Optional[Dict[str, Tensor]] = None
) -> Dict[str, float]:
    """
    Calculate metrics for expert specialization.

    Args:
        routing_logits: List of routing logits tensors
        token_features: Optional dictionary mapping feature names to token features

    Returns:
        Dict[str, float]: Specialization metrics
    """
    metrics = {}

    # Process all layers and average the metrics
    entropy_sum = 0.0
    gini_sum = 0.0
    specialization_sum = 0.0

    for logits in routing_logits:
        # Calculate router probability
        router_probs = torch.softmax(
            logits, dim=-1
        )  # [batch_size, seq_len, num_experts]

        # Average routing probability across batch and sequence dimensions
        mean_probs = (
            router_probs.mean(dim=[0, 1]).detach().cpu().numpy()
        )  # [num_experts]

        # Calculate entropy (higher entropy = more balanced)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
        entropy_sum += entropy

        # Calculate Gini coefficient (0 = perfect equality, 1 = perfect inequality)
        sorted_probs = np.sort(mean_probs)
        n = len(sorted_probs)
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * sorted_probs)) / (n * np.sum(sorted_probs))
        gini_sum += gini

        # Expert specialization score (how specialized are experts to particular inputs)
        if token_features is not None:
            # This would require token feature vectors and expert assignments
            # Just placeholder for now
            specialization_sum += 0.5

    # Average across layers
    num_layers = len(routing_logits)
    if num_layers > 0:
        metrics["entropy"] = entropy_sum / num_layers
        metrics["gini_coefficient"] = gini_sum / num_layers
        if token_features is not None:
            metrics["specialization_score"] = specialization_sum / num_layers

    return metrics


def track_expert_assignments_by_operation(
    routing_logits: List[Tensor],
    reasoning_operations: Dict[str, Tensor],
    config: RouterConfig,
) -> Dict[str, List[float]]:
    """
    Track expert assignments for different reasoning operations.

    Args:
        routing_logits: List of routing logits tensors
        reasoning_operations: Dict mapping operation names to binary masks
        config: Router configuration

    Returns:
        Dict[str, List[float]]: Mapping from operation types to expert probabilities
    """
    operation_to_expert = {}

    # Process all layers
    for layer_idx, logits in enumerate(routing_logits):
        # Calculate router probability
        router_probs = torch.softmax(
            logits, dim=-1
        )  # [batch_size, seq_len, num_experts]
        num_experts = router_probs.size(-1)

        # Process each reasoning operation
        for op_name, op_mask in reasoning_operations.items():
            # Apply operation mask to get relevant routing probabilities
            # op_mask shape: [batch_size, seq_len]
            # We want to average over just the tokens marked with this operation
            masked_probs = router_probs * op_mask.unsqueeze(
                -1
            )  # [batch_size, seq_len, num_experts]

            # Count tokens with this operation
            op_token_count = op_mask.sum().item()

            if op_token_count > 0:
                # Average routing probability for this operation
                op_expert_probs = (
                    masked_probs.sum(dim=[0, 1]).detach().cpu().numpy() / op_token_count
                )

                # Update or initialize operation to expert mapping
                if op_name not in operation_to_expert:
                    operation_to_expert[op_name] = [0.0] * num_experts

                # Accumulate probabilities across layers
                for expert_idx in range(num_experts):
                    operation_to_expert[op_name][expert_idx] += op_expert_probs[
                        expert_idx
                    ] / len(routing_logits)

    return operation_to_expert


def track_expert_transitions(
    expert_indices: List[Tensor], config: TrainingConfig
) -> np.ndarray:
    """
    Track expert transitions in sequential processing.

    Args:
        expert_indices: List of expert indices tensors
        config: Training configuration

    Returns:
        np.ndarray: Expert transition matrix
    """
    # Get number of experts
    num_experts = config.model.num_experts

    # Initialize transition matrix
    transition_matrix = np.zeros((num_experts, num_experts))

    # Process all layers
    for layer_idx, indices in enumerate(expert_indices):
        # Get expert indices
        expert_idx = indices.detach().cpu().numpy()  # [batch_size, seq_len, top_k]

        # For each sequence, count transitions between consecutive tokens
        batch_size, seq_len, top_k = expert_idx.shape

        for batch in range(batch_size):
            for seq in range(seq_len - 1):
                # Count transitions for each expert pair
                for k1 in range(top_k):
                    curr_expert = expert_idx[batch, seq, k1]
                    # Transition to experts in next token
                    for k2 in range(top_k):
                        next_expert = expert_idx[batch, seq + 1, k2]
                        transition_matrix[curr_expert, next_expert] += 1

    # Normalize matrix by row sums (if non-zero)
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    transition_matrix = transition_matrix / row_sums

    return transition_matrix


def analyze_router(
    model: nn.Module,
    batch: Dict[str, Tensor],
    config: TrainingConfig,
    reasoning_op_masks: Optional[Dict[str, Tensor]] = None,
) -> RouterAnalysisResult:
    """
    Analyze router behavior for a batch of data.

    Args:
        model: Mixtral model
        batch: Input batch
        config: Training configuration
        reasoning_op_masks: Optional dict mapping operation names to binary masks

    Returns:
        RouterAnalysisResult: Router analysis results
    """
    # Extract routing logits
    routing_logits = extract_routing_logits(model, batch)

    if not routing_logits:
        logger.warning("No routing logits captured. Check model architecture.")
        # Return empty result
        return RouterAnalysisResult(
            expert_loads=[0.0] * config.model.num_experts,
            expert_importance=[0.0] * config.model.num_experts,
            gini_coefficient=0.0,
            entropy=0.0,
            reasoning_op_assignments={},
            reasoning_op_confidence={},
        )

    # Get expert assignments
    expert_indices, expert_weights = get_expert_assignments(
        routing_logits, top_k=2  # Mixtral default top-k
    )

    # Calculate expert loads
    expert_loads = [0.0] * config.model.num_experts

    for logits in routing_logits:
        # Calculate router probability
        router_probs = torch.softmax(
            logits, dim=-1
        )  # [batch_size, seq_len, num_experts]

        # Calculate fraction of tokens routing to each expert
        layer_expert_probs = router_probs.mean(dim=[0, 1]).detach().cpu().numpy()

        # Accumulate expert loads across layers
        for expert_idx in range(config.model.num_experts):
            expert_loads[expert_idx] += layer_expert_probs[expert_idx] / len(
                routing_logits
            )

    # Calculate expert importance (simplified version based on load)
    expert_importance = expert_loads.copy()

    # Calculate expert specialization metrics
    specialization_metrics = calculate_expert_specialization_metrics(routing_logits)

    # Track expert assignments by reasoning operation
    reasoning_op_assignments = {}
    reasoning_op_confidence = {}

    if reasoning_op_masks is not None:
        reasoning_op_assignments = track_expert_assignments_by_operation(
            routing_logits, reasoning_op_masks, config.model.router
        )

        # Calculate routing confidence per operation
        for op_name, expert_probs in reasoning_op_assignments.items():
            # Higher max prob = more confident routing
            max_prob = max(expert_probs)
            reasoning_op_confidence[op_name] = max_prob

    # Track expert transitions if enabled
    expert_transition_matrix = None
    if config.model.expert_collaboration.analyze_expert_transitions:
        expert_transition_matrix = track_expert_transitions(expert_indices, config)

    # Return analysis results
    return RouterAnalysisResult(
        expert_loads=expert_loads,
        expert_importance=expert_importance,
        gini_coefficient=specialization_metrics.get("gini_coefficient", 0.0),
        entropy=specialization_metrics.get("entropy", 0.0),
        reasoning_op_assignments=reasoning_op_assignments,
        reasoning_op_confidence=reasoning_op_confidence,
        expert_transition_matrix=expert_transition_matrix,
        specialization_scores=specialization_metrics,
    )


def extract_reasoning_operation_masks(
    input_ids: Tensor,
    attention_mask: Tensor,
    tokenizer,
    reasoning_ops: List[str],
) -> Dict[str, Tensor]:
    """
    Extract binary masks for different reasoning operations in the input.

    This is a simplified implementation. A more advanced implementation
    would use a reasoning operation classifier model.

    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        tokenizer: Tokenizer
        reasoning_ops: List of reasoning operation types

    Returns:
        Dict[str, Tensor]: Mapping from operation types to binary masks
    """
    # Decode input_ids to text
    texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    # Initialize masks for each operation
    batch_size, seq_len = input_ids.shape
    masks = {}
    for op in reasoning_ops:
        masks[op] = torch.zeros((batch_size, seq_len), device=input_ids.device)

    # Simple keyword-based heuristics for reasoning operations
    op_keywords = {
        "deduction": ["therefore", "thus", "hence", "so", "because", "as a result"],
        "induction": ["generally", "usually", "tend to", "in most cases", "observed"],
        "abduction": ["best explanation", "could be", "might be", "possibility"],
        "analogy": ["like", "similar to", "resembles", "comparable to", "as with"],
        "causal": [
            "causes",
            "leads to",
            "results in",
            "due to",
            "effect of",
            "because",
        ],
        "mathematical": ["calculate", "compute", "=", "+", "-", "*", "/", "equals"],
        "temporal": ["before", "after", "during", "when", "while", "then", "firstly"],
        "spatial": ["above", "below", "behind", "next to", "inside", "outside"],
    }

    # Process each text
    for batch_idx, text in enumerate(texts):
        text_lower = text.lower()

        # Get valid token positions (where attention_mask is 1)
        valid_positions = attention_mask[batch_idx].bool()

        # For each operation, mark tokens if keywords are present
        for op in reasoning_ops:
            if op in op_keywords:
                # Check for each keyword
                for keyword in op_keywords[op]:
                    if keyword in text_lower:
                        # This is a very simplified approach
                        # A more sophisticated approach would use token alignment
                        # to identify the exact token positions for the operation

                        # Mark a random subset of tokens as this operation
                        # (This is just a placeholder - in a real implementation,
                        # you would use a classifier or more precise heuristics)
                        valid_token_positions = torch.nonzero(valid_positions).squeeze(
                            -1
                        )
                        if len(valid_token_positions) > 0:
                            num_tokens = min(5, len(valid_token_positions))
                            indices = torch.randperm(len(valid_token_positions))[
                                :num_tokens
                            ]
                            for idx in valid_token_positions[indices]:
                                masks[op][batch_idx, idx] = 1

    return masks


def calculate_router_loss(
    routing_logits: List[Tensor], config: TrainingConfig
) -> Tensor:
    """
    Calculate combined router optimization loss.

    Args:
        routing_logits: List of routing logits tensors
        config: Training configuration

    Returns:
        Tensor: Combined router loss
    """
    router_config = config.model.router

    # Calculate load balancing loss
    load_balancing_loss = torch.tensor(0.0, device=routing_logits[0].device)
    if router_config.enable_router_optimization:
        load_balancing_loss = calculate_load_balancing_loss(
            routing_logits, router_config
        )

    # Calculate z-loss
    z_loss = torch.tensor(0.0, device=routing_logits[0].device)
    if router_config.enable_router_optimization:
        z_loss = calculate_z_loss(
            routing_logits, router_config.router_z_loss_coefficient
        )

    # Calculate importance loss
    importance_loss = torch.tensor(0.0, device=routing_logits[0].device)
    if router_config.enable_router_optimization:
        importance_loss = calculate_expert_importance_loss(
            routing_logits, router_config.router_importance_loss_coefficient
        )

    # Combine losses
    combined_loss = (
        load_balancing_loss * router_config.router_aux_loss_coefficient
        + z_loss
        + importance_loss
    )

    return combined_loss


def save_router_analysis(
    analysis_result: RouterAnalysisResult, output_dir: str, step: int
) -> str:
    """
    Save router analysis results to file.

    Args:
        analysis_result: Router analysis results
        output_dir: Output directory
        step: Training step

    Returns:
        str: Path to saved file
    """
    # Create analysis directory
    analysis_dir = os.path.join(output_dir, "router_analysis")
    ensure_directory_exists(analysis_dir)

    # Create file path
    file_path = os.path.join(analysis_dir, f"router_analysis_step_{step}.json")

    # Save analysis results
    success = analysis_result.save(file_path)

    if success:
        logger.info(f"Router analysis saved to {file_path}")
        return file_path
    else:
        logger.warning(f"Failed to save router analysis to {file_path}")
        return ""


def visualize_expert_utilization(
    analysis_results: List[RouterAnalysisResult],
    output_dir: str,
) -> str:
    """
    Generate visualization of expert utilization over time.

    Args:
        analysis_results: List of router analysis results
        output_dir: Output directory

    Returns:
        str: Path to visualization file
    """
    # This would typically generate a visualization using matplotlib
    # For this implementation, we'll just save the data in a format
    # that can be visualized later

    # Create visualization directory
    vis_dir = os.path.join(output_dir, "visualizations")
    ensure_directory_exists(vis_dir)

    # Prepare data
    visualization_data = {
        "expert_loads": [],
        "gini_coefficients": [],
        "entropies": [],
    }

    for result in analysis_results:
        visualization_data["expert_loads"].append(result.expert_loads)
        visualization_data["gini_coefficients"].append(result.gini_coefficient)
        visualization_data["entropies"].append(result.entropy)

    # Save data
    data_path = os.path.join(vis_dir, "expert_utilization_data.json")
    success = safe_write_json(visualization_data, data_path)

    if success:
        logger.info(f"Expert utilization data saved to {data_path}")
        return data_path
    else:
        logger.warning(f"Failed to save expert utilization data to {data_path}")
        return ""
