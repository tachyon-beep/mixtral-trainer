"""Configuration module for Mixtral training framework."""

import os
import uuid
import enum
import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict

from mixtral_training.utils.storage import safe_write_json, safe_read_json
from mixtral_training.utils.exceptions import ConfigError


class PrecisionType(enum.Enum):
    """Precision types for model training."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"

    def is_float16(self) -> bool:
        """Check if this is FP16 precision."""
        return self == PrecisionType.FP16

    def is_bfloat16(self) -> bool:
        """Check if this is BF16 precision."""
        return self == PrecisionType.BF16

    def is_float32(self) -> bool:
        """Check if this is FP32 precision."""
        return self == PrecisionType.FP32

    def is_int8(self) -> bool:
        """Check if this is INT8 precision."""
        return self == PrecisionType.INT8

    def is_int4(self) -> bool:
        """Check if this is INT4 precision."""
        return self == PrecisionType.INT4


class DeepSpeedZeroStage(enum.Enum):
    """DeepSpeed ZeRO stage."""

    DISABLED = 0
    OPTIMIZER_STATE = 1
    OPTIMIZER_GRADIENT = 2
    FULL = 3


class ReasoningOperationType(enum.Enum):
    """Types of reasoning operations that can be tracked during training."""

    DEDUCTION = "deduction"  # Logical deduction from premises
    INDUCTION = "induction"  # Inference from specific to general
    ABDUCTION = "abduction"  # Inference to best explanation
    ANALOGY = "analogy"  # Comparison-based reasoning
    CAUSAL = "causal"  # Cause-effect reasoning
    MATHEMATICAL = "mathematical"  # Mathematical computations
    TEMPORAL = "temporal"  # Time-based reasoning
    SPATIAL = "spatial"  # Space-based reasoning

    @classmethod
    def from_string(cls, value: str) -> "ReasoningOperationType":
        """Convert string to reasoning operation type."""
        for op_type in cls:
            if op_type.value == value.lower():
                return op_type
        raise ValueError(f"Unknown reasoning operation type: {value}")


@dataclass
class RouterConfig:
    """Configuration for router optimization and analysis."""

    # Router optimization settings
    enable_router_optimization: bool = True
    router_z_loss_coefficient: float = (
        1e-3  # Z-loss coefficient to prevent router saturation
    )
    router_aux_loss_coefficient: float = 0.01  # Auxiliary load balancing loss
    router_importance_loss_coefficient: float = 0.01  # Expert importance loss

    # Router analysis settings
    track_expert_assignments: bool = True
    track_token_to_expert_mapping: bool = True
    save_router_logits: bool = (
        False  # Whether to save raw router logits (high storage cost)
    )
    expert_assignment_logging_steps: int = 100  # Log expert assignments every N steps

    # Expert specialization analysis
    analyze_expert_specialization: bool = True
    specialization_metrics: List[str] = field(
        default_factory=lambda: ["entropy", "gini_coefficient", "expert_focus"]
    )

    # Tracked reasoning operations
    track_reasoning_operations: bool = True
    reasoning_operations: List[str] = field(
        default_factory=lambda: [
            "deduction",
            "induction",
            "abduction",
            "analogy",
            "causal",
            "mathematical",
            "temporal",
            "spatial",
        ]
    )


@dataclass
class ExpertCollaborationConfig:
    """Configuration for tracking expert collaboration during training."""

    enable_collaboration_tracking: bool = True

    # Collaboration analysis settings
    track_sequential_activations: bool = True
    max_sequence_length: int = 10  # Max length of expert activation sequences to track

    # Expert co-activation tracking
    track_expert_co_activation: bool = True
    co_activation_window_size: int = 5  # Window size for co-activation tracking

    # Visualization and logging
    save_collaboration_graphs: bool = True
    collaboration_logging_steps: int = 100  # Log collaboration metrics every N steps

    # Expert transition analysis
    analyze_expert_transitions: bool = True
    transition_smoothing_coefficient: float = 0.01  # Encourage smooth transitions


@dataclass
class ModelConfig:
    """Model configuration."""

    model_name: str = "mistralai/Mixtral-8x7B-v0.1"
    max_length: int = 2048
    precision_type: PrecisionType = PrecisionType.INT4
    lora_enabled: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Model architecture parameters
    num_experts: int = 8  # Number of experts in Mixtral model
    expert_model_parallel: bool = (
        False  # Whether to place different experts on different GPUs
    )

    # Router configuration for MoE optimization
    router: RouterConfig = field(default_factory=RouterConfig)

    # Expert collaboration tracking
    expert_collaboration: ExpertCollaborationConfig = field(
        default_factory=ExpertCollaborationConfig
    )

    # Dynamic parameter calculation
    calculate_model_size_dynamically: bool = (
        True  # Replace hardcoded parameter counts with dynamic calculation
    )


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    dataset_name: str = "timdettmers/openassistant-guanaco"
    max_samples: Optional[int] = None
    train_split: str = "train"
    eval_split: str = "validation"
    prompt_template: str = "[INST] {instruction} [/INST]"
    response_template: str = "{response}"
    max_example_length: int = 1024

    # Reasoning-specific dataset settings
    reasoning_templates: Dict[str, str] = field(
        default_factory=lambda: {
            "chain_of_thought": "[INST] {instruction} [/INST]\nLet me think step by step.",
            "tree_of_thought": "[INST] {instruction} [/INST]\nLet me explore multiple approaches:",
            "reasoning_with_verification": "[INST] {instruction} [/INST]\nI'll solve this and verify my answer.",
        }
    )

    # Reasoning operations annotation
    annotate_reasoning_operations: bool = (
        False  # Auto-annotate reasoning operations in dataset
    )
    reasoning_operation_classification_model: Optional[str] = (
        None  # Model to use for classification
    )


@dataclass
class GPUConfig:
    """GPU configuration."""

    gpu_count: int = 0
    per_gpu_memory_gb: float = 0.0

    def __post_init__(self):
        """Initialize with actual GPU info if available."""
        try:
            import torch

            if torch.cuda.is_available():
                self.gpu_count = torch.cuda.device_count()
                # Estimate memory per GPU (use first GPU as reference)
                if self.gpu_count > 0:
                    self.per_gpu_memory_gb = torch.cuda.get_device_properties(
                        0
                    ).total_memory / (1024**3)
        except ImportError:
            pass


@dataclass
class DeepSpeedConfig:
    """DeepSpeed configuration."""

    enabled: bool = False
    zero_stage: DeepSpeedZeroStage = DeepSpeedZeroStage.OPTIMIZER_STATE
    offload_optimizer: bool = False
    offload_param: bool = False

    def to_dict(
        self, batch_size: int, gradient_accumulation_steps: int, gpu_count: int
    ) -> Dict[str, Any]:
        """
        Convert to DeepSpeed config dict.

        Args:
            batch_size: Batch size per GPU
            gradient_accumulation_steps: Gradient accumulation steps
            gpu_count: Number of GPUs

        Returns:
            Dict[str, Any]: DeepSpeed config
        """
        if not self.enabled:
            return {}

        train_batch_size = batch_size * gradient_accumulation_steps * max(1, gpu_count)

        config = {
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "zero_optimization": {
                "stage": self.zero_stage.value,
                "offload_optimizer": {
                    "device": "cpu" if self.offload_optimizer else "none"
                },
                "offload_param": {"device": "cpu" if self.offload_param else "none"},
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
            },
            "fp16": {
                "enabled": True,
                "auto_cast": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            "bf16": {"enabled": False},
            "zero_allow_untested_optimizer": True,
            "wall_clock_breakdown": False,
        }

        return config


@dataclass
class WandBConfig:
    """Weights & Biases configuration."""

    enabled: bool = False
    project: str = "mixtral-training"
    entity: Optional[str] = None
    log_model: bool = True


@dataclass
class HuggingFaceConfig:
    """Hugging Face configuration."""

    push_to_hub: bool = False
    repo_id: Optional[str] = None
    private: bool = False
    token: Optional[str] = None


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    # Basic evaluation settings
    generate_samples: bool = True
    num_samples: int = 10

    # Metrics for reasoning quality
    evaluate_reasoning_quality: bool = True
    reasoning_metrics: List[str] = field(
        default_factory=lambda: [
            "step_validity",
            "logical_consistency",
            "conclusion_correctness",
        ]
    )

    # Expert utilization evaluation
    evaluate_expert_utilization: bool = True
    expert_utilization_metrics: List[str] = field(
        default_factory=lambda: [
            "expert_balance",
            "expert_specialization",
            "routing_consistency",
        ]
    )

    # Reasoning benchmarks
    use_reasoning_benchmarks: bool = False
    reasoning_benchmarks: List[str] = field(
        default_factory=lambda: ["gsm8k", "mmlu", "bbh", "math"]
    )
    benchmark_subset_size: Optional[int] = (
        100  # Number of examples to use from each benchmark
    )


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Basic settings
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    run_name: Optional[str] = None
    output_dir: str = "outputs"
    seed: int = 42
    is_debug: bool = False

    # Training parameters
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 1000
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 10

    # Router optimization parameters
    router_aux_loss_coef: float = 0.01  # Coefficient for auxiliary router loss
    use_expert_choice_routing: bool = (
        False  # Use expert-choice routing instead of token-choice
    )

    # Advanced optimization
    use_gradient_checkpointing: bool = (
        True  # Enable gradient checkpointing to save memory
    )
    use_flash_attention: bool = True  # Use flash attention implementation if available
    gradient_accumulation_device: str = "gpu"  # 'cpu' or 'gpu'

    # Expert utilization parameters
    balance_experts: bool = True  # Apply expert balancing loss
    expert_capacity_factor: float = (
        1.0  # Capacity factor for each expert (>1 allows more tokens per expert)
    )

    # Reasoning-specific training
    reasoning_loss_weighting: bool = False  # Apply higher weight to reasoning examples
    reasoning_example_weight: float = (
        1.5  # Weight for reasoning examples vs regular examples
    )

    # Component configs
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self):
        """Initialize dependent fields."""
        # Generate run name if not provided
        if not self.run_name:
            model_name = self.model.model_name.split("/")[-1]
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.run_name = f"{model_name}-{timestamp}-{self.run_id}"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.

        Returns:
            Dict[str, Any]: Configuration as dictionary
        """
        # Convert dataclasses to dictionaries
        config_dict = asdict(self)

        # Handle enums (convert to string values)
        config_dict["model"]["precision_type"] = self.model.precision_type.value
        config_dict["deepspeed.zero_stage"] = self.deepspeed.zero_stage.value

        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """
        Create config from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            TrainingConfig: Configuration object
        """
        # Make a copy to avoid modifying the input
        config = dict(config_dict)

        # Extract nested configs
        model_config = config.pop("model", {})
        dataset_config = config.pop("dataset", {})
        gpu_config = config.pop("gpu", {})
        deepspeed_config = config.pop("deepspeed", {})
        wandb_config = config.pop("wandb", {})
        huggingface_config = config.pop("huggingface", {})

        # Handle enums
        if "precision_type" in model_config:
            model_config["precision_type"] = PrecisionType(
                model_config["precision_type"]
            )

        zero_stage = config.pop("deepspeed.zero_stage", None)
        if zero_stage is not None:
            deepspeed_config["zero_stage"] = DeepSpeedZeroStage(zero_stage)
        elif "zero_stage" in deepspeed_config:
            deepspeed_config["zero_stage"] = DeepSpeedZeroStage(
                deepspeed_config["zero_stage"]
            )

        # Create config object
        training_config = cls(
            **config,
            model=ModelConfig(**model_config),
            dataset=DatasetConfig(**dataset_config),
            gpu=GPUConfig(**gpu_config),
            deepspeed=DeepSpeedConfig(**deepspeed_config),
            wandb=WandBConfig(**wandb_config),
            huggingface=HuggingFaceConfig(**huggingface_config),
        )

        return training_config

    def save(self, path: str) -> bool:
        """
        Save config to file.

        Args:
            path: File path

        Returns:
            bool: True if successful
        """
        return safe_write_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """
        Load config from file.

        Args:
            path: File path

        Returns:
            TrainingConfig: Configuration object

        Raises:
            ConfigError: If loading fails
        """
        config_dict = safe_read_json(path)
        if not config_dict:
            raise ConfigError(f"Failed to load configuration from {path}")

        return cls.from_dict(config_dict)

    def validate(self) -> List[str]:
        """
        Validate configuration.

        Returns:
            List[str]: List of validation issues (empty if valid)
        """
        issues = []

        # Check required fields
        if not self.output_dir:
            issues.append("Output directory is required")

        if self.batch_size <= 0:
            issues.append(f"Batch size must be positive, got {self.batch_size}")

        if not self.model.model_name:
            issues.append("Model name is required")

        if not self.dataset.dataset_name:
            issues.append("Dataset name is required")

        # Check LoRA settings
        if self.model.lora_enabled:
            if self.model.lora_r <= 0:
                issues.append(f"LoRA rank must be positive, got {self.model.lora_r}")
            if self.model.lora_alpha <= 0:
                issues.append(
                    f"LoRA alpha must be positive, got {self.model.lora_alpha}"
                )
            if not self.model.lora_target_modules:
                issues.append("LoRA target modules are required when LoRA is enabled")

        return issues
