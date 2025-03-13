# Mixtral-8x7B R1 Reasoning Training Framework

A comprehensive toolkit for fine-tuning Mixtral-8x7B models for specialized reasoning tasks, focused on optimizing Mixture-of-Experts (MoE) model architecture. This framework enables efficient training on consumer-grade hardware through advanced memory optimization techniques.

## 🔑 Key Features

- **MoE Architecture Optimization**

  - Expert-specific parameter optimization
  - Router optimization with load balancing and specialization tracking
  - Expert collaboration analysis
  - Dynamic routing analysis for reasoning operations

- **Memory Efficiency**

  - QLoRA fine-tuning for parameter efficiency
  - 4-bit & 8-bit quantization
  - Automatic memory-based configuration
  - Optimized for consumer hardware (16GB+ GPUs)

- **Training Capabilities**

  - Distributed training across multiple GPUs
  - DeepSpeed ZeRO optimization
  - Gradient checkpointing
  - Reasoning-specific loss weighting

- **Comprehensive Evaluation**
  - Reasoning quality metrics
  - Expert utilization analysis
  - Router behavior visualization
  - Performance benchmarking

## 📋 System Requirements

### Minimum Requirements

- Python 3.9+
- CUDA-capable GPU with 16GB+ VRAM
- 16GB+ system RAM
- 100GB+ free disk space

### Recommended Setup

- Multiple GPUs (24GB+ each)
- 32GB+ system RAM
- 500GB+ SSD storage
- Linux operating system

## 🔧 Installation

```bash
# Installation via pip (once released)
pip install mixtral-training

# Or install from the repository
git clone https://github.com/tachyon-beep/mixtral-trainer
cd mixtral-trainer

# Install the package
pip install -e .

# Optional dependencies
pip install -e ".[wandb,deepspeed]"  # For Weights & Biases and DeepSpeed support
pip install -e ".[dev]"  # For development and testing
```

## 🚀 Quick Start

### Setup Configuration

```bash
# Create a basic configuration
mixtral-train setup --output config.json \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --dataset_name timdettmers/openassistant-guanaco
```

### Train a Model

```bash
# Train with default settings (auto-detects GPU memory)
mixtral-train train --config config.json

# Train with LoRA and DeepSpeed
mixtral-train train --config config.json --use_lora --use_deepspeed

# Train with debug mode for faster iteration
mixtral-train train --config config.json --debug
```

### Evaluate a Model

```bash
# Evaluate a trained model
mixtral-train evaluate --checkpoint ./outputs/final_model \
    --dataset databricks/databricks-dolly-15k
```

## 📊 Memory Requirements & Scaling

The framework automatically adapts to available GPU memory:

| Configuration        | Memory Required | Hardware Example             |
| -------------------- | --------------- | ---------------------------- |
| Full Fine-tuning     | 80+ GB          | 4x A100 40GB or 2x A100 80GB |
| 8-bit LoRA           | ~24 GB          | RTX 4090 or A10              |
| 4-bit LoRA           | ~16 GB          | RTX 3090 or 4080             |
| 4-bit LoRA + Offload | ~8 GB           | RTX 3060 or 2080             |

## 📖 Advanced Usage

### Configuring Router Optimization

Router optimization is critical for MoE models. Configure specialized parameters:

```json
{
  "model": {
    "router": {
      "enable_router_optimization": true,
      "router_z_loss_coefficient": 0.001,
      "router_aux_loss_coefficient": 0.01,
      "track_expert_assignments": true,
      "analyze_expert_specialization": true
    }
  }
}
```

### Expert Collaboration Analysis

Track how experts collaborate on complex reasoning tasks:

```json
{
  "model": {
    "expert_collaboration": {
      "enable_collaboration_tracking": true,
      "track_sequential_activations": true,
      "track_expert_co_activation": true,
      "co_activation_window_size": 5
    }
  }
}
```

### Reasoning Operation Tracking

Configure the system to track specific types of reasoning operations:

```json
{
  "model": {
    "router": {
      "track_reasoning_operations": true,
      "reasoning_operations": [
        "deduction",
        "induction",
        "abduction",
        "analogy",
        "causal",
        "mathematical"
      ]
    }
  }
}
```

## 🔍 Known Limitations and Future Work

- **Router Operation Detection**: Currently uses simple keyword matching; a classifier-based approach is planned
- **Visualization Tools**: Router analysis visualization relies on external tools for now
- **Testing Coverage**: Test suite is currently limited to configuration testing
- **Memory Estimation**: Memory requirement estimates may be inaccurate for some hardware configurations
- **Multi-GPU Scaling**: Expert model parallelism needs further optimization for very large models

## 🛡️ License

MIT License

## 📚 Citation

If you use this framework in your research, please cite:

```
@software{mixtral_training,
  author = {MTG AI Team},
  title = {Mixtral-8x7B R1 Reasoning Training Framework},
  year = {2025},
  url = {https://github.com/tachyon-beep/mixtral-trainer}
}
```

## 🤝 Contributing

Contributions are welcome! Please check out our contributing guidelines for details on how to submit pull requests, report issues, or request features.
