"""Tests for configuration module."""

import os
import json
import tempfile
import unittest
from pathlib import Path

from mixtral_training.config import (
    TrainingConfig,
    ModelConfig,
    DatasetConfig,
    GPUConfig,
    DeepSpeedConfig,
    WandBConfig,
    HuggingFaceConfig,
    PrecisionType,
    DeepSpeedZeroStage,
)


class TestTrainingConfig(unittest.TestCase):
    """Test the TrainingConfig class."""

    def test_default_config(self):
        """Test creating a default configuration."""
        config = TrainingConfig()

        # Check that required attributes are present
        self.assertIsNotNone(config.run_id)
        self.assertIsNotNone(config.model)
        self.assertIsNotNone(config.dataset)
        self.assertIsNotNone(config.gpu)
        self.assertIsNotNone(config.deepspeed)
        self.assertIsNotNone(config.wandb)
        self.assertIsNotNone(config.huggingface)

        # Check default values
        self.assertEqual(config.output_dir, "outputs")
        self.assertEqual(config.seed, 42)

        # Default model should be Mixtral
        self.assertTrue("Mixtral" in config.model.model_name)

        # Should generate run_name if not provided
        self.assertIsNotNone(config.run_name)
        self.assertTrue(config.model.model_name.split("/")[-1] in config.run_name)

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = TrainingConfig()
        config_dict = config.to_dict()

        # Check that all sections are present
        self.assertIn("run_id", config_dict)
        self.assertIn("run_name", config_dict)
        self.assertIn("model", config_dict)
        self.assertIn("dataset", config_dict)
        self.assertIn("gpu", config_dict)
        self.assertIn("deepspeed", config_dict)
        self.assertIn("wandb", config_dict)
        self.assertIn("huggingface", config_dict)

        # Check that model name is preserved
        self.assertEqual(config_dict["model"]["model_name"], config.model.model_name)

        # Check that precision type is converted to string
        self.assertEqual(
            config_dict["model"]["precision_type"], config.model.precision_type.value
        )

        # Check DeepSpeed stage is converted
        self.assertEqual(
            config_dict["deepspeed.zero_stage"], config.deepspeed.zero_stage.value
        )

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        # Create a custom config dictionary
        config_dict = {
            "run_id": "test123",
            "output_dir": "test_outputs",
            "batch_size": 4,
            "model": {
                "model_name": "test/model",
                "max_length": 1024,
                "precision_type": "bf16",
                "lora_enabled": True,
                "lora_r": 16,
            },
            "dataset": {"dataset_name": "test/dataset", "max_samples": 1000},
            "deepspeed": {"enabled": True},
            "deepspeed.zero_stage": 2,
        }

        # Create config from dictionary
        config = TrainingConfig.from_dict(config_dict)

        # Check that values are set correctly
        self.assertEqual(config.run_id, "test123")
        self.assertEqual(config.output_dir, "test_outputs")
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.model.model_name, "test/model")
        self.assertEqual(config.model.max_length, 1024)
        self.assertEqual(config.model.precision_type, PrecisionType.BF16)
        self.assertEqual(config.model.lora_r, 16)
        self.assertEqual(config.dataset.dataset_name, "test/dataset")
        self.assertEqual(config.dataset.max_samples, 1000)
        self.assertTrue(config.deepspeed.enabled)
        self.assertEqual(
            config.deepspeed.zero_stage, DeepSpeedZeroStage.OPTIMIZER_GRADIENT
        )

    def test_config_save_load(self):
        """Test saving and loading config."""
        # Create a config
        original_config = TrainingConfig(
            run_id="test_save_load",
            output_dir="test_outputs",
            batch_size=2,
            model=ModelConfig(
                model_name="test/model",
                max_length=2048,
                precision_type=PrecisionType.FP16,
            ),
            dataset=DatasetConfig(dataset_name="test/dataset"),
        )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Save
            success = original_config.save(temp_path)
            self.assertTrue(success)

            # Load
            loaded_config = TrainingConfig.load(temp_path)

            # Compare
            self.assertEqual(loaded_config.run_id, original_config.run_id)
            self.assertEqual(loaded_config.output_dir, original_config.output_dir)
            self.assertEqual(loaded_config.batch_size, original_config.batch_size)
            self.assertEqual(
                loaded_config.model.model_name, original_config.model.model_name
            )
            self.assertEqual(
                loaded_config.model.max_length, original_config.model.max_length
            )
            self.assertEqual(
                loaded_config.model.precision_type, original_config.model.precision_type
            )
            self.assertEqual(
                loaded_config.dataset.dataset_name, original_config.dataset.dataset_name
            )

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_nested_configs(self):
        """Test nested configuration classes."""
        # Model config
        model_config = ModelConfig(
            model_name="test/model", max_length=1024, lora_enabled=True, lora_r=32
        )

        # Dataset config
        dataset_config = DatasetConfig(dataset_name="test/dataset", max_samples=500)

        # GPU config
        gpu_config = GPUConfig(gpu_count=2, per_gpu_memory_gb=16.0)

        # Create training config with nested configs
        config = TrainingConfig(
            model=model_config, dataset=dataset_config, gpu=gpu_config
        )

        # Check nested configs
        self.assertEqual(config.model.model_name, "test/model")
        self.assertEqual(config.model.max_length, 1024)
        self.assertTrue(config.model.lora_enabled)
        self.assertEqual(config.model.lora_r, 32)
        self.assertEqual(config.dataset.dataset_name, "test/dataset")
        self.assertEqual(config.dataset.max_samples, 500)
        self.assertEqual(config.gpu.gpu_count, 2)
        self.assertEqual(config.gpu.per_gpu_memory_gb, 16.0)

    def test_validate_config(self):
        """Test configuration validation."""
        # Create valid config
        valid_config = TrainingConfig(
            output_dir="outputs",
            batch_size=1,
            model=ModelConfig(model_name="test/model"),
            dataset=DatasetConfig(dataset_name="test/dataset"),
        )

        # Should pass validation
        issues = valid_config.validate()
        self.assertFalse(issues)  # Empty list = no issues

        # Create invalid config
        invalid_config = TrainingConfig(
            output_dir="",  # Empty output dir
            batch_size=0,  # Invalid batch size
            model=ModelConfig(model_name=""),  # Empty model name
            dataset=DatasetConfig(dataset_name=""),  # Empty dataset name
        )

        # Should fail validation
        issues = invalid_config.validate()
        self.assertTrue(len(issues) >= 4)  # At least 4 issues


if __name__ == "__main__":
    unittest.main()
