#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

# Get version without importing
version = "0.1.0"  # Hardcoded default version
try:
    with open("mixtral_training/version.py", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                version = line.split("=")[1].strip().strip("\"'")
                break
except (IOError, FileNotFoundError):
    pass  # Use default version if file not found

# Get long description from README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Core dependencies
install_requires = [
    "torch>=2.0.0",
    "transformers>=4.30.0,<4.40.0",
    "peft>=0.5.0",
    "datasets>=2.11.0",
    "numpy>=1.24.0",
    "tqdm>=4.65.0",
    "psutil>=5.9.0",
    "accelerate>=0.21.0",
    "bitsandbytes>=0.39.0",
    "safetensors>=0.3.1",
]

# Optional dependencies
extras_require = {
    "dev": [
        "black",
        "isort",
        "flake8",
        "mypy",
        "pytest",
        "pytest-cov",
    ],
    "wandb": [
        "wandb>=0.15.0",
    ],
    "deepspeed": [
        "deepspeed>=0.9.0",
    ],
}

setup(
    name="mixtral-training",
    version=version,
    description="Training framework for Mixtral-8x7B R1 Reasoning models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MTG AI Team",
    author_email="info@mixtralai.com",
    url="https://github.com/mixtralai/mixtral-training",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.9.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "mixtral-train=mixtral_training.main:main",
        ],
    },
    exclude_package_data={"": ["src/*"]},
)
