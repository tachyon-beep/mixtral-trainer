# Contributing to Mixtral Training Framework

Thank you for your interest in contributing to the Mixtral Training Framework! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Features](#suggesting-features)
  - [Pull Requests](#pull-requests)
- [Development Guidelines](#development-guidelines)
  - [Code Style](#code-style)
  - [Testing](#testing)
  - [Documentation](#documentation)
- [Project Structure](#project-structure)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and considerate of others when participating in discussions, submitting code, or engaging with the community.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```
   git clone https://github.com/your-username/mixtral-trainer.git
   cd mixtral-trainer
   ```
3. **Install dependencies**:
   ```
   pip install -e .
   pip install -e ".[dev]"  # For development dependencies
   ```
4. **Create a branch**:
   ```
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Reporting Bugs

If you find a bug in the project, please create an issue on GitHub with the following information:

- Clear and descriptive title
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Screenshots or logs (if applicable)
- Environment information (OS, Python version, GPU, etc.)

### Suggesting Features

We welcome feature suggestions! To suggest a feature:

1. Check if the feature has already been suggested or implemented
2. Create an issue describing the feature and its benefits
3. Include any relevant details, examples, or use cases

### Pull Requests

1. Ensure your code follows our [Code Style](#code-style) guidelines
2. Include tests for any new functionality
3. Update documentation as needed
4. Make sure all tests pass
5. Create a pull request with a clear description of the changes

Pull requests should target the `main` branch and include:

- A reference to any related issues
- A description of the changes
- Any additional context or information

## Development Guidelines

### Code Style

This project follows the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code. We use the following tools for code formatting and linting:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting

To format your code:

```
black .
isort .
flake8 .
```

### Testing

We use `pytest` for testing. All new code should include appropriate tests:

- Unit tests for individual functions and classes
- Integration tests for interactions between components
- Performance tests for critical functionality

To run tests:

```
pytest
```

### Documentation

Documentation is a critical part of this project:

- Use docstrings for all functions, classes, and modules
- Follow the [NumPy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html)
- Keep the README and other documentation up to date with changes

## Project Structure

The project is organized as follows:

- `mixtral_training/`: Main package
  - `__init__.py`: Package initialization
  - `config.py`: Configuration management
  - `data.py`: Dataset handling
  - `evaluate.py`: Evaluation utilities
  - `main.py`: Command-line interface
  - `model.py`: Model definition and loading
  - `routing.py`: Router optimization and analysis
  - `train.py`: Training loops and utilities
  - `utils/`: Utility modules
    - `checkpoint.py`: Checkpoint management
    - `exceptions.py`: Custom exceptions
    - `logging.py`: Logging configuration
    - `memory.py`: Memory management
    - `security.py`: Security utilities
    - `storage.py`: Storage utilities
  - `tests/`: Test modules
    - `test_config.py`: Configuration tests
    - ...

## Working on Tasks

We maintain a detailed task list in [TASKS.md](TASKS.md) which outlines current development priorities and plans. If you're interested in contributing, please check this file for tasks that are available to work on.

For each task you take on:

1. Comment on the relevant issue or create a new one
2. Reference the task from TASKS.md
3. Update the task status when you start working on it

## Questions?

If you have any questions or need further guidance, please don't hesitate to create an issue or reach out to the maintainers.

Thank you for contributing to the Mixtral Training Framework!
