# nonfig Examples

This directory contains example code demonstrating various features of nonfig.

## Usage

1. **Install dependencies**:
   ```bash
   # Make sure nonfig is installed
   cd /path/to/nonfig
   uv add -e .
   ```

2. **Run the ML pipeline example**:
   ```bash
   uv run python examples/ml_pipeline.py
   ```

## Examples

### ml_pipeline.py

A complete machine learning training pipeline demonstrating:
- Nested configurations (Model → Optimizer)
- Data preprocessing with hyperparameters
- Configuration serialization for experiment tracking
- Hyperparameter search

### nested_functions.py

Demonstrates nesting configurable functions using the unified pattern:
- Using `fn: inner.Type = inner` for nested functions
- Works for both direct calls and Config.make()
- Override nested function parameters via Config
- Serialization of nested function configs

## Adding More Examples

When adding new examples:
1. Use clear, self-contained code
2. Add docstrings explaining what's demonstrated
3. Include type hints
4. Show both basic and advanced usage
5. Update this README
