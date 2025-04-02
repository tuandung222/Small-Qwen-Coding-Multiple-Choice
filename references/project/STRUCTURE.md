# Qwen Multiple Choice Project Structure

## Model Components (`src/model/`)
- `qwen_handler.py`
  - `QwenModelHandler` class
    - Model initialization and loading
    - Generation and inference methods
    - Model saving and pushing to hub
  - Model utility functions
  - Weight initialization helpers

## Data Processing (`src/data/`)
- `prompt_creator.py`
  - `PromptCreator` class
    - Basic prompt formatting
    - YAML reasoning format
    - Teacher-reasoned format
  - Prompt utility functions
- `response_parser.py`
  - `ResponseParser` class
    - Basic answer parsing
    - YAML response parsing
    - Reasoning extraction
- `dataset.py`
  - Dataset creation functions
  - Data loading utilities
  - Quality validation dataset creation

## Training (`src/training/`)
- `trainer.py`
  - `QwenTrainer` class
    - Training loop implementation
    - Validation logic
    - Checkpoint management
    - Metrics logging
- `callbacks.py`
  - `ValidationCallback` class
    - Validation during training
    - Example logging
    - Metric tracking
  - `EarlyStoppingCallback` class
    - Training early stopping
    - Best model saving

## Testing (`src/testing/`)
- `tester.py`
  - `MultipleChoiceTester` class
    - Batch inference
    - Example testing
    - Results saving
- `metrics.py`
  - Accuracy calculation
  - Reasoning quality evaluation
  - Combined metrics

## Utilities (`src/utils/`)
- `common.py`
  - Shared utility functions
  - File handling
  - Logging setup
  - Configuration management

## Scripts
- `train.py`
  - Training entry point
  - Hyperparameter setup
  - Model initialization
- `evaluate.py`
  - Model evaluation
  - Results generation
- `run_pipeline.py`
  - Full pipeline execution
  - End-to-end workflow

## Configuration (`configs/`)
- `config.yaml`
  - Model configuration
  - Training parameters
  - Evaluation settings
  - Pipeline configuration

## Requirements
- `requirements.txt`
  - Project dependencies
  - Version specifications
