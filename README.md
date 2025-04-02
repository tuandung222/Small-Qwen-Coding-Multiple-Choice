# Qwen-Coder-MCQ: Fine-tuning Qwen2.5 for Multiple-Choice Coding Questions

This project provides a framework for fine-tuning Qwen2.5 Coder models on multiple-choice coding questions with structured reasoning. It uses LoRA (Low-Rank Adaptation) for efficient training and includes a comprehensive pipeline for data processing, training, and evaluation.

## Features

- **Parameter-efficient fine-tuning** with LoRA
- **Optimized training** using Unsloth
- **Structured reasoning** with YAML-format outputs for inference
- **Teacher-reasoned approach** for training
- **Comprehensive evaluation** framework
- **HuggingFace Hub integration** for model sharing
- **Flexible configuration** via command line arguments
- **Automatic repository creation**
- **Test modes** for rapid iteration and debugging
- **Callbacks** for validation, early stopping, and WandB logging

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- HuggingFace account with access token
- Weights & Biases account (optional, for experiment tracking)

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` file for authentication:

```bash
cp .env.example .env
# Edit the .env file with your API keys
```

The environment variables required are:
- `HF_TOKEN`: Your HuggingFace access token
- `WANDB_API_KEY`: Your Weights & Biases API key
- `OPENAI_API_KEY`: Only needed if generating teacher completions

## Usage

### Training

The project provides a Python module with a command-line interface for training. You can import the module in your code or run it directly from the command line.

#### Python Module Import

```python
# Import the main module
from src.run import main

# Run the training with default parameters
if __name__ == "__main__":
    main()
```

#### Direct Script Execution

```python
# Run standard training
python -m src.run --experiment-name full_training --epochs 3 --batch-size 16

# Run quick test with minimal data
python -m src.run --test-mode --epochs 1 --experiment-name quick_test

# Run test with a single batch of data
python -m src.run --test-training-mode --epochs 1 --experiment-name test_batch
```

#### Training Arguments

You can configure training by defining arguments either programmatically or through the command line:

```python
# Programmatic configuration example
import sys
from src.run import main

# Define command-line arguments
sys.argv = [
    "run.py",
    "--experiment-name", "custom_experiment",
    "--source-model", "unsloth/Qwen2.5-Coder-1.5B-Instruct",
    "--batch-size", "16",
    "--epochs", "3",
    "--learning-rate", "2e-4",
    "--early-stopping-patience", "3"
]

# Run with these arguments
main()
```

### Training Output Structure

All experiment outputs are organized in the `outputs` folder. Each experiment creates its own subdirectory:

```
outputs/
├── experiment_name/           # Your experiment output
│   ├── checkpoint-XXX/        # Model checkpoints
│   ├── best_model/            # Best model checkpoint
│   ├── training.log           # Detailed logs
│   ├── experiment_config.txt  # Configuration used
│   └── training_metrics.txt   # Final metrics
├── another_experiment/        # Another experiment
└── latest -> experiment_name  # Symlink to latest run
```

You can always access your most recent experiment using the `outputs/latest` symlink.

### Using Weights & Biases for Experiment Tracking

Training automatically integrates with Weights & Biases (WandB):

```bash
wandb login
```

This enables real-time tracking of:
- Training and validation loss
- Learning rate schedule
- GPU memory usage
- Model gradients
- Training samples with predictions

### Prompt Format

This project uses two specific prompt formats:

1. **TEACHER_REASONED**: Default format for training the model. This provides a teacher-guided reasoning approach that helps the model learn step-by-step problem solving.

2. **YAML_REASONING**: Default format for inference. This provides structured YAML output that's easy to parse and use in applications.

These prompt formats are designed to work together, with the TEACHER_REASONED format providing rich training data and the YAML_REASONING format providing clean, structured outputs during inference.

### Complete Command Line Interface

Here's the full syntax for the training module:

```
usage: run.py [-h] [--source-model SOURCE_MODEL] [--destination-repo DESTINATION_REPO]
              [--max-seq-length MAX_SEQ_LENGTH] [--quantization {4bit,8bit,none}]
              [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--grad-accum GRAD_ACCUM]
              [--learning-rate LEARNING_RATE] [--warmup-ratio WARMUP_RATIO]
              [--weight-decay WEIGHT_DECAY] [--output-dir OUTPUT_DIR]
              [--early-stopping-patience EARLY_STOPPING_PATIENCE]
              [--early-stopping-delta EARLY_STOPPING_DELTA]
              [--test-mode] [--test-training-mode] [--experiment-name EXPERIMENT_NAME]
              [--debug-samples DEBUG_SAMPLES] [--logging-steps LOGGING_STEPS]
              [--save-steps SAVE_STEPS] [--test-logging-steps TEST_LOGGING_STEPS]
              [--test-save-steps TEST_SAVE_STEPS] [--lora-r LORA_R]
              [--lora-alpha LORA_ALPHA] [--lora-dropout LORA_DROPOUT] [--private]
              [--save-method {lora,merged_16bit,merged_4bit,gguf}]
              [--push-strategy {best,end,all,no}]
              [--save-total-limit SAVE_TOTAL_LIMIT] [--dataset DATASET]
              [--val-split VAL_SPLIT] [--random-seed RANDOM_SEED]
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--experiment-name` | Name for this experiment | auto-generated timestamp |
| `--source-model` | Base model to fine-tune | unsloth/Qwen2.5-Coder-1.5B-Instruct |
| `--destination-repo` | HF Hub repo for the model | tuandunghcmut/Qwen25_Coder_MultipleChoice_v2 |
| `--batch-size` | Per device batch size | 16 |
| `--grad-accum` | Gradient accumulation steps | 4 |
| `--learning-rate` | Learning rate | 2e-4 |
| `--warmup-ratio` | Proportion of steps for warmup | 0.1 |
| `--weight-decay` | Weight decay for optimizer | 0.01 |
| `--lora-r` | LoRA attention dimension | 8 |
| `--lora-alpha` | LoRA alpha parameter | 32 |
| `--quantization` | Model quantization level | 4bit |
| `--push-strategy` | When to push to HF Hub | best |
| `--private` | Make the repository private | False |
| `--test-mode` | Use only 2 examples | False |
| `--test-training-mode` | Use only one batch of data | False |

For the complete list of parameters, run `python -m src.run --help`.

### Inference

You can use the trained model for inference:

```python
from model.qwen_handler import QwenModelHandler
from data.prompt_creator import PromptCreator
from data.response_parser import ResponseParser

# Initialize model handler
model_handler = QwenModelHandler(
    model_name="tuandunghcmut/Qwen25_Coder_MultipleChoice_v2",  # Default trained model
    max_seq_length=2048,
    quantization="4bit"  # For efficient inference
)

# Create prompt (using YAML_REASONING as the default inference prompt type)
prompt_creator = PromptCreator(PromptCreator.YAML_REASONING)
question = "Which of these is a valid Python dictionary comprehension?"
choices = [
    "{x: x**2 for x in range(10)}",
    "{for x in range(10): x**2}",
    "{x**2 for x in range(10)}",
    "[x: x**2 for x in range(10)]"
]
prompt = prompt_creator.create_inference_prompt(question, choices)

# Generate response
response = model_handler.generate_with_streaming(prompt, temperature=0.1)

# Parse response
parser = ResponseParser()
result = parser.parse_yaml_response(response)
print(f"Answer: {result['answer']}")
print(f"Reasoning: {result['reasoning']}")
```

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── prompt_creator.py  # Creates TEACHER_REASONED training and YAML_REASONING inference prompts
│   │   └── response_parser.py  # Parses model responses
│   ├── model/
│   │   └── qwen_handler.py    # Handles model loading and inference
│   ├── training/
│   │   ├── callbacks.py       # Custom training callbacks
│   │   └── trainer.py         # QwenTrainer for fine-tuning
│   ├── testing/
│   │   └── tester.py          # Framework for model evaluation
│   ├── utils/
│   │   ├── auth.py            # Authentication utilities
│   │   └── wandb_logger.py    # Weights & Biases integration
│   └── run.py                 # Main training script
├── outputs/                   # Training outputs folder
│   └── latest                 # Symlink to latest run
├── .env.example               # Example environment variables
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses [Unsloth](https://github.com/unslothai/unsloth) for optimized training
- The base model is [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) by Alibaba Cloud
