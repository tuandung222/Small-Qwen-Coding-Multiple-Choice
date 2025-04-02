# Qwen-Coder-MCQ: Fine-tuning Qwen2.5 for Multiple-Choice Coding Questions

This project provides a framework for fine-tuning Qwen2.5 Coder models on multiple-choice coding questions with structured YAML reasoning. It uses LoRA (Low-Rank Adaptation) for efficient training and includes a comprehensive pipeline for data processing, training, and evaluation.

## Features

- **Parameter-efficient fine-tuning** with LoRA
- **Optimized training** using Unsloth
- **Structured reasoning** with YAML-format outputs
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

This project provides several scripts for easy model training:

1. **Standard Training** with full customization:
   ```bash
   python src/run.py --experiment-name full_training --epochs 3 --batch-size 16
   ```

2. **Test Training Mode** for validating with minimal data (one batch):
   ```bash
   python src/run.py --test-training-mode --epochs 1 --experiment-name test_batch
   ```

3. **Shell Script Training** with all callbacks enabled:
   ```bash
   ./scripts/train_with_callbacks.sh comprehensive_training --batch-size 16
   ```

4. **Ultra-Fast Test Mode** with just 2 examples:
   ```bash
   ./scripts/train_with_callbacks.sh quick_test --test-mode --epochs 1
   ```

All training methods support extensive configuration. View options with:

```bash
python src/run.py --help
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

All training modes automatically integrate with Weights & Biases (WandB):

```bash
wandb login
```

This enables real-time tracking of:
- Training and validation loss
- Learning rate schedule
- GPU memory usage
- Model gradients
- Training samples with predictions

### Advanced Configuration

The training script provides extensive configuration options for experimentation:

```bash
python src/run.py \
  --experiment-name lora_config_test \
  --source-model "unsloth/Qwen2.5-Coder-1.5B-Instruct" \
  --destination-repo "tuandunghcmut/Qwen25_Coder_MultipleChoice_v2" \
  --batch-size 16 \
  --lora-r 16 \
  --lora-alpha 64 \
  --prompt-template "teacher_reasoned" \
  --push-strategy "best"
```

#### Key Parameters

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
| `--prompt-template` | Prompt format to use | yaml_reasoning |
| `--lora-r` | LoRA attention dimension | 8 |
| `--lora-alpha` | LoRA alpha parameter | 32 |
| `--quantization` | Model quantization level | 4bit |
| `--push-strategy` | When to push to HF Hub | best |
| `--private` | Make the repository private | False |
| `--test-mode` | Use only 2 examples | False |
| `--test-training-mode` | Use only one batch of data | False |

For the complete list of parameters, run `python src/run.py --help`.

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

# Create prompt
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
│   │   ├── prompt_creator.py  # Creates prompts for the model
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
├── scripts/                   # Utility scripts
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
