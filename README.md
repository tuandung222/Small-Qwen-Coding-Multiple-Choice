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

1. **Basic Training Script**: Run training with standard parameters
   ```bash
   ./scripts/train_experiment.sh my_experiment
   ```

2. **Batch-32 Training Script**: Train with batch size 32 for faster convergence
   ```bash
   ./scripts/train_batch32.sh
   ```

3. **Comprehensive Training with All Callbacks**: Train with validation, early stopping and WandB integration
   ```bash
   ./scripts/train_with_callbacks.sh full_experiment
   ```

4. **Quick Test Mode**: Train on just 2 dataset instances for rapid testing and debugging
   ```bash
   ./scripts/train_with_callbacks.sh quick_test --test-mode --epochs 2
   ```

All training scripts support various command-line arguments to customize the training process. Use the `--help` flag to see all available options:

```bash
./scripts/train_with_callbacks.sh --help
```

### Training Output Structure

All experiment outputs are organized in the `experiment_output` directory by default. Each run creates a timestamped subdirectory with:

- Training logs
- Model checkpoints
- Experiment configuration
- Training metrics
- Status markers

### Using Weights & Biases for Experiment Tracking

The comprehensive training script automatically integrates with Weights & Biases (WandB). Ensure you have logged in with your WandB credentials:

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

You can customize the training with various command-line arguments:

```bash
python src/run.py \
  --source-model "unsloth/Qwen2.5-Coder-1.5B-Instruct" \
  --destination-repo "username/model-name" \
  --max-seq-length 2048 \
  --epochs 3 \
  --batch-size 4 \
  --grad-accum 4 \
  --learning-rate 2e-4 \
  --dataset "tuandunghcmut/coding-mcq-reasoning" \
  --val-split 0.1 \
  --save-method "lora"
```

#### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--source-model` | Base model to fine-tune | unsloth/Qwen2.5-Coder-1.5B-Instruct |
| `--destination-repo` | HF Hub repo for the trained model | auto-generated |
| `--max-seq-length` | Maximum sequence length | 2048 |
| `--epochs` | Number of training epochs | 3 |
| `--batch-size` | Per device batch size | 4 |
| `--grad-accum` | Gradient accumulation steps | 4 |
| `--learning-rate` | Learning rate | 2e-4 |
| `--dataset` | Dataset ID on HuggingFace Hub | tuandunghcmut/coding-mcq-reasoning |
| `--val-split` | Validation split ratio | 0.1 |
| `--save-method` | Model saving method (lora, merged_16bit, merged_4bit, gguf) | lora |
| `--private` | Make the repository private | True |

### Inference

You can use the trained model for inference:

```python
from model.qwen_handler import QwenModelHandler
from data.prompt_creator import PromptCreator
from data.response_parser import ResponseParser

# Initialize model handler
model_handler = QwenModelHandler(
    model_name="username/model-name",  # Your trained model
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
│   │   └── auth.py            # Authentication utilities
│   └── run.py                 # Main training script
├── configs/                   # Configuration files
├── scripts/                   # Utility scripts
├── tests/                     # Unit tests
├── .env.example               # Example environment variables
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses [Unsloth](https://github.com/unslothai/unsloth) for optimized training
- The base model is [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) by Alibaba Cloud
