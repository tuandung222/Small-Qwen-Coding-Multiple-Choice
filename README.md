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
              [--optimizer OPTIMIZER] [--adam-beta1 ADAM_BETA1] [--adam-beta2 ADAM_BETA2]
              [--adam-epsilon ADAM_EPSILON] [--max-grad-norm MAX_GRAD_NORM]
              [--optim-bits OPTIM_BITS] [--lr-scheduler LR_SCHEDULER]
              [--lr-scheduler-num-cycles LR_SCHEDULER_NUM_CYCLES]
              [--lr-scheduler-power LR_SCHEDULER_POWER]
              [--lr-scheduler-last-epoch LR_SCHEDULER_LAST_EPOCH]
              [--peft-type PEFT_TYPE] [--adalora-target-r ADALORA_TARGET_R]
              [--adalora-init-r ADALORA_INIT_R] [--adalora-tinit ADALORA_TINIT]
              [--adalora-tfinal ADALORA_TFINAL] [--adalora-delta-t ADALORA_DELTA_T]
              [--adalora-beta1 ADALORA_BETA1] [--adalora-beta2 ADALORA_BETA2]
              [--target-modules TARGET_MODULES] [--fan-in-fan-out FAN_IN_FAN_OUT]
              [--use-gradient-checkpointing USE_GRADIENT_CHECKPOINTING]
              [--modules-to-save MODULES_TO_SAVE]
              [--train-on-responses-only] [--instruction-token INSTRUCTION_TOKEN]
              [--response-token RESPONSE_TOKEN] [--instruction-token-id INSTRUCTION_TOKEN_ID]
              [--response-token-id RESPONSE_TOKEN_ID]
              [--attention-implementation {default,flash_attention_2,sdpa,eager,xformers}]
              [--use-flash-attention] [--force-attn-implementation]
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--experiment-name` | Name for this experiment | auto-generated timestamp |
| `--source-model` | Base model to fine-tune | unsloth/Qwen2.5-Coder-1.5B-Instruct |
| `--destination-repo` | HF Hub repo for the model | tuandunghcmut/Qwen25_Coder_MultipleChoice_v2 |
| `--batch-size` | Per device batch size | 24 |
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
| `--train-on-responses-only` | Enable response-only training | False |
| `--instruction-token` | Token/prefix indicating start of instruction | `<|im_start|>user\n` |
| `--response-token` | Token/prefix indicating start of response | `<|im_start|>assistant\n` |
| `--instruction-token-id` | Token ID for instruction start (optional) | None |
| `--response-token-id` | Token ID for response start (optional) | None |

### Optimizer Configuration

You can customize the optimizer used during training with the following parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--optimizer` | Optimizer type | adamw_torch |
| `--weight-decay` | Weight decay for regularization | 0.01 |
| `--adam-beta1` | Beta1 for Adam-based optimizers | 0.9 |
| `--adam-beta2` | Beta2 for Adam-based optimizers | 0.999 |
| `--adam-epsilon` | Epsilon for Adam-based optimizers | 1e-8 |
| `--max-grad-norm` | Maximum gradient norm for clipping | 1.0 |
| `--optim-bits` | Quantization bits for 8-bit optimizers | 8 |

The `--optimizer` parameter supports several options:
- `adamw_torch`: PyTorch's AdamW implementation (default)
- `adamw_hf`: Hugging Face's AdamW implementation
- `adam8bit`: 8-bit Adam for memory efficiency
- `pagedadam`: Paged Adam for large models
- `lion`: Lion optimizer (less memory, potentially better generalization)
- `adafactor`: Adafactor optimizer (memory efficient alternative to Adam)

### Learning Rate Scheduler Configuration

You can customize the learning rate scheduler used during training with the following parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--lr-scheduler` | Learning rate scheduler type | cosine |
| `--lr-scheduler-num-cycles` | Number of cycles for cosine_with_restarts | 1 |
| `--lr-scheduler-power` | Power factor for polynomial scheduler | 1.0 |
| `--lr-scheduler-last-epoch` | Index of last epoch when resuming training | -1 |

The `--lr-scheduler` parameter supports several options:
- `cosine`: Cosine decay scheduler (default) - gradually reduces LR following a cosine curve
- `linear`: Linear decay scheduler - reduces LR linearly to zero
- `cosine_with_restarts`: Cosine decay with restarts - follows cosine curve but restarts periodically
- `polynomial`: Polynomial decay scheduler - reduces LR following a polynomial function
- `constant`: Constant scheduler - maintains a constant LR after warmup
- `constant_with_warmup`: Constant with warmup - increases during warmup then stays constant
- `inverse_sqrt`: Inverse square root scheduler - decays proportionally to inverse square root of step

Example usage with custom scheduler:
```python
python -m src.run --lr-scheduler polynomial --lr-scheduler-power 2.0 --warmup-ratio 0.2
```

### PEFT Configuration

You can customize the Parameter-Efficient Fine-Tuning (PEFT) approach with the following parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--peft-type` | PEFT method to use | lora |
| `--lora-r` | LoRA attention dimension | 8 |
| `--lora-alpha` | LoRA alpha parameter | 32 |
| `--lora-dropout` | LoRA dropout rate | 0.05 |
| `--target-modules` | Comma-separated list of target modules | q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj |
| `--fan-in-fan-out` | Set fan_in_fan_out for Conv1D | False |
| `--use-gradient-checkpointing` | Use gradient checkpointing | False |
| `--modules-to-save` | Modules to save in full precision | None |

The `--peft-type` parameter supports several methods:
- `lora`: Low-Rank Adaptation (default) - efficient parameter-saving technique
- `adalora`: Adaptive LoRA - dynamically adjusts ranks during training
- `prefix`: Prefix Tuning - adds trainable continuous prompts
- `prompt`: Prompt Tuning - adds trainable prompt vectors
- `ia3`: IA³ - scales activations with learned vectors
- `lokr`: LoKr - combines LoRA with Kronecker product
- `oft`: OFT - orthogonal fine-tuning approach

**AdaLoRA-specific parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--adalora-target-r` | Target rank for AdaLoRA | 8 |
| `--adalora-init-r` | Initial rank for AdaLoRA | 12 |
| `--adalora-tinit` | Initial step before sparsification begins | 200 |
| `--adalora-tfinal` | Final step when sparsification ends | 1000 |
| `--adalora-delta-t` | Steps between rank updates | 10 |
| `--adalora-beta1` | EMA hyperparameter | 0.85 |
| `--adalora-beta2` | EMA hyperparameter | 0.85 |

Example usage with different PEFT configurations:

```python
# Using LoRA with custom settings
python -m src.run --peft-type lora --lora-r 16 --lora-alpha 64 --target-modules "q_proj,k_proj,v_proj"

# Using AdaLoRA
python -m src.run --peft-type adalora --adalora-target-r 4 --adalora-init-r 8 --adalora-tfinal 2000

# Using Prefix Tuning
python -m src.run --peft-type prefix
```

For the complete list of parameters, run `python -m src.run --help`.

### Response-Only Training

Unsloth provides a feature called `train_on_responses_only` that allows you to focus the training specifically on the assistant's responses rather than the entire conversation. This can be more efficient for instruction tuning as it concentrates the learning on the output generation rather than the input understanding.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--train-on-responses-only` | Enable response-only training | False |
| `--instruction-token` | Token/prefix indicating start of instruction | `<|im_start|>user\n` |
| `--response-token` | Token/prefix indicating start of response | `<|im_start|>assistant\n` |
| `--instruction-token-id` | Token ID for instruction start (optional) | None |
| `--response-token-id` | Token ID for response start (optional) | None |

This mode works by identifying instruction and response segments in the training data and applying a special mask to focus the training loss calculation only on the assistant's responses.

Example usage for response-only training:

```python
# Basic response-only training with default tokens
python -m src.run --train-on-responses-only

# Custom token configuration for Qwen2.5 format
python -m src.run --train-on-responses-only --instruction-token "<|im_start|>user\n" --response-token "<|im_start|>assistant\n"

# Using token IDs instead of strings (if you know the exact token IDs in your tokenizer)
python -m src.run --train-on-responses-only --instruction-token-id 83769 --response-token-id 83942
```

For the complete list of parameters, run `python -m src.run --help`.

### Attention Implementation

The training script provides options for configuring the attention mechanism implementation. Different attention implementations can significantly affect training speed and memory usage. The framework supports multiple attention implementations:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--attention-implementation` | Type of attention implementation to use | `default` |
| `--use-flash-attention` | Use Flash Attention 2 if available (shortcut for setting attention-implementation=flash_attention_2) | False |
| `--force-attn-implementation` | Force the attention implementation even if not optimal for the hardware | False |

Available attention implementations:
- `default`: The default implementation provided by the model
- `flash_attention_2`: Flash Attention 2, which provides significant speedups but requires appropriate hardware/CUDA support
- `sdpa`: PyTorch's Scaled Dot Product Attention, which offers good performance on modern GPUs
- `eager`: Standard eager execution mode attention
- `xformers`: xFormers library's memory-efficient attention (requires xformers to be installed)

Example usage:

```bash
# Use Flash Attention 2 for faster training
python -m src.run --use-flash-attention

# Use SDPA implementation
python -m src.run --attention-implementation sdpa

# Force a specific implementation even if not optimal
python -m src.run --attention-implementation flash_attention_2 --force-attn-implementation
```

The system will automatically check for hardware compatibility with your chosen attention implementation and fall back to the default if needed (unless `--force-attn-implementation` is specified).

### Inference

You can use the trained model for inference:

```python
from model.qwen_handler import QwenModelHandler
from prompt_processors.prompt_creator import PromptCreator
from prompt_processors.response_parser import ResponseParser

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
│   ├── prompt_processors/
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
