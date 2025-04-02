# Qwen-Coder-MCQ: Fine-tuning Qwen2.5 for Multiple-Choice Coding Questions

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</div>

This project provides a framework for fine-tuning **Qwen2.5-Coder-1.5B-Instruct** models on multiple-choice coding questions with structured reasoning. It uses LoRA (Low-Rank Adaptation) for efficient training and includes a comprehensive pipeline for data processing, training, and evaluation.

## 📑 Table of Contents

- [Features](#features)
  - [Parameter-Efficient Fine-Tuning](#parameter-efficient-fine-tuning)
  - [Optimized Training](#optimized-training)
  - [Advanced Optimizers and Schedulers](#advanced-optimizers-and-schedulers)
  - [Structured Reasoning](#structured-reasoning)
  - [Comprehensive Evaluation](#comprehensive-evaluation)
  - [Advanced Monitoring](#advanced-monitoring)
  - [HuggingFace Hub Integration](#huggingface-hub-integration)
  - [Development and Testing](#development-and-testing)
- [Advanced Features](#advanced-features)
  - [Parameter-Efficient Fine-Tuning (PEFT)](#parameter-efficient-fine-tuning-peft)
  - [Attention Implementations](#attention-implementations)
  - [Response-Only Training](#response-only-training)
  - [Optimizer and Scheduler Options](#optimizer-and-scheduler-options)
  - [Monitoring and Visualization](#monitoring-and-visualization)
  - [Teacher Synthesis](#teacher-synthesis)
- [Dataset](#dataset)
  - [Dataset Structure](#dataset-structure)
  - [Data Examples](#data-examples)
- [Command-Line Arguments](#command-line-arguments)
  - [Prompt Monitoring Arguments](#prompt-monitoring-arguments)
  - [Training Arguments](#training-arguments)
- [Callbacks](#callbacks)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Teacher Synthesis](#teacher-synthesis-1)

## ✨ Features

### Parameter-Efficient Fine-Tuning

- **LoRA (Low-Rank Adaptation)** with configurable parameters
- **AdaLoRA** with dynamic rank adjustment
- Support for multiple **PEFT methods** (prefix, prompt, ia3, lokr, oft)
- **Gradient checkpointing** for memory efficiency

### Optimized Training

- **Unsloth integration** for faster training and reduced memory usage
- Multiple **attention implementations** (Flash Attention 2, SDPA, xFormers)
- **Mixed precision training** (FP16/BF16)
- **Gradient accumulation** for effective batch size control

### Advanced Optimizers and Schedulers

- Multiple **optimizer options** (adamw_torch, adam8bit, pagedadam, lion, adafactor)
- Configurable **learning rate schedulers** (cosine, linear, polynomial, etc.)
- **Warmup strategies** with customizable ratios
- **Gradient clipping** and weight decay

### Structured Reasoning

- **YAML-format outputs** for clear reasoning steps
- Multiple **prompt templates** for different approaches
- **Teacher-reasoned training** methodology
- **Response-only training** option for focused learning

### Comprehensive Evaluation

- Multiple **evaluation metrics**
- **Validation strategies** with configurable frequency
- **Best model checkpointing**
- **Early stopping** with customizable patience

### Advanced Monitoring

#### Prompt Monitoring
- **Real-time display** of random training prompts
- **Token distribution analysis** and visualization
- **Prompt diversity tracking** with similarity metrics
- **Quality metrics** (length, complexity, readability)
- **Automatic prompt categorization**
- **Interactive prompt selection** and comparison
- **WandB integration** for prompt analytics
- **Configurable logging** frequency

#### Training Metrics
- **Learning rate tracking**
- **Model loading alerts**
- **GPU memory and gradient monitoring**
- **WandB integration** for experiment tracking

### HuggingFace Hub Integration

- **Automatic repository creation**
- **Configurable push strategies**
- Support for **private repositories**
- Multiple **save formats** (LoRA, merged 16bit, merged 4bit, GGUF)

### Development and Testing

- **Test modes** for rapid iteration
- **Debug sampling** for data inspection
- **Comprehensive logging**
- **Flexible configuration** via CLI

## 🚀 Advanced Features

### Parameter-Efficient Fine-Tuning (PEFT)

The framework supports multiple PEFT methods for efficient model adaptation:

| Method | Description |
|--------|-------------|
| **LoRA** | Low-Rank Adaptation with configurable rank, alpha, and dropout |
| **AdaLoRA** | Adaptive LoRA that dynamically adjusts ranks during training |
| **Prefix Tuning** | Adds trainable continuous prompts |
| **Prompt Tuning** | Adds trainable prompt vectors |
| **IA³** | Scales activations with learned vectors |
| **LoKr** | Combines LoRA with Kronecker product |
| **OFT** | Orthogonal fine-tuning approach |

### Attention Implementations

Multiple attention implementations are supported for optimal performance:

| Implementation | Description |
|----------------|-------------|
| **Flash Attention 2** | Significantly faster training with appropriate hardware |
| **SDPA** | PyTorch's Scaled Dot Product Attention |
| **xFormers** | Memory-efficient attention implementation |
| **Eager** | Standard eager execution mode |
| **Default** | Model's default implementation |

### Response-Only Training

Unsloth's response-only training feature allows focusing on assistant responses:

- Identifies instruction and response segments
- Applies special masking for focused learning
- Configurable instruction and response tokens
- Optional token ID specification

### Optimizer and Scheduler Options

Comprehensive training optimization options:

| Category | Options |
|----------|---------|
| **Optimizers** | adamw_torch, adamw_hf, adam8bit, pagedadam, lion, adafactor |
| **Schedulers** | cosine, linear, cosine_with_restarts, polynomial, constant, constant_with_warmup, inverse_sqrt |
| **Warmup** | Configurable warmup ratio and steps |
| **Regularization** | Weight decay and gradient clipping |

### Monitoring and Visualization

Advanced monitoring capabilities:

#### Prompt Monitoring
- **Real-time display** of random training prompts
- **Token distribution analysis** and visualization
- **Prompt diversity tracking** with similarity metrics
- **Quality metrics** (length, complexity, readability)
- **Automatic prompt categorization**
- **Interactive prompt selection** and comparison
- **WandB integration** for prompt analytics
- **Configurable logging** frequency

#### Training Metrics
- **Learning rate tracking**
- **Model loading alerts**
- **GPU memory and gradient monitoring**
- **WandB integration** for experiment tracking

### Teacher Synthesis

The project includes a teacher synthesis framework for generating high-quality explanations for multiple-choice questions. See [Teacher Synthesis Documentation](src/data_synthesis/README.md) for detailed information about:

- **Supported OpenAI models** (GPT-4o, GPT-4, GPT-3.5-turbo)
- **Generation parameters** and configuration
- **Concurrent processing** capabilities
- **Metrics tracking** and analysis
- **Output formats** and structure

## 📊 Dataset

The project uses a curated dataset of multiple-choice coding questions with structured reasoning. The dataset is published on HuggingFace at [tuandunghcmut/coding-mcq-reasoning](https://huggingface.co/datasets/tuandunghcmut/coding-mcq-reasoning).

### Dataset Structure

The dataset contains 3,549 selected coding multiple-choice questions derived from the CodeMMLU benchmark, enriched with detailed reasoning steps provided by a GPT-4o teacher model. Each example includes:

- **Task ID**: Unique identifier for each question
- **Question**: The coding problem or concept being tested
- **Choices**: Multiple choice answers (A, B, C, D, etc.)
- **Answer**: The correct option
- **Teacher Understanding**: Detailed breakdown of the problem statement
- **Teacher Analysis**: Systematic evaluation of each option
- **Teacher Reasoning**: Step-by-step logical process
- **Teacher Conclusion**: Final explanation of the correct answer
- **YAML String**: Structured format of the reasoning process

### Prompt Formats

The project uses two distinct YAML-formatted prompts: one for student reasoning during inference and another for teacher synthesis during training data generation.

#### Student Reasoning Format

This format is used during model inference, encouraging structured thinking without knowledge of the correct answer:

```yaml
Question: [question text]

Choices:
A. [choice 1]
B. [choice 2]
C. [choice 3]
D. [choice 4]

Think through this step-by-step:
- Understand what the question is asking
- Analyze each option carefully
- Reason about why each option might be correct or incorrect
- Select the most appropriate answer

Your response MUST be in YAML format:
understanding: |
  <your understanding of the question>
analysis: |
  <your analysis of each option>
reasoning: |
  <your reasoning about the correct answer>
conclusion: |
  <your final conclusion>
answer: <single letter A through D>
```

#### Teacher Synthesis Format

This format is used to generate high-quality training data, where the model acts as a teacher with knowledge of the correct answer:

```yaml
TASK: You are a teacher creating a concise, precise explanation for a multiple-choice question.

QUESTION:
[question text]

CHOICES:
A. [choice 1]
B. [choice 2]
C. [choice 3]
D. [choice 4]

CORRECT ANSWER: [correct_answer]

INSTRUCTIONS:
Create a focused explanation that demonstrates why [correct_answer] is correct
and why other options are incorrect. Be thorough but concise.

Your response MUST be in YAML format:
understanding: |
  <brief explanation of key concepts>
analysis: |
  <concise analysis of each option>
reasoning: |
  <focused reasoning for correct answer>
conclusion: |
  <brief summary>
answer: [correct_answer]
```

Key differences between the formats:
1. **Knowledge of Answer**: Student format encourages exploration, while teacher format focuses on explaining the known correct answer
2. **Focus**: Student format emphasizes step-by-step thinking, teacher format prioritizes concise, precise explanations
3. **Purpose**: Student format for inference, teacher format for generating training data
4. **Style**: Student format is exploratory, teacher format is authoritative and educational

The structured YAML format ensures:
- Consistent formatting across responses
- Easy parsing and validation
- Clear separation of reasoning components
- Systematic approach to problem-solving and explanation

### Data Examples

Here are some examples from the dataset:

#### Example 1: SQL Function Question

```
Task ID: k08183
Question: What does the SQL function "ROUND()" do?
Choices: ['Rounds a number to the nearest integer', 'Concatenates two or more strings', 'Converts a string to lowercase', 'Returns the length of a string']
Answer: A
Teacher Understanding: The question is asking about the purpose of the SQL function "ROUND()". This function is used in SQL to manipulate numerical data, specifically to adjust the precision of numbers.
Teacher Analysis:
- A. Rounds a number to the nearest integer: This is correct. The "ROUND()" function is used to round a numeric value to the nearest integer or to a specified number of decimal places.
- B. Concatenates two or more strings: This is incorrect. Concatenation of strings is typically done using the "CONCAT()" function in SQL.
- C. Converts a string to lowercase: This is incorrect. Converting a string to lowercase is done using the "LOWER()" function in SQL.
- D. Returns the length of a string: This is incorrect. The length of a string is determined using the "LENGTH()" function in SQL.
Teacher Reasoning: The "ROUND()" function is specifically designed to handle numerical values by rounding them to the nearest integer or specified decimal places, which aligns with option A. The other options describe functions that manipulate strings, not numbers.
Teacher Conclusion: Answer A is correct because the "ROUND()" function's primary purpose is to round numbers, which is distinct from the string operations described in the other options.
```

#### Example 2: Algorithm Problem

```
Task ID: k08183
Question: Given a sequence of rolls of a k-sided dice, what is the length of the shortest sequence that cannot be formed?
Choices: ['ans += k - len(seen) + 1', 'ans += 1', 'ans = min(ans + 1, k)', 'ans = ans + 1']
Answer: B
Teacher Understanding: The problem asks for the length of the shortest sequence that cannot be formed from the given rolls of a k-sided dice. The solution involves tracking unique rolls and incrementing a counter when all k numbers have been seen.
Teacher Analysis:
- A. This option incorrectly adjusts the answer based on the difference between k and the size of the set, which is unnecessary since the goal is to increment when all k numbers are seen.
- B. This option correctly increments the answer by 1 when all k numbers have been seen, indicating a complete sequence.
- C. This option uses the min function, which is unnecessary and incorrect because the answer should simply increment by 1 when all k numbers are seen.
- D. This option is similar to B but is redundant because it doesn't add any new logic beyond incrementing by 1.
Teacher Reasoning: The solution needs to increment the sequence count (ans) each time a complete set of k unique numbers is seen. Option B correctly increments the count by 1 when the set size equals k, which signifies that a complete sequence of k numbers has been formed and another sequence can start.
Teacher Conclusion: Answer B is correct because it directly and correctly increments the sequence count by 1 when all k numbers have been seen, aligning with the problem's requirement to find the shortest sequence that cannot be formed.
```

## 🛠️ Command-Line Arguments

The framework supports extensive configuration via command-line arguments:

### Prompt Monitoring Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--prompt-track-diversity` | Enable/disable prompt diversity tracking | True |
| `--prompt-track-quality` | Enable/disable prompt quality metrics | True |
| `--prompt-interactive` | Enable/disable interactive prompt selection mode | False |
| `--prompt-categorize` | Enable/disable automatic prompt categorization | True |
| `--prompt-comparison` | Enable/disable prompt comparison features | True |
| `--max-prompts-to-save` | Maximum number of prompts to save for analysis | 100 |

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--experiment-name` | Name for this experiment | auto-generated timestamp |
| `--source-model` | Base model to fine-tune | unsloth/Qwen2.5-Coder-1.5B-Instruct |
| `--destination-repo` | HF Hub repo for the model | tuandunghcmut/Qwen25_Coder_MultipleChoice_v3 |
| `--max-seq-length` | Maximum sequence length for input | 2048 |
| `--quantization` | Model quantization level | 4bit |
| `--epochs` | Number of training epochs | 3 |
| `--batch-size` | Per device batch size | 24 |
| `--grad-accum` | Gradient accumulation steps | 4 |
| `--learning-rate` | Learning rate | 2e-4 |
| `--warmup-ratio` | Proportion of steps for warmup | 0.1 |
| `--weight-decay` | Weight decay for optimizer | 0.01 |
| `--lora-r` | LoRA attention dimension | 8 |
| `--lora-alpha` | LoRA alpha parameter | 32 |
| `--push-strategy` | When to push to HF Hub | best |
| `--private` | Make the repository private | False |
| `--test-mode` | Use only 2 examples | False |
| `--test-training-mode` | Use only one batch of data | False |
| `--train-on-responses-only` | Enable response-only training | False |
| `--instruction-token` | Token/prefix indicating start of instruction | `<|im_start|>user\n` |
| `--response-token` | Token/prefix indicating start of response | `<|im_start|>assistant\n` |
| `--instruction-token-id` | Token ID for instruction start (optional) | None |
| `--response-token-id` | Token ID for response start (optional) | None |
| `--attention-implementation` | Type of attention implementation to use | default |
| `--use-flash-attention` | Use Flash Attention 2 if available | False |
| `--force-attn-implementation` | Force the attention implementation | False |

## 🔄 Callbacks

The training process includes several specialized callbacks for monitoring and optimization:

| Callback | Description |
|----------|-------------|
| **LRMonitorCallback** | Tracks learning rates and optimizer parameters during training |
| **PromptMonitorCallback** | Displays random training prompts after each logging step |
| **ModelLoadingAlertCallback** | Alerts when model loading method changes |
| **EarlyStoppingCallback** | Implements early stopping to prevent overfitting |
| **ValidationCallback** | Manages validation metrics and best model checkpointing |

## 🚀 Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- HuggingFace account with access token
- Weights & Biases account (optional, for experiment tracking)
- OpenAI API key (required for teacher synthesis)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/tuandung222/Small-Qwen-Coding-Multiple-Choice.git
cd Small-Qwen-Coding-Multiple-Choice
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:

You can set up environment variables in two ways:

#### a. Using a `.env` file (recommended):
```bash
# Copy the example .env file
cp .env.example .env

# Edit the .env file with your API keys
nano .env  # or use your preferred text editor
```

Required environment variables in `.env`:
```
HF_TOKEN=your_huggingface_token_here
WANDB_API_KEY=your_wandb_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Required for teacher synthesis
```

#### b. Using environment variables directly:
```bash
export HF_TOKEN=your_huggingface_token_here
export WANDB_API_KEY=your_wandb_api_key_here
export OPENAI_API_KEY=your_openai_api_key_here  # Required for teacher synthesis
```

> **Note**: The OpenAI API key is only required if you plan to use the teacher synthesis framework. For regular training without teacher synthesis, this key is optional.

## 📚 Usage

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

```bash
# Basic usage
python src/run.py

# Advanced usage with custom parameters
python src/run.py --experiment-name "my_experiment" --source-model "unsloth/Qwen2.5-Coder-1.5B-Instruct" --epochs 5 --batch-size 16 --learning-rate 1e-4
```

### Teacher Synthesis

To generate synthetic explanations for multiple-choice questions using OpenAI models:

```bash
# Basic usage
python src/data_synthesis/gpt4o_generated.py --model gpt-4o --data-path /path/to/dataset --api-key YOUR_API_KEY

# Advanced usage
python src/data_synthesis/gpt4o_generated.py --model gpt-4o --data-path /path/to/dataset --sample-size 100 --temperature 0.2 --max-tokens 2048 --concurrent-requests 5 --output-dir ./my_results
```

See [Teacher Synthesis Documentation](src/data_synthesis/README.md) for more details.
