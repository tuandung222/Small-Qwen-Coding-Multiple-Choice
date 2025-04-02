# Qwen-Coder-MCQ: Fine-tuning Qwen2.5 for Multiple-Choice Coding Questions

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</div>

This project provides a framework for fine-tuning **Qwen2.5-Coder-1.5B-Instruct** models on multiple-choice coding questions with structured reasoning. It uses LoRA (Low-Rank Adaptation) for efficient training and includes a comprehensive pipeline for data processing, training, and evaluation.

## 📑 Table of Contents

- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Basic Usage](#basic-usage)
  - [Environment Setup](#environment-setup)
- [Prompt Formats](#prompt-formats)
  - [Student Reasoning Format](#student-reasoning-format)
  - [Teacher Synthesis Format](#teacher-synthesis-format)
- [Command-Line Interface](#command-line-interface)
  - [Training Arguments](#training-arguments)
  - [Synthesis Arguments](#synthesis-arguments)
  - [Monitoring Arguments](#monitoring-arguments)
- [Features](#features)
- [Dataset](#dataset)
- [Advanced Features](#advanced-features)
- [Examples and Showcase](#examples-and-showcase)

## 🏗️ Architecture

### Overall System Architecture

```mermaid
graph TB
    classDef pipeline fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef component fill:#fff3e0,stroke:#ff6f00,stroke-width:2px
    classDef output fill:#f1f8e9,stroke:#33691e,stroke-width:2px

    subgraph Data["Data Pipeline"]
        D1[("CodeMMLU<br/>Dataset")]:::component --> D2["Data Processing<br/>(YAML Format)"]:::component
        D2 --> D3["Teacher Synthesis<br/>(GPT-4o/3.5)"]:::component
        D3 --> D4[("Processed MCQ<br/>Dataset")]:::output
    end

    subgraph Training["Training Pipeline"]
        T1[("Qwen2.5<br/>Base Model")]:::component --> T2["LoRA Fine-tuning<br/>(8-bit Quantization)"]:::component
        D4 --> T2
        T2 --> T3["Fine-tuned Model<br/>(LoRA Weights)"]:::output
        T3 --> T4["Model Evaluation<br/>(Accuracy/Loss)"]:::component
    end

    subgraph Monitoring["Monitoring & Callbacks"]
        M1["WandB Logger<br/>(Real-time)"]:::component --> M2["Metrics Tracking<br/>(Loss/Accuracy)"]:::component
        M2 --> M3["Early Stopping<br/>(Patience: 3)"]:::component
        M2 --> M4["Validation<br/>(10% Split)"]:::component
        M2 --> M5["Prompt Monitor<br/>(Quality/Diversity)"]:::component
    end

    Data --> Training
    Training --> Monitoring

    class Data,Training,Monitoring pipeline
```

### Teacher Synthesis Pipeline

```mermaid
graph LR
    classDef process fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef data fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef output fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef monitor fill:#fce4ec,stroke:#c2185b,stroke-width:2px

    subgraph Input["Input Processing"]
        I1[("Raw MCQ<br/>Data")]:::data --> I2["Task Queue<br/>(Batch Size: 5)"]:::process
        I2 --> I3["Concurrent<br/>Processing"]:::process
    end

    subgraph Synthesis["Synthesis Process"]
        S1["GPT-4o/3.5<br/>API Calls"]:::process --> S2["YAML<br/>Generation"]:::process
        S2 --> S3["Answer<br/>Verification"]:::process
        S3 --> S4["Quality<br/>Check"]:::process
    end

    subgraph Output["Output & Monitoring"]
        O1["Save<br/>Explanations"]:::output --> O2["Calculate<br/>Metrics"]:::monitor
        O2 --> O3["Track<br/>Progress"]:::monitor
        O3 --> O4["WandB<br/>Logging"]:::monitor
    end

    Input --> Synthesis
    Synthesis --> Output

    style Input fill:#f8f9fa,stroke:#343a40,stroke-width:2px
    style Synthesis fill:#f8f9fa,stroke:#343a40,stroke-width:2px
    style Output fill:#f8f9fa,stroke:#343a40,stroke-width:2px
```

### Data Processing Pipeline

```mermaid
graph TB
    classDef input fill:#e8eaf6,stroke:#283593,stroke-width:2px
    classDef process fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef validation fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    classDef output fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px

    subgraph Input["Input Data Processing"]
        D1[("CodeMMLU<br/>Dataset")]:::input --> D2["Question<br/>Extraction"]:::process
        D2 --> D3["Choice<br/>Formatting"]:::process
        D3 --> D4["Token<br/>Encoding"]:::process
    end

    subgraph Processing["Data Enhancement"]
        P1["YAML<br/>Formatting"]:::process --> P2["Token<br/>Analysis"]:::process
        P2 --> P3["Quality<br/>Metrics"]:::process
        P3 --> P4["Diversity<br/>Tracking"]:::process
    end

    subgraph Validation["Quality Control"]
        V1["Answer<br/>Preservation"]:::validation --> V2["Format<br/>Verification"]:::validation
        V2 --> V3["Metrics<br/>Logging"]:::validation
        V3 --> V4["Error<br/>Handling"]:::validation
    end

    subgraph Output["Data Output"]
        O1["Save<br/>Dataset"]:::output --> O2["Generate<br/>Statistics"]:::output
        O2 --> O3["Create<br/>Visualizations"]:::output
    end

    Input --> Processing
    Processing --> Validation
    Validation --> Output

    style Input fill:#f8f9fa,stroke:#343a40,stroke-width:2px
    style Processing fill:#f8f9fa,stroke:#343a40,stroke-width:2px
    style Validation fill:#f8f9fa,stroke:#343a40,stroke-width:2px
    style Output fill:#f8f9fa,stroke:#343a40,stroke-width:2px
```

### Training Architecture

```mermaid
classDiagram
    class QwenTrainer {
        +model: PreTrainedModel
        +tokenizer: PreTrainedTokenizer
        +prompt_creator: PromptCreator
        +train(dataset, args)
        +evaluate(dataset)
        +save_checkpoint(path)
        +push_to_hub(repo_id)
        -setup_optimizer()
        -setup_scheduler()
    }

    class PromptCreator {
        +YAML_REASONING: str
        +TEACHER_REASONED: str
        +BASIC: str
        +create_inference_prompt(question, choices)
        +create_training_prompt(question, choices)
        -format_choices(choices)
        -validate_format(prompt)
    }

    class TeacherSynthesisFramework {
        +model_config: ModelConfig
        +output_dir: str
        +concurrent_requests: int
        +generate_synthetic_explanation()
        +process_dataset(dataset)
        +_calculate_metrics()
        -_save_results()
        -_handle_errors()
    }

    class Callbacks {
        +ValidationCallback
        +EarlyStoppingCallback
        +PromptMonitorCallback
        +LRMonitorCallback
        +ModelLoadingCallback
        -track_metrics()
        -log_to_wandb()
    }

    class ModelConfig {
        +name: str
        +temperature: float
        +max_tokens: int
        +api_key: str
        +validate()
        +to_dict()
    }

    QwenTrainer --> PromptCreator: uses
    QwenTrainer --> Callbacks: manages
    TeacherSynthesisFramework --> PromptCreator: uses
    TeacherSynthesisFramework --> ModelConfig: configures
    Callbacks --> QwenTrainer: monitors

    note for QwenTrainer "Main training orchestrator"
    note for PromptCreator "Handles prompt generation"
    note for TeacherSynthesisFramework "Manages synthetic data"
    note for Callbacks "Monitors training process"
```

### Monitoring System

```mermaid
graph LR
    classDef metrics fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef callbacks fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef viz fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px

    subgraph Metrics["Metrics Collection"]
        M1["Training Loss<br/>(Per Step)"]:::metrics --> M4["WandB<br/>Logger"]:::metrics
        M2["Validation<br/>Metrics"]:::metrics --> M4
        M3["Prompt<br/>Quality"]:::metrics --> M4
    end

    subgraph Callbacks["Training Control"]
        C1["Early<br/>Stopping"]:::callbacks --> C4["Training<br/>Control"]:::callbacks
        C2["Learning Rate<br/>Monitor"]:::callbacks --> C4
        C3["Prompt<br/>Monitor"]:::callbacks --> C4
    end

    subgraph Visualization["Analytics Dashboard"]
        V1["Loss<br/>Curves"]:::viz --> V4["WandB<br/>Dashboard"]:::viz
        V2["Prompt<br/>Statistics"]:::viz --> V4
        V3["Model<br/>Performance"]:::viz --> V4
    end

    Metrics --> Callbacks
    Callbacks --> Visualization

    style Metrics fill:#f8f9fa,stroke:#343a40,stroke-width:2px
    style Callbacks fill:#f8f9fa,stroke:#343a40,stroke-width:2px
    style Visualization fill:#f8f9fa,stroke:#343a40,stroke-width:2px
```

## 🚀 Quick Start

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

### Environment Setup

Set up environment variables in either of two ways:

1. Using a `.env` file (recommended):
```bash
# Copy the example .env file
cp .env.example .env

# Edit the .env file with your API keys
nano .env
```

Required variables in `.env`:
```
HF_TOKEN=your_huggingface_token_here
WANDB_API_KEY=your_wandb_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Required for teacher synthesis
```

2. Using environment variables directly:
```bash
export HF_TOKEN=your_huggingface_token_here
export WANDB_API_KEY=your_wandb_api_key_here
export OPENAI_API_KEY=your_openai_api_key_here
```

### Basic Usage

1. **Training the Model**:
```bash
# Basic training
python src/run.py

# Advanced training with custom parameters
python src/run.py \
  --experiment-name "my_experiment" \
  --source-model "unsloth/Qwen2.5-Coder-1.5B-Instruct" \
  --epochs 5 \
  --batch-size 16 \
  --learning-rate 1e-4
```

2. **Generating Synthetic Explanations**:
```bash
# Basic synthesis
python src/data_synthesis/gpt4o_generated.py \
  --model gpt-4o \
  --data-path /path/to/dataset

# Advanced synthesis with options
python src/data_synthesis/gpt4o_generated.py \
  --model gpt-4o \
  --data-path /path/to/dataset \
  --sample-size 100 \
  --temperature 0.2 \
  --max-tokens 2048 \
  --concurrent-requests 5
```

## 📝 Prompt Formats

The project uses two distinct YAML-formatted prompts: one for student reasoning during inference and another for teacher synthesis during training data generation.

### Student Reasoning Format

Used during model inference to encourage structured thinking:

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

### Teacher Synthesis Format

Used for generating high-quality training data:

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

Key differences between formats:
1. **Knowledge of Answer**: Student format encourages exploration, teacher format explains known answer
2. **Focus**: Student format emphasizes step-by-step thinking, teacher format prioritizes conciseness
3. **Purpose**: Student format for inference, teacher format for generating training data
4. **Style**: Student format is exploratory, teacher format is authoritative

## ⚙️ Command-Line Interface

### Training Arguments

```bash
python src/run.py [arguments]
```

Key training arguments:
| Argument | Description | Default |
|----------|-------------|---------|
| `--experiment-name` | Name for this experiment | auto-generated |
| `--source-model` | Base model to fine-tune | unsloth/Qwen2.5-Coder-1.5B-Instruct |
| `--epochs` | Number of training epochs | 3 |
| `--batch-size` | Per device batch size | 24 |
| `--learning-rate` | Learning rate | 2e-4 |
| `--lora-r` | LoRA attention dimension | 8 |
| `--lora-alpha` | LoRA alpha parameter | 32 |

### Synthesis Arguments

```bash
python src/data_synthesis/gpt4o_generated.py [arguments]
```

Key synthesis arguments:
| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | OpenAI model to use | gpt-4o |
| `--data-path` | Path to dataset | ./data/train |
| `--sample-size` | Number of examples | None (all) |
| `--temperature` | Generation temperature | 0.2 |
| `--concurrent-requests` | Parallel requests | 5 |

### Monitoring Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--prompt-track-diversity` | Track prompt diversity | True |
| `--prompt-track-quality` | Track quality metrics | True |
| `--prompt-interactive` | Interactive selection | False |
| `--max-prompts-to-save` | Max prompts to save | 100 |

## 🎯 Features

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

## 📊 Dataset

The project uses a curated dataset of multiple-choice coding questions with structured reasoning, published at [tuandunghcmut/coding-mcq-reasoning](https://huggingface.co/datasets/tuandunghcmut/coding-mcq-reasoning).

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

```yaml
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

```yaml
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

## 🎯 Examples and Showcase

### Example 1: SQL Function Question

```yaml
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

### Example 2: Algorithm Problem

```yaml
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
