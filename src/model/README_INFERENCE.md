# Optimized Inference for Qwen Models

This module provides optimized inference utilities for faster model inference with Qwen models.

## Features

- `torch.inference_mode()` for faster inference
- Batched inference for better throughput
- Half-precision inference (fp16/bf16)
- Flash Attention for faster attention computation
- KV cache optimization
- Unsloth optimization with `FastLanguageModel.for_inference`

## Installation

Make sure you have the following dependencies installed:

```bash
pip install torch transformers peft
pip install unsloth  # Optional, for additional optimization
```

## Usage

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model.optimized_inference import OptimizedInference

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("unsloth/Qwen2.5-Coder-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-Coder-1.5B-Instruct")

# Create optimized inference
optimizer = OptimizedInference(
    model=model,
    tokenizer=tokenizer,
    precision="bf16",  # Options: "fp32", "fp16", "bf16"
    batch_size=4,
)

# Single prompt generation
prompt = "Write a Python function to calculate Fibonacci numbers."
generated_text = optimizer.generate(prompt, max_new_tokens=256)[0]
print(generated_text)

# Chat completion
messages = [
    {"role": "user", "content": "Write a Python function to calculate Fibonacci numbers."}
]
response = optimizer.create_chat_completion(messages, max_new_tokens=256)
print(response)
```

### Inference with LoRA Adapter

```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model.optimized_inference import OptimizedInference

# Load base model
model = AutoModelForCausalLM.from_pretrained("unsloth/Qwen2.5-Coder-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-Coder-1.5B-Instruct")

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "path/to/lora/adapter")

# Create optimized inference
optimizer = OptimizedInference(
    model=model,
    tokenizer=tokenizer,
    precision="bf16",
    batch_size=4,
)

# Generate text
response = optimizer.create_chat_completion([
    {"role": "user", "content": "Write a Python function to calculate Fibonacci numbers."}
])
print(response)
```

### Batch Inference

```python
from src.model.optimized_inference import OptimizedInference
from src.model.qwen_handler import QwenModelHandler

# Load model with QwenModelHandler
model_handler = QwenModelHandler(
    model_name="unsloth/Qwen2.5-Coder-1.5B-Instruct",
    device_map="auto",
)

# Create optimized inference
optimizer = OptimizedInference(
    model=model_handler.model,
    tokenizer=model_handler.tokenizer,
    precision="bf16",
    batch_size=8,  # Increase batch size for better throughput
)

# Process multiple prompts in batches
prompts = [
    "Write a Python function to calculate Fibonacci numbers.",
    "Explain the difference between lists and tuples in Python.",
    "How do I implement a binary search tree in Python?",
    "Write a Python function to sort a list using quicksort.",
]

# Generate responses for all prompts
responses = optimizer.generate(
    prompts,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
)

# Print results
for prompt, response in zip(prompts, responses):
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print("-" * 80)
```

### Benchmarking

```python
from src.model.optimized_inference import OptimizedInference
from src.model.qwen_handler import QwenModelHandler

# Load model
model_handler = QwenModelHandler(
    model_name="unsloth/Qwen2.5-Coder-1.5B-Instruct",
    device_map="auto",
)

# Create optimized inference
optimizer = OptimizedInference(
    model=model_handler.model,
    tokenizer=model_handler.tokenizer,
    precision="bf16",
    batch_size=1,
)

# Run benchmark
results = optimizer.benchmark(
    prompt="Write a Python function to calculate Fibonacci numbers.",
    max_new_tokens=100,
    num_runs=5,
)

# Print results
print(f"Average latency: {results['avg_latency']:.2f} seconds")
print(f"Average tokens per second: {results['avg_tokens_per_second']:.2f}")
```

## 4-bit and 8-bit Quantization

For even faster inference with reduced memory usage, you can use 4-bit or 8-bit quantization:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.model.optimized_inference import OptimizedInference

# 4-bit quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen2.5-Coder-1.5B-Instruct",
    quantization_config=quantization_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-Coder-1.5B-Instruct")

# Create optimized inference
# Note: For quantized models, use "fp32" as the precision
optimizer = OptimizedInference(
    model=model,
    tokenizer=tokenizer,
    precision="fp32",  # The model is already quantized
    batch_size=4,
)

# Generate text
response = optimizer.create_chat_completion([
    {"role": "user", "content": "Write a Python function to calculate Fibonacci numbers."}
])
print(response)
```

## Demo Script

You can also use the provided demo script:

```bash
# Basic usage
python src/inference_demo.py --prompt "Write a Python function to calculate Fibonacci numbers."

# With LoRA adapter
python src/inference_demo.py --model_path "unsloth/Qwen2.5-Coder-1.5B-Instruct" --adapter_path "path/to/lora/adapter"

# With 4-bit quantization
python src/inference_demo.py --precision "4bit"

# Run benchmark
python src/inference_demo.py --benchmark --num_runs 10

# Load prompt from file
python src/inference_demo.py --prompt_file "path/to/prompt.txt"
```

## Performance Tips

1. Use the highest precision your hardware supports: bf16 > fp16 > fp32
2. Increase batch size for better throughput when processing multiple prompts
3. Enable Flash Attention if your model supports it
4. Use 4-bit or 8-bit quantization for larger models to reduce memory usage
5. Set `device_map="auto"` for multi-GPU inference
6. Use `torch.compile()` for additional performance (requires PyTorch 2.0+) 