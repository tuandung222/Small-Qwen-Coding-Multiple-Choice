# Qwen Training Framework

A comprehensive framework for training and fine-tuning Qwen language models with advanced features including LoRA, QLoRA, and efficient training optimizations.

## Features

- **Model Management**
  - Support for Qwen-1.5 and Qwen-2.5 models
  - Automatic model downloading and caching
  - HuggingFace Hub integration for model pushing
  - Efficient model loading and initialization

- **Training Optimizations**
  - LoRA and QLoRA support for efficient fine-tuning
  - Gradient checkpointing for memory efficiency
  - Mixed precision training (fp16/bf16)
  - Gradient accumulation for larger effective batch sizes
  - CPU offloading for training on limited GPU memory
  - Flash Attention 2 support for faster training

- **Advanced Training Features**
  - Early stopping with configurable patience
  - Learning rate monitoring and scheduling
  - Prompt monitoring and analysis
  - Comprehensive validation metrics
  - Response-only loss calculation for better evaluation
  - Reproducible training with fixed random seeds

- **Monitoring and Logging**
  - Weights & Biases integration
  - Training progress tracking
  - Resource utilization monitoring
  - Validation metrics logging
  - Model checkpointing

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/qwen-training.git
cd qwen-training

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python src/run.py \
    --model_name Qwen/Qwen1.5-7B-Chat \
    --dataset_name your_dataset \
    --output_dir ./outputs \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --seed 42
```

### Advanced Training with All Features

Here's a comprehensive example showcasing all available features:

```bash
#!/bin/bash

# Set environment variables for reproducibility
export PYTHONHASHSEED=42
export CUDA_LAUNCH_BLOCKING=1

# Run the training script with comprehensive features
python src/run.py \
    --experiment-name "Qwen25_Coder_MCQ_5Epochs" \
    --source-model "unsloth/Qwen2.5-Coder-1.5B-Instruct" \
    --destination-repo "your-username/model-name" \
    --epochs 5 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --grad-accum 4 \
    --warmup-ratio 0.1 \
    --weight-decay 0.01 \
    --max-seq-length 2048 \
    --quantization "4bit" \
    \
    --lora-r 8 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --peft-type "lora" \
    --target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    \
    --optimizer "adamw_torch" \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-epsilon 1e-8 \
    --max-grad-norm 1.0 \
    --optim-bits 8 \
    \
    --lr-scheduler "cosine" \
    --lr-scheduler-num-cycles 1 \
    --lr-scheduler-power 1.0 \
    \
    --early-stopping-patience 5 \
    --early-stopping-delta 0.01 \
    --validation-steps 50 \
    --metric-for-best "eval_loss" \
    --greater-is-better false \
    --validate-at-start true \
    \
    --prompt-template "teacher_reasoned" \
    --logging-steps 100 \
    --save-steps 500 \
    --save-total-limit 3 \
    --push-strategy "best" \
    --push-to-hub true \
    \
    --dataset "your-dataset-name" \
    --val-split 0.04 \
    --random-seed 42 \
    --output-dir "model_output" \
    \
    --use-flash-attention true \
    --attention-implementation "flash_attention_2" \
    --force-attn-implementation true \
    \
    --train-on-responses-only true \
    --instruction-token "<|im_start|>user\n" \
    --response-token "<|im_start|>assistant\n" \
    \
    --prompt-track-diversity true \
    --prompt-track-quality true \
    --prompt-categorize true \
    --prompt-comparison true \
    --max-prompts-to-save 100 \
    --debug-samples 3 \
    2>&1 | tee training.log
```

This comprehensive example includes:

1. **Environment Setup**
   - Fixed random seeds for reproducibility
   - CUDA launch blocking for better error tracking

2. **Training Configuration**
   - 5 epochs with batch size 32
   - 4-bit quantization
   - Gradient accumulation
   - Cosine learning rate schedule

3. **LoRA Settings**
   - Rank 8 with alpha 32
   - Comprehensive module targeting
   - Optimized dropout

4. **Optimization**
   - 8-bit AdamW optimizer
   - Gradient clipping
   - Early stopping
   - Regular validation

5. **Advanced Features**
   - Flash Attention 2
   - Response-only training
   - Prompt monitoring and analysis
   - Automatic model pushing

6. **Monitoring**
   - Regular logging
   - Checkpoint management
   - Debug samples
   - Comprehensive logging

Save this as `train.sh`, make it executable with `chmod +x train.sh`, and run with `./train.sh`.

## Configuration

### Model Configuration

- `--model_name`: HuggingFace model name or local path
- `--model_revision`: Specific model revision to use
- `--trust_remote_code`: Trust remote code when loading models
- `--use_safetensors`: Use safetensors for model loading

### Training Configuration

- `--num_train_epochs`: Number of training epochs
- `--per_device_train_batch_size`: Batch size per device
- `--gradient_accumulation_steps`: Number of gradient accumulation steps
- `--learning_rate`: Learning rate
- `--max_grad_norm`: Maximum gradient norm
- `--warmup_ratio`: Warmup ratio for learning rate scheduler
- `--lr_scheduler_type`: Learning rate scheduler type
- `--logging_steps`: Logging frequency
- `--save_steps`: Model saving frequency
- `--save_total_limit`: Maximum number of checkpoints to keep

### LoRA Configuration

- `--lora_r`: LoRA attention dimension
- `--lora_alpha`: LoRA alpha parameter
- `--lora_dropout`: LoRA dropout probability
- `--lora_target_modules`: Target modules for LoRA
- `--use_qlora`: Enable QLoRA training

### Optimization Configuration

- `--gradient_checkpointing`: Enable gradient checkpointing
- `--flash_attention_2`: Enable Flash Attention 2
- `--cpu_offload`: Enable CPU offloading
- `--bf16`: Enable bfloat16 training
- `--fp16`: Enable float16 training

### Validation Configuration

- `--validation_steps`: Validation frequency in steps
- `--validate_at_start`: Run validation before training
- `--metric_for_best`: Metric to use for model selection
- `--greater_is_better`: Whether greater metric values are better
- `--push_to_hub`: Push best model to HuggingFace Hub
- `--hub_model_id`: HuggingFace Hub model ID

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
