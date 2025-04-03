#!/bin/bash

# =============================================================================
# Qwen2.5-Coder-1.5B-Instruct Fine-tuning Script with QLoRA Optimizations
# =============================================================================
# This script configures and runs the fine-tuning process using QLoRA for:
# 1. Minimal Memory Usage (4-bit quantization)
# 2. High Quality Training
# 3. Efficient GPU Utilization
# 4. Stable Training Process
#
# QLoRA Optimizations:
# ------------------
# 1. Quantization:
#    - 4-bit NormalFloat quantization
#    - Double quantization for memory savings
#    - NF4 data type for better quality
#
# 2. Memory Management:
#    - Gradient checkpointing
#    - Paged optimizer states
#    - Efficient attention patterns
#
# 3. Training Efficiency:
#    - Optimized batch sizes
#    - Gradient accumulation
#    - Memory-aware processing
#
# =============================================================================

# Set environment variables for performance and stability
export PYTHONHASHSEED=42
export CUDA_LAUNCH_BLOCKING=0
export PYTHONPATH=$PYTHONPATH:$(pwd)
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# QLoRA-optimized default values with Lion 8-bit settings
SOURCE_MODEL="unsloth/Qwen2.5-Coder-1.5B-Instruct"
DESTINATION_REPO="tuandunghcmut/Qwen25_Coder_MultipleChoice_v4"
BATCH_SIZE=4
GRAD_ACCUM=8
LEARNING_RATE=5e-5
EPOCHS=3
WARMUP_STEPS=200
VALIDATION_STEPS=50
DEBUG_SAMPLES=3
MINIMAL_VALIDATING=true
MAX_VALIDATION_SAMPLES=60
SAVE_STEPS=30
SAVE_TOTAL_LIMIT=5

# Add timestamp to experiment name for uniqueness
TIMESTAMP=$(date +"%m%d_%H%M")
EXPERIMENT_NAME="Qwen25_Coder_MCQ_QLoRA_${TIMESTAMP}"

# Create output dirs
mkdir -p model_output

# Clean up any running processes (optional)
pkill -f "python src/run.py"

# Run the QLoRA-optimized training script with Lion 8-bit
python src/run.py \
    --experiment-name "${EXPERIMENT_NAME}" \
    --source-model "$SOURCE_MODEL" \
    --destination-repo "$DESTINATION_REPO" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --optimizer "lion_8bit" \
    --learning-rate "$LEARNING_RATE" \
    --weight-decay 0.1 \
    --lion-beta1 0.95 \
    --lion-beta2 0.98 \
    --max-grad-norm 0.3 \
    --warmup-steps "$WARMUP_STEPS" \
    --lr-scheduler "cosine" \
    --lr-scheduler-num-cycles 1 \
    --validation-steps "$VALIDATION_STEPS" \
    --minimal-validating \
    --max-validation-samples "$MAX_VALIDATION_SAMPLES" \
    --validate-at-start \
    --metric-for-best "eval_loss" \
    --early-stopping-patience 5 \
    --early-stopping-delta 0.01 \
    --save-steps "$SAVE_STEPS" \
    --save-total-limit "$SAVE_TOTAL_LIMIT" \
    --save-strategy "steps" \
    --quantization "4bit" \
    --double-quant \
    --quant-type "nf4" \
    --bits 4 \
    --lora-r 64 \
    --lora-alpha 16 \
    --lora-dropout 0.1 \
    --target-modules "q_proj,k_proj,v_proj,o_proj" \
    --debug-samples "$DEBUG_SAMPLES" \
    --push-to-hub \
    --prompt-track-diversity \
    --prompt-track-quality \
    --prompt-categorize \
    --prompt-comparison \
    --max-prompts-to-save 100 \
    --logging-steps 3 \
    --max-seq-length 2048 \
    --prompt-template "teacher_reasoned" \
    --push-strategy "best" \
    --dataset "tuandunghcmut/coding-mcq-reasoning" \
    --val-split 0.035 \
    --random-seed 42 \
    --output-dir "model_output" \
    --gradient-checkpointing \
    --use-flash-attention \
    --attention-implementation "flash_attention_2" \
    --force-attn-implementation \
    --train-on-responses-only \
    --instruction-token "<|im_start|>user\n" \
    --response-token "<|im_start|>assistant\n" \
    --dataloader-num-workers 4 \
    --dataloader-pin-memory \
    --bf16

echo "QLoRA training started"
echo "You can monitor the training progress in the console output"
