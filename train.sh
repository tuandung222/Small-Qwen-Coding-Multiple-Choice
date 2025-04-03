#!/bin/bash

# =============================================================================
# Qwen2.5-Coder-1.5B-Instruct Fine-tuning Script with LoRA
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

# Make sure we're in the correct directory and Python can find the modules
cd "$(dirname "$0")"
export PYTHONPATH="$PYTHONPATH:$(pwd)"
echo "Setting PYTHONPATH to include: $(pwd)"

# LoRA-optimized default values with Lion 8-bit settings
SOURCE_MODEL="unsloth/Qwen2.5-Coder-1.5B-Instruct"
DESTINATION_REPO="tuandunghcmut/Qwen25_Coder_MultipleChoice_v4"
BATCH_SIZE=12
GRAD_ACCUM=2
LEARNING_RATE=5e-5
EPOCHS=3
WARMUP_STEPS=20
VALIDATION_STEPS=3
DEBUG_SAMPLES=3
MINIMAL_VALIDATING=true
MAX_VALIDATION_SAMPLES=120
SAVE_STEPS=40
SAVE_TOTAL_LIMIT=5

# Data loading configuration
DATALOADER_NUM_WORKERS=2
DATALOADER_PIN_MEMORY=true
FULL_DETERMINISM=false
TORCH_COMPILE=false
USE_CPU=false

# Evaluation configuration
EVAL_STRATEGY="steps"
REPORT_TO="wandb"
REMOVE_UNUSED_COLUMNS=false

# Add timestamp to experiment name for uniqueness
TIMESTAMP=$(date +"%m%d_%H%M")
EXPERIMENT_NAME="Qwen25_Coder_MCQ_LoRA_${TIMESTAMP}"

# Kill all processes that match the pattern
pkill -f "python3 src/run.py"

# Create output dirs
mkdir -p model_output

# Remove train.log if it exists
rm -f train.log
# Create train.log if it doesn't exist
touch train.log

# Clean up any running processes (optional)
pkill -f "python3 src/run.py"

nohup python3 src/run.py \
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
    --adam-epsilon 1e-8 \
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
    --no-load-best-model-at-end \
    --lora-r 16 \
    --lora-alpha 16 \
    --lora-dropout 0.05 \
    --target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj" \
    --debug-samples "$DEBUG_SAMPLES" \
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS" \
    --dataloader-pin-memory \
    --full-determinism \
    --torch-compile \
    --use-cpu \
    --evaluation-strategy "$EVAL_STRATEGY" \
    --report-to "$REPORT_TO" \
    --remove-unused-columns \
    --push-to-hub \
    --logging-steps 1 \
    --max-seq-length 2048 \
    --prompt-template "teacher_reasoned" \
    --push-strategy "best" \
    --dataset "tuandunghcmut/coding-mcq-reasoning" \
    --val-split 0.05 \
    --random-seed 42 \
    --output-dir "model_output" \
    --use-gradient-checkpointing \
    --use-flash-attention \
    --attention-implementation "flash_attention_2" \
    --force-attn-implementation \
    --train-on-responses-only \
    --instruction-token "<|im_start|>user\n" \
    --response-token "<|im_start|>assistant\n" | tee -a train.log
