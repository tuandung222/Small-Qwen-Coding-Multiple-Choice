#!/bin/bash

# =============================================================================
# Qwen2.5-Coder-1.5B-Instruct Fine-tuning Script with QLoRA Optimizations
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

# Remove any existing PID files
rm -f training_*.pid

# Function to kill existing training processes
kill_existing_processes() {
    echo "Checking for existing training processes..."
    pids=$(pgrep -f "python src/run.py")
    if [ ! -z "$pids" ]; then
        echo "Found existing training processes. Killing them..."
        kill -9 $pids
        sleep 2
    else
        echo "No existing training processes found."
    fi
}

# Reset training log file if it exists
if [ -f "training.log" ]; then
    echo "Resetting training.log file..."
    > training.log
    echo "$(date): Starting new training run with QLoRA and Lion 8-bit" > training.log
else
    echo "Creating new training.log file..."
    echo "$(date): Starting new training run with QLoRA and Lion 8-bit" > training.log
fi

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

# Safety checkpoint configuration
SAVE_STEPS=30
SAVE_TOTAL_LIMIT=5

# Add timestamp to experiment name for uniqueness
TIMESTAMP=$(date +"%m%d_%H%M")
EXPERIMENT_NAME="Qwen25_Coder_MCQ_QLoRA_${TIMESTAMP}"

# Kill any existing training processes
kill_existing_processes

# Create a unique process ID file
PID_FILE="training_${TIMESTAMP}.pid"

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
    --double-quant true \
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
    --gradient-checkpointing true \
    --use-flash-attention true \
    --attention-implementation "flash_attention_2" \
    --force-attn-implementation true \
    --train-on-responses-only \
    --instruction-token "<|im_start|>user\n" \
    --response-token "<|im_start|>assistant\n" \
    --dataloader-num-workers 4 \
    --dataloader-pin-memory true \
    --bf16 true > training.log 2>&1 &

# Save the process ID
echo $! > $PID_FILE

echo "QLoRA training started in background with PID: $(cat $PID_FILE)"
echo "Logs are being written to training.log"
echo "To monitor progress, use: tail -f training.log"
echo "To stop training, run: kill $(cat $PID_FILE)"