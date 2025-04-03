#!/bin/bash

# Set environment variables for reproducibility
export PYTHONHASHSEED=42
export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH=$PYTHONPATH:$(pwd)
export TOKENIZERS_PARALLELISM=false

# Install additional dependencies if needed
pip install -q wandb tqdm numpy pandas prettytable scikit-learn lion-pytorch

# Reset training log file if it exists
if [ -f "training.log" ]; then
    echo "Resetting training.log file..."
    > training.log
    echo "$(date): Starting new training run" > training.log
else
    echo "Creating new training.log file..."
    echo "$(date): Starting new training run" > training.log
fi

# Default values
SOURCE_MODEL="unsloth/Qwen2.5-Coder-1.5B-Instruct"
DESTINATION_REPO="tuandunghcmut/Qwen25_Coder_MultipleChoice_v3"
BATCH_SIZE=4
LEARNING_RATE=3e-5  # Adjusted for Lion
EPOCHS=3
WARMUP_STEPS=30
VALIDATION_STEPS=30
DEBUG_SAMPLES=3
MINIMAL_VALIDATING=true
MAX_VALIDATION_SAMPLES=60

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --source-model)
      SOURCE_MODEL="$2"
      shift 2
      ;;
    --destination-repo)
      DESTINATION_REPO="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --learning-rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --warmup-steps)
      WARMUP_STEPS="$2"
      shift 2
      ;;
    --validation-steps)
      VALIDATION_STEPS="$2"
      shift 2
      ;;
    --debug-samples)
      DEBUG_SAMPLES="$2"
      shift 2
      ;;
    --minimal-validating)
      MINIMAL_VALIDATING=true
      shift
      ;;
    --max-validation-samples)
      MAX_VALIDATION_SAMPLES="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Print configuration
echo "=== Training Configuration ==="
echo "Source model: $SOURCE_MODEL"
echo "Destination repo: $DESTINATION_REPO"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE (Lion optimizer)"
echo "Epochs: $EPOCHS"
echo "Warmup steps: $WARMUP_STEPS"
echo "Validation steps: $VALIDATION_STEPS"
echo "Debug samples: $DEBUG_SAMPLES"
echo "Minimal validating: $MINIMAL_VALIDATING (max $MAX_VALIDATION_SAMPLES samples)"
echo "==========================="

# Add timestamp to experiment name for uniqueness
TIMESTAMP=$(date +"%m%d_%H%M")
EXPERIMENT_NAME="Qwen25_Coder_MCQ_Lion_${TIMESTAMP}"

# Run the training script with Lion optimizer configuration
python src/run.py \
    --experiment-name "${EXPERIMENT_NAME}" \
    --source-model "$SOURCE_MODEL" \
    --destination-repo "$DESTINATION_REPO" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    \
    # Lion optimizer configuration
    --optimizer "lion" \
    --learning-rate 3e-5 \
    --weight-decay 0.1 \
    --lion-beta1 0.95 \
    --lion-beta2 0.98 \
    --max-grad-norm 1.0 \
    \
    # Learning rate scheduler
    --warmup-steps "$WARMUP_STEPS" \
    --lr-scheduler "cosine_with_warmup" \
    --lr-scheduler-num-cycles 1 \
    \
    # Validation configuration
    --validation-steps "$VALIDATION_STEPS" \
    --minimal-validating \
    --max-validation-samples "$MAX_VALIDATION_SAMPLES" \
    --validate-at-start \
    --metric-for-best "eval_loss" \
    --early-stopping-patience 3 \
    --early-stopping-delta 0.01 \
    \
    # LoRA configuration (adjusted for Lion)
    --lora-r 32 \
    --lora-alpha 64 \
    --lora-dropout 0.1 \
    --peft-type "lora" \
    --target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    \
    # Training monitoring
    --debug-samples "$DEBUG_SAMPLES" \
    --push-to-hub \
    --prompt-track-diversity \
    --prompt-track-quality \
    --prompt-categorize \
    --prompt-comparison \
    --max-prompts-to-save 100 \
    --logging-steps 10 \
    --save-steps 100 \
    --save-total-limit 3 \
    \
    # Model configuration
    --max-seq-length 2048 \
    --quantization "4bit" \
    --prompt-template "teacher_reasoned" \
    --save-total-limit 5 \
    --push-strategy "best" \
    \
    # Dataset configuration
    --dataset "tuandunghcmut/coding-mcq-reasoning" \
    --val-split 0.04 \
    --random-seed 42 \
    --output-dir "model_output" \
    \
    # Attention configuration
    --use-flash-attention \
    --attention-implementation "flash_attention_2" \
    --force-attn-implementation \
    \
    # Response-only training
    --train-on-responses-only \
    --instruction-token "<|im_start|>user\n" \
    --response-token "<|im_start|>assistant\n" \
    2>&1 | tee training.log &
