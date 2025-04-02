#!/bin/bash

# Set environment variables for reproducibility
export PYTHONHASHSEED=42
export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH=$PYTHONPATH:$(pwd)
export TOKENIZERS_PARALLELISM=false

# Install additional dependencies if needed
pip install -q wandb tqdm numpy pandas prettytable scikit-learn

# Reset training log file if it exists
if [ -f "training.log" ]; then
    echo "Resetting training.log file..."
    > training.log
    echo "$(date): Starting new training run" > training.log
else
    echo "Creating new training.log file..."
    echo "$(date): Starting new training run" > training.log
fi


# Add timestamp to experiment name for uniqueness
TIMESTAMP=$(date +"%m%d_%H%M")
EXPERIMENT_NAME="Qwen25_Coder_MCQ_5Epochs_${TIMESTAMP}"

# Run the training script with comprehensive features
python src/run.py \
    --experiment-name "${EXPERIMENT_NAME}" \
    --source-model "unsloth/Qwen2.5-Coder-1.5B-Instruct" \
    --destination-repo "tuandunghcmut/Qwen25_Coder_MultipleChoice_v3" \
    --epochs 5 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --grad-accum 1 \
    --warmup-ratio 0.15 \
    --weight-decay 0.02 \
    --max-seq-length 2048 \
    --quantization "4bit" \
    \
    --lora-r 16 \
    --lora-alpha 16 \
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
    --early-stopping-patience 7 \
    --early-stopping-delta 0.01 \
    --validation-steps 50 \
    --metric-for-best "eval_loss" \
    --validate-at-start \
    \
    --prompt-template "teacher_reasoned" \
    --logging-steps 2 \
    --save-steps 500 \
    --save-total-limit 3 \
    --push-strategy "best" \
    --push-to-hub \
    \
    --dataset "tuandunghcmut/coding-mcq-reasoning" \
    --val-split 0.04 \
    --random-seed 42 \
    --output-dir "model_output" \
    \
    --use-flash-attention \
    --attention-implementation "flash_attention_2" \
    --force-attn-implementation \
    \
    --train-on-responses-only \
    --instruction-token "<|im_start|>user\n" \
    --response-token "<|im_start|>assistant\n" \
    \
    --prompt-track-diversity \
    --prompt-track-quality \
    --prompt-categorize \
    --prompt-comparison \
    --max-prompts-to-save 200 \
    --debug-samples 3 \
    2>&1 | tee training.log &
