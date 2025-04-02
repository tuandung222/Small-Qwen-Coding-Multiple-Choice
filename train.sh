#!/bin/bash

# Set environment variables for reproducibility
export PYTHONHASHSEED=42
export CUDA_LAUNCH_BLOCKING=1

# Run the training script with comprehensive features
python src/run.py \
    --experiment-name "Qwen25_Coder_MCQ_5Epochs" \
    --source-model "unsloth/Qwen2.5-Coder-1.5B-Instruct" \
    --destination-repo "tuandunghcmut/Qwen25_Coder_MultipleChoice_v3" \
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
    --dataset "tuandunghcmut/coding-mcq-reasoning" \
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
