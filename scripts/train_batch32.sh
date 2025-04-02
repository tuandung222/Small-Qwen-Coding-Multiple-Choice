#!/bin/bash
# train_batch32.sh
# Script to train Qwen model with batch size 32 and optimized settings

# Display help information
function show_help {
    echo "Usage: ./train_batch32.sh [OPTIONS]"
    echo ""
    echo "Train Qwen2.5-Coder model with batch size 32 and optimized settings"
    echo ""
    echo "Options:"
    echo "  -m, --model NAME       Source model name (default: unsloth/Qwen2.5-Coder-1.5B-Instruct)"
    echo "  -r, --repo NAME        Destination repo name (default: auto-generated)"
    echo "  -e, --epochs NUM       Number of training epochs (default: 3)"
    echo "  -l, --lr RATE          Learning rate (default: 2e-4)"
    echo "  -g, --grad-accum NUM   Gradient accumulation steps (default: 2)"
    echo "  -d, --dataset NAME     Dataset to use (default: tuandunghcmut/coding-mcq-reasoning)"
    echo "  -s, --seq-len NUM      Maximum sequence length (default: 2048)"
    echo "  -n, --name NAME        Experiment name (default: batch32_run)"
    echo "  -o, --output-root DIR  Root directory for experiment outputs (default: ./experiment_output)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Example:"
    echo "  ./train_batch32.sh --epochs 5 --lr 1e-4 --name my_experiment"
}

# Default values
MODEL="unsloth/Qwen2.5-Coder-1.5B-Instruct"
REPO=""
EPOCHS=3
LEARNING_RATE=2e-4
GRAD_ACCUM=2
DATASET="tuandunghcmut/coding-mcq-reasoning"
SEQ_LEN=2048
EXP_NAME="batch32_run"
OUTPUT_ROOT="./experiment_output"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -r|--repo)
            REPO="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -l|--lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -g|--grad-accum)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -s|--seq-len)
            SEQ_LEN="$2"
            shift 2
            ;;
        -n|--name)
            EXP_NAME="$2"
            shift 2
            ;;
        -o|--output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Generate timestamp for unique experiment directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}_${TIMESTAMP}"

# Create output directory with parents if needed
mkdir -p "$OUTPUT_DIR"

echo "=========================================================="
echo "Starting training with batch size 32"
echo "=========================================================="
echo "Experiment name:   $EXP_NAME"
echo "Source model:      $MODEL"
echo "Destination repo:  ${REPO:-auto-generated}"
echo "Epochs:            $EPOCHS"
echo "Learning rate:     $LEARNING_RATE"
echo "Gradient accum:    $GRAD_ACCUM"
echo "Effective batch:   $((32 * GRAD_ACCUM))"
echo "Dataset:           $DATASET"
echo "Sequence length:   $SEQ_LEN"
echo "Output directory:  $OUTPUT_DIR"
echo "=========================================================="

# Build command with optional arguments
CMD="python src/run.py \
  --source-model \"$MODEL\" \
  --batch-size 32 \
  --epochs $EPOCHS \
  --learning-rate $LEARNING_RATE \
  --grad-accum $GRAD_ACCUM \
  --dataset \"$DATASET\" \
  --max-seq-length $SEQ_LEN \
  --output-dir \"$OUTPUT_DIR\""

# Add destination repo if specified
if [ -n "$REPO" ]; then
  CMD="$CMD --destination-repo \"$REPO\""
fi

# Display the command
echo "Executing command:"
echo "$CMD"
echo "=========================================================="

# Execute the command
eval $CMD

# Check if training was successful
if [ $? -eq 0 ]; then
  echo "=========================================================="
  echo "Training completed successfully!"
  echo "Model saved to: $OUTPUT_DIR"
  echo "=========================================================="
else
  echo "=========================================================="
  echo "Training failed! Please check the logs for errors."
  echo "=========================================================="
  exit 1
fi
